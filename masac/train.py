import torch
import torch.nn.functional as F
import vmas
import actor
import critic
from buffer import ReplayBuffer

from torch.utils.tensorboard import SummaryWriter
import os

import utils
import config
import logging


def get_permuted_env_random_numbers(env_random_numbers, number_agents, num_envs):
    """
    Permute the random numbers for each agent in the environment.
    """
    permutation_indices = torch.zeros(number_agents, number_agents, dtype=torch.long, device=device)
    for i in range(number_agents):
        other_agents = sorted([j for j in range(number_agents) if j != i])
        permutation_indices[i] = torch.tensor([i] + other_agents)
    expanded_rand = env_random_numbers.unsqueeze(1).expand(-1, number_agents, -1)
    permuted_rand = torch.gather(expanded_rand, dim=2, index=permutation_indices.unsqueeze(0).expand(num_envs, -1, -1))
    return permuted_rand


class Trainer:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(**config.BASIC_CONFIG)
        self._setup_done = False
        
        # Model parameters
        self.actor = None
        self.critic = None
        self.target_critic = None
        self.actor_optimizer = None
        self.critic_optimizer = None
        self.alpha_optimizer = None
        self.alpha = None

        # Training parameters
        self.buffer = None
        self.buffer_device = None
        self.env = None
        self.global_step = 0
        self.update_step = 0
        self.best_reward = None
        self.device = None

        # Logging parameters
        self.writer = None

        # Internal parameters for ease
        self._obs_dim = None
        self._action_dim = None
    
    def do_setup(self):
        if self._setup_done:
            self.logger.warning("Setup already done. Skipping setup.")
            return
        
        self._setup_done = True

        # Set up all devices

        if self.config.DEVICE is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.DEVICE)
        
        self.logger.info("Using device: %s", self.device)

        if self.config.BUFFER_DEVICE is None:
            self.buffer_device = self.device
        else:
            self.buffer_device = torch.device(self.config.BUFFER_DEVICE)
        
        self.logger.info("Buffer device: %s", self.buffer_device)
        
        # Set up Buffer

        self.buffer = ReplayBuffer(
            self.config.BUFFER_SIZE,
            self.config.NUMBER_AGENTS,
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
            self.buffer_device
        )

        # Set up environment

        self.env = vmas.make_env(
            scenario_name=self.config.ENV_NAME,
            num_agents=self.config.NUMBER_AGENTS,
            num_envs=self.config.NUM_ENVS,
            continuous_action=self.config.ENV_CONTINUOUS_ACTION,
            max_episode_length=self.config.MAX_EPISODE_LENGTH,
            seed=self.config.SEED,
            device=self.buffer_device,
            terminated_truncated=True,
        )
        self.logger.info("""
            Environment created with the following parameters:
            Scenario Name: %s
            Number of Agents: %d
            Number of Environments: %d
            Continuous Action: %s
            Max Episode Length: %d
            Seed: %d
        """, self.config.ENV_NAME, self.config.NUMBER_AGENTS, self.config.NUM_ENVS, self.config.ENV_CONTINUOUS_ACTION, self.config.MAX_EPISODE_LENGTH, self.config.SEED)

        self._obs_dim = self.env.observation_space.shape[0]
        self._action_dim = self.env.action_space.shape[0]

        # Set up actor and critic
        self.actor_cls = getattr(actor, self.config.ACTOR_MODULE)
        self.critic_cls = getattr(critic, self.config.CRITIC_MODULE)

        self.actor = self.actor_cls(
            self.config.NUMBER_AGENTS,
            agent_dim=self._action_dim,
            landmark_dim=self._action_dim,
            hidden_dim=self.config.ACTOR_HIDDEN_DIM
        ).to(self.device)

        qvalue_config = {
            "device": self.device,
            "n_agents": self.config.NUMBER_AGENTS,
            "observation_dim_per_agent": self._obs_dim,
            "action_dim_per_agent": self._action_dim,
        }

        self.critic = self.critic_cls(qvalue_config).to(self.device)
        self.target_critic = self.critic_cls(qvalue_config).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_critic.eval()

        self.alpha = torch.tensor(self.config.INITIAL_ALPHA, device=self.device, requires_grad=True)

        # Set up Optimizers

        self.actor_optimizer = getattr(torch.optim, self.config.ACTOR_OPTIMIZER)(
            self.actor.parameters(),
            lr=self.config.ACTOR_LR,
            **self.config.ACTOR_OPTIMIZER_PARAMS
        )
        self.critic_optimizer = getattr(torch.optim, self.config.CRITIC_OPTIMIZER)(
            self.critic.parameters(),
            lr=self.config.CRITIC_LR,
            **self.config.CRITIC_OPTIMIZER_PARAMS
        )
        self.alpha_optimizer = getattr(torch.optim, self.config.ALPHA_OPTIMIZER)(
            [self.alpha],
            lr=self.config.ALPHA_LR,
            **self.config.ALPHA_OPTIMIZER_PARAMS
        )

        # Set up TensorBoard writer
        self.writer = SummaryWriter(
            log_dir=self.config.LOG_DIR,
        )
    
    def save_checkpoint(self):
        if not self._setup_done:
            self.logger.warning("Setup not done. Running setup before saving checkpoint.")
            self.do_setup()

        checkpoint_dir = self.config.CHECKPOINT_DIR
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        # Remove old checkpoints if number of checkpoints exceeds limit
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
        if len(checkpoint_files) >= self.config.MAX_CHECKPOINTS:
            oldest_checkpoint = sorted(checkpoint_files, key=lambda x: int(x.split('_')[2].split('.')[0]))[0]
            os.remove(os.path.join(checkpoint_dir, oldest_checkpoint))
            self.logger.info("Removed old checkpoint: %s", oldest_checkpoint)

        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_step_{self.global_step}.pt')
        self.logger.info("Saving checkpoint to %s", checkpoint_path)

        utils.save_checkpoint(
            checkpoint_dir,
            self.actor,
            self.critic,
            self.target_critic,
            self.actor_optimizer,
            self.critic_optimizer,
            self.alpha,
            self.alpha_optimizer,
            self.buffer,
            self.global_step,
            self.update_step,
            best_reward=self.best_reward
        )

    
    def load_from_checkpoint(self):
        if not self._setup_done:
            self.logger.warning("Setup not done. Running setup before loading checkpoint.")
            self.do_setup()

        checkpoint_dir = self.config.CHECKPOINT_DIR
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint directory {checkpoint_dir} does not exist.")

        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
        if not checkpoint_files:
            raise FileNotFoundError("No checkpoint files found in the directory.")
        # Load the latest checkpoint
        latest_checkpoint = sorted(checkpoint_files, key=lambda x: int(x.split('_')[2].split('.')[0]), reverse=True)[0]
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)

        self.logger.info("Loading checkpoint from %s on device %s", checkpoint_path, self.device)

        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)


        self.global_step, self.update_step, self.best_reward = utils.load_checkpoint(
            checkpoint,
            self.config.SEED,
            self.actor,
            self.critic,
            self.target_critic,
            self.actor_optimizer,
            self.critic_optimizer,
            self.alpha,
            self.alpha_optimizer,
            self.buffer,
            self.device
        )
        del checkpoint
        self.logger.info("Checkpoint loaded successfully.")
        self.logger.info("Global step: %d, Update step: %d, Best reward: %s", self.global_step, self.update_step, self.best_reward)
    
    def calculate_actor_pass(self, obs, rand_nums=None):
        obs = obs.view(-1, self._obs_dim)
        # When sampling next actions, also need to sample random numbers
        if rand_nums is None:
            rand_nums = torch.rand(
                obs.shape[0], self.config.NUMBER_AGENTS, device=self.device
            )
        actions, log_probs = self.actor(obs, rand_nums)
        actions = actions.view(-1, self.config.NUMBER_AGENTS, self._action_dim)
        log_probs = log_probs.view(-1, self.config.NUMBER_AGENTS)
        return actions, log_probs
    
    def compute_target_values(self, next_obs, rewards, dones):
        with torch.no_grad():
            next_actions, next_log_probs = self.calculate_actor_pass(next_obs)
            target_q1, target_q2 = self.target_critic(next_obs, next_actions)
            target_v = torch.min(target_q1, target_q2).view(-1)
            target_v -= self.alpha * next_log_probs.sum(dim=-1)
            target_values = rewards + (1 - dones) * self.config.GAMMA * target_v
        return target_values
    
    def compute_critic_loss(self, obs, actions, target_values):
        q1, q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(q1.view(-1), target_values) + F.mse_loss(q2.view(-1), target_values)
        return critic_loss
    
    def compute_actor_alpha_loss(self, obs, rand_nums):
        actions, log_probs = self.calculate_actor_pass(obs, rand_nums)
        # Compute the Q-values for the actions taken
        q1, q2 = self.critic(obs, actions)
        min_q = torch.min(q1, q2).view(-1, 1)
        # Compute the actor loss
        actor_loss = (self.alpha *log_probs - min_q).sum(dim=-1).mean()
        # Detach log_probs to avoid gradient flow
        log_probs = log_probs.detach()
        # Compute the alpha loss
        alpha_loss = -(self.alpha.log() * (log_probs - self.config.TARGET_ENTROPY)).mean()
        return actor_loss, alpha_loss

        
    def batch_update(self):        
        # Sample a batch from the buffer
        obs, actions, rewards, next_obs, dones, rand_nums = self.buffer.sample(self.config.BATCH_SIZE, self.device)
        rand_nums = rand_nums.view(-1, self.config.NUMBER_AGENTS)
        
        # Reshape the batch for actor pass

        # Update counter
        self.update_step += 1
    
    def collect_experience(self, obs):
        # Generate random numbers for the environment
        rand_nums = torch.rand(
            self.config.NUM_ENVS, self.config.NUMBER_AGENTS, device=self.device
        )
        rand_nums = get_permuted_env_random_numbers(rand_nums, self.config.NUMBER_AGENTS, self.config.NUM_ENVS)
        rand_nums = rand_nums.view(-1, self.config.NUMBER_AGENTS)

        obs = obs.view(-1, self._obs_dim)

        actions, log_probs = self.actor(obs, rand_nums)
        actions = actions.view(self.config.NUMBER_AGENTS, self.config.NUM_ENVS, self._action_dim)

        obs, rewards, terminated, truncated, _ = self.env.step(actions)

        obs = torch.stack(obs, dim=0)


if __name__ == "__main__":
    
    # Set the random seed for reproducibility
    utils.set_seed(config.SEED)

