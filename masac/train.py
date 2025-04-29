import torch
import torch.nn.functional as F
import vmas
import actor
import critic
from buffer import ReplayBuffer

from torch.utils.tensorboard import SummaryWriter
import os

import utils
import logging

import tqdm


def get_permuted_env_random_numbers(env_random_numbers, number_agents, num_envs, device):
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
        
        # Set up environment

        self.env = vmas.make_env(
            scenario=self.config.ENV_NAME,
            n_agents=self.config.NUMBER_AGENTS,
            num_envs=self.config.NUM_ENVS,
            continuous_actions=self.config.ENV_CONTINUOUS_ACTION,
            max_steps=self.config.MAX_EPISODE_LENGTH,
            seed=self.config.SEED,
            device=self.device,
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

        self._obs_dim = self.env.observation_space[0].shape[0]
        self._action_dim = self.env.action_space[0].shape[0]

        # Set up Buffer

        self.buffer = ReplayBuffer(
            self.config.BUFFER_SIZE,
            self._obs_dim,
            self._action_dim,
            self.config.NUMBER_AGENTS,
            self.buffer_device
        )

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
            self.config.SEED
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


        self.global_step, self.update_step = utils.load_checkpoint(
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
        self.logger.info("Global step: %d, Update step: %d", self.global_step, self.update_step)
    
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
        actor_loss = (self.alpha * log_probs - min_q).sum(dim=-1).mean()
        # Detach log_probs to avoid gradient flow
        log_probs = log_probs.clone().detach()
        # Compute the alpha loss
        alpha_loss = -(self.alpha.log() * (log_probs + self.config.TARGET_ENTROPY)).mean()
        return actor_loss, alpha_loss

        
    def batch_update(self):        
        # Sample a batch from the buffer
        obs, actions, rewards, next_obs, dones, rand_nums = self.buffer.sample(self.config.BATCH_SIZE, self.device)
        rand_nums = rand_nums.view(-1, self.config.NUMBER_AGENTS)

        # Compute target values
        target_values = self.compute_target_values(next_obs, rewards, dones)
        # Compute critic loss
        critic_loss = self.compute_critic_loss(obs, actions, target_values)
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        # Compute actor and alpha loss if updating actor and alpha
        if self.update_step % self.config.UPDATE_ACTOR_EVERY_CRITIC == 0:
            # Calculate actor loss and alpha loss
            actor_loss, alpha_loss = self.compute_actor_alpha_loss(obs, rand_nums)
            # Sum up the loss for common backpropagation
            total_loss = actor_loss + alpha_loss
            # Update actor and alpha
            self.actor_optimizer.zero_grad()
            self.alpha_optimizer.zero_grad()
            total_loss.backward()
            self.actor_optimizer.step()
            self.alpha_optimizer.step()

            # Clip alpha value
            with torch.no_grad():
                self.alpha.clamp_(self.config.ALPHA_MIN, self.config.ALPHA_MAX)

            actor_loss = actor_loss.item()
            alpha_loss = alpha_loss.item()
        else:
            actor_loss = None
            alpha_loss = None
        
        critic_loss = critic_loss.item()
        
        # Soft update the target critic
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.config.TAU * param.data + (1.0 - self.config.TAU) * target_param.data)
            
        # Update counter
        self.update_step += 1

        return critic_loss, actor_loss, alpha_loss, target_values.mean().item()
    
    def collect_experience(self, obs):
        # Generate random numbers for the environment
        rand_nums = torch.rand(
            self.config.NUM_ENVS, self.config.NUMBER_AGENTS, device=self.device
        )
        rand_nums = get_permuted_env_random_numbers(rand_nums, self.config.NUMBER_AGENTS, self.config.NUM_ENVS, self.device)
        rand_nums = rand_nums.view(-1, self.config.NUMBER_AGENTS)

        obs = obs.view(-1, self._obs_dim)

        actions, log_probs = self.actor(obs, rand_nums)
        actions = actions.view(self.config.NUMBER_AGENTS, self.config.NUM_ENVS, self._action_dim)

        next_obs, rewards, terminated, truncated, _ = self.env.step(actions)

        next_obs = torch.stack(next_obs, dim=1).view(
            self.config.NUMBER_AGENTS, self.config.NUM_ENVS, -1
        ).transpose(1,0).to(self.device)

        dones = torch.logical_or(terminated, truncated).float().to(self.device)

        rand_nums = rand_nums.view(
            self.config.NUMBER_AGENTS, self.config.NUM_ENVS, -1
        ).transpose(1,0)

        actions = actions.transpose(1,0)

        # Store the experience in the buffer
        for i in range(self.config.NUM_ENVS):
            self.buffer.add(
                obs[i],
                actions[i],
                rewards[0][i],
                next_obs[i],
                dones[i],
                rand_nums[i]
            )
        
        # Update the observation
        obs = next_obs.transpose(1,0)

        # Reset done environments
        all_obs = None
        for i in range(self.config.NUM_ENVS):
            if dones[i]:
                all_obs = self.env.reset_at(i)
        if all_obs is not None:
            obs = torch.stack(all_obs, dim=1).view(
                self.config.NUMBER_AGENTS, self.config.NUM_ENVS, -1
            ).transpose(1,0).to(self.device).contiguous()

        # Update the global step
        self.global_step += self.config.NUM_ENVS

        return obs, rewards, terminated, truncated, log_probs.mean().item()
    
    def train(self):
        self.do_setup()
        
        if self.config.RESUME_FROM_CHECKPOINT:
            self.logger.info("Resuming from checkpoint.")
            self.load_from_checkpoint()

        obs = self.env.reset()
        obs = torch.stack(obs, dim=1).view(
            self.config.NUMBER_AGENTS, self.config.NUM_ENVS, -1
        ).transpose(1,0).to(self.device).contiguous()

        # Setup tqdm for progress bar
        pbar = tqdm.tqdm(total=self.config.NUM_TIMESTEPS, desc="Training", unit="step")
        pbar.update(self.global_step)

        # Set everything to eval mode
        self.actor.eval()
        self.critic.eval()

        # Main training loop
        while self.global_step < self.config.NUM_TIMESTEPS:
            with torch.no_grad():
                # Collect experience
                obs, rewards, terminated, truncated, logprobs = self.collect_experience(obs)
                # Log the rewards
                rewards = torch.stack(rewards, dim=1)
                self.writer.add_scalar("Values/Reward", rewards.mean().item(), self.global_step)
                # Log logporbs
                self.writer.add_scalar("Values/LogProb", logprobs, self.global_step)
                # Log the success rate (# env terminated / # envs terminated + # envs truncated)
                dones = torch.logical_or(terminated, truncated).float()
                success_rate = terminated.sum() / ((terminated + truncated).sum() + 1e-6)
                self.writer.add_scalar("Values/Success Rate", success_rate.item(), self.global_step)

            if self.global_step >= self.config.UPDATE_START and self.global_step % self.config.UPDATE_EVERY == 0:
                # Set the actor and critic to training mode
                self.actor.train()
                self.critic.train()
                # Update the actor and critic
                for _ in range(self.config.UPDATE_TIMES):
                    critic_loss, actor_loss, alpha_loss, target_value = self.batch_update()
                    # Log the losses
                    self.writer.add_scalar("Loss/Critic", critic_loss, self.global_step)
                    if actor_loss is not None:
                        self.writer.add_scalar("Loss/Actor", actor_loss, self.global_step)
                    if alpha_loss is not None:
                        self.writer.add_scalar("Loss/Alpha", alpha_loss, self.global_step)
                    # Log the target value
                    self.writer.add_scalar("Values/Target Value", target_value, self.global_step)
                    # Log the alpha value
                    self.writer.add_scalar("Values/alpha", self.alpha.item(), self.global_step)
            
                    # Save the model checkpoint
                    if self.update_step % self.config.CHECKPOINT_INTERVAL_UPDATE == 0:
                        self.save_checkpoint()
                        self.logger.info("Checkpoint saved at step %d", self.global_step)
                    
                    # Set the actor and critic to eval mode
                    self.actor.eval()
                    self.critic.eval()
            
            # Update the progress bar
            pbar.update(self.config.NUM_ENVS)

if __name__ == "__main__":
    import config
    # Set the random seed for reproducibility
    utils.set_seed(config.SEED)

    # Create the trainer
    trainer = Trainer(config)
    # Start training
    trainer.train()
    # Close the environment
    trainer.env.close()
    # Close the TensorBoard writer
    trainer.writer.close()
    # Close the logger
    logging.shutdown()