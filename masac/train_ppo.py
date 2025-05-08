import torch
import torch.nn as nn
import torch.nn.functional as F
import vmas
from actor import PPOAgentPolicy
import utils
import os
import logging
from torch.utils.tensorboard import SummaryWriter
from buffer import PPORolloutBuffer

import config
from tqdm import tqdm

def get_permuted_env_random_numbers(env_random_numbers, number_agents, num_envs, device):
    """
    Permute the random numbers for each agent in the environment.
    Same as in original train.py
    """
    permutation_indices = torch.zeros(number_agents, number_agents, dtype=torch.long, device=device)
    for i in range(number_agents):
        other_agents = sorted([j for j in range(number_agents) if j != i])
        permutation_indices[i] = torch.tensor([i] + other_agents)
    expanded_rand = env_random_numbers.unsqueeze(1).expand(-1, number_agents, -1)
    permuted_rand = torch.gather(expanded_rand, dim=2, index=permutation_indices.unsqueeze(0).expand(num_envs, -1, -1))
    return permuted_rand


class PPOTrainer:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(**config.BASIC_CONFIG)
        self._setup_done = False
        
        self.actor = None
        self.actor_optimizer = None

        self.buffer = None
        self.buffer_device = None
        self.env = None
        self.global_step = 0
        self.update_step = 0
        self.device = None

        self.writer = None

        self._obs_dim = None
        self._action_dim = None
    
    def do_setup(self):
        if self._setup_done:
            self.logger.warning("Setup already done. Skipping setup.")
            return
        
        self._setup_done = True

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

        self.buffer = PPORolloutBuffer(
            self.config.ROLLOUT_STEPS,
            self._obs_dim,
            self._action_dim,
            self.config.NUMBER_AGENTS,
            self.buffer_device,
            gamma=self.config.GAMMA,
            gae_lambda=self.config.GAE_LAMBDA
        )

        self.actor = PPOAgentPolicy(
            self.config.NUMBER_AGENTS,
            agent_dim=self._action_dim,
            landmark_dim=self._action_dim,
            hidden_dim=self.config.ACTOR_HIDDEN_DIM
        ).to(self.device)

        self.actor_optimizer = getattr(torch.optim, self.config.ACTOR_OPTIMIZER)(
            self.actor.parameters(),
            lr=self.config.ACTOR_LR,
            **self.config.ACTOR_OPTIMIZER_PARAMS
        )

        if hasattr(self.config, 'LR_SCHEDULER') and self.config.LR_SCHEDULER:
            self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.actor_optimizer, 
                start_factor=1.0, 
                end_factor=0.1, 
                total_iters=self.config.NUM_TIMESTEPS//self.config.NUM_ENVS
            )
        else:
            self.lr_scheduler = None

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

        # Save the checkpoint
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'global_step': self.global_step,
            'update_step': self.update_step,
            'seed': self.config.SEED,
        }
        
        torch.save(checkpoint, checkpoint_path)
    
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
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Restore model state
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        self.update_step = checkpoint['update_step']
        
        # Check seed matching
        if checkpoint['seed'] != self.config.SEED:
            self.logger.warning("Checkpoint seed %d does not match the current seed %d.", 
                               checkpoint['seed'], self.config.SEED)
        
        del checkpoint
        self.logger.info("Checkpoint loaded successfully.")
        self.logger.info("Global step: %d, Update step: %d", self.global_step, self.update_step)
    
    def collect_rollouts(self):
        """
        Collect experience rollouts for PPO training
        """
        self.actor.eval()
        
        # Get initial observations
        obs = self.env.reset()
        obs = torch.stack(obs, dim=1).view(
            self.config.NUMBER_AGENTS, self.config.NUM_ENVS, -1
        ).transpose(1, 0).to(self.device).contiguous()
        
        self.buffer.clear()
        
        total_steps = 0
        
        # Start collecting rollouts
        with torch.no_grad():
            while total_steps < self.config.ROLLOUT_STEPS:
                rand_nums = torch.rand(
                    self.config.NUM_ENVS, self.config.NUMBER_AGENTS, device=self.device
                )
                rand_nums = get_permuted_env_random_numbers(
                    rand_nums, self.config.NUMBER_AGENTS, self.config.NUM_ENVS, self.device
                )
                rand_nums_flat = rand_nums.view(-1, self.config.NUMBER_AGENTS)
                
                obs_flat = obs.view(-1, self._obs_dim)
                
                actions, log_probs, values = self.actor(obs_flat, rand_nums_flat)
                
                actions_env = actions.view(self.config.NUMBER_AGENTS, self.config.NUM_ENVS, self._action_dim)
                
                next_obs, rewards, terminated, truncated, _ = self.env.step(actions_env)
                
                # Calculate sum of rewards across agents
                reward_sum = sum(rewards)
                
                next_obs = torch.stack(next_obs, dim=1).view(
                    self.config.NUMBER_AGENTS, self.config.NUM_ENVS, -1
                ).transpose(1, 0).to(self.device).contiguous()
                
                dones = torch.logical_or(terminated, truncated).float().to(self.device)
                
                # Reshape actions and values for storage
                actions_shaped = actions.view(self.config.NUM_ENVS, self.config.NUMBER_AGENTS, self._action_dim)
                log_probs_shaped = log_probs.view(self.config.NUM_ENVS, self.config.NUMBER_AGENTS)
                values_shaped = values.view(self.config.NUM_ENVS, self.config.NUMBER_AGENTS)
                
                # Store transition in buffer
                for i in range(self.config.NUM_ENVS):
                    self.buffer.add(
                        obs[i],
                        actions_shaped[i],
                        log_probs_shaped[i],
                        values_shaped[i],
                        reward_sum[i] * self.config.REWARD_SCALE,
                        dones[i],
                        rand_nums[i]
                    )
                    
                    # Track episode returns and lengths
                    if dones[i]:
                        # Reset this environment
                        env_obs = self.env.reset_at(i)
                        if env_obs is not None:
                            env_obs_tensor = torch.stack(env_obs, dim=1).view(
                                self.config.NUMBER_AGENTS, 1, -1
                            ).transpose(1, 0).to(self.device).contiguous()
                            # Update observation for this environment
                            obs[i] = env_obs_tensor[0]
                
                # Update observation
                obs = next_obs
                
                # Update counters
                total_steps += self.config.NUM_ENVS
                self.global_step += self.config.NUM_ENVS
        
        with torch.no_grad():
            rand_nums = torch.rand(
                self.config.NUM_ENVS, self.config.NUMBER_AGENTS, device=self.device
            )
            rand_nums = get_permuted_env_random_numbers(
                rand_nums, self.config.NUMBER_AGENTS, self.config.NUM_ENVS, self.device
            )
            rand_nums_flat = rand_nums.view(-1, self.config.NUMBER_AGENTS)
            
            obs_flat = obs.view(-1, self._obs_dim)
            _, _, last_values = self.actor(obs_flat, rand_nums_flat)
            last_values = last_values.view(self.config.NUM_ENVS, self.config.NUMBER_AGENTS)
        
        self.buffer.finalize_buffer(last_values)
        
        return total_steps
    
    def update_policy(self):
        """
        Update policy using PPO algorithm
        """
        self.actor.train()
        
        # Track statistics
        mean_policy_loss = 0
        mean_value_loss = 0
        mean_entropy = 0
        update_count = 0
        
        for epoch in range(self.config.PPO_EPOCHS):
            # Get batches from buffer
            for batch_data in self.buffer.get_batches(self.config.BATCH_SIZE, normalize_advantages=True):
                obs_batch, actions_batch, old_log_probs_batch, old_values_batch, returns_batch, advantages_batch, random_numbers_batch = batch_data
                
                batch_size = obs_batch.shape[0]
                obs_flat = obs_batch.view(-1, self._obs_dim)
                actions_flat = actions_batch.view(-1, self._action_dim)
                random_numbers_flat = random_numbers_batch.view(-1, self.config.NUMBER_AGENTS)
                
                new_log_probs, new_values, entropy = self.actor.evaluate_actions(
                    obs_flat, actions_flat, random_numbers_flat
                )
                
                new_log_probs = new_log_probs.view(batch_size, self.config.NUMBER_AGENTS)
                new_values = new_values.view(batch_size, self.config.NUMBER_AGENTS)
                entropy = entropy.view(batch_size, self.config.NUMBER_AGENTS)
                
                # Calculate ratios for PPO
                ratios = torch.exp(new_log_probs - old_log_probs_batch)
                
                # Calculate surrogate losses
                surr1 = ratios * advantages_batch
                surr2 = torch.clamp(ratios, 1.0 - self.config.PPO_CLIP_PARAM, 1.0 + self.config.PPO_CLIP_PARAM) * advantages_batch
                
                # Calculate policy loss
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Calculate value loss
                if self.config.PPO_USE_CLIP_VALUE:
                    value_pred_clipped = old_values_batch + torch.clamp(
                        new_values - old_values_batch, 
                        -self.config.PPO_CLIP_PARAM, 
                        self.config.PPO_CLIP_PARAM
                    )
                    value_loss_clipped = (returns_batch - value_pred_clipped).pow(2)
                    value_loss_unclipped = (returns_batch - new_values).pow(2)
                    value_loss = 0.5 * torch.max(value_loss_clipped, value_loss_unclipped).mean()
                else:
                    value_loss = 0.5 * F.mse_loss(returns_batch, new_values)
                
                # Calculate entropy loss
                entropy_loss = entropy.mean()
                
                # Total loss
                loss = (
                    policy_loss 
                    + self.config.PPO_VALUE_COEF * value_loss 
                    - self.config.PPO_ENTROPY_COEF * entropy_loss
                )
                
                self.actor_optimizer.zero_grad()
                loss.backward()
                
                if self.config.PPO_MAX_GRAD_NORM > 0:
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.PPO_MAX_GRAD_NORM)
                
                self.actor_optimizer.step()
                
                # Track statistics
                mean_policy_loss += policy_loss.item()
                mean_value_loss += value_loss.item()
                mean_entropy += entropy_loss.item()
                update_count += 1
                self.update_step += 1
        
        # Update learning rate if scheduler is used
        if self.lr_scheduler:
            self.lr_scheduler.step()
            current_lr = self.lr_scheduler.get_last_lr()[0]
            self.writer.add_scalar("Training/LearningRate", current_lr, self.global_step)
        
        # Calculate averages
        if update_count > 0:
            mean_policy_loss /= update_count
            mean_value_loss /= update_count
            mean_entropy /= update_count
        
        return mean_policy_loss, mean_value_loss, mean_entropy
    
    def train(self):
        self.do_setup()
        
        if self.config.RESUME_FROM_CHECKPOINT:
            self.logger.info("Resuming from checkpoint.")
            try:
                self.load_from_checkpoint()
            except FileNotFoundError as e:
                self.logger.warning(f"Could not load checkpoint: {e}")
                self.logger.info("Starting training from scratch.")
        
        # Setup tqdm for progress bar
        pbar = tqdm(total=self.config.NUM_TIMESTEPS, desc="Training", unit="step")
        pbar.update(self.global_step)
        
        # Main training loop
        while self.global_step < self.config.NUM_TIMESTEPS:
            # Collect rollouts
            self.logger.info(f"Collecting rollouts (Step {self.global_step})...")
            steps_collected = self.collect_rollouts()
            
            # Update policy using PPO
            self.logger.info(f"Updating policy (Step {self.global_step})...")
            policy_loss, value_loss, entropy = self.update_policy()
            
            # Log statistics
            self.writer.add_scalar("Loss/Policy", policy_loss, self.global_step)
            self.writer.add_scalar("Loss/Value", value_loss, self.global_step)
            self.writer.add_scalar("Loss/Entropy", entropy, self.global_step)
            
            # Save checkpoint
            if self.global_step % self.config.CHECKPOINT_INTERVAL_UPDATE == 0:
                self.save_checkpoint()
                self.logger.info(f"Checkpoint saved at step {self.global_step}")
            
            # Update progress bar
            pbar.update(steps_collected)
            
            # Log training progress
            self.logger.info(f"Step: {self.global_step}/{self.config.NUM_TIMESTEPS} | "
                            f"Policy Loss: {policy_loss:.4f} | Value Loss: {value_loss:.4f} | "
                            f"Entropy: {entropy:.4f}")
        
        # Save final checkpoint
        self.save_checkpoint()
        self.logger.info("Training completed.")
        
        # Close progress bar
        pbar.close()


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(**config.BASIC_CONFIG)
    logger = logging.getLogger(__name__)
    
    # Set random seed for reproducibility
    utils.set_seed(config.SEED)
    logger.info(f"Random seed set to {config.SEED}")
    
    # Create the PPO trainer
    trainer = PPOTrainer(config)
    logger.info("PPO trainer created")
    
    # Start training
    logger.info("Starting PPO training...")
    trainer.train()
    
    # Save the final checkpoint
    trainer.save_checkpoint()
    logger.info("Final checkpoint saved")
    
    # Close the TensorBoard writer
    trainer.writer.close()
    
    # Close the logger
    logging.shutdown()
    
    print("Training completed!")