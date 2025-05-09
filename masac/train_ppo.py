import torch
import torch.nn as nn
import torch.nn.functional as F
import vmas
from actor import RandomAgentPolicy
import utils
import os
import logging
from torch.utils.tensorboard import SummaryWriter
from buffer import PPORolloutBuffer
import numpy as np
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
        
        # Model parameters
        self.actor = None
        self.actor_optimizer = None

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

        # Set up devices
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

        # Set up Buffer for PPO
        self.buffer = PPORolloutBuffer(
            self.config.ROLLOUT_STEPS,
            self._obs_dim,
            self._action_dim,
            self.config.NUMBER_AGENTS,
            self.buffer_device,
            gamma=self.config.GAMMA,
            gae_lambda=self.config.GAE_LAMBDA
        )

        # Set up the actor-critic network
        self.actor = RandomAgentPolicy(
            self.config.NUMBER_AGENTS,
            agent_dim=self._action_dim,
            landmark_dim=self._action_dim,
            hidden_dim=self.config.ACTOR_HIDDEN_DIM
        ).to(self.device)

        # Set up Optimizer
        self.actor_optimizer = getattr(torch.optim, self.config.ACTOR_OPTIMIZER)(
            self.actor.parameters(),
            lr=self.config.ACTOR_LR,
            **self.config.ACTOR_OPTIMIZER_PARAMS
        )

        # Set up learning rate scheduler if needed
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
        Collect experience rollouts for PPO training with enhanced debugging
        """
        self.logger.info("Starting rollout collection...")
        
        # Set actor to evaluation mode during rollout collection
        self.actor.eval()
        
        # Reset episode returns and lengths tracking
        episode_returns = []
        episode_lengths = []
        
        # Get initial observations
        obs = self.env.reset()
        
        # Debug - print initial observations
        self.logger.info(f"Initial obs shape: {len(obs)}, First agent obs shape: {obs[0].shape}")
        self.logger.info(f"Sample initial obs values: {obs[0][0, :5]}")  # Print first 5 values
        
        obs = torch.stack(obs, dim=1).view(
            self.config.NUMBER_AGENTS, self.config.NUM_ENVS, -1
        ).transpose(1, 0).to(self.device).contiguous()
        
        # Clear the buffer for new rollouts
        self.buffer.clear()
        
        # Track total steps collected
        total_steps = 0
        
        # Track raw rewards (before scaling) for debugging
        raw_rewards = []
        scaled_rewards = []
        
        # Start collecting rollouts
        self.logger.info("Beginning rollout loop...")
        with torch.no_grad():
            while total_steps < self.config.ROLLOUT_STEPS:
                # Generate random numbers for the environment
                rand_nums = torch.rand(
                    self.config.NUM_ENVS, self.config.NUMBER_AGENTS, device=self.device
                )
                rand_nums = get_permuted_env_random_numbers(
                    rand_nums, self.config.NUMBER_AGENTS, self.config.NUM_ENVS, self.device
                )
                rand_nums_flat = rand_nums.view(-1, self.config.NUMBER_AGENTS)
                
                # Flatten observations for the actor
                obs_flat = obs.view(-1, self._obs_dim)
                
                # Get actions, log probs, and values from the actor
                actions, log_probs, values = self.actor(obs_flat, rand_nums_flat)
                
                # Debug - occasionally print action and value statistics
                if total_steps % 500 == 0 or total_steps == 0:
                    self.logger.info(f"Step {total_steps} action stats - Mean: {actions.mean().item():.6f}, "
                                f"Min: {actions.min().item():.6f}, Max: {actions.max().item():.6f}")
                    self.logger.info(f"Step {total_steps} value stats - Mean: {values.mean().item():.6f}, "
                                f"Min: {values.min().item():.6f}, Max: {values.max().item():.6f}")
                
                # Reshape for environment step
                actions_env = actions.view(self.config.NUMBER_AGENTS, self.config.NUM_ENVS, self._action_dim)
                
                # Take a step in the environment
                next_obs, rewards, terminated, truncated, info = self.env.step(actions_env)
                
                # Debug - check rewards
                if total_steps % 500 == 0 or total_steps == 0:
                    reward_list = [r.mean().item() for r in rewards]
                    self.logger.info(f"Step {total_steps} per-agent rewards: {reward_list}")
                
                # Calculate sum of rewards across agents
                reward_sum = sum(rewards)
                
                # Store raw rewards for debugging
                raw_rewards.extend(reward_sum.cpu().numpy())
                scaled_rewards.extend((reward_sum * self.config.REWARD_SCALE).cpu().numpy())
                
                # Process observations
                next_obs = torch.stack(next_obs, dim=1).view(
                    self.config.NUMBER_AGENTS, self.config.NUM_ENVS, -1
                ).transpose(1, 0).to(self.device).contiguous()
                
                # Calculate done flags
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
                    
                
                # Update observation
                obs = next_obs
                
                # Update counters
                total_steps += self.config.NUM_ENVS
                self.global_step += self.config.NUM_ENVS
        
        # Print reward statistics
        if raw_rewards:
            self.logger.info(f"Rollout reward stats - Raw: mean={np.mean(raw_rewards):.6f}, "
                        f"min={np.min(raw_rewards):.6f}, max={np.max(raw_rewards):.6f}")
            self.logger.info(f"Rollout reward stats - Scaled: mean={np.mean(scaled_rewards):.6f}, "
                        f"min={np.min(scaled_rewards):.6f}, max={np.max(scaled_rewards):.6f}")
                        
        # Finalize the buffer with last values
        # Get last values for incomplete episodes
        self.logger.info("Computing final values for incomplete episodes...")
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
            
            self.logger.info(f"Final values - Mean: {last_values.mean().item():.6f}, "
                        f"Min: {last_values.min().item():.6f}, Max: {last_values.max().item():.6f}")
        
        # Finalize buffer with computed last values
        self.buffer.finalize_buffer(last_values)
        
        self.logger.info(f"Rollout collection completed. Total steps: {total_steps}")
        return total_steps
    
    def update_policy(self):
        """Update policy using PPO algorithm with enhanced debugging"""
        # Set actor to training mode
        self.actor.train()
        
        # Track statistics
        mean_policy_loss = 0
        mean_value_loss = 0
        mean_entropy = 0
        mean_approx_kl = 0
        mean_clipfrac = 0
        update_count = 0
        
        # Print buffer stats
        if self.buffer.size > 0:
            returns = self.buffer.returns_buf[:self.buffer.size]
            values = self.buffer.values_buf[:self.buffer.size]
            advantages = self.buffer.advantages_buf[:self.buffer.size]
            log_probs = self.buffer.old_log_probs_buf[:self.buffer.size]
            rewards = self.buffer.rewards_buf[:self.buffer.size]
            
            self.logger.info(f"Buffer stats before update:")
            self.logger.info(f"  Size: {self.buffer.size}")
            self.logger.info(f"  Rewards: mean={rewards.mean().item():.6f}, std={rewards.std().item():.6f}, "
                        f"min={rewards.min().item():.6f}, max={rewards.max().item():.6f}")
            self.logger.info(f"  Returns: mean={returns.mean().item():.6f}, std={returns.std().item():.6f}, "
                        f"min={returns.min().item():.6f}, max={returns.max().item():.6f}")
            self.logger.info(f"  Values: mean={values.mean().item():.6f}, std={values.std().item():.6f}, "
                        f"min={values.min().item():.6f}, max={values.max().item():.6f}")
            self.logger.info(f"  Advantages: mean={advantages.mean().item():.6f}, std={advantages.std().item():.6f}, "
                        f"min={advantages.min().item():.6f}, max={advantages.max().item():.6f}")
            self.logger.info(f"  Log probs: mean={log_probs.mean().item():.6f}, std={log_probs.std().item():.6f}, "
                        f"min={log_probs.min().item():.6f}, max={log_probs.max().item():.6f}")
        
        # Add a check for advantages - if they're all zero, print a warning
        if self.buffer.advantages_buf[:self.buffer.size].abs().sum().item() < 1e-6:
            self.logger.warning("ALL ADVANTAGES ARE EFFECTIVELY ZERO! Training will not progress.")
            
        # Perform multiple epochs of updates
        for epoch in range(self.config.PPO_EPOCHS):
            self.logger.info(f"Starting PPO update epoch {epoch+1}/{self.config.PPO_EPOCHS}")
            epoch_policy_loss = 0
            epoch_value_loss = 0
            epoch_entropy = 0
            epoch_update_count = 0
            
            # Get batches from buffer
            for batch_idx, batch_data in enumerate(self.buffer.get_batches(self.config.BATCH_SIZE, normalize_advantages=True)):
                obs_batch, actions_batch, old_log_probs_batch, old_values_batch, returns_batch, advantages_batch, random_numbers_batch = batch_data
                
                # Debug info - print stats for a few batches
                if batch_idx == 0 or batch_idx == 5:
                    self.logger.info(f"Batch {batch_idx} - Advantages: mean={advantages_batch.mean().item():.6f}, "
                                f"std={advantages_batch.std().item():.6f}, min={advantages_batch.min().item():.6f}, "
                                f"max={advantages_batch.max().item():.6f}")
                
                # Flatten batch data for processing
                batch_size = obs_batch.shape[0]
                obs_flat = obs_batch.view(-1, self._obs_dim)
                actions_flat = actions_batch.view(-1, self._action_dim)
                random_numbers_flat = random_numbers_batch.view(-1, self.config.NUMBER_AGENTS)
                
                # Get new log probs, values, and entropy
                new_log_probs, new_values, entropy = self.actor.evaluate_actions(
                    obs_flat, actions_flat, random_numbers_flat
                )
                
                # Reshape to match batch dimensions
                new_log_probs = new_log_probs.view(batch_size, self.config.NUMBER_AGENTS)
                new_values = new_values.view(batch_size, self.config.NUMBER_AGENTS)
                entropy = entropy.view(batch_size, self.config.NUMBER_AGENTS)
                
                # Calculate approximate KL divergence for monitoring
                with torch.no_grad():
                    log_ratio = new_log_probs - old_log_probs_batch
                    approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean().item()
                    mean_approx_kl += approx_kl
                
                # Calculate ratios for PPO
                ratios = torch.exp(log_ratio)
                
                # Calculate surrogate losses
                surr1 = ratios * advantages_batch
                surr2 = torch.clamp(ratios, 1.0 - self.config.PPO_CLIP_PARAM, 1.0 + self.config.PPO_CLIP_PARAM) * advantages_batch
                
                # Calculate clipping fraction for monitoring
                with torch.no_grad():
                    clipfracs = ((ratios - 1.0).abs() > self.config.PPO_CLIP_PARAM).float().mean().item()
                    mean_clipfrac += clipfracs
                
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
                
                # Print loss components for the first batch
                if batch_idx == 0:
                    self.logger.info(f"Loss components - Policy: {policy_loss.item():.6f}, "
                                f"Value: {value_loss.item():.6f}, Entropy: {entropy_loss.item():.6f}")
                    self.logger.info(f"Policy metrics - KL: {approx_kl:.6f}, Clip fraction: {clipfracs:.6f}")
                    self.logger.info(f"Ratio stats - Mean: {ratios.mean().item():.6f}, Min: {ratios.min().item():.6f}, "
                                f"Max: {ratios.max().item():.6f}")
                
                # Update actor-critic network
                self.actor_optimizer.zero_grad()
                loss.backward()
                
                # Debug gradient norms
                if batch_idx == 0:
                    grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.actor.parameters() if p.grad is not None) ** 0.5
                    self.logger.info(f"Gradient norm: {grad_norm:.6f}")
                    
                    # Check for any NaN gradients
                    has_nan = False
                    for name, param in self.actor.named_parameters():
                        if param.grad is not None and torch.isnan(param.grad).any():
                            has_nan = True
                            self.logger.error(f"NaN gradient detected in {name}")
                    
                    if has_nan:
                        self.logger.error("NaN gradients detected! This will disrupt training.")
                
                # Optional gradient clipping
                if self.config.PPO_MAX_GRAD_NORM > 0:
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.PPO_MAX_GRAD_NORM)
                
                self.actor_optimizer.step()
                
                # Track statistics
                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()
                epoch_entropy += entropy_loss.item()
                epoch_update_count += 1
                
                mean_policy_loss += policy_loss.item()
                mean_value_loss += value_loss.item()
                mean_entropy += entropy_loss.item()
                update_count += 1
                self.update_step += 1
            
            # Print epoch stats
            if epoch_update_count > 0:
                self.logger.info(f"Epoch {epoch+1} stats - Policy loss: {epoch_policy_loss/epoch_update_count:.6f}, "
                            f"Value loss: {epoch_value_loss/epoch_update_count:.6f}, "
                            f"Entropy: {epoch_entropy/epoch_update_count:.6f}")
        
        # Update learning rate if scheduler is used
        if self.lr_scheduler:
            old_lr = self.lr_scheduler.get_last_lr()[0]
            self.lr_scheduler.step()
            new_lr = self.lr_scheduler.get_last_lr()[0]
            self.writer.add_scalar("Training/LearningRate", new_lr, self.global_step)
            self.logger.info(f"Learning rate updated: {old_lr:.6f} -> {new_lr:.6f}")
        
        # Calculate averages
        if update_count > 0:
            mean_policy_loss /= update_count
            mean_value_loss /= update_count
            mean_entropy /= update_count
            mean_approx_kl /= update_count
            mean_clipfrac /= update_count
            
            # Log additional metrics
            self.writer.add_scalar("Loss/KL", mean_approx_kl, self.global_step)
            self.writer.add_scalar("Loss/ClipFraction", mean_clipfrac, self.global_step)
            
            # Log additional training information
            self.logger.info(f"Training stats summary - KL: {mean_approx_kl:.6f}, Clip fraction: {mean_clipfrac:.6f}")
        
        return mean_policy_loss, mean_value_loss, mean_entropy
    
    def train(self):
        """Main training loop with enhanced debugging"""
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
            
            # Debug - Check if we're collecting any rewards
            avg_rewards = self.buffer.rewards_buf[:self.buffer.size].mean().item()
            min_rewards = self.buffer.rewards_buf[:self.buffer.size].min().item()
            max_rewards = self.buffer.rewards_buf[:self.buffer.size].max().item()
            self.logger.info(f"Rewards stats - Mean: {avg_rewards:.6f}, Min: {min_rewards:.6f}, Max: {max_rewards:.6f}")
            
            # Debug - Check if done flags are being set
            dones_ratio = self.buffer.dones_buf[:self.buffer.size].float().mean().item()
            self.logger.info(f"Dones ratio: {dones_ratio:.6f} (Episodes completing: {dones_ratio * 100:.2f}%)")
            
            # Update policy using PPO
            self.logger.info(f"Updating policy (Step {self.global_step})...")
            policy_loss, value_loss, entropy = self.update_policy()
            
            # Save checkpoint
            if self.global_step % self.config.CHECKPOINT_INTERVAL_UPDATE == 0:
                self.save_checkpoint()
                self.logger.info(f"Checkpoint saved at step {self.global_step}")
            
            # Update progress bar
            pbar.update(steps_collected)
            
            # Log training progress
            self.logger.info(f"Step: {self.global_step}/{self.config.NUM_TIMESTEPS} | "
                            f"Policy Loss: {policy_loss:.6f} | Value Loss: {value_loss:.6f} | "
                            f"Entropy: {entropy:.6f}")
        
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