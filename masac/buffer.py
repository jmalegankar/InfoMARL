import torch
from collections import deque
import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity, obs_dim, action_dim, n_agents, device):
        self.capacity = capacity
        self.device = device
        self.obs_buf = torch.zeros((capacity, n_agents, obs_dim), dtype=torch.float32, device=device)
        self.next_obs_buf = torch.zeros((capacity, n_agents, obs_dim), dtype=torch.float32, device=device)
        self.actions_buf = torch.zeros((capacity, n_agents, action_dim), dtype=torch.float32, device=device)
        self.rewards_buf = torch.zeros((capacity), dtype=torch.float32, device=device)
        self.dones_buf = torch.zeros((capacity), dtype=torch.float32, device=device)
        self.random_numbers_buf = torch.zeros((capacity, n_agents, n_agents), dtype=torch.float32, device=device)
        self.ptr, self.size = 0, 0

    def add(self, obs, action, reward, next_obs, done, random_numbers):
        self.obs_buf[self.ptr, ...] = obs
        self.actions_buf[self.ptr, ...] = action
        self.rewards_buf[self.ptr] = reward
        self.next_obs_buf[self.ptr, ...] = next_obs
        self.dones_buf[self.ptr] = done
        self.random_numbers_buf[self.ptr, ...] = random_numbers 
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device=None):
        if self.size == 0:
            raise ValueError("Replay buffer is empty. Cannot sample from an empty buffer.")
        if batch_size > self.size:
            raise ValueError(f"Batch size {batch_size} exceeds buffer size {self.size}.")

        idxs = torch.randint(0, self.size, size=(batch_size,), device=self.device)

        if device is None or device == self.device:
            return (
                self.obs_buf[idxs].to(device),
                self.actions_buf[idxs].to(device),
                self.rewards_buf[idxs].to(device),
                self.next_obs_buf[idxs].to(device),
                self.dones_buf[idxs].to(device),
                self.random_numbers_buf[idxs].to(device),
            )
        else:
            # If device is different, we need to move the tensors to the new device
            return (
                self.obs_buf[idxs].to(device),
                self.actions_buf[idxs].to(device),
                self.rewards_buf[idxs].to(device),
                self.next_obs_buf[idxs].to(device),
                self.dones_buf[idxs].to(device),
                self.random_numbers_buf[idxs].to(device),
            )
        
    def get_save_state(self):
        if self.size == 0:
            return {
                'size': 0,
                'ptr': 0
            }
            
        # circular buffers,handle the wrapping correctly
        if self.ptr < self.size:
            obs_part1 = self.obs_buf[self.ptr:self.size].cpu()
            obs_part2 = self.obs_buf[:self.ptr].cpu()
            actions_part1 = self.actions_buf[self.ptr:self.size].cpu()
            actions_part2 = self.actions_buf[:self.ptr].cpu()
            rewards_part1 = self.rewards_buf[self.ptr:self.size].cpu()
            rewards_part2 = self.rewards_buf[:self.ptr].cpu()
            next_obs_part1 = self.next_obs_buf[self.ptr:self.size].cpu()
            next_obs_part2 = self.next_obs_buf[:self.ptr].cpu()
            dones_part1 = self.dones_buf[self.ptr:self.size].cpu()
            dones_part2 = self.dones_buf[:self.ptr].cpu()
            random_numbers_part1 = self.random_numbers_buf[self.ptr:self.size].cpu()
            random_numbers_part2 = self.random_numbers_buf[:self.ptr].cpu()
            
            return {
                'obs_part1': obs_part1,
                'obs_part2': obs_part2,
                'actions_part1': actions_part1,
                'actions_part2': actions_part2,
                'rewards_part1': rewards_part1,
                'rewards_part2': rewards_part2,
                'next_obs_part1': next_obs_part1,
                'next_obs_part2': next_obs_part2,
                'dones_part1': dones_part1,
                'dones_part2': dones_part2,
                'random_numbers_part1': random_numbers_part1,
                'random_numbers_part2': random_numbers_part2,
                'size': self.size,
                'ptr': self.ptr,
                'wrapped': True
            }
        else:

            return {
                'obs': self.obs_buf[:self.size].cpu(),
                'actions': self.actions_buf[:self.size].cpu(),
                'rewards': self.rewards_buf[:self.size].cpu(),
                'next_obs': self.next_obs_buf[:self.size].cpu(),
                'dones': self.dones_buf[:self.size].cpu(),
                'random_numbers': self.random_numbers_buf[:self.size].cpu(),
                'size': self.size,
                'ptr': self.ptr,
                'wrapped': False
            }
    
    def load_save_state(self, save_state):
        if save_state['size'] == 0:
            self.size = 0
            self.ptr = 0
            return
            
        self.size = save_state['size']
        self.ptr = save_state['ptr']
        
        if save_state.get('wrapped', False):
            size_part1 = save_state['obs_part1'].size(0)
            size_part2 = save_state['obs_part2'].size(0)
            
            # Load part 1 (from ptr to end)
            self.obs_buf[self.ptr:self.ptr+size_part1] = save_state['obs_part1'].to(self.device)
            self.actions_buf[self.ptr:self.ptr+size_part1] = save_state['actions_part1'].to(self.device)
            self.rewards_buf[self.ptr:self.ptr+size_part1] = save_state['rewards_part1'].to(self.device)
            self.next_obs_buf[self.ptr:self.ptr+size_part1] = save_state['next_obs_part1'].to(self.device)
            self.dones_buf[self.ptr:self.ptr+size_part1] = save_state['dones_part1'].to(self.device)
            self.random_numbers_buf[self.ptr:self.ptr+size_part1] = save_state['random_numbers_part1'].to(self.device)
            
            # Load part 2 (from 0 to ptr)
            self.obs_buf[:size_part2] = save_state['obs_part2'].to(self.device)
            self.actions_buf[:size_part2] = save_state['actions_part2'].to(self.device)
            self.rewards_buf[:size_part2] = save_state['rewards_part2'].to(self.device)
            self.next_obs_buf[:size_part2] = save_state['next_obs_part2'].to(self.device)
            self.dones_buf[:size_part2] = save_state['dones_part2'].to(self.device)
            self.random_numbers_buf[:size_part2] = save_state['random_numbers_part2'].to(self.device)
        else:
            # Buffer wasn't wrapped
            self.obs_buf[:self.size] = save_state['obs'].to(self.device)
            self.actions_buf[:self.size] = save_state['actions'].to(self.device)
            self.rewards_buf[:self.size] = save_state['rewards'].to(self.device)
            self.next_obs_buf[:self.size] = save_state['next_obs'].to(self.device)
            self.dones_buf[:self.size] = save_state['dones'].to(self.device)
            self.random_numbers_buf[:self.size] = save_state['random_numbers'].to(self.device)
        
        print(f"Loaded replay buffer with {self.size} transitions")



class PPORolloutBuffer:
    def __init__(self, buffer_size, obs_dim, action_dim, n_agents, device, gamma=0.99, gae_lambda=0.95):
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        # Initialize buffers
        self.obs_buf = torch.zeros((buffer_size, n_agents, obs_dim), dtype=torch.float32, device=device)
        self.actions_buf = torch.zeros((buffer_size, n_agents, action_dim), dtype=torch.float32, device=device)
        self.old_log_probs_buf = torch.zeros((buffer_size, n_agents), dtype=torch.float32, device=device)
        self.rewards_buf = torch.zeros((buffer_size), dtype=torch.float32, device=device)
        self.values_buf = torch.zeros((buffer_size, n_agents), dtype=torch.float32, device=device)
        self.returns_buf = torch.zeros((buffer_size, n_agents), dtype=torch.float32, device=device)
        self.advantages_buf = torch.zeros((buffer_size, n_agents), dtype=torch.float32, device=device)
        self.dones_buf = torch.zeros((buffer_size), dtype=torch.float32, device=device)
        self.random_numbers_buf = torch.zeros((buffer_size, n_agents, n_agents), dtype=torch.float32, device=device)
        
        self.ptr = 0
        self.size = 0
        self.episode_start_idx = 0
    
    def add(self, obs, actions, log_probs, values, reward, done, random_numbers):
        # Store transition
        self.obs_buf[self.ptr] = obs
        self.actions_buf[self.ptr] = actions
        self.old_log_probs_buf[self.ptr] = log_probs
        self.values_buf[self.ptr] = values
        self.rewards_buf[self.ptr] = reward
        self.dones_buf[self.ptr] = done
        self.random_numbers_buf[self.ptr] = random_numbers
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
        
        if done:
            self.compute_advantages_and_returns(self.episode_start_idx, self.ptr - 1)
            self.episode_start_idx = self.ptr
    
    def compute_advantages_and_returns(self, start_idx, end_idx, last_value=None):
        """Compute Generalized Advantage Estimation (GAE) and returns with debugging"""
        print(f"\n--- DEBUG: Computing advantages from {start_idx} to {end_idx} ---")
        
        # Extract episode data
        if end_idx >= start_idx:
            # Extract continuous episode
            rewards = self.rewards_buf[start_idx:end_idx+1]
            values = self.values_buf[start_idx:end_idx+1]
            dones = self.dones_buf[start_idx:end_idx+1]
        else:
            # Handle wrapped buffer
            rewards = torch.cat([self.rewards_buf[start_idx:], self.rewards_buf[:end_idx+1]])
            values = torch.cat([self.values_buf[start_idx:], self.values_buf[:end_idx+1]])
            dones = torch.cat([self.dones_buf[start_idx:], self.dones_buf[:end_idx+1]])
        
        # Print episode statistics
        print(f"Episode length: {len(rewards)}")
        print(f"Rewards - Mean: {rewards.mean().item():.6f}, Min: {rewards.min().item():.6f}, Max: {rewards.max().item():.6f}")
        print(f"Values - Mean: {values.mean().item():.6f}, Min: {values.min().item():.6f}, Max: {values.max().item():.6f}")
        print(f"Dones - Count: {dones.sum().item()}/{len(dones)}")
        
        # Calculate GAE
        episode_length = len(rewards)
        advantages = torch.zeros((episode_length, self.n_agents), device=self.device)
        last_gae = torch.zeros(self.n_agents, device=self.device)
        
        # Print gamma and lambda parameters
        print(f"GAE parameters - gamma: {self.gamma}, lambda: {self.gae_lambda}")
        
        # Set the next value for the final step
        if last_value is not None:
            next_value = last_value
            print(f"Using provided last_value: {next_value.mean().item():.6f}")
        else:
            if dones[-1]:
                next_value = torch.zeros(self.n_agents, device=self.device)
                print("Last step is terminal, using zero next_value")
            else:
                next_value = values[-1]
                print(f"Last step is non-terminal, using last value: {next_value.mean().item():.6f}")
        
        # Debug info for first and last few steps
        debug_indices = [0, 1, min(5, episode_length-1), max(0, episode_length-2), max(0, episode_length-1)]
        
        # Iterate backwards through episode
        for t in reversed(range(episode_length)):
            if t == episode_length - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value_step = next_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value_step = values[t+1]
            
            # Calculate TD error (delta)
            delta = rewards[t].unsqueeze(-1) + self.gamma * next_value_step * next_non_terminal - values[t]
            
            # Calculate GAE
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
            
            # Print debug info for selected steps
            if t in debug_indices:
                print(f"Step {t} - Reward: {rewards[t].item():.6f}, Value: {values[t, 0].item():.6f}, "
                    f"Next value: {next_value_step[0].item():.6f}, Terminal: {bool(dones[t].item())}, "
                    f"Delta: {delta[0].item():.6f}, Advantage: {last_gae[0].item():.6f}")
        
        # Calculate returns (advantages + values)
        returns = advantages + values
        
        # Print final advantage and returns statistics
        print(f"Advantages - Mean: {advantages.mean().item():.6f}, Min: {advantages.min().item():.6f}, "
            f"Max: {advantages.max().item():.6f}, Std: {advantages.std().item():.6f}")
        print(f"Returns - Mean: {returns.mean().item():.6f}, Min: {returns.min().item():.6f}, "
            f"Max: {returns.max().item():.6f}")
        
        # Store back in buffer
        if end_idx >= start_idx:
            self.advantages_buf[start_idx:end_idx+1] = advantages
            self.returns_buf[start_idx:end_idx+1] = returns
        else:
            # Handle wrapped buffer
            split_idx = len(self.advantages_buf) - start_idx
            self.advantages_buf[start_idx:] = advantages[:split_idx]
            self.advantages_buf[:end_idx+1] = advantages[split_idx:]
            self.returns_buf[start_idx:] = returns[:split_idx]
            self.returns_buf[:end_idx+1] = returns[split_idx:]
        
        print("--- Advantage computation completed ---\n")
    
    def finalize_buffer(self, last_values=None):
        """
        Compute advantages and returns for the last incomplete episode
        """
        if self.ptr != self.episode_start_idx:
            # We have an incomplete episode
            if self.ptr > self.episode_start_idx:
                self.compute_advantages_and_returns(self.episode_start_idx, self.ptr - 1, last_values)
            else:
                # Wrapped buffer
                self.compute_advantages_and_returns(self.episode_start_idx, self.buffer_size - 1, last_values)
                if self.ptr > 0:
                    self.compute_advantages_and_returns(0, self.ptr - 1, last_values)
    
    def normalize_advantages(self):
        """
        Normalize advantages for more stable training
        """
        if self.size > 0:
            # Access only valid data
            advantages = self.advantages_buf[:self.size]
            
            # Normalize across all dimensions to handle multi-agent case properly
            mean = advantages.mean()
            std = advantages.std() + 1e-8
            
            # Apply normalization
            self.advantages_buf[:self.size] = (advantages - mean) / std
            
            # Log stats for debugging
            print(f"Advantage stats - Mean: {mean.item():.4f}, Std: {std.item():.4f}, "
                  f"Min: {advantages.min().item():.4f}, Max: {advantages.max().item():.4f}")
    
    def get_batches(self, batch_size, normalize_advantages=True):
        """
        Generate random batches from the buffer for PPO updates
        """
        if normalize_advantages:
            self.normalize_advantages()
            
        indices = torch.randperm(self.size, device=self.device)
        
        for start_idx in range(0, self.size, batch_size):
            end_idx = min(start_idx + batch_size, self.size)
            batch_indices = indices[start_idx:end_idx]
            
            yield (
                self.obs_buf[batch_indices],
                self.actions_buf[batch_indices],
                self.old_log_probs_buf[batch_indices],
                self.values_buf[batch_indices],
                self.returns_buf[batch_indices],
                self.advantages_buf[batch_indices],
                self.random_numbers_buf[batch_indices]
            )
    
    def clear(self):
        """
        Clear the buffer
        """
        self.ptr = 0
        self.size = 0
        self.episode_start_idx = 0
    
    def get_save_state(self):
        if self.size == 0:
            return {
                'size': 0,
                'ptr': 0,
                'episode_start_idx': 0
            }
            
        # Save all current data
        return {
            'obs': self.obs_buf[:self.size].cpu(),
            'actions': self.actions_buf[:self.size].cpu(),
            'old_log_probs': self.old_log_probs_buf[:self.size].cpu(),
            'rewards': self.rewards_buf[:self.size].cpu(),
            'values': self.values_buf[:self.size].cpu(),
            'returns': self.returns_buf[:self.size].cpu(),
            'advantages': self.advantages_buf[:self.size].cpu(),
            'dones': self.dones_buf[:self.size].cpu(),
            'random_numbers': self.random_numbers_buf[:self.size].cpu(),
            'size': self.size,
            'ptr': self.ptr,
            'episode_start_idx': self.episode_start_idx
        }
    
    def load_save_state(self, save_state):
        if save_state['size'] == 0:
            self.size = 0
            self.ptr = 0
            self.episode_start_idx = 0
            return
            
        self.size = save_state['size']
        self.ptr = save_state['ptr']
        self.episode_start_idx = save_state['episode_start_idx']
        
        self.obs_buf[:self.size] = save_state['obs'].to(self.device)
        self.actions_buf[:self.size] = save_state['actions'].to(self.device)
        self.old_log_probs_buf[:self.size] = save_state['old_log_probs'].to(self.device)
        self.rewards_buf[:self.size] = save_state['rewards'].to(self.device)
        self.values_buf[:self.size] = save_state['values'].to(self.device)
        self.returns_buf[:self.size] = save_state['returns'].to(self.device)
        self.advantages_buf[:self.size] = save_state['advantages'].to(self.device)
        self.dones_buf[:self.size] = save_state['dones'].to(self.device)
        self.random_numbers_buf[:self.size] = save_state['random_numbers'].to(self.device)
        
        print(f"Loaded PPO buffer with {self.size} transitions")