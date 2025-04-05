"""
Randomized Replay Buffer for storing experiences with random numbers
"""

import numpy as np
import torch

class RandomizedReplayBuffer(object):
    """Buffer to store environment transitions with random numbers."""
    def __init__(self, obs_shape_per_agent, action_shape_per_agent, num_envs, num_agents, capacity, device):
        self.capacity = capacity
        self.device = device
        self.num_envs = num_envs
        self.num_agents = num_agents

        # Store by agent dimension separately to handle variable observations
        self.obs_shape_per_agent = obs_shape_per_agent
        self.action_shape_per_agent = action_shape_per_agent
        
        # Correctly allocate buffer space
        self.obses = np.empty((capacity, num_agents, obs_shape_per_agent), dtype=np.float32)
        self.next_obses = np.empty((capacity, num_agents, obs_shape_per_agent), dtype=np.float32)
        self.actions = np.empty((capacity, num_agents, action_shape_per_agent), dtype=np.float32)
        self.rewards = np.empty((capacity, num_agents), dtype=np.float32)
        self.dones = np.empty((capacity, 1), dtype=np.float32)
        self.random_nums = np.empty((capacity, num_agents), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done, random_nums):
        for env_idx in range(self.num_envs):
            # Reshape observation to [n_agents, obs_dim]
            obs_reshaped = obs[env_idx].reshape(self.num_agents, -1)
            action_reshaped = action[env_idx].reshape(self.num_agents, -1)
            next_obs_reshaped = next_obs[env_idx].reshape(self.num_agents, -1)
            
            # Copy to buffer
            np.copyto(self.obses[self.idx], obs_reshaped)
            np.copyto(self.actions[self.idx], action_reshaped)
            np.copyto(self.rewards[self.idx], reward[env_idx])
            np.copyto(self.next_obses[self.idx], next_obs_reshaped)
            np.copyto(self.dones[self.idx], done[env_idx])
            np.copyto(self.random_nums[self.idx], random_nums[env_idx])

            self.idx = (self.idx + 1) % self.capacity
            self.full = self.full or self.idx == 0

    def sample(self, batch_size):
            idxs = np.random.randint(0,
                                    self.capacity if self.full else self.idx,
                                    size=batch_size)

            obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
            actions = torch.as_tensor(self.actions[idxs], device=self.device)
            rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
            next_obses = torch.as_tensor(self.next_obses[idxs], device=self.device).float()
            dones = torch.as_tensor(self.dones[idxs], device=self.device)
            random_nums = torch.as_tensor(self.random_nums[idxs], device=self.device)

            return obses, actions, rewards, next_obses, dones, random_nums