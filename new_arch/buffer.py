import torch
from collections import deque
import random

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

    def sample(self, batch_size):
        idxs = torch.randint(0, self.size, size=(batch_size,), device=self.device)
        
        return (
            self.obs_buf[idxs],
            self.actions_buf[idxs],
            self.rewards_buf[idxs],
            self.next_obs_buf[idxs],
            self.dones_buf[idxs],
            self.random_numbers_buf[idxs]
        )