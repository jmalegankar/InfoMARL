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