"""
Randomized Attention Q-Value Network
"""

import torch
import torch.nn as nn

class RandomizedAttention_qvalue(nn.Module):
    """Q-value network compatible with the randomized attention actor"""
    def __init__(self, qvalue_config):
        super().__init__()
        self.device = qvalue_config["device"]
        self.na = qvalue_config["n_agents"]
        self.observation_dim_per_agent = qvalue_config["observation_dim_per_agent"]
        self.action_dim_per_agent = qvalue_config["action_dim_per_agent"]
        
        # Create a Q-network that takes concatenated observations and actions
        self.mlp = nn.Sequential(
            nn.Linear((self.observation_dim_per_agent + self.action_dim_per_agent) * self.na, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 1)
        ).to(self.device)
    
    def forward(self, observation, action):
        """
        Compute Q-value for given observation and action
        
        Args:
            observation: Tensor of shape [batch_size, n_agents, obs_dim]
            action: Tensor of shape [batch_size, n_agents, action_dim]
            
        Returns:
            q_value: Tensor of shape [batch_size, 1]
        """
        # Flatten and concatenate observations and actions
        obs_flat = observation.reshape(observation.shape[0], -1)
        action_flat = action.reshape(action.shape[0], -1)
        x = torch.cat([obs_flat, action_flat], dim=1)
        
        # Compute Q-value
        return self.mlp(x)