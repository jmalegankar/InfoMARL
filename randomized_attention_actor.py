"""
Randomized Attention Actor Network
"""

import torch
import torch.nn as nn
from reinforcement_functions import SquashedNormal

class RandomizedAttention_actor(nn.Module):
    """Actor network based on Randomized Attention mechanism"""
    def __init__(self, actor_config):
        super().__init__()
        self.device = actor_config["device"]
        self.na = actor_config["n_agents"]
        self.observation_dim_per_agent = actor_config["observation_dim_per_agent"]
        self.action_dim_per_agent = actor_config["action_dim_per_agent"]
        self.r_communication = actor_config["r_communication"]
        self.batch_size = actor_config["batch_size"]
        self.num_envs = actor_config["num_envs"]
        self.epsilon = 1e-6
        self.log_std_min = -5
        self.log_std_max = 2
        self.hidden_dim = 64  # Can be adjusted as needed
        
        # Initialize random number generator
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(12345)
        
        # Initial attention components
        self.init_embed = nn.Linear(self.observation_dim_per_agent, self.hidden_dim)
        self.init_attention = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=4, batch_first=True)
        
        # Consensus attention components
        self.consensus_attention = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=4, batch_first=True)
        
        # Output layers for action distribution parameters
        self.mean_layer = nn.Linear(self.hidden_dim, self.action_dim_per_agent)
        self.logstd_layer = nn.Linear(self.hidden_dim, self.action_dim_per_agent)

    def initial_state_estimation(self, obs_batch):
        """
        Process initial observation to create temp state and message
        
        Args:
            obs_batch: Tensor of shape [batch_size, obs_dim]
            
        Returns:
            temp_state: Tensor for temp state
            message: Tensor for message to share
        """
        # Create embedding from observations
        embed = self.init_embed(obs_batch).unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # Apply initial attention
        att_out, _ = self.init_attention(embed, embed, embed, need_weights=False)
        
        temp_state = att_out  # [batch_size, 1, hidden_dim]
        message = att_out.mean(dim=1, keepdim=True)  # [batch_size, 1, hidden_dim]
        
        return temp_state, message
    
    def create_mask(self, rnd, rnd_list):
        """
        Create attention mask based on random numbers
        
        Args:
            rnd: Random number for current agent (batch_size, 1)
            rnd_list: Random numbers for all agents (batch_size, n_agents)
            
        Returns:
            mask: Attention mask
        """
        batch_size = rnd.size(0)
        n_agents = rnd_list.size(1)
        
        # Expand dimensions for broadcasting
        rnd_expanded = rnd.view(batch_size, 1, 1).expand(batch_size, 1, n_agents)
        rnd_list_expanded = rnd_list.unsqueeze(1).expand(batch_size, n_agents, n_agents)
        
        # Create mask: M_ij = (rnd_i > rnd) || (rnd_j > rnd)
        mask = (rnd_list_expanded > rnd_expanded) | (rnd_list_expanded.transpose(1, 2) > rnd_expanded)
        
        # For attention, True means masked (excluded), so we need to invert
        return ~mask
    
    def consensus_process(self, temp_state, messages, random_numbers, agent_idx, batch_size):
        """
        Process messages using consensus attention mechanism
        
        Args:
            temp_state: Tensor of shape [batch_size, 1, hidden_dim]
            messages: Tensor of shape [batch_size, n_agents, hidden_dim]
            random_numbers: Tensor of shape [batch_size, n_agents]
            agent_idx: Index of the current agent
            batch_size: Batch size
            
        Returns:
            mean: Action mean
            log_std: Action log standard deviation
        """
        # Get agent's random number
        agent_rnd = random_numbers[:, agent_idx].view(batch_size, 1)
        
        # Create attention mask
        mask = self.create_mask(agent_rnd, random_numbers)
        
        # Apply consensus attention
        att_out, _ = self.consensus_attention(
            temp_state,  # query
            messages,    # key
            messages,    # value
            attn_mask=mask.float(),
            need_weights=False
        )
        
        # Process attention output
        att_out = att_out.mean(dim=1)  # [batch_size, hidden_dim]
        
        # Generate action distribution parameters
        mean = self.mean_layer(att_out)
        log_std = self.logstd_layer(att_out)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std

    def forward(self, x):
        """
        Process batch of observations to generate actions
        
        Args:
            x: Tensor of shape [batch_size, n_agents, obs_dim]
            
        Returns:
            action_dist: SquashedNormal distribution
            random_numbers: Tensor of random numbers used
        """
        batch_size = x.shape[0]
        
        # Generate random numbers for all agents
        random_numbers = torch.rand(batch_size, self.na, device=self.device, generator=self.rng)
        
        # Process each agent's observation
        temp_states = []
        messages = []
        
        for i in range(self.na):
            obs_i = x[:, i, :]
            temp_state_i, message_i = self.initial_state_estimation(obs_i)
            temp_states.append(temp_state_i)
            messages.append(message_i)
        
        # Stack messages from all agents
        messages_tensor = torch.cat(messages, dim=1)  # [batch_size, n_agents, hidden_dim]
        
        # Process consensus for each agent
        all_means = []
        all_log_stds = []
        
        for i in range(self.na):
            mean_i, log_std_i = self.consensus_process(
                temp_states[i],
                messages_tensor,
                random_numbers,
                i,
                batch_size
            )
            all_means.append(mean_i)
            all_log_stds.append(log_std_i)
        
        # Stack results for all agents
        means = torch.stack(all_means, dim=1)  # [batch_size, n_agents, action_dim]
        log_stds = torch.stack(all_log_stds, dim=1)  # [batch_size, n_agents, action_dim]
        
        return SquashedNormal(means, torch.exp(log_stds)), random_numbers