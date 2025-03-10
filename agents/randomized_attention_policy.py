import torch as th
import torch.nn as nn
import math
from typing import Dict, Tuple, Union

class SquashedNormal:
    """
    Squashed normal distribution for action representation
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        
    def sample(self):
        eps = th.randn_like(self.mean)
        unscaled = self.mean + self.std * eps
        return th.tanh(unscaled)
    
    def rsample(self):
        return self.sample()
    
    def log_prob(self, value):
        # Pre-tanh value
        pre_tanh = 0.5 * th.log((1 + value) / (1 - value + 1e-6) + 1e-6)
        
        # Log prob of normal distribution
        log_prob_normal = -0.5 * ((pre_tanh - self.mean) / (self.std + 1e-6))**2 - self.std.log() - 0.5 * math.log(2 * math.pi)
        
        # Log det of Jacobian
        log_det_jacobian = 2 * (math.log(2) - pre_tanh - th.nn.functional.softplus(-2 * pre_tanh))
        
        # Return sum across last dimension
        return (log_prob_normal - log_det_jacobian).sum(dim=-1)

class RandomizedAttentionPolicy(nn.Module):
    """
    Implementation of the RandomizedAttention policy
    """
    def __init__(
        self, 
        observation_dim: int,
        action_dim: int, 
        hidden_dim: int = 64, 
        device: Union[str, th.device] = "cpu"
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.device = device if isinstance(device, th.device) else th.device(device)

        self.relu = nn.LeakyReLU(0.2)

        # Initial state estimation
        # Adapt the embedding layer to match the actual observation dimension
        self.embed = nn.Linear(observation_dim, hidden_dim)
        self.init_attn = nn.MultiheadAttention(hidden_dim, 1, batch_first=True)
        self.msg_layer = nn.Linear(hidden_dim, hidden_dim)
        self.state_layer = nn.Linear(hidden_dim, hidden_dim)

        # State consensus
        self.concensus_fc = nn.Linear(hidden_dim, hidden_dim)
        self.consensus_attn1 = nn.MultiheadAttention(hidden_dim, 1, batch_first=True)
        self.consensus_attn2 = nn.MultiheadAttention(hidden_dim, 1, batch_first=True)

        # Action output layers
        self.mean_layer = nn.Linear(2*hidden_dim, action_dim)
        self.logstd_layer = nn.Linear(2*hidden_dim, action_dim)
    
    def tanh_normal_sample(self, mean, log_std):
        """
        Sample from a tanh-transformed normal distribution
        """
        std = th.exp(log_std)
        eps = th.randn_like(mean)
        pre_tanh = mean + std * eps
        action = th.tanh(pre_tanh)

        # Diagonal Gaussian log-prob
        var = std.pow(2)
        log_prob_gauss = -0.5 * (((pre_tanh - mean)**2) / (var + 1e-8) + 2*log_std + math.log(2*math.pi))
        log_prob_gauss = log_prob_gauss.sum(dim=-1)  # => shape [batch_size, n_agents]

        # Tanh correction
        correction = 2.0 * (math.log(2.0) - pre_tanh - th.nn.functional.softplus(-2.0 * pre_tanh))
        correction = correction.sum(dim=-1)  # => [batch_size, n_agents]

        log_prob = log_prob_gauss - correction
        return action, log_prob
    
    def initial_state_estimation(self, obs: Dict[str, th.Tensor]) -> Tuple[th.Tensor, th.Tensor]:
        """
        Performs initial state estimation based on observations
        """
        # Directly use the observation tensor
        x = obs['obs']
        batch_size, n_agents, obs_dim = x.shape
        
        # Reshape for processing
        x_flat = x.reshape(batch_size * n_agents, obs_dim)
        
        # Process through embed layer
        embed = self.relu(self.embed(x_flat))  # [batch_size*n_agents, hidden_dim]
        
        # Reshape for attention
        embed = embed.view(batch_size, n_agents, self.hidden_dim)  # [batch_size, n_agents, hidden_dim]
        
        # Process with self-attention
        attn_out, _ = self.init_attn(embed, embed, embed, need_weights=False)
        attn_out = th.tanh(attn_out)  # [batch_size, n_agents, hidden_dim]
        
        # Generate messages and state
        messages = self.msg_layer(attn_out)  # [batch_size, n_agents, hidden_dim]
        temp_state = self.state_layer(attn_out)  # [batch_size, n_agents, hidden_dim]
        
        return temp_state, messages
    
    def _create_mask(self, rnd_nums: th.Tensor, rnd: th.Tensor) -> th.Tensor:
        """
        Creates a mask based on random numbers
        """
        batch_size = rnd.shape[0]
        n_agents = rnd_nums.shape[1]
        
        # Expand dimensions for broadcasting
        rnd_expanded = rnd.view(batch_size, 1).unsqueeze(2)  # [batch_size, 1, 1]
        rnd_nums_i = rnd_nums.unsqueeze(2)  # [batch_size, n_agents, 1]
        rnd_nums_j = rnd_nums.unsqueeze(1)  # [batch_size, 1, n_agents]
        
        # Create mask: M_ij = (rnd_i > rnd) OR (rnd_j > rnd)
        mask = (rnd_nums_i > rnd_expanded) | (rnd_nums_j > rnd_expanded)
        return mask
    
    def consensus(self, temp_states: th.Tensor, obs: Dict[str, th.Tensor], messages: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Performs consensus process based on messages and temp states
        """
        batch_size, n_agents, _ = temp_states.shape
        
        rnd_nums = obs['rnd_nums']  # [batch_size, n_agents]
        idx = obs['idx']  # [batch_size, n_agents]
        
        # Initialize outputs container
        outputs = []
        
        # Process each agent
        for i in range(n_agents):
            agent_idx = idx[:, i]  # [batch_size]
            agent_rnd = rnd_nums[th.arange(batch_size), agent_idx]  # [batch_size]
            
            # Create masks for this agent
            masks = self._create_mask(rnd_nums, agent_rnd)  # [batch_size, n_agents, n_agents]
            
            # Extract mask for current agent i
            mask_greater = masks[:, i, :]  # [batch_size, n_agents]
            mask_lesser = ~mask_greater  # [batch_size, n_agents]
            
            # Create masked messages
            msgs_greater = messages * mask_greater.unsqueeze(-1)  # [batch_size, n_agents, hidden_dim]
            msgs_lesser = messages * mask_lesser.unsqueeze(-1)  # [batch_size, n_agents, hidden_dim]
            
            # Current agent's temp state as query
            agent_temp_state = temp_states[:, i:i+1, :]  # [batch_size, 1, hidden_dim]
            
            # Process with attention
            attn1_out, _ = self.consensus_attn1(
                msgs_greater, 
                agent_temp_state, 
                agent_temp_state,  # Using temp_state as both query and value
                need_weights=False
            )  # [batch_size, 1, hidden_dim]
            
            attn2_out, _ = self.consensus_attn2(
                msgs_lesser, 
                agent_temp_state, 
                agent_temp_state,  # Using temp_state as both query and value
                need_weights=False
            )  # [batch_size, 1, hidden_dim]
            
            # Combine outputs
            concat_out = th.cat([
                attn1_out.squeeze(1),  # [batch_size, hidden_dim]
                attn2_out.squeeze(1)   # [batch_size, hidden_dim]
            ], dim=1)  # [batch_size, 2*hidden_dim]
            
            outputs.append(concat_out)
        
        # Stack outputs
        out = th.stack(outputs, dim=1)  # [batch_size, n_agents, 2*hidden_dim]
        out = self.relu(out)
        
        # Generate mean and logstd
        mean = self.mean_layer(out)  # [batch_size, n_agents, action_dim]
        logstd = self.logstd_layer(out)
        
        # Constrain logstd for stability
        logstd = th.clamp(logstd, -20, 2)
        
        return mean, logstd
    
    def sample_actions_and_logp(self, obs: Dict[str, th.Tensor]) -> Tuple[th.Tensor, th.Tensor]:
        """
        Samples actions and computes log probabilities
        """
        temp_states, messages = self.initial_state_estimation(obs)
        mean, logstd = self.consensus(temp_states, obs, messages)
        actions, logp = self.tanh_normal_sample(mean, logstd)
        return actions, logp
    
    def forward(self, obs: Dict[str, th.Tensor]):
        """
        Forward pass through the policy
        """
        temp_states, messages = self.initial_state_estimation(obs)
        mean, logstd = self.consensus(temp_states, obs, messages)
        std = th.exp(logstd)
        return SquashedNormal(mean, std)

class RandomizedAttentionAgent(nn.Module):
    """
    Agent wrapper for RandomizedAttentionPolicy
    """
    def __init__(
        self, 
        observation_dim: int, 
        action_dim: int, 
        hidden_dim: int = 64, 
        device: Union[str, th.device] = "cpu"
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.device = device if isinstance(device, th.device) else th.device(device)
        
        # Initialize the policy with the actual observation dimension
        self.policy = RandomizedAttentionPolicy(
            observation_dim=observation_dim // 5,  # Per-agent observation dimension
            action_dim=action_dim // 5,            # Per-agent action dimension
            hidden_dim=hidden_dim,
            device=self.device
        )
    
    def prepare_obs_dict(self, obs: th.Tensor) -> Dict[str, th.Tensor]:
        """
        Prepares observation dictionary expected by RandomizedAttentionPolicy
        """
        batch_size, n_agents, _ = obs.shape
        
        # Generate random numbers for each agent
        rnd_nums = th.rand(batch_size, n_agents, device=self.device)
        
        # Create indices for each agent
        idx = th.arange(n_agents, device=self.device).repeat(batch_size, 1)
        
        return {
            'obs': obs,  # Shape: [batch_size, n_agents, obs_dim]
            'rnd_nums': rnd_nums,  # Shape: [batch_size, n_agents]
            'idx': idx  # Shape: [batch_size, n_agents]
        }
    
    def forward(self, obs_dict: Union[Dict[str, th.Tensor], th.Tensor]) -> SquashedNormal:
        """
        Forward pass to get action distribution
        """
        # Handle tensor input by converting to dict if needed
        if isinstance(obs_dict, th.Tensor):
            obs_dict = self.prepare_obs_dict(obs_dict)
            
        return self.policy(obs_dict)