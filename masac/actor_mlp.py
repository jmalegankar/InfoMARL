import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

class MLPActor(nn.Module):
    def __init__(
        self,
        agent_dim: int,
        landmark_dim: int,
        hidden_dim: int,
        action_dim: int,
    ):
        super().__init__()

        self.input_dim = agent_dim + landmark_dim  

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs: Dict[str, th.Tensor]) -> Tuple[th.Tensor, th.Tensor]:
        x = obs["obs"].view(obs["obs"].size(0), -1)
        h = self.net(x)
        mean = self.mean_layer(h)
        log_std = self.log_std_layer(h)
        return mean, log_std
    
    @th.jit.export
    def sample_actions_and_logp(
        self, obs: Dict[str, th.Tensor]
    ) -> Tuple[th.Tensor, th.Tensor]:
        mean, log_std = self.forward(obs)
        std = th.exp(log_std)

        eps = th.randn_like(mean)
        pre_tanh = mean + std * eps
        actions = th.tanh(pre_tanh)

        
        var = std.pow(2)
        log_prob_gauss = -0.5 * (
            ((pre_tanh - mean).pow(2) / (var + 1e-8))
            + 2 * log_std
            + math.log(2 * math.pi)
        )
        log_prob_gauss = log_prob_gauss.sum(dim=-1, keepdim=True)

        correction = 2.0 * (
            math.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh)
        )
        correction = correction.sum(dim=-1, keepdim=True)

        logp = log_prob_gauss - correction
        return actions, logp.view(-1) # => [n_agents, action_dim], [n_agents]