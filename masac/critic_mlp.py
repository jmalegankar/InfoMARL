# critic_mlp.py
import torch as th
import torch.nn as nn
from typing import Dict

class MLPQCritic(nn.Module):
    """
    Simple MLP critic with two hidden layers (64 units each).
    Takes obs + action as input, outputs scalar Q.
    """

    def __init__(
        self,
        agent_dim: int,
        action_dim: int,
        landmark_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        # We'll match the actor's assumption about obs dimension:
        self.input_dim = agent_dim + landmark_dim + action_dim

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    @th.jit.export
    def forward(self, obs: Dict[str, th.Tensor], action: th.Tensor) -> th.Tensor:
        # Flatten obs
        x_obs = obs["obs"].view(obs["obs"].size(0), -1)
        x_act = action.view(action.size(0), -1)  # ensure shape [batch_size, action_dim]
        x = th.cat([x_obs, x_act], dim=-1)
        q_value = self.net(x)
        return q_value.view(-1)
