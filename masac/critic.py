import torch as th
import torch.nn as nn
from typing import Dict

class CustomQFuncCritic(nn.Module):
    def __init__(self, agent_dim, action_dim, landmark_dim, hidden_dim):
        super().__init__()
        self.agent_dim = agent_dim
        self.action_dim = action_dim
        self.landmark_dim = landmark_dim
        self.hidden_dim = hidden_dim

        self.fc = nn.Linear(agent_dim + action_dim + landmark_dim + action_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, 1)

    @th.jit.export
    def forward(self, obs:Dict[str, th.Tensor]):
        all_features = th.cat((obs['agent_states'], obs['obs']), dim=-1) # => [n_agents, agent_dim + action_dim + landmark_dim + action_dim]
        all_features = self.fc(all_features)
        attn_output, _ = self.attention(all_features, all_features, all_features)
        aggregated = attn_output.mean(dim=1)  # => shape [1, hidden_dim]
        q_value = self.output_layer(aggregated)  # => shape [1,1]
        return q_value.squeeze()
