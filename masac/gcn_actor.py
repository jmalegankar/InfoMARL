import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from typing import Dict

class GCNActor(nn.Module):
    def __init__(self, agent_dim: int, landmark_dim: int, hidden_dim: int, action_dim: int, device="cpu"):
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        # GCN Layers
        self.gcn1 = GCNConv(agent_dim + landmark_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)

        # Fully Connected Layers for output
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_logstd = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs: Dict[str, th.Tensor]):
        x = obs['obs']  # Node features
        edge_index = obs['edge_index']  # Edge indices

        # Process with GCN layers
        x = F.relu(self.gcn1(x, edge_index))
        x = F.relu(self.gcn2(x, edge_index))

        # Ensure output matches the number of agents
        assert x.shape[0] == obs['obs'].shape[0], "Mismatch in number of agents and actions"
        mean = self.fc_mean(x)
        logstd = self.fc_logstd(x)
        return mean, logstd

    @th.jit.export
    def sample_actions_and_logp(self, obs: Dict[str, th.Tensor]):
        mean, logstd = self.forward(obs)
        std = th.exp(logstd)
        eps = th.randn_like(mean)
        pre_tanh = mean + std * eps
        actions = th.tanh(pre_tanh)

        # Squeeze to ensure correct shape [num_agents, action_dim]
        actions = actions.squeeze(0)
        mean = mean.squeeze(0)
        logstd = logstd.squeeze(0)

        # Compute log probabilities
        var = std.pow(2)
        log_prob_gauss = -0.5 * (((pre_tanh - mean) ** 2) / (var + 1e-8) + 2 * logstd + th.log(2 * th.tensor(th.pi, device=mean.device)))
        log_prob_gauss = log_prob_gauss.sum(dim=-1)

        correction = 2.0 * (th.log(th.tensor(2.0, device=pre_tanh.device)) - pre_tanh - F.softplus(-2.0 * pre_tanh))
        correction = correction.sum(dim=-1)

        logp = log_prob_gauss - correction
        return actions, logp

if __name__ == "__main__":
    # Example usage
    from torch_geometric.data import Data

    # Define graph-based observation
    num_nodes = 5
    feature_dim = 4
    action_dim = 2
    hidden_dim = 64

    obs = {
        'obs': th.randn((num_nodes, feature_dim)),
        'edge_index': th.tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3], [1, 2, 3, 4, 0, 3, 4, 0, 1]], dtype=th.long),
    }

    actor = GCNActor(agent_dim=2, landmark_dim=2, hidden_dim=hidden_dim, action_dim=action_dim)
    actions, logp = actor.sample_actions_and_logp(obs)
    print("Actions:", actions)
    print("Log probabilities:", logp)
