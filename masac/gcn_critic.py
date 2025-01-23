import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from typing import Dict

class GCNQCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        # Ensure the input size of gcn1 matches the feature size of `obs['obs']`.
        self.gcn1 = GCNConv(state_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    @torch.jit.export
    def forward(self, obs: Dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
        # # Debug the shapes of the inputs
        # print("obs['obs'] shape (node features):", obs['obs'].shape)
        # print("obs['edge_index'] shape:", obs['edge_index'].shape)
        # print("action shape before processing:", action.shape)

        # # Ensure obs['obs'] has the expected shape [num_nodes, feature_dim]
        # x = obs['obs']
        # edge_index = obs['edge_index']

        # if x.ndim != 2:
        #     raise ValueError(f"Expected obs['obs'] to have shape [num_nodes, feature_dim], but got {x.shape}")

        # # Ensure edge_index has the correct shape [2, num_edges]
        # if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        #     raise ValueError(f"Expected obs['edge_index'] to have shape [2, num_edges], but got {edge_index.shape}")

        # # Process state with GCN
        # x = F.relu(self.gcn1(x, edge_index))
        # print("Shape of x after GCN1:", x.shape)
        # x = F.relu(self.gcn2(x, edge_index))
        # print("Shape of x after GCN2:", x.shape)

        # # Ensure action is correctly shaped
        # if action.ndim == 1:  # Fix for 1D action tensors
        #     action = action.unsqueeze(0)

        # # Match action dimensions with node dimensions
        # if x.shape[0] != action.shape[0]:
        #     if action.shape[0] == 1:
        #         action = action.expand(x.shape[0], -1)  # Broadcast actions to match nodes
        #     else:
        #         raise ValueError(f"Mismatch in number of nodes ({x.shape[0]}) and actions ({action.shape[0]})")

        # print("Action shape after expansion (if applied):", action.shape)

        # # Concatenate state and action
        # x = torch.cat([x, action], dim=-1)  # Shape: [num_nodes, hidden_dim + action_dim]
        # print("Shape of x after concatenation with action:", x.shape)

        # # Compute Q-values
        # x = F.relu(self.fc1(x))  # Shape: [num_nodes, hidden_dim]
        # q_values = self.fc2(x)  # Shape: [num_nodes, 1]
        # print("Shape of q_values:", q_values.shape)

        #return q_values.squeeze(-1)  # Shape: [num_nodes]
        x = obs['obs']  # Node features
        edge_index = obs['edge_index']  # Edge indices

        # Ensure obs['obs'] has the correct shape
        if x.ndim == 3 and x.shape[0] == 1:  # Handle [1, num_nodes, feature_dim]
            x = x.squeeze(0)  # Remove the batch dimension

        if x.ndim != 2:
            raise ValueError(f"Expected obs['obs'] to have shape [num_nodes, feature_dim], but got {x.shape}")

        # Ensure edge_index has the correct shape [2, num_edges]
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"Expected obs['edge_index'] to have shape [2, num_edges], but got {edge_index.shape}")

        # Process state with GCN
        x = F.relu(self.gcn1(x, edge_index))
        x = F.relu(self.gcn2(x, edge_index))

        # Ensure action shape matches nodes
        if action.ndim == 1:  # If action is 1D, unsqueeze it
            action = action.unsqueeze(0)

        if x.shape[0] != action.shape[0]:
            if action.shape[0] == 1:
                action = action.expand(x.shape[0], -1)  # Broadcast actions to match nodes
            else:
                raise ValueError(f"Mismatch in number of nodes ({x.shape[0]}) and actions ({action.shape[0]})")

        # Concatenate state and action
        x = torch.cat([x, action], dim=-1)

        # Compute Q-values
        x = F.relu(self.fc1(x))
        q_values = self.fc2(x)
        return q_values.squeeze(-1)


# Test the critic with an example input to ensure dimensional consistency
if __name__ == "__main__":
    # Example input: 1 node with 4 features, 1 edge, and action_dim=2
    state_dim = 4  # This must match the feature size of obs['obs']
    action_dim = 2
    hidden_dim = 512

    critic = GCNQCritic(state_dim, action_dim, hidden_dim)

    # Example observation and action tensors
    obs = {
        'obs': torch.randn((1, state_dim)),  # 1 node with 4 features
        'edge_index': torch.tensor([[0], [0]])  # Single self-loop edge
    }
    action = torch.randn((1, action_dim))  # Action for 1 node

    try:
        q_values = critic(obs, action)
        print("Q-values:", q_values)
    except ValueError as e:
        print("Error:", e)
