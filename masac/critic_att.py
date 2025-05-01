import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        layers = [nn.Linear(self.in_channels, self.hidden_channels[0]), nn.SiLU()]
        for i in range(len(self.hidden_channels) - 1):
            layers.append(nn.Linear(self.hidden_channels[i], self.hidden_channels[i + 1]))
            if i < len(self.hidden_channels) - 2:
                layers.append(nn.SiLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class RAP_att_qvalue(nn.Module):
    def __init__(self, qvalue_config):
        super().__init__()
        
        self.device = qvalue_config["device"]
        self.na = qvalue_config["n_agents"]
        self.obs_dim = qvalue_config["observation_dim_per_agent"]
        self.action_dim = qvalue_config["action_dim_per_agent"]
        
        # Calculate total dimensions
        self.total_obs_dim = self.obs_dim * self.na
        self.total_action_dim = self.action_dim * self.na
        self.total_input_dim = self.total_obs_dim + self.total_action_dim
        
        # Process observations and actions separately before concatenation
        self.obs_processor = nn.Sequential(
            nn.Linear(self.total_obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        self.action_processor = nn.Sequential(
            nn.Linear(self.total_action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Attention layer to process combined input
        self.attention = nn.MultiheadAttention(
            embed_dim=64, 
            num_heads=4, 
            batch_first=True
        )
        
        # Q networks with same input structure as original
        hidden_dim = 128
        self.q1 = MLP(
            in_channels=self.total_input_dim,
            hidden_channels=[hidden_dim * 2,
                            hidden_dim,
                            hidden_dim // 2,
                            1]
        ).to(self.device)
        
        self.q2 = MLP(
            in_channels=self.total_input_dim,
            hidden_channels=[hidden_dim * 2,
                            hidden_dim,
                            hidden_dim // 2,
                            1]
        ).to(self.device)

    def forward(self, observation, action):
        batch_size = observation.shape[0]
        
        # Reshape inputs as in the original implementation
        obs_flat = observation.reshape([observation.shape[0], -1])
        action_flat = action.reshape([action.shape[0], -1])
        
        # Process observation and action
        obs_features = self.obs_processor(obs_flat)
        action_features = self.action_processor(action_flat)
        
        # Prepare for attention - reshape to sequence format
        combined = torch.cat([obs_features, action_features], dim=1)
        seq_len = self.na
        feature_dim = combined.shape[1] // seq_len
        sequence = combined.view(batch_size, seq_len, feature_dim)
        
        # Apply attention for agent interactions
        attn_out, _ = self.attention(sequence, sequence, sequence)
        
        # Flatten back to original format
        attn_flat = attn_out.reshape(batch_size, -1)
        
        # Concatenate with original inputs for skip connection
        obs_action = torch.cat((obs_flat, action_flat), dim=1)
        
        # Calculate Q-values using original input format
        q1 = self.q1(obs_action)
        q2 = self.q2(obs_action)
        
        return q1, q2