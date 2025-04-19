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

class RAP_qvalue(nn.Module):
    def __init__(self, qvalue_config):
        super().__init__()
        
        self.device = qvalue_config["device"]
        self.na = qvalue_config["n_agents"]
        self.obs_dim = qvalue_config["observation_dim_per_agent"]
        self.act_dim = qvalue_config["action_dim_per_agent"]
        
        # Input processing layers - process each agent separately first
        self.obs_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.obs_dim, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.LayerNorm(64)
            ) for _ in range(self.na)
        ])
        
        self.act_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.act_dim, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.LayerNorm(64)
            ) for _ in range(self.na)
        ])
        
        # Multi-head attention to model agent relationships
        self.attention = nn.MultiheadAttention(
            embed_dim=128,  # Combined obs+act features
            num_heads=4,
            batch_first=True,
            dropout=0.1
        )
        
        # Joint processing after attention
        joint_dim = 128 * self.na
        
        # Q1 network
        self.q1 = nn.Sequential(
            nn.Linear(joint_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Q2 network
        self.q2 = nn.Sequential(
            nn.Linear(joint_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, observation, action):
        batch_size = observation.shape[0]
        
        # Ensure proper dimensions
        # observation should be [batch_size, n_agents, obs_dim]
        # action should be [batch_size, n_agents, act_dim]
        if observation.dim() == 2:
            observation = observation.view(batch_size, self.na, self.obs_dim)
        if action.dim() == 2:
            action = action.view(batch_size, self.na, self.act_dim)
        
        # Split by agent
        obs_per_agent = [observation[:, i] for i in range(self.na)]
        act_per_agent = [action[:, i] for i in range(self.na)]
        
        # Encode observations and actions for each agent
        encoded_obs = [self.obs_encoders[i](obs_per_agent[i]) for i in range(self.na)]
        encoded_act = [self.act_encoders[i](act_per_agent[i]) for i in range(self.na)]
        
        # Combine obs and action features
        combined_features = [torch.cat([encoded_obs[i], encoded_act[i]], dim=1) for i in range(self.na)]
        combined_features = torch.stack(combined_features, dim=1)  # [batch, n_agents, 128]
        
        # Apply self-attention to model agent interactions
        attended_features, _ = self.attention(
            combined_features, 
            combined_features, 
            combined_features
        )
        
        # Add residual connection
        attended_features = attended_features + combined_features
        
        # Flatten for Q networks
        flat_features = attended_features.reshape(batch_size, -1)
        
        # Get Q values
        q1 = self.q1(flat_features)
        q2 = self.q2(flat_features)
        
        return q1, q2