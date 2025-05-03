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
        self.observation_dim_per_agent = qvalue_config["observation_dim_per_agent"]
        self.action_dim_per_agent = qvalue_config["action_dim_per_agent"]

        self.q1 = MLP(
            in_channels=(self.observation_dim_per_agent + self.action_dim_per_agent) * self.na,
            hidden_channels=[(self.observation_dim_per_agent + self.action_dim_per_agent) * 2 * self.na,
                             (self.observation_dim_per_agent + self.action_dim_per_agent) * self.na,
                             (self.observation_dim_per_agent + self.action_dim_per_agent),
                             1]).to(self.device)
        self.q2 = MLP(
            in_channels=(self.observation_dim_per_agent + self.action_dim_per_agent) * self.na,
            hidden_channels=[(self.observation_dim_per_agent + self.action_dim_per_agent) * 2 * self.na,
                             (self.observation_dim_per_agent + self.action_dim_per_agent) * self.na,
                             (self.observation_dim_per_agent + self.action_dim_per_agent),
                             1]).to(self.device)

    def forward(self, observation, action):
        obs_action = torch.cat((observation.reshape([observation.shape[0], -1]),
                                         action.reshape([action.shape[0], -1])), dim=1)
        q1 = self.q1(obs_action)
        q2 = self.q2(obs_action)
        return q1, q2

import torch
import torch.nn as nn
import torch.nn.functional as F


class AgentEncoder(nn.Module):
    def __init__(self, obs_dim, action_dim, embed_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim + action_dim, embed_dim),
            nn.SiLU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, obs, actions):
        x = torch.cat([obs, actions], dim=-1)  # [B, N, obs+act]
        return self.encoder(x)  # [B, N, embed_dim]


class AttentionCritic(nn.Module):
    def __init__(self, qvalue_config):
        super().__init__()
        self.device = qvalue_config["device"]
        self.n_heads = qvalue_config["n_heads"]
        self.n_agents = qvalue_config["n_agents"]
        self.obs_dim = qvalue_config["observation_dim_per_agent"]
        self.action_dim = qvalue_config["action_dim_per_agent"]
        self.embed_dim = qvalue_config["embed_dim"]

        self.agent_encoder = AgentEncoder(self.obs_dim, self.action_dim, self.embed_dim)

        self.attn = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.n_heads, batch_first=True)

        self.processor = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.SiLU(),
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.SiLU(),
            nn.LayerNorm(self.embed_dim),
        )

        self.q_head_1 = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.SiLU(),
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, 1),
        )

        self.q_head_2 = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.SiLU(),
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, 1),
        )

    def forward(self, obs, actions):
        # obs/actions: [B, N, D]
        x = self.agent_encoder(obs, actions)  # [B, N, embed_dim]
        attn_out, _ = self.attn(x, x, x)  # self-attention across agents
        x = x + attn_out  # residual connection
        x = self.processor(x)  
        pooled = x.mean(dim=1)  # [B, embed_dim]
        q1 = self.q_head_1(pooled)  # [B, 1]
        q2 = self.q_head_2(pooled)
        return q1, q2
