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
    