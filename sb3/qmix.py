from torch import nn
import torch
from policy import env_parser

class QMixNetwork(nn.Module):
    def __init__(self, agent_dim, hidden_dim, num_agents):
        super(QMixNetwork, self).__init__()
        self.num_agents = num_agents
        self.hidden_dim = hidden_dim
        
        # Define individual Q-value networks for each agent
        self.q_values = nn.ModuleList([nn.Linear(agent_dim * 2, hidden_dim) for _ in range(num_agents)])
        
        # Define the mixing network
        self.mixer = nn.Sequential(
            nn.Linear(hidden_dim * num_agents, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, state, actions):
        q_vals = [q(state) for q, state in zip(self.q_values, actions)]
        q_vals = torch.stack(q_vals, dim=-1)
        return self.mixer(q_vals.view(-1, self.hidden_dim * self.num_agents))
    

class QMixAgentPolicy(nn.Module):
    def __init__(self, number_agents, agent_dim, hidden_dim):
        super().__init__()
        self.number_agents = number_agents
        self.agent_dim = agent_dim
        self.hidden_dim = hidden_dim

        # Define the QMix network
        self.qmix_network = QMixNetwork(agent_dim, hidden_dim, number_agents)

    def forward(self, obs):
        cur_pos, cur_vel, landmarks, other_agents, random_numbers = env_parser(obs, self.number_agents)
        actions = torch.cat((cur_pos, cur_vel), dim=-1)
        # Forward pass through QMIX network
        q_values = self.qmix_network(cur_pos, actions)

        return q_values