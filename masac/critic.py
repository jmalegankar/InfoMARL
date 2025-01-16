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
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=1, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    @th.jit.export
    def forward(self, obs:Dict[str, th.Tensor], action:th.Tensor):
        all_features = th.cat((obs['agent_states'], obs['obs'], action.repeat(obs['agent_states'].size(0), 1, 1)), dim=-1) # => [n_agents, n_landmarks, agent_dim + action_dim + landmark_dim + action_dim]
        all_features = self.relu(self.fc(all_features)) # => [n_agents, n_landmarks, hidden_dim]
        attn_output, _ = self.attention(all_features, all_features, all_features)
        aggregated = attn_output.mean(dim=1) # => [n_agents, hidden_dim]
        q_value = self.output_layer(aggregated) # => [n_agents, 1]
        return q_value.squeeze() # => [n_agents]

if __name__ == "__main__":
    from masac.simple_spread import Scenario
    from masac.env import RandomAgentCountEnv
    from masac.actor import RandomizedAttentionPolicy
    env = RandomAgentCountEnv(
        scenario_name=Scenario(),
        agent_count_dict={1: 0.2, 3: 0.4, 5: 0.4},
        seed=42,
        device="cpu",
    )
    policy = th.jit.script(RandomizedAttentionPolicy(
        agent_dim=2,
        landmark_dim=2,
        hidden_dim=64,
        action_dim=2,
        device="cpu",
    ))
    obs = env.reset()
    actions, logp = policy.sample_actions_and_logp(obs)
    q_func = th.jit.script(CustomQFuncCritic(agent_dim=2, action_dim=2, landmark_dim=2, hidden_dim=64))
    q_value = q_func(obs, actions)
    print(q_value)