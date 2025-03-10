import torch as th
import torch.nn as nn
from typing import Dict, List

class CustomQFuncCritic(nn.Module):
    def __init__(self, agent_dim, action_dim, landmark_dim, hidden_dim, keys):
        super().__init__()
        self.agent_dim = agent_dim
        self.action_dim = action_dim
        self.landmark_dim = landmark_dim
        self.hidden_dim = hidden_dim

        self.fc1 = nn.ModuleList()
        self.fc2 = nn.ModuleList()
        self.fc3 = nn.ModuleList()
        counter = 0
        self.map = th.jit.annotate(Dict[int, int], {})
        for key in keys:
            self.map[key] = counter
            self.fc1.append(nn.Linear(key*(agent_dim + 2*action_dim + landmark_dim), hidden_dim))
            self.fc2.append(nn.Linear(hidden_dim, hidden_dim))
            self.fc3.append(nn.Linear(hidden_dim, 1))
            counter += 1
        self.relu = nn.ReLU()

    @th.jit.export
    def forward(self, obs:Dict[str, th.Tensor], action:th.Tensor):
        x = th.cat((obs['agent_states'][0], action.view(-1)), dim=-1) # => [n_agents*(agent_dim+action_dim) + n_landmarks*landmark_dim + n_agents*action_dim]
        key = action.size(0)
        key = self.map[key]
        for idx, (fc1, fc2, fc3) in enumerate(zip(self.fc1, self.fc2, self.fc3)):
            if idx == key:
                x = self.relu(fc1(x))
                x = self.relu(fc2(x))
                x = fc3(x)
        return x

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
    q_func = th.jit.script(CustomQFuncCritic(agent_dim=2, action_dim=2, landmark_dim=2, hidden_dim=64, keys=[1, 3, 5]))
    q_value = q_func(obs, actions)
    print(q_value)