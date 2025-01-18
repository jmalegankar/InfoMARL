import torch as th
import torch.nn as nn
from typing import Union, Dict

import math

class RandomizedAttentionPolicy(nn.Module):
    def __init__(self, agent_dim: int, landmark_dim: int, hidden_dim: int, action_dim: int, device: Union[str, th.device] ="cpu"):
        super().__init__()
        self.agent_dim = agent_dim
        self.landmark_dim = landmark_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.device = device

        self.relu = nn.ReLU()
        self.logsoftmax = nn.LogSoftmax(dim=-2)

        # Initial state estimation
        self.embed1 = nn.Linear(landmark_dim + action_dim, hidden_dim)
        self.embed2 = nn.Linear(hidden_dim, hidden_dim)
        self.init_score = nn.Linear(hidden_dim, hidden_dim)

        # State consensus
        self.concencus1 = nn.Linear(hidden_dim, hidden_dim)
        self.concencus2 = nn.Linear(hidden_dim, hidden_dim)
        self.concencus_score = nn.Linear(hidden_dim, hidden_dim)

        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.logstd_layer = nn.Linear(hidden_dim, action_dim)

    
    @th.jit.export
    def tanh_normal_sample(self, mean, log_std):
        std = th.exp(log_std)
        eps = th.randn_like(mean)
        pre_tanh = mean + std * eps
        action = th.tanh(pre_tanh)

        # diagonal Gaussian log-prob
        var = std.pow(2)
        log_prob_gauss = -0.5 * (((pre_tanh - mean)**2) / (var + 1e-8) + 2*log_std + math.log(2*math.pi))
        log_prob_gauss = log_prob_gauss.sum(dim=-1)  # => shape [1,1]

        # Tanh correction
        # - sum( log( d/dx tanh(x) ) ) = sum( log( 1 - tanh^2(x) ) ), but we do the stable version:
        # formula: 2 * ( log(2) - x - softplus(-2x) )
        correction = 2.0 * (math.log(2.0) - pre_tanh - th.nn.functional.softplus(-2.0 * pre_tanh))
        correction = correction.sum(dim=-1)  # => [1,1]

        log_prob = log_prob_gauss - correction
        return action, log_prob
    
    ################################
    # initial_state_estimation
    ################################
    @th.jit.export
    def initial_state_estimation(self, obs: Dict[str, th.Tensor]):
        embed = self.relu(self.embed2(self.relu(self.embed1(obs['obs']))))  # => [n_agents, n_landmarks, hidden_dim]
        score = self.logsoftmax(self.init_score(embed))  # => [n_agents, n_landmarks, hidden_dim]
        temp_state = embed + score  # => [n_agents, n_landmarks, hidden_dim]
        message = temp_state.mean(dim=-2)  # => [n_agents, hidden_dim]
        return temp_state, message # => [n_agents, n_landmarks, hidden_dim], [n_agents, hidden_dim]
    
    @th.jit.export
    def _create_mask(self, rnd_nums: th.Tensor, rnd: th.Tensor):
        mask = ~(rnd_nums < rnd)  # Compare as per M_ij definition
        return mask.view(-1) # => [n_agents]
    
    @th.jit.export
    def _get_scored_messages(self, messages: th.Tensor, idx: int, rnd: th.Tensor):
        mask = self._create_mask(rnd, rnd[idx])  # => [n_agents]
        messages = messages[mask]  # => [n_agents, hidden_dim]
        messages += self.logsoftmax(self.concencus_score(messages))  # => [n_agents, hidden_dim]
        return messages.sum(dim=-2)  # => [hidden_dim]

    ################################
    # consensus
    ################################
    @th.jit.export
    def consensus(self, temp_states: th.Tensor, obs: Dict[str, th.Tensor], messages: th.Tensor):
        rnd_nums = obs['rnd_nums']  # => [n_agents]
        idx = obs['idx'] # => [n_agents]
        rnd = rnd_nums[idx].view(-1) # => [n_agents]
        messages = self.concencus2(self.relu(self.concencus1(messages)))  # => [n_agents, hidden_dim]
        out = temp_states.mean(dim=-2) + th.stack(
            [self._get_scored_messages(messages, i, rnd) for i in range(rnd.size(0))],
        ) # => [n_agents, hidden_dim]
        mean = self.mean_layer(out)
        logstd = self.logstd_layer(out)
        return mean, logstd
    
    @th.jit.export
    def sample_actions_and_logp(self, obs: Dict[str, th.Tensor]):
        temp_states, messages = self.initial_state_estimation(obs)
        mean, logstd = self.consensus(temp_states, obs, messages)
        actions, logp = self.tanh_normal_sample(mean, logstd)
        return actions, logp

if __name__ == "__main__":
    from masac.simple_spread import Scenario
    from masac.env import RandomAgentCountEnv
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
    print(actions, logp)
    obs, rewards, dones, infos = env.step(actions.unsqueeze(1))
    print(obs, rewards, dones, infos)