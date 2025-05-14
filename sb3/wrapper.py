import gymnasium
import torch
import numpy as np

from stable_baselines3.common.vec_env import DummyVecEnv

from typing import Any, Dict, List

class VMASVecEnv(DummyVecEnv):
    def __init__(self, env, rnd_nums=False):
        self.env = env
        self.rnd_nums = rnd_nums
        self.num_envs = env.num_envs
        self.num_agents = env.n_agents
        self.reset_infos: List[Dict[str, Any]] = [{} for _ in range(self.num_envs)]
        if self.rnd_nums:
            self.observation_space = gymnasium.spaces.Box(
                low=np.ones((self.num_agents, env.observation_space[0].shape[-1] + 1)) * env.observation_space[0].low[0],
                high=np.ones((self.num_agents, env.observation_space[0].shape[-1] + 1)) * env.observation_space[0].high[0],
                shape=(self.num_agents, env.observation_space[0].shape[-1] + 1),
                dtype=env.observation_space[0].dtype
            )
        else:
            self.observation_space = gymnasium.spaces.Box(
                low=np.ones((self.num_agents, env.observation_space[0].shape[-1])) * env.observation_space[0].low,
                high=np.ones((self.num_agents, env.observation_space[0].shape[-1])) * env.observation_space[0].high,
                shape=(self.num_agents, env.observation_space[0].shape[1]),
                dtype=env.observation_space[0].dtype
            )
        self.action_space = gymnasium.spaces.Box(
            low=np.ones((self.num_agents, env.action_space[0].shape[-1])) * env.action_space[0].low,
            high=np.ones((self.num_agents, env.action_space[0].shape[-1])) * env.action_space[0].high,
            shape=(self.num_agents, env.action_space[0].shape[-1]),
            dtype=env.action_space[0].dtype
        )
    
    def reset(self):
        obs = self.env.reset()
        if self.rnd_nums:
            rnd_nums = torch.rand(self.num_envs, self.num_agents, device=self.env.device).unsqueeze(-1)
            obs = torch.stack(obs, dim=0).transpose(1, 0)
            obs = torch.cat([obs, rnd_nums], dim=-1)
        else:
            obs = torch.stack(obs, dim=0).transpose(1, 0)
        return obs.cpu().numpy()
    
    def step(self, actions):
        actions = torch.from_numpy(actions).to(self.env.device).transpose(1, 0)
        obs, rewards, truncated, terminated, _ = self.env.step(actions)
        infos = [{} for _ in range(self.num_envs)]
        rewards = sum(rewards) / self.num_agents
        if self.rnd_nums:
            rnd_nums = torch.rand(self.num_envs, self.num_agents, device=self.env.device).unsqueeze(-1)
            obs = torch.stack(obs, dim=0).transpose(1, 0)
            obs = torch.cat([obs, rnd_nums], dim=-1)
        else:
            obs = torch.stack(obs, dim=0).transpose(1, 0)
        dones = torch.logical_or(terminated, truncated)
        if dones.any():
            for i in range(self.num_envs):
                infos[i]["terminal_observation"] = obs[i].cpu()
                infos[i]["TimeLimit.truncated"] = True
            obs = self.reset()
            return obs, rewards.cpu().numpy(), dones.cpu().numpy(), infos
        else:        
            return obs.cpu().numpy(), rewards.cpu().numpy(), dones.cpu().numpy(), infos
    
    def render(self, mode='human'):
        return self.env.render(mode=mode)