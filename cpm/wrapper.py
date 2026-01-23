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
        self.action_space = env.action_space
        self.observation_space = env.observation_space
    
    def reset(self):
        obs = self.env.reset()
        obs = torch.stack(obs, dim=0).transpose(1, 0)  # (total_envs, n_agents, obs_dim)
        return obs
    
    def step(self, actions):
        actions = torch.from_numpy(actions).to(self.env.device).transpose(1, 0).unsqueeze(-1)
        obs, rews, dones, infos = self.env.step(actions)
        dones = dones.unsqueeze(-1).repeat(1, self.num_agents)  # (total_envs, n_agents)
        infos = [{} for _ in range(self.num_envs)]
        rewards = torch.stack(rews, dim=0).transpose(1, 0)  # (total_envs, n_agents)
        obs = torch.stack(obs, dim=0).transpose(1, 0)  # (total_envs, n_agents, obs_dim)
        return obs, rewards.cpu(), dones.cpu(), infos
    
    def render(self, mode='human', agent_index_focus=None):
        return self.env.render(mode=mode, agent_index_focus=agent_index_focus )


class SMACVecEnv(DummyVecEnv):
    def __init__(self, env_name:str, num_envs:int, max_steps:int, device: str):
        self.env_name = env_name
        self.num_envs = num_envs
        self.max_steps = max_steps
        self.device = device
        self._envs = [gymnasium.make(env_name) for _ in range(num_envs)]
        self._steps = np.zeros(num_envs, dtype=int)
        self.num_agents = self._envs[0].unwrapped.n_agents
        self.reset_infos: List[Dict[str, Any]] = [{} for _ in range(self.num_envs)]
        self.observation_space = gymnasium.spaces.Tuple(
            [gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=self._envs[0].observation_space[i].shape, dtype=self._envs[0].observation_space[i].dtype) for i in range(self.num_agents)]
        )
        self.action_space = gymnasium.spaces.MultiDiscrete(
            [self._envs[0].action_space[0].n for _ in range(self.num_agents)],
        )
    
    def reset(self):
        obs = torch.zeros((self.num_envs, self.num_agents, self.observation_space[0].shape[0]), device=self.device)
        self._steps[:] = 0
        for i, env in enumerate(self._envs):
            ob, _ = env.reset()
            obs[i, ...].copy_(torch.from_numpy(np.stack(ob, axis=0)))
        return obs
    
    def step(self, actions):
        actions = actions.reshape(self.num_envs, self.num_agents)
        rews = torch.zeros(self.num_envs)
        dones = torch.zeros(self.num_envs, dtype=torch.bool)
        obs = torch.zeros((self.num_envs, self.num_agents, self.observation_space[0].shape[0]), device=self.device)
        infos = [{} for _ in range(self.num_envs)]
        for i, env in enumerate(self._envs):
            ob, rew, done, _, info = env.step(actions[i].tolist())
            rews[i] = rew
            dones[i] = done
            infos[i].update(info)
            obs[i, ...].copy_(torch.from_numpy(np.stack(ob, axis=0)))

        self._steps += 1
        rews /= self.num_agents
        rews = rews.unsqueeze(-1).repeat(1, self.num_agents)  # (total_envs, n_agents)
        for i in range(self.num_envs):
            if dones[i] or self._steps[i] >= self.max_steps:
                infos[i]["terminal_observation"] = obs[i]
                infos[i]["TimeLimit.truncated"] = self._steps[i] >= self.max_steps
                self._steps[i] = 0
                dones[i] = True
                ob, _ = self._envs[i].reset()
                obs[i, ...].copy_(torch.from_numpy(np.stack(ob, axis=0)))
        
        dones = dones.unsqueeze(-1).repeat(1, self.num_agents)  # (total_envs, n_agents)
        
        return obs, rews, dones, infos
    
    def render(self, mode='human', agent_index_focus=None):
        pass