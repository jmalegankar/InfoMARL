import gymnasium
import numpy as np

from stable_baselines3.common.vec_env import DummyVecEnv

from typing import Any, Dict, List

class SMACVecEnv(DummyVecEnv):
    def __init__(self, env_name:str, num_envs:int, max_steps:int, rnd_nums=False):
        self.env_name = env_name
        self.num_envs = num_envs
        self.max_steps = max_steps
        self.rnd_nums = rnd_nums
        self._envs = [gymnasium.make(env_name) for _ in range(num_envs)]
        self._steps = np.zeros(num_envs, dtype=int)
        self.rnd_nums = rnd_nums
        self.num_agents = self._envs[0].unwrapped.n_agents
        self.reset_infos: List[Dict[str, Any]] = [{} for _ in range(self.num_envs)]
        if self.rnd_nums:
            self.observation_space = gymnasium.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.num_agents, self._envs[0].observation_space[0].shape[-1] + 1),
                dtype=self._envs[0].observation_space[0].dtype
            )
        else:
            self.observation_space = gymnasium.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.num_agents, self._envs[0].observation_space[0].shape[-1]),
                dtype=self._envs[0].observation_space[0].dtype
            )
        self.action_space = gymnasium.spaces.MultiDiscrete(
            [self._envs[0].action_space[0].n for _ in range(self.num_agents)],
        )
    
    def reset(self):
        obs = np.zeros((self.num_envs, *self.observation_space.shape), dtype=self.observation_space.dtype)
        self._steps[:] = 0
        for i, env in enumerate(self._envs):
            ob, _ = env.reset()
            if self.rnd_nums:
                obs[i, :, :-1] = np.stack(ob, axis=0)
                obs[i, :, -1] = np.random.rand(self.num_agents)
            else:
                obs[i, ...] = np.stack(ob, axis=0)
        return obs
    
    def step(self, actions):
        actions = actions.reshape(self.num_envs, self.num_agents)
        rews = np.zeros(self.num_envs)
        dones = np.zeros(self.num_envs, dtype=bool)
        obs = np.zeros((self.num_envs, *self.observation_space.shape), dtype=self.observation_space.dtype)
        infos = [{} for _ in range(self.num_envs)]
        for i, env in enumerate(self._envs):
            ob, rew, done, _, info = env.step(actions[i].tolist())
            rews[i] = rew
            dones[i] = done
            infos[i].update(info)
            if self.rnd_nums:
                obs[i, :, :-1] = np.stack(ob, axis=0)
                obs[i, :, -1] = np.random.rand(self.num_agents)
            else:
                obs[i, ...] = np.stack(ob, axis=0)

        self._steps += 1
        rews /= self.num_agents
        for i in range(self.num_envs):
            if dones[i] or self._steps[i] >= self.max_steps:
                infos[i]["terminal_observation"] = obs[i]
                infos[i]["TimeLimit.truncated"] = self._steps[i] >= self.max_steps
                self._steps[i] = 0
                dones[i] = True
                ob, _ = self._envs[i].reset()
                if self.rnd_nums:
                    obs[i, :, :-1] = np.stack(ob, axis=0)
                    obs[i, :, -1] = np.random.rand(self.num_agents)
                else:
                    obs[i, ...] = np.stack(ob, axis=0)

        return obs, rews, dones, infos
    
    def render(self, mode='human', agent_index_focus=None):
        pass