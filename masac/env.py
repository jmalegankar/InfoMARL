import numpy as np
import vmas
import torch as th

class RandomAgentCountEnv:
    def __init__(
        self,
        scenario_name="simple_spread",
        agent_count_dict={1: 0.4, 3: 0.2, 5: 0.2, 7: 0.2},
        device="cpu",
        continuous_actions=True,
        max_steps=None,
        seed=None
    ):
        self.scenario_name = scenario_name
        self.agent_count_dict = agent_count_dict
        self.device = device
        self.continuous_actions = continuous_actions
        self.max_steps = max_steps
        self.seed = seed
        self.agent_counts = list(agent_count_dict.keys())
        self.agent_probs = list(agent_count_dict.values())
        self.agent_rng = np.random.default_rng(seed)
        self.num_landmarks = None
        self.current_num_agents = None
        self.env = None
        self._render = False
        self._env_rng = np.random.default_rng(seed)
        self.rnd_rng = th.Generator(device=device)
        self.rnd_rng.manual_seed(seed)

        self.reset()

    def sample_agent_count(self):
        return self.agent_rng.choice(self.agent_counts, p=self.agent_probs)
    
    def make_env(self, n_agents):
        return vmas.make_env(
            scenario=self.scenario_name,
            num_envs=1,
            n_agents=n_agents,
            continuous_actions=self.continuous_actions,
            device=self.device,
            seed=self.seed,
            max_steps=self.max_steps,
        )
    
    def reset(self):
        self.current_num_agents = int(self.sample_agent_count())
        self.num_landmarks = self.current_num_agents
        self.env = self.make_env(self.current_num_agents)
        obs = self.env.reset(seed=self._env_rng.integers(0, 2**32, dtype=np.uint32))  # list of length n_agents
        obs = {k: th.stack(tuple(o[k] for o in obs)) for k in obs[0].keys()}
        obs['rnd_nums'] = th.rand(self.current_num_agents, device=self.device, generator=self.rnd_rng)
        return obs
    
    def render(self, value=True):
        self._render = value
    
    def step(self, actions):
        obs, rewards, dones, infos = self.env.step(actions)
        if self._render:
            self.env.render()
        obs = {k: th.stack(tuple(o[k] for o in obs)) for k in obs[0].keys()}
        obs['rnd_nums'] = th.rand(self.current_num_agents, device=self.device, generator=self.rnd_rng)
        return obs, th.stack(rewards), dones, infos
    
    def close(self):
        if self.env is not None:
            # Seems to be an issue pf pyglet not closing the window properly
            if self._render:
                self.env._env.viewer.close()
            self.env = None
    
    @property
    def num_agents(self):
        return self.current_num_agents


if __name__ == '__main__':
    from masac.simple_spread import Scenario
    import torch
    import time
    env = RandomAgentCountEnv(
        scenario_name=Scenario(),
        agent_count_dict={1: 0.2, 3: 0.4, 5: 0.4},
        seed=42,
        device="cuda",
    )
    env.render(True)
    obs = env.reset()
    for _ in range(100):
        actions =  torch.from_numpy(
            np.stack(tuple(np.stack(env.env.action_space.sample()) for _ in range(env.num_envs)), axis=1)
        )
        obs, rewards, dones, infos = env.step(actions)
        time.sleep(0.01)
    print(obs)
    obs = env.reset()
    for _ in range(100):
        actions =  torch.from_numpy(
            np.stack(tuple(np.stack(env.env.action_space.sample()) for _ in range(env.num_envs)), axis=1)
        )
        obs, rewards, dones, infos = env.step(actions)
        time.sleep(0.01)