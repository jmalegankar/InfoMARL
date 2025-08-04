import gymnasium
import numpy as np

class XORGameEnv(gymnasium.Env):
    def __init__(self, n_agents=2, n_actions=2):
        super(XORGameEnv, self).__init__()
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.action_space = gymnasium.spaces.MultiDiscrete([n_actions] * n_agents)
        self.observation_space = gymnasium.spaces.Box(low=-1, high=1, shape=(n_agents, n_actions + 1), dtype=np.float32)
        self.done = False
    
    def reset(self, seed=None):
        np.random.seed(seed)
        self.obs = np.zeros((self.n_agents, self.n_actions + 1), dtype=np.float32)
        self.obs[:, -1] = np.random.rand(self.n_agents)  # Random noise
        self.done = False
        return self.obs, {}
    
    def step(self, actions):
        if self.done:
            raise RuntimeError("Environment is done. Please reset it before stepping.")
        
        # XOR logic: only one agent can take action 1 to get a reward
        actions = actions.reshape(self.n_agents)
        taken = set()
        for i, act in enumerate(actions):
            self.obs[i, act] = 1.0
            if act in taken:
                reward = 0.0
                break
            taken.add(act)
        else:
            reward = 1.0
        self.done = True
        obs = self.obs.copy()
        obs[:, -1] = np.random.rand(self.n_agents)  # Random noise
        return obs, reward, self.done, False, {}