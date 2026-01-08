import numpy as np
import torch
from gymnasium.spaces import Box, Discrete
from typing import List, Tuple, Dict


class VMAS_CPM_Wrapper:
    """
    Wrapper to make VMAS environments compatible with CPM/MAAC code.

    By default, uses VMAS discrete actions natively (5-action space):
    [no-op, up, down, left, right]

    Can optionally convert CPM discrete actions to VMAS continuous actions
    by setting continuous_actions=True.
    """
    
    def __init__(self, vmas_env, continuous_actions=False):
        self.env = vmas_env
        self.n_agents = vmas_env.n_agents
        self.continuous_actions = continuous_actions
        self.device = vmas_env.device
        self.num_vmas_envs = vmas_env.num_envs
        self.n = self.n_agents

        # Extract observation dimension
        obs_dim = vmas_env.observation_space[0].shape[-1]
        self.observation_space = [
            Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
            for _ in range(self.n_agents)
        ]

        # Determine action dimension and create action spaces
        if continuous_actions:
            # For continuous actions, we need to know the action dimension for conversion
            if not hasattr(vmas_env.action_space[0], 'shape'):
                raise ValueError("VMAS environment must have continuous action space when continuous_actions=True")
            self.action_dim = vmas_env.action_space[0].shape[-1]
            if self.action_dim < 2:
                print(f"Warning: action_dim={self.action_dim} < 2. Discrete-to-continuous mapping may not work correctly.")
            # CPM expects discrete actions, so create discrete action space
            self.num_discrete_actions = 5  # Standard discrete action mapping
            self.action_space = [Discrete(self.num_discrete_actions) for _ in range(self.n_agents)]
        else:
            # For discrete actions, VMAS handles the action space internally
            # Get the actual number of discrete actions from VMAS
            if hasattr(vmas_env.action_space[0], 'n'):
                self.num_discrete_actions = vmas_env.action_space[0].n
            else:
                self.num_discrete_actions = 5  # Default for 2D movement
            self.action_dim = self.num_discrete_actions
            self.action_space = [Discrete(self.num_discrete_actions) for _ in range(self.n_agents)]
            print(f"Using VMAS discrete actions: {self.num_discrete_actions} actions per agent")
    
    def reset(self) -> List[np.ndarray]:
        """Reset and return observations as list of (num_vmas_envs, obs_dim) arrays."""
        obs = self.env.reset()
        
        obs_list = []
        for agent_obs_tensor in obs:
            agent_obs = agent_obs_tensor.cpu().numpy()
            if agent_obs.ndim == 1:
                agent_obs = agent_obs.reshape(1, -1)
            obs_list.append(agent_obs)
        
        return obs_list
    
    def step(self, actions: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], Dict]:
        """Step with discrete actions, return (obs, rewards, dones, info)."""

        # Convert discrete to continuous if needed (for backwards compatibility)
        if self.continuous_actions:
            vmas_actions = self._discrete_to_continuous(actions)
        else:
            # VMAS uses discrete actions natively - pass through
            vmas_actions = actions

        # Convert to tensors for VMAS
        vmas_actions_list = []
        for act in vmas_actions:
            # Handle one-hot encoded actions from CPM
            if act.ndim == 2:
                # Convert one-hot to discrete indices: (n_envs, action_dim) -> (n_envs,)
                act = np.argmax(act, axis=1)

            act_tensor = torch.from_numpy(act).to(self.device).long()  # Use .long() for discrete actions
            if act_tensor.ndim == 0:
                act_tensor = act_tensor.reshape(self.num_vmas_envs)
            vmas_actions_list.append(act_tensor)

        # Step VMAS
        obs, rewards, dones, infos = self.env.step(vmas_actions_list)
        
        # Convert observations
        obs_list = []
        for agent_obs_tensor in obs:
            agent_obs = agent_obs_tensor.cpu().numpy()
            if agent_obs.ndim == 1:
                agent_obs = agent_obs.reshape(1, -1)
            obs_list.append(agent_obs)
        
        # Convert rewards
        if isinstance(rewards, torch.Tensor):
            reward_np = rewards.cpu().numpy()
            if reward_np.ndim == 0:
                reward_np = np.array([reward_np])
            reward_list = [reward_np.copy() for _ in range(self.n_agents)]
        else:
            reward_list = []
            for agent_reward_tensor in rewards:
                agent_reward = agent_reward_tensor.cpu().numpy()
                if agent_reward.ndim == 0:
                    agent_reward = np.array([agent_reward])
                reward_list.append(agent_reward)
        
        # Convert dones (replicate for all agents since VMAS ends episodes simultaneously)
        dones_np = dones.cpu().numpy().astype(bool)
        if dones_np.ndim == 0:
            dones_np = np.array([dones_np])
        done_list = [dones_np.copy() for _ in range(self.n_agents)]
        
        info_dict = {'n': []}
        return obs_list, reward_list, done_list, info_dict
    
    def _discrete_to_continuous(self, discrete_actions: List[np.ndarray]) -> List[np.ndarray]:
        """
        Convert discrete actions to continuous for VMAS.
        Handles both one-hot encoded actions and discrete indices.

        Action mapping:
        0: [0, 0]     (no-op)
        1: [0, 1]     (up)
        2: [0, -1]    (down)
        3: [-1, 0]    (left)
        4: [1, 0]     (right)
        """
        continuous_actions = []

        for agent_actions in discrete_actions:
            # Handle scalar case
            if agent_actions.ndim == 0:
                n_envs = 1
                agent_actions = np.array([agent_actions])
            else:
                n_envs = agent_actions.shape[0]

            # Convert one-hot to discrete indices if needed
            if agent_actions.ndim == 2:
                # One-hot encoded: (n_envs, action_dim) -> (n_envs,)
                agent_actions = np.argmax(agent_actions, axis=1)

            cont_acts = np.zeros((n_envs, self.action_dim), dtype=np.float32)

            if self.action_dim >= 2:
                mask_up = (agent_actions == 1)
                mask_down = (agent_actions == 2)
                mask_left = (agent_actions == 3)
                mask_right = (agent_actions == 4)

                cont_acts[mask_up, 1] = 1.0     # up
                cont_acts[mask_down, 1] = -1.0  # down
                cont_acts[mask_left, 0] = -1.0  # left
                cont_acts[mask_right, 0] = 1.0  # right

            continuous_actions.append(cont_acts)

        return continuous_actions
    
    def render(self, mode='human'):
        return self.env.render(mode=mode)
    
    def close(self):
        pass


class DummyVecEnvForCPM:
    """
    Vectorized environment wrapper for CPM that handles multiple VMAS instances.
    """
    
    def __init__(self, env_fns: List):
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        
        first_env = self.envs[0]
        self.observation_space = first_env.observation_space
        self.action_space = first_env.action_space
        self.n = first_env.n
        self.num_vmas_envs_per_wrapper = first_env.num_vmas_envs
        
        self.ts = np.zeros(self.num_envs, dtype=int)
        self.actions = None
    
    def reset(self) -> np.ndarray:
        """Reset all environments, return (total_envs, n_agents, obs_dim)."""
        obs_list = [env.reset() for env in self.envs]
        
        all_obs = []
        for env_obs_list in obs_list:
            env_obs_array = np.stack(env_obs_list, axis=1)  # (num_vmas_envs, n_agents, obs_dim)
            for vmas_env_idx in range(self.num_vmas_envs_per_wrapper):
                all_obs.append(env_obs_array[vmas_env_idx])
        
        obs_array = np.stack(all_obs, axis=0)
        self.ts = np.zeros(self.num_envs * self.num_vmas_envs_per_wrapper, dtype=int)
        return obs_array
    
    def step_async(self, actions: np.ndarray):
        """Store actions for next step."""
        self.actions = actions
    
    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List]:
        """Execute stored actions and return results."""
        
        # Reshape actions for each wrapper
        actions_per_env = []
        start_idx = 0
        for _ in range(self.num_envs):
            end_idx = start_idx + self.num_vmas_envs_per_wrapper
            env_actions = self.actions[start_idx:end_idx]
            
            agent_actions = []
            for agent_idx in range(self.n):
                agent_actions.append(env_actions[:, agent_idx])
            
            actions_per_env.append(agent_actions)
            start_idx = end_idx
        
        # Step each environment
        results = [env.step(act_list) for env, act_list in zip(self.envs, actions_per_env)]
        
        obs_list = [r[0] for r in results]
        rewards_list = [r[1] for r in results]
        dones_list = [r[2] for r in results]
        infos = [r[3] for r in results]
        
        # Reshape observations
        all_obs = []
        for env_obs_list in obs_list:
            env_obs_array = np.stack(env_obs_list, axis=1)
            for vmas_env_idx in range(self.num_vmas_envs_per_wrapper):
                all_obs.append(env_obs_array[vmas_env_idx])
        obs = np.stack(all_obs, axis=0)
        
        # Reshape rewards
        all_rewards = []
        for env_rewards_list in rewards_list:
            env_rewards_array = np.stack(env_rewards_list, axis=1)
            for vmas_env_idx in range(self.num_vmas_envs_per_wrapper):
                all_rewards.append(env_rewards_array[vmas_env_idx])
        rewards = np.stack(all_rewards, axis=0)
        
        # Reshape dones
        all_dones = []
        for env_dones_list in dones_list:
            env_dones_array = np.stack(env_dones_list, axis=1)
            for vmas_env_idx in range(self.num_vmas_envs_per_wrapper):
                all_dones.append(env_dones_array[vmas_env_idx])
        dones = np.stack(all_dones, axis=0).astype(bool)
        
        # Handle resets
        self.ts += 1
        for env_idx in range(len(all_obs)):
            if dones[env_idx].all():
                wrapper_idx = env_idx // self.num_vmas_envs_per_wrapper
                wrapper_obs = self.envs[wrapper_idx].reset()
                
                wrapper_start = wrapper_idx * self.num_vmas_envs_per_wrapper
                wrapper_obs_array = np.stack(wrapper_obs, axis=1)
                for vmas_idx in range(self.num_vmas_envs_per_wrapper):
                    obs[wrapper_start + vmas_idx] = wrapper_obs_array[vmas_idx]
                
                self.ts[env_idx] = 0
        
        self.actions = None
        return obs, rewards, dones, infos
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List]:
        """Synchronous step."""
        self.step_async(actions)
        return self.step_wait()
    
    def close(self):
        """Close all environments."""
        for env in self.envs:
            env.close()


def make_vmas_env_for_cpm(scenario: str, n_agents: int, num_envs: int = 1,
                          device: str = "cpu", seed: int = 0,
                          max_steps: int = 100, **kwargs):
    """Create a VMAS environment wrapped for CPM/MAAC.

    Uses discrete actions natively from VMAS (no conversion needed).
    """
    import vmas

    if scenario == "food_collection":
        vmas_env = vmas.make_env(
            scenario=scenario,
            n_agents=n_agents,
            n_food=kwargs.get("n_food", n_agents),
            num_envs=num_envs,
            continuous_actions=False,  # Use discrete actions directly
            max_steps=max_steps,
            seed=seed,
            device=device,
            terminated_truncated=False,
        )
    else:
        vmas_env = vmas.make_env(
            scenario=scenario,
            n_agents=n_agents,
            num_envs=num_envs,
            continuous_actions=False,  # Use discrete actions directly
            max_steps=max_steps,
            seed=seed,
            device=device,
            terminated_truncated=False,
        )

    return VMAS_CPM_Wrapper(vmas_env, continuous_actions=False)


def make_parallel_env_for_cpm(scenario: str, n_agents: int, 
                              n_rollout_threads: int, seed: int,
                              max_steps: int = 100, device: str = "cpu",
                              **kwargs):
    """Create parallel environments for CPM training."""
    def get_env_fn(rank):
        def init_env():
            env = make_vmas_env_for_cpm(
                scenario=scenario,
                n_agents=n_agents,
                num_envs=1,
                device=device,
                seed=seed + rank * 1000,
                max_steps=max_steps,
                **kwargs
            )
            return env
        return init_env
    
    return DummyVecEnvForCPM([get_env_fn(i) for i in range(n_rollout_threads)])