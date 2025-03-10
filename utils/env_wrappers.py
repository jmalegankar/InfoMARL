import torch as th
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union

class EnvironmentWrapper:
    """
    Wrapper for standardizing environment interfaces
    """
    def __init__(self, env, n_agents: int, device: Union[str, th.device] = "cpu"):
        self.env = env
        self.n_agents = n_agents
        self.device = device if isinstance(device, th.device) else th.device(device)
        
        # Determine if environment is using VMAS or MPE interface
        self.is_vmas = hasattr(env, 'observation_space')
        
        # Determine observation and action dimensions
        if self.is_vmas:
            # VMAS environment
            self.observation_dim = env.observation_space[0].shape[0]
            self.action_dim = env.action_space[0].shape[0]
            self.continuous_actions = True
        else:
            # MPE environment
            self.observation_dim = len(env.observation_space)
            self.action_dim = env.action_space[0].shape[0]
            self.continuous_actions = not isinstance(env.action_space[0], np.ndarray)
    
    def reset(self) -> th.Tensor:
        """
        Reset the environment
        
        Returns:
            obs: Observation tensor [batch_size, n_agents, obs_dim]
        """
        if self.is_vmas:
            # VMAS
            obs = th.stack(self.env.reset()).squeeze(1).to(self.device)
        else:
            # MPE
            obs = self.env.reset()
            obs = th.tensor(obs, dtype=th.float32).to(self.device)
        
        return obs
    
    def step(self, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor, Dict[str, Any]]:
        """
        Take a step in the environment
        
        Args:
            actions: Action tensor [batch_size, n_agents, action_dim]
            
        Returns:
            obs: Next observation tensor [batch_size, n_agents, obs_dim]
            rewards: Reward tensor [batch_size, n_agents]
            dones: Done flag tensor [batch_size]
            info: Info dictionary
        """
        if self.is_vmas:
            # VMAS
            actions_np = actions.cpu().numpy()
            next_obs, rewards, dones, info = self.env.step(actions_np)
            next_obs = th.stack(next_obs).to(self.device)
            rewards = th.tensor(rewards, device=self.device)
            dones = th.tensor(dones, device=self.device)
        else:
            # MPE
            actions_np = actions.cpu().numpy()
            next_obs, rewards, dones, info = self.env.step(actions_np)
            next_obs = th.tensor(next_obs, dtype=th.float32).to(self.device)
            rewards = th.tensor(rewards, dtype=th.float32).to(self.device)
            dones = th.tensor([all(dones)], device=self.device)
        
        return next_obs, rewards, dones, info
    
    def reset_at(self, idx: int) -> th.Tensor:
        """
        Reset a specific environment in case of parallel environments
        
        Args:
            idx: Index of environment to reset
            
        Returns:
            obs: Observation tensor [n_agents, obs_dim]
        """
        if hasattr(self.env, 'reset_at'):
            # VMAS
            obs = th.stack(self.env.reset_at(idx)).squeeze(1).to(self.device)
        else:
            # MPE
            obs = self.env.reset()
            obs = th.tensor(obs, dtype=th.float32).to(self.device)
        
        return obs
    
    def render(self, mode: str = 'human') -> Union[None, np.ndarray]:
        """
        Render the environment
        
        Args:
            mode: Rendering mode
            
        Returns:
            frame: Rendered frame if mode is 'rgb_array', else None
        """
        if hasattr(self.env, 'render'):
            return self.env.render(mode=mode)
        return None

def make_env_for_training(args, device):
    """
    Create environment for training
    
    Args:
        args: Command line arguments
        device: Device to place tensors on
        
    Returns:
        envs: Environment wrapper
    """
    try:
        # Try to import VMAS
        from vmas import make_env
        
        # Create VMAS environment
        env = make_env(
            scenario=args.scenario,  # Note: the parameter name is 'scenario' not 'scenario_name'
            num_envs=args.num_envs,
            device=device,
            continuous_actions=True,
            seed=args.seed,
            max_steps=args.max_episode_len,
            n_agents=args.n_agents
        )
        
        # Wrap environment
        return EnvironmentWrapper(env, args.n_agents, device)
    except ImportError:
        raise ImportError("Could not import either VMAS or MPE. Please install one of them.")

def make_env_for_evaluation(args, device):
    """
    Create environment for evaluation
    
    Args:
        args: Command line arguments
        device: Device to place tensors on
        
    Returns:
        env: Environment wrapper
    """
    try:
        # Try to import VMAS
        from vmas import make_env
        
        # Create VMAS environment - note the parameter name change
        env = make_env(
            scenario=args.scenario,  # Changed from scenario_name to scenario
            num_envs=1,
            device=device,
            continuous_actions=True,
            seed=args.seed + 100,  # Different seed for evaluation
            max_steps=args.max_episode_len,
            n_agents=args.n_agents
        )
        
        # Wrap environment
        return EnvironmentWrapper(env, args.n_agents, device)
    except ImportError:
        raise ImportError("Could not import either VMAS or MPE. Please install one of them.")