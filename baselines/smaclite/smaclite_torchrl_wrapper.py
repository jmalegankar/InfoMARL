import copy
from typing import Any, Callable, Dict, List, Optional
from enum import Enum
from dataclasses import dataclass

import torch
import numpy as np
import gymnasium as gym

from tensordict import TensorDict, TensorDictBase
from torchrl.data import Composite, Bounded, Categorical, Binary, Unbounded
from torchrl.envs import EnvBase, Transform, RewardSum, StepCounter
from torchrl.envs.utils import check_env_specs
from torchrl.envs import step_mdp


# ============================================================================
# TorchRL Wrapper for SMACLite
# ============================================================================

class SMACliteTorchRLWrapper(EnvBase):
    """
    Output TensorDict structure:
        agents:
            observation: [n_agents, obs_dim]
            action_mask: [n_agents, n_actions]
            action: [n_agents] (categorical)
        state: [state_dim]
        done: [1]
        terminated: [1]
        truncated: [1]
        reward: [1]  # shared reward
    """
    
    def __init__(
        self,
        env_name: str = "smaclite/2s3z-v0",
        device: torch.device = None,
        seed: Optional[int] = None,
        categorical_actions: bool = True,
        **kwargs
    ):
        super().__init__(device=device, batch_size=torch.Size([]))
        
        self._env_name = env_name
        self._categorical_actions = categorical_actions
        
        # Create environment
        self._env = gym.make(env_name, **kwargs)
        self._unwrapped = self._env.unwrapped
        
        # Extract environment info
        self._n_agents = len(self._env.observation_space)
        self._obs_dim = self._env.observation_space[0].shape[0]
        self._n_actions = self._env.action_space[0].n
        
        # Get state dim by doing a reset and calling get_state
        self._env.reset(seed=seed)
        state = np.asarray(self._unwrapped.get_state())
        self._state_dim = state.shape[0]
        
        # Store seed
        self._seed = seed
        
        # Agent group name -> for BenchMARL
        self.group_map = {"agents": [f"agent_{i}" for i in range(self._n_agents)]}
        
        # Build specs
        self._make_specs()
        
    def _make_specs(self):
        device = self.device
        
        # Observation spec: [n_agents, obs_dim]
        obs_space = self._env.observation_space[0]
        obs_spec = Bounded(
            low=obs_space.low[0],
            high=obs_space.high[0],
            shape=(self._n_agents, self._obs_dim),
            dtype=torch.float32,
            device=device,
        )
        
        # Action mask spec: [n_agents, n_actions]
        action_mask_spec = Binary(
            n=self._n_actions,
            shape=(self._n_agents, self._n_actions),
            dtype=torch.bool,
            device=device,
        )
        
        # State spec: [state_dim]
        state_spec = Unbounded(
            shape=(self._state_dim,),
            dtype=torch.float32,
            device=device,
        )
        
        # Full observation spec (includes state at top level for CTDE)
        self.observation_spec = Composite(
            agents=Composite(
                observation=obs_spec,
                action_mask=action_mask_spec,
                shape=(self._n_agents,),
                device=device,
            ),
            state=state_spec,
            device=device,
        )
        
        # Action spec: [n_agents] categorical or [n_agents, n_actions] one-hot
        if self._categorical_actions:
            action_spec = Categorical(
                n=self._n_actions,
                shape=(self._n_agents,),
                dtype=torch.int64,
                device=device,
            )
        else:
            action_spec = Binary(
                n=self._n_actions,
                shape=(self._n_agents, self._n_actions),
                dtype=torch.int64,
                device=device,
            )
        
        self.action_spec = Composite(
            agents=Composite(
                action=action_spec,
                shape=(self._n_agents,),
                device=device,
            ),
            device=device,
        )
        
        # Reward spec: shared reward
        self.reward_spec = Composite(
            agents=Composite(
                reward=Unbounded(shape=(self._n_agents, 1), dtype=torch.float32, device=device),
                shape=(self._n_agents,),
                device=device,
            ),
            device=device,
        )
        
        # Done spec
        self.done_spec = Composite(
            done=Binary(n=1, shape=(1,), dtype=torch.bool, device=device),
            terminated=Binary(n=1, shape=(1,), dtype=torch.bool, device=device),
            truncated=Binary(n=1, shape=(1,), dtype=torch.bool, device=device),
            device=device,
        )
        
    def _set_seed(self, seed: Optional[int]):
        """Set the random seed."""
        self._seed = seed
        
    def _reset(self, tensordict: TensorDictBase = None) -> TensorDictBase:
        """Reset the environment."""
        obs_tuple, info = self._env.reset(seed=self._seed)
        
        # Convert observations: tuple of arrays → [n_agents, obs_dim]
        obs = torch.tensor(np.stack(obs_tuple), dtype=torch.float32, device=self.device)
        
        # Get global state
        state = torch.tensor(
            np.asarray(self._unwrapped.get_state()), 
            dtype=torch.float32, 
            device=self.device
        )
        
        # Get action masks: [n_agents, n_actions]
        avail = np.asarray(self._unwrapped.get_avail_actions())
        action_mask = torch.tensor(avail > 0, dtype=torch.bool, device=self.device)
        
        return TensorDict(
            {
                "agents": TensorDict(
                    {
                        "observation": obs,
                        "action_mask": action_mask,
                    },
                    batch_size=(self._n_agents,),
                    device=self.device,
                ),
                "state": state,
                "done": torch.tensor([False], dtype=torch.bool, device=self.device),
                "terminated": torch.tensor([False], dtype=torch.bool, device=self.device),
                "truncated": torch.tensor([False], dtype=torch.bool, device=self.device),
            },
            batch_size=self.batch_size,
            device=self.device,
        )
    
    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Execute one environment step."""
        # Extract actions from tensordict
        actions = tensordict["agents", "action"]
        
        if self._categorical_actions:
            # actions: [n_agents] → list of ints
            action_list = actions.cpu().numpy().tolist()
        else:
            # one-hot: [n_agents, n_actions] → list of ints
            action_list = actions.argmax(dim=-1).cpu().numpy().tolist()
        
        # Enforce action masking -> SMACLite will reject invalid actions
        # Get current available actions and clamp if necessary
        avail = np.asarray(self._unwrapped.get_avail_actions())  # [n_agents, n_actions]
        for i in range(self._n_agents):
            if avail[i, action_list[i]] == 0:
                # Action is invalid, pick first valid action
                valid_actions = np.where(avail[i] > 0)[0]
                if len(valid_actions) > 0:
                    action_list[i] = int(valid_actions[0])
                else:
                    action_list[i] = 0  # fallback -> shouldn't happen
        
        # Step the environment
        obs_tuple, reward, terminated, truncated, info = self._env.step(action_list)
        
        # Convert observations
        obs = torch.tensor(np.stack(obs_tuple), dtype=torch.float32, device=self.device)
        
        # Get global state
        state = torch.tensor(
            np.asarray(self._unwrapped.get_state()),
            dtype=torch.float32,
            device=self.device
        )
        
        # Get action masks
        avail = np.asarray(self._unwrapped.get_avail_actions())
        action_mask = torch.tensor(avail > 0, dtype=torch.bool, device=self.device)
        
        # Shared reward → broadcast to all agents [n_agents, 1]
        reward_tensor = torch.full(
            (self._n_agents, 1), 
            reward, 
            dtype=torch.float32, 
            device=self.device
        )
        
        done = terminated or truncated
        
        return TensorDict(
            {
                "agents": TensorDict(
                    {
                        "observation": obs,
                        "action_mask": action_mask,
                        "reward": reward_tensor,
                    },
                    batch_size=(self._n_agents,),
                    device=self.device,
                ),
                "state": state,
                "done": torch.tensor([done], dtype=torch.bool, device=self.device),
                "terminated": torch.tensor([terminated], dtype=torch.bool, device=self.device),
                "truncated": torch.tensor([truncated], dtype=torch.bool, device=self.device),
            },
            batch_size=self.batch_size,
            device=self.device,
        )
    
    def close(self, raise_if_closed: bool = True):
        """Close the environment."""
        if hasattr(self, '_env') and self._env is not None:
            self._env.close()
    
    def render(self, mode: str = "rgb_array"):
        """Render the environment."""
        if hasattr(self._env, 'render'):
            return self._env.render()
        return None
        
    @property
    def n_agents(self) -> int:
        return self._n_agents
    
    @property
    def reward_key(self) -> tuple:
        """The key for rewards in the tensordict."""
        return ("agents", "reward")
    
    @property
    def episode_limit(self) -> int:
        """Return episode limit (check env spec or use default)."""
        limit = getattr(self._unwrapped, 'episode_limit', None)
        if limit is None:
            # Fallback: check env spec
            if self._env.spec and self._env.spec.max_episode_steps:
                return self._env.spec.max_episode_steps
            return 200  # not sure what a good default is
        return limit
    
    def sample_action(self, tensordict: TensorDictBase = None) -> TensorDict:
        """
        Sample a valid random action respecting action masks.
        
        Args:
            tensordict: Current state tensordict containing action_mask.
                       If None, fetches current action mask from env.
        
        Returns:
            TensorDict with valid random actions.
        """
        if tensordict is not None and ("agents", "action_mask") in tensordict.keys(include_nested=True):
            action_mask = tensordict["agents", "action_mask"]
        else:
            # Get current action mask from environment
            avail = np.asarray(self._unwrapped.get_avail_actions())
            action_mask = torch.tensor(avail > 0, dtype=torch.bool, device=self.device)
        
        actions = []
        for i in range(self._n_agents):
            valid_actions = torch.where(action_mask[i])[0]
            if len(valid_actions) > 0:
                idx = torch.randint(len(valid_actions), (1,)).item()
                action = valid_actions[idx].item()
            else:
                action = 0  # fallback
            actions.append(action)
        
        return TensorDict({
            "agents": TensorDict({
                "action": torch.tensor(actions, dtype=torch.int64, device=self.device)
            }, batch_size=(self._n_agents,), device=self.device)
        }, batch_size=self.batch_size, device=self.device)


# ============================================================================
# BenchMARL Task Definition
# ============================================================================

@dataclass
class SMACliteTaskConfig:
    """Configuration for SMACLite tasks."""
    map_name: str = "2s3z"
    max_steps: int = 500


class SMACliteTask(Enum):
    """
    BenchMARL Task enum for SMACLite scenarios.
    
    Usage:
        task = SMACliteTask.TWO_S_THREE_Z.get_from_yaml()
        experiment = Experiment(task=task, algorithm=..., ...)
    """
    TWO_S_THREE_Z = None    # 2s3z
    THREE_S_FIVE_Z = None   # 3s5z
    MMM = None              # MMM
    MMM2 = None             # MMM2
    THREE_M = None          # 3m
    EIGHT_M = None          # 8m
    TWENTY_FIVE_M = None    # 25m
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
    def update_config(self, config: Dict[str, Any]) -> "SMACliteTask":
        if self.config is None:
            self.config = config
        else:
            self.config.update(config)
        return self
    
    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: str,
    ) -> Callable[[], EnvBase]:
        """Return a callable that creates the TorchRL environment."""
        config = copy.deepcopy(self.config)
        map_name = self._get_map_name()
        env_id = f"smaclite/{map_name}-v0"
        
        def make_env():
            return SMACliteTorchRLWrapper(
                env_name=env_id,
                device=device,
                seed=seed,
                categorical_actions=True,  # BenchMARL uses categorical
            )
        
        return make_env
    
    def _get_map_name(self) -> str:
        """Map enum name to SMACLite map name."""
        name_map = {
            "TWO_S_THREE_Z": "2s3z",
            "THREE_S_FIVE_Z": "3s5z",
            "MMM": "MMM",
            "MMM2": "MMM2",
            "THREE_M": "3m",
            "EIGHT_M": "8m",
            "TWENTY_FIVE_M": "25m",
        }
        return name_map.get(self.name, self.name.lower())
    
    def supports_continuous_actions(self) -> bool:
        return False
    
    def supports_discrete_actions(self) -> bool:
        return True
    
    def has_render(self, env: EnvBase) -> bool:
        return False  # SMACLite rendering is optional; disable for now
    
    def max_steps(self, env: EnvBase) -> int:
        return self.config.get("max_steps", env.episode_limit)
    
    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        return env.group_map
    
    def state_spec(self, env: EnvBase) -> Optional[Composite]:
        return Composite({"state": env.observation_spec["state"].clone()})
    
    def action_mask_spec(self, env: EnvBase) -> Optional[Composite]:
        return Composite({
            "agents": Composite({
                "action_mask": env.observation_spec["agents", "action_mask"].clone()
            })
        })
    
    def observation_spec(self, env: EnvBase) -> Composite:
        obs_spec = env.observation_spec.clone()
        # Remove state and action_mask from observation spec
        # they're accessed separately
        del obs_spec["state"]
        del obs_spec["agents", "action_mask"]
        return obs_spec
    
    def info_spec(self, env: EnvBase) -> Optional[Composite]:
        return None
    
    def action_spec(self, env: EnvBase) -> Composite:
        return env.full_action_spec
    
    def get_env_transforms(self, env: EnvBase) -> List[Transform]:
        """Returns a list of transforms to be applied to the env."""
        # Add StepCounter to enforce episode limit (truncation after max_steps)
        max_steps = self.config.get("max_steps", 120)
        return [StepCounter(max_steps=max_steps)]
    
    def get_replay_buffer_transforms(self, env: EnvBase, group: str = None) -> List[Transform]:
        """Returns a list of transforms to be applied to the ReplayBuffer."""
        return []
    
    def get_reward_sum_transform(self, env: EnvBase) -> Transform:
        """Returns the RewardSum transform for the environment."""
        return RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")])
    
    @staticmethod
    def log_info(batch: TensorDictBase) -> Dict[str, float]:
        """Return a str->float dict with extra items to log."""
        return {}
    
    @staticmethod
    def render_callback(experiment, env: EnvBase, data: TensorDictBase):
        """Callback for rendering the environment."""
        # Try to get base_env if it's wrapped
        base_env = env
        while hasattr(base_env, 'base_env'):
            base_env = base_env.base_env
        
        if hasattr(base_env, 'render'):
            try:
                return base_env.render(mode="rgb_array")
            except TypeError:
                try:
                    return base_env.render()
                except Exception:
                    return None
        return None
    
    @staticmethod
    def env_name() -> str:
        return "smaclite"
    
    def get_from_yaml(self, path: Optional[str] = None) -> "SMACliteTask":
        """Load config from YAML (or use defaults)."""
        # Default configs for each scenario
        defaults = {
            "TWO_S_THREE_Z": {"map_name": "2s3z", "max_steps": 120},
            "THREE_S_FIVE_Z": {"map_name": "3s5z", "max_steps": 150},
            "MMM": {"map_name": "MMM", "max_steps": 150},
            "MMM2": {"map_name": "MMM2", "max_steps": 180},
            "THREE_M": {"map_name": "3m", "max_steps": 60},
            "EIGHT_M": {"map_name": "8m", "max_steps": 120},
            "TWENTY_FIVE_M": {"map_name": "25m", "max_steps": 150},
        }
        self.config = defaults.get(self.name, {"max_steps": 200})
        return self


# # ============================================================================
# # Quick Test / Validation
# # ============================================================================

# def test_wrapper():
#     """Test the TorchRL wrapper independently."""
#     print("=" * 60)
#     print("Testing SMACliteTorchRLWrapper")
#     print("=" * 60)
    
#     env = SMACliteTorchRLWrapper(env_name="smaclite/2s3z-v0", seed=42)
    
#     print(f"\n[INFO] n_agents: {env.n_agents}")
#     print(f"[INFO] obs_dim: {env._obs_dim}")
#     print(f"[INFO] n_actions: {env._n_actions}")
#     print(f"[INFO] state_dim: {env._state_dim}")
#     print(f"[INFO] episode_limit: {env.episode_limit}")
    
#     print("\n[SPECS]")
#     print(f"observation_spec:\n{env.observation_spec}")
#     print(f"\naction_spec:\n{env.action_spec}")
#     print(f"\nreward_spec:\n{env.reward_spec}")
    
#     # Run spec check
#     print("\n[CHECK] Running check_env_specs...")
#     try:
#         check_env_specs(env)
#         print("[OK] Environment specs are valid!")
#     except Exception as e:
#         # Action masking environments often fail spec check because
#         # check_env_specs samples random actions without respecting masks
#         err_str = str(e)
#         if "Invalid action" in err_str or "invalid action" in err_str.lower():
#             print(f"[OK] Spec check failed due to action masking (expected): {err_str}")
#             print("     This is normal - the wrapper handles invalid actions internally.")
#         else:
#             print(f"[WARN] Spec check issue: {e}")
    
#     # Test rollout
#     print("\n[ROLLOUT] Testing 5-step rollout (TED format)...")
#     print("         TED = TorchRL Episode Data: step() output has 'next' key")
#     td = env.reset()
#     print(f"Reset output keys: {list(td.keys(include_nested=True, leaves_only=True))}")
    
#     for step in range(5):
#         # Sample valid action using the helper method
#         action_td = env.sample_action(td)
        
#         # Merge action into tensordict
#         td = td.update(action_td)
        
#         # Step returns TED format: input + "next" containing step output
#         td_step = env.step(td)
        
#         # Show TED structure on first step
#         if step == 0:
#             print(f"Step output keys (TED): {list(td_step.keys())}")
#             print(f"  'next' subkeys: {list(td_step['next'].keys(include_nested=True, leaves_only=True))}")
        
#         # Access done from the "next" subtensordict
#         done = td_step["next", "done"].item()
#         reward = td_step["next", "agents", "reward"][0, 0].item()
        
#         print(f"  Step {step+1}: reward={reward:.3f}, done={done}")
        
#         if done:
#             print("  Episode ended!")
#             break
        
#         # Use step_mdp to advance: moves "next" contents to root for next iteration
#         td = step_mdp(td_step)
    
#     env.close()
#     print("\n[OK] Wrapper test complete!")
#     return True


# def test_benchmarl_integration():
#     """Test full BenchMARL integration."""
#     print("\n" + "=" * 60)
#     print("Testing BenchMARL Integration")
#     print("=" * 60)
    
#     # Create task
#     task = SMACliteTask.TWO_S_THREE_Z.get_from_yaml()
#     print(f"\n[TASK] {task}")
#     print(f"[CONFIG] {task.config}")
    
#     # Get env function
#     env_fn = task.get_env_fun(
#         num_envs=1,
#         continuous_actions=False,
#         seed=42,
#         device="cpu"
#     )
    
#     # Create environment
#     env = env_fn()
#     print(f"\n[ENV] Created environment: {env}")
#     print(f"[ENV] group_map: {task.group_map(env)}")
#     print(f"[ENV] max_steps: {task.max_steps(env)}")
#     print(f"[ENV] supports_discrete: {task.supports_discrete_actions()}")
    
#     # Verify specs
#     print(f"\n[SPEC] observation_spec:\n{task.observation_spec(env)}")
#     print(f"\n[SPEC] action_mask_spec:\n{task.action_mask_spec(env)}")
#     print(f"\n[SPEC] state_spec:\n{task.state_spec(env)}")
    
#     env.close()
#     print("\n[OK] BenchMARL integration test complete!")
#     return True


# if __name__ == "__main__":
#     import smaclite  # noqa: F401
    
#     test_wrapper()
#     test_benchmarl_integration()
    
#     print("\n" + "=" * 60)
#     print("All tests passed! Ready for BenchMARL experiments.")
#     print("=" * 60)