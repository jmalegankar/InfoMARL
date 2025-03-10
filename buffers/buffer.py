import torch as th
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

@th.jit.script
class StateBuffer:
    """
    Container for storing state transitions
    """
    def __init__(
        self,
        obs: Dict[str, th.Tensor],
        action: th.Tensor,
        reward: th.Tensor,
        next_obs: Dict[str, th.Tensor],
        done: th.Tensor,
    ):
        self.obs = obs
        self.action = action
        self.reward = reward
        self.next_obs = next_obs
        self.done = done
    
    def to(self, device: Union[str, th.device]):
        device = device if isinstance(device, th.device) else th.device(device)
        self.obs = {k: v.to(device) for k, v in self.obs.items()}
        self.action = self.action.to(device)
        self.reward = self.reward.to(device)
        self.next_obs = {k: v.to(device) for k, v in self.next_obs.items()}
        self.done = self.done.to(device)
        return self

from typing import Any
NoneFuture = Any 

class ReplayBuffer(th.nn.Module):
    """
    Optimized replay buffer for multi-agent reinforcement learning with dictionary observations
    """
    def __init__(
        self,
        max_size: int,
        device: Union[str, th.device] = "cpu",
    ):
        super().__init__()
        self.max_size = max_size
        self.device = device if isinstance(device, th.device) else th.device(device)
        
        # Initialize buffer with empty state buffers
        self.buffer = tuple(
            StateBuffer(
                obs={},
                action=th.zeros(0, device=self.device),
                reward=th.zeros(0, device=self.device),
                next_obs={},
                done=th.zeros(0, device=self.device),
            )
            for _ in range(max_size)
        )
        
        self.idx = 0
        self.size = 0
    
    def _add(
        self,
        obs: Dict[str, th.Tensor],
        action: th.Tensor,
        reward: th.Tensor,
        next_obs: Dict[str, th.Tensor],
        done: th.Tensor,
        idx: int,
    ) -> None:
        """Add a single transition to the buffer at the specified index"""
        self.buffer[idx].obs = obs
        self.buffer[idx].action = action
        self.buffer[idx].reward = reward
        self.buffer[idx].next_obs = next_obs
        self.buffer[idx].done = done
    
    def add(
        self,
        obs: Union[Dict[str, th.Tensor], List[Dict[str, th.Tensor]]],
        action: Union[th.Tensor, List[th.Tensor]],
        reward: Union[th.Tensor, List[th.Tensor]],
        next_obs: Union[Dict[str, th.Tensor], List[Dict[str, th.Tensor]]],
        done: Union[th.Tensor, List[th.Tensor]],
    ) -> None:
        """Add one or multiple transitions to the buffer"""
        if isinstance(obs, dict) and isinstance(action, th.Tensor) and isinstance(reward, th.Tensor) and isinstance(next_obs, dict) and isinstance(done, th.Tensor):
            # Single transition case
            self._add(
                obs=obs,
                action=action,
                reward=reward,
                next_obs=next_obs,
                done=done,
                idx=self.idx,
            )
            self.idx = (self.idx + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)
        elif isinstance(obs, list) and isinstance(action, list) and isinstance(reward, list) and isinstance(next_obs, list) and isinstance(done, list):
            # Multiple transitions case - process in parallel
            futures: List[NoneFuture] = []
            for i in range(len(obs)):
                futures.append(
                    th.jit.fork(
                        self._add,
                        obs=obs[i],
                        action=action[i],
                        reward=reward[i],
                        next_obs=next_obs[i],
                        done=done[i],
                        idx=(self.idx + i) % self.max_size,
                    )
                )
            
            # Update buffer index and size
            self.idx = (self.idx + len(obs)) % self.max_size
            self.size = min(self.size + len(obs), self.max_size)
            
            # Wait for all parallel operations to complete
            for future in futures:
                th.jit.wait(future)
        else:
            raise TypeError("Input types don't match expected formats")
    
    def sample(self, batch_size: int, device: Optional[th.device] = None) -> StateBuffer:
        """Sample a batch of transitions from the buffer"""
        if self.size < batch_size:
            raise ValueError(f"Not enough samples in buffer ({self.size}) to sample batch of size {batch_size}")
        
        if device is None:
            device = self.device
            
        # Sample random indices
        idxs = th.randint(0, self.size, (batch_size,)).tolist()
        
        # Collect states from sampled indices
        obs_batch = {}
        action_batch = []
        reward_batch = []
        next_obs_batch = {}
        done_batch = []
        
        # Initialize obs_batch and next_obs_batch dictionaries with empty tensors
        for key in self.buffer[0].obs.keys():
            obs_batch[key] = []
            next_obs_batch[key] = []
        
        # Collect data from each sampled state
        for idx in idxs:
            state = self.buffer[idx]
            
            # Add to action, reward, and done lists
            action_batch.append(state.action)
            reward_batch.append(state.reward)
            done_batch.append(state.done)
            
            # Add to obs and next_obs dictionaries
            for key in state.obs.keys():
                obs_batch[key].append(state.obs[key])
                next_obs_batch[key].append(state.next_obs[key])
        
        # Stack tensors along new batch dimension
        action_batch = th.stack(action_batch).to(device)
        reward_batch = th.stack(reward_batch).to(device)
        done_batch = th.stack(done_batch).to(device)
        
        # Stack observation tensors
        for key in obs_batch.keys():
            obs_batch[key] = th.stack(obs_batch[key]).to(device)
            next_obs_batch[key] = th.stack(next_obs_batch[key]).to(device)
        
        # Create and return StateBuffer
        return StateBuffer(
            obs=obs_batch,
            action=action_batch,
            reward=reward_batch,
            next_obs=next_obs_batch,
            done=done_batch
        )
    
    def __len__(self) -> int:
        return self.size