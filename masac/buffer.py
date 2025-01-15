import torch as th
from typing import Dict, Optional, Tuple, Union, List

NoneFuture = th.jit.Future[None]

@th.jit.script
class StateBuffer:
    def __init__(
        self,
        obs: Dict[str, th.Tensor],
        reward: th.Tensor,
        done: th.Tensor,
        next_obs: Dict[str, th.Tensor],
        action: th.Tensor,
    ):
        self.obs = obs
        self.reward = reward
        self.done = done
        self.next_obs = next_obs
        self.action = action
    
    def to(self, device: th.device):
        self.obs = {k: v.to(device) for k, v in self.obs.items()}
        self.reward = self.reward.to(device)
        self.done = self.done.to(device)
        self.next_obs = {k: v.to(device) for k, v in self.next_obs.items()}
        self.action = self.action.to(device)
        return self

class ReplayBuffer(th.nn.Module):
    def __init__(
        self,
        max_size: int,
        device: th.device,
    ):
        super().__init__()
        self.max_size = max_size
        self.device = device
        self.buffer = tuple(
            StateBuffer(
                obs={},
                reward=th.zeros(0, device=device),
                done=th.zeros(0, device=device),
                next_obs=th.zeros(0, device=device),
                action=th.zeros(0, device=device),
            )
            for _ in range(max_size)
        )
        self.idx: int = 0
        self.size: int = 0
    
    @th.jit.export
    def _add(
        self,
        obs: Dict[str, th.Tensor],
        reward: th.Tensor,
        done: th.Tensor,
        next_obs: Dict[str, th.Tensor],
        action: th.Tensor,
        idx: int,
    ) -> None:
        self.buffer[idx].obs = obs
        self.buffer[idx].reward = reward
        self.buffer[idx].done = done
        self.buffer[idx].next_obs = next_obs
        self.buffer[idx].action = action
    
    @th.jit.export
    def add(
        self,
        obs: Union[List[Dict[str, th.Tensor]], Dict[str, th.Tensor]],
        action: Union[List[th.Tensor], th.Tensor],
        reward: Union[List[th.Tensor], th.Tensor],
        next_obs: Union[List[Dict[str, th.Tensor]], Dict[str, th.Tensor]],
        done: Union[List[th.Tensor], th.Tensor],
    ) -> None:
        if isinstance(obs, Dict[str, th.Tensor]) and isinstance(reward, th.Tensor) and isinstance(done, th.Tensor) and isinstance(next_obs, Dict[str, th.Tensor]) and isinstance(action, th.Tensor):
            self._add(
                obs=obs,
                reward=reward,
                done=done,
                next_obs=next_obs,
                action=action,
                idx=self.idx,
            )
            self.idx = (self.idx + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)
            return
        elif isinstance(obs, list) and isinstance(reward, list) and isinstance(done, list) and isinstance(next_obs, list) and isinstance(action, list):
            futures: List[NoneFuture] = []
            for i in range(len(obs)):
                futures.append(
                    th.jit.fork(
                        self._add,
                        obs=obs[i],
                        reward=reward[i],
                        done=done[i],
                        next_obs=next_obs[i],
                        action=action[i],
                        idx=self.idx,
                    )
                )
                self.idx = (self.idx + 1) % self.max_size
                self.size = min(self.size + 1, self.max_size)
            for future in futures:
                th.jit.wait(future)
        
    @th.jit.export
    def sample(self, batch_size: int, device: Optional[th.device] = None) -> List[StateBuffer]:
        if batch_size > self.size:
            raise ValueError(f"Batch size {batch_size} is greater than buffer size {self.size}")
        if device is None:
            device = self.device
        idxs: List[int] = th.randint(0, self.size, (batch_size,), device=device).tolist()
        return list(self.buffer[idx].to(device=device) for idx in idxs)
    

if __name__ == '__main__':
    buffer = th.jit.script(ReplayBuffer(10, th.device('cpu')))
    obs = []
    reward = []
    done = []
    next_obs = []
    action = []
    for i in range(15):
        obs.append({'obs': th.rand(3, 3)})
        reward.append(th.rand(1))
        done.append(th.rand(1))
        next_obs.append({'obs': th.rand(3, 3)})
        action.append(th.rand(1))
    buffer.add(obs, reward, done, next_obs, action)
    print(buffer.sample(5, device=th.device('cuda')))
