import torch as th
from typing import Dict, Optional, List, Tuple

@th.jit.script
class StateBuffer:
    def __init__(
        self,
        obs: Dict[str, th.Tensor],
        reward: th.Tensor,
        done: th.Tensor,
        next_obs: th.Tensor,
        action: th.Tensor,
        agent_idx: int,
    ):
        self.obs = obs
        self.reward = reward
        self.done = done
        self.next_obs = next_obs
        self.action = action
        self.agent_idx = agent_idx
    
    def to(self, device: th.device):
        self.obs = {k: v.to(device) for k, v in self.obs.items()}
        self.reward = self.reward.to(device)
        self.done = self.done.to(device)
        self.next_obs = self.next_obs.to(device)
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
                agent_idx=0,
            )
            for _ in range(max_size)
        )
        self.idx: int = 0
        self.size: int = 0
    
    @th.jit.export
    def add(
        self,
        obs: Dict[str, th.Tensor],
        reward: th.Tensor,
        done: th.Tensor,
        next_obs: th.Tensor,
        action: th.Tensor,
        agent_idx: int,
    ) -> None:
        self.buffer[self.idx].obs = obs
        self.buffer[self.idx].reward = reward
        self.buffer[self.idx].done = done
        self.buffer[self.idx].next_obs = next_obs
        self.buffer[self.idx].action = action
        self.buffer[self.idx].agent_idx = agent_idx
        self.idx = (self.idx + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    @th.jit.export
    def sample(self, batch_size: int, device: Optional[th.device] = None) -> Tuple[Dict[str, th.Tensor], th.Tensor, th.Tensor, Dict[str, th.Tensor], th.Tensor, th.Tensor]:
        if batch_size > self.size:
            raise ValueError(f"Batch size {batch_size} is greater than buffer size {self.size}")
        if device is None:
            device = self.device
        idxs: List[int] = th.randint(0, self.size, (batch_size,), dtype=th.long).tolist()
        obs = {k: th.stack([self.buffer[idx].obs[k] for idx in idxs], dim=0) for k in self.buffer[0].obs.keys()}
        reward = th.stack([self.buffer[idx].reward for idx in idxs], dim=0).to(device)
        done = th.stack([self.buffer[idx].done for idx in idxs], dim=0).to(device)
        next_obs = {k: th.stack([self.buffer[idx].next_obs for idx in idxs], dim=0).to(device) for k in self.buffer[0].obs.keys()}
        action = th.stack([self.buffer[idx].action for idx in idxs], dim=0).to(device)
        agent_idx = th.tensor([self.buffer[idx].agent_idx for idx in idxs], device=device)
        return obs, reward, done, next_obs, action, agent_idx
    

if __name__ == '__main__':
    buffer = th.jit.script(ReplayBuffer(10, th.device('cpu')))
    for i in range(15):
        buffer.add(
            obs={'obs': th.tensor([i], dtype=th.float32)},
            reward=th.tensor(i, dtype=th.float32),
            done=th.tensor(0, dtype=th.float32),
            next_obs=th.tensor([i + 1], dtype=th.float32),
            action=th.tensor(i, dtype=th.float32),
            agent_idx=i,
        )
    print(buffer.sample(5, device=th.device('cuda')))
