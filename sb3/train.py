import policy
import vmas
import wrapper

from stable_baselines3 import PPO

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

env = vmas.make_env(
    scenario="simple_spread",
    n_agents=4,
    num_envs=64,
    continuous_actions=True,
    max_steps=400,
    seed=42,
    device="cuda",
    terminated_truncated=True,
)

env = wrapper.VMASVecEnv(env, rnd_nums=True)

model = PPO(
    policy=policy.InfoMARLActorCriticPolicy,
    env=env,
    device=device,
    verbose=1,
)

model.learn(total_timesteps=1000000, progress_bar=True)