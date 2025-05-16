import policy
import vmas
import wrapper

from stable_baselines3 import PPO

import torch
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

env = vmas.make_env(
    scenario="simple_spread",
    n_agents=4,
    num_envs=40,
    continuous_actions=True,
    max_steps=400,
    seed=42,
    device="cuda",
    terminated_truncated=True,
)

env = wrapper.VMASVecEnv(env, rnd_nums=True)

if os.path.exists("ppo_infomarl.zip"):
    print("Loading existing model...")
    model = PPO.load("ppo_infomarl.zip", env=env, device=device)
else:
    print("Creating new model...")
    # Create the model
    model = PPO(
        policy=policy.InfoMARLActorCriticPolicy,
        env=env,
        device=device,
        verbose=1,
        batch_size=400,
        n_epochs=10,
        max_grad_norm=10.0,
        gamma=0.95,
        n_steps=100,
    )

for _ in range(10):
    model.learn(total_timesteps=1000000, progress_bar=True)
    model.save("ppo_infomarl")