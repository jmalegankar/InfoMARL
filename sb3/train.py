import policy
import smaclite
import wrapper

from stable_baselines3 import PPO

import torch
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

env = wrapper.SMACVecEnv(
    env_name="smaclite/2s3z-v0",
    num_envs=64,
    max_steps=108000,
    rnd_nums=True,
)
obs = env.reset()

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
        batch_size=1024,
        n_epochs=10,
        gamma=0.99,
        n_steps=160,
        vf_coef=0.5,
        ent_coef=0.001,
        target_kl=0.25,
        max_grad_norm=10.0,
        learning_rate=1e-4,
    )

while True:
    model.learn(total_timesteps=300000, progress_bar=True)
    model.save("ppo_infomarl")