import policy
import xor_game
from stable_baselines3.common.vec_env import DummyVecEnv

from stable_baselines3 import PPO

import torch
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

env = DummyVecEnv([lambda: xor_game.XORGameEnv(n_agents=2, n_actions=3) for _ in range(64)])

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
        batch_size=128,
        n_epochs=1,
        gamma=0.99,
        n_steps=2,
        vf_coef=0.5,
        target_kl=0.25,
        max_grad_norm=10.0,
        learning_rate=1e-4,
    )

model.learn(total_timesteps=50000, progress_bar=True)

env = DummyVecEnv([lambda: xor_game.XORGameEnv(n_agents=3, n_actions=3) for _ in range(2)])

model.policy.action_space = env.action_space
model.policy.pi_features_extractor.actor.number_agents = 3
model.policy.observation_space = env.observation_space
model.policy.action_dist.action_dims = [3] * 3

obs = env.reset()
actions = model.predict(obs, deterministic=True)[0]
print(f"Actions: {actions}")
_, rewards, dones, infos = env.step(actions)
print(f"Rewards: {rewards}")