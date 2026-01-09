import policy
import smaclite
import wrapper

from stable_baselines3 import PPO

import torch
import os

device = "cpu"

env = wrapper.SMACVecEnv(
    env_name="smaclite/3s5z-v0",
    num_envs=2,
    max_steps=108000,
    rnd_nums=True,
)

# Create the model
model = PPO(
        policy=policy.InfoMARLActorCriticPolicy,
        env=env,
        device=device,
        verbose=1,
        batch_size=320,
        n_epochs=10,
        gamma=0.99,
        n_steps=160,
        vf_coef=0.5,
        ent_coef=0.001,
        target_kl=0.25,
        max_grad_norm=10.0,
        learning_rate=1e-4,
    )
del model.policy.vf_features_extractor
del model.policy.value_net
model.policy.load_state_dict(PPO.load("ppo_infomarl.zip", device=device).policy.state_dict(), strict=False)

with torch.no_grad():
    obs = env.reset()
    wins = 0
    for _ in range(100):
        while True:
            actions, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(actions)
            env._envs[0].render()
            if dones[0]:
                wins += int(infos[0].get("battle_won"))
                break
    
        print(f"Win rate over 100 episodes: {wins}%")