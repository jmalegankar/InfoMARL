import policy
import vmas
import wrapper

from stable_baselines3 import PPO
from qmix import QMixVMAS
import torch
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

env = vmas.make_env(
    scenario="simple_spread",
    n_agents=4,
    num_envs=64,
    continuous_actions=True,
    max_steps=400,
    seed=42,
    device=device,
    terminated_truncated=False,
)

env = wrapper.VMASVecEnv(env, rnd_nums=True)

flag = "qmix"  # Change to "ppo" for PPO training
if flag == "qmix":
    if os.path.exists("qmix_model_qmix.zip"):
        print("Loading existing QMix model...")
        model = QMixVMAS(env, device=device, lr=1e-4)
        model.load("qmix_model")
    else:
        print("Creating new QMix model...")
        model = QMixVMAS(
            env=env,
            lr=1e-4,
            gamma=0.99,
            batch_size=1024,
            device=device
        )
elif flag == "ppo":
    qmix_model = None
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
            target_kl=0.25,
            max_grad_norm=10.0,
            learning_rate=1e-4,
        )

while True:
    model.learn(total_timesteps=300000, progress_bar=True)
    model.save("ppo_infomarl")