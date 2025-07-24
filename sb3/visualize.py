import vmas
import wrapper
import policy
import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import PPO
from matplotlib.animation import FuncAnimation
import os
from matplotlib import gridspec
from matplotlib.colors import Normalize
from matplotlib.animation import FFMpegWriter

from tqdm import tqdm


class AttentionAnimator:
    def __init__(self):
        self.env = None
        self.model = None
        self.landmark_weights = None
        self.scenario = None
        self.max_steps = None
        self.n_agents = None
        self.env_idx = 0
    
    def create_env(self, sim, env_idx, **kwargs):
        if sim == "vmas":
            env = vmas.make_env(
                scenario=kwargs["scenario"],
                num_envs=kwargs["num_envs"],
                continuous_actions=kwargs["continuous_actions"],
                max_steps=kwargs["max_steps"],
                seed=kwargs["seed"],
                device=kwargs["device"],
                terminated_truncated=kwargs["terminated_truncated"],
                n_agents_good=kwargs.get("n_agents_good", 5),
                n_agents_adversaries=kwargs.get("n_agents_adversaries", 3),
            )
            env = wrapper.VMASVecEnv(env, rnd_nums=True)
            self.env = env
            self.env_idx = env_idx
            self.scenario = kwargs["scenario"]
            self.max_steps = kwargs["max_steps"]
            self.num_envs = kwargs["num_envs"]
            self.num_adversaries = kwargs.get("n_agents_adversaries", 3)
            self.num_good = kwargs.get("n_agents_good", 5)
            self.n_agents = self.num_adversaries + self.num_good
            
    
    def attach_and_load_model(self, model_name, path, **kwargs):
        if model_name == "ppo":
            self.model = PPO.load(path, **kwargs)
            if self.model.policy.observation_space != self.env.observation_space:
                self.model.policy.observation_space = self.env.observation_space
                self.model.policy.action_space = self.env.action_space
                self.model.policy.pi_features_extractor.actor.num_adversaries = self.num_adversaries
                self.model.policy.pi_features_extractor.actor.num_good = self.num_good
                with torch.no_grad():
                    self.model.policy.log_std = torch.nn.Parameter(
                        torch.zeros(np.prod(self.env.action_space.shape), dtype=torch.float32)
                    )
        else:
            raise ValueError(f"Model {model_name} not supported.")

    def collect_data(self):
        obs = self.env.reset()
        for step in range(self.max_steps):
            action, _ = self.model.predict(obs, deterministic=True)
            actor = self.model.policy.features_extractor.actor
            
            adv_good_weights, good_lmk_weights = actor.cross_attention_weights

            # print(f"adv_good_weights shape: {adv_good_weights.shape}")
            # print(f"good_lmk_weights shape: {good_lmk_weights.shape}")

            adv_attention = adv_good_weights.view(self.num_envs, self.num_adversaries, self.num_good, self.num_adversaries)
            adv_attention = adv_attention[self.env_idx].cpu().numpy()  # Shape: [num_adversaries, num_good, num_adversaries]
            
            
            good_attention = good_lmk_weights.view(self.num_envs, self.num_good, good_lmk_weights.shape[1], good_lmk_weights.shape[2])
            good_attention = good_attention[self.env_idx].cpu().numpy()  # Shape: [num_good, seq_len, num_landmarks]


            frame = self.env.render(
                mode="rgb_array",
                agent_index_focus=None
            )
            
            # take a step in the environment
            obs, reward, _, _ = self.env.step(action)

            yield frame, adv_attention, good_attention, reward
            
    
    def create_mp4(self, path, fps=10, dpi=100):
        data_collector = self.collect_data()
        print(f"Creating mp4 file at {path}...")

        max_agents_per_row = 4

        adv_rows = int(np.ceil(self.num_adversaries / max_agents_per_row)) if self.num_adversaries > 0 else 0
        good_rows = int(np.ceil(self.num_good / max_agents_per_row)) if self.num_good > 0 else 0
        
        
        total_rows = max(1, adv_rows + 1 + good_rows)
        total_cols = max_agents_per_row + 1

        print(f"Layout: adv_rows={adv_rows}, good_rows={good_rows}, total_rows={total_rows}, total_cols={total_cols}")

        fig = plt.figure(figsize=(18, 12))  
        gs = gridspec.GridSpec(total_rows, total_cols,
                            width_ratios=[4] + [1] * max_agents_per_row, 
                            height_ratios=[1] * total_rows,
                            wspace=0.3, hspace=0.4)

        # Environment subplot (spans multiple rows)
        ax_env = fig.add_subplot(gs[:, 0])
        ax_env.axis("off")
        ax_env.set_title(f"{self.scenario} Scenario", fontsize=14, fontweight='bold')

        # Adversary attention subplots
        adv_axes = []
        for idx in range(self.num_adversaries):
            row = idx // max_agents_per_row
            col = 1 + (idx % max_agents_per_row)
            ax = fig.add_subplot(gs[row, col])
            ax.set_title(f"Predator {idx}", fontsize=10, color='red')
            ax.axis("off")
            adv_axes.append(ax)
        

        if self.num_adversaries > 0:
            fig.text(0.75, 0.91, "Predator Attention to Prey", ha="center", va="center",
                    fontsize=12, fontweight="bold", color="red")

        # Good agent attention subplots  
        good_axes = []
        good_start_row = adv_rows + 1  # Start after adversary section + spacer
        for idx in range(self.num_good):
            row = good_start_row + (idx // max_agents_per_row)
            col = 1 + (idx % max_agents_per_row)
            ax = fig.add_subplot(gs[row, col])
            ax.set_title(f"Prey {idx}", fontsize=10, color='blue')
            ax.axis("off")
            good_axes.append(ax)
            
        if self.num_good > 0:
            fig.text(0.75, 0.48, "Prey Attention to Food", ha="center", va="center",
                    fontsize=12, fontweight="bold", color="blue")

        # Frame counter
        txt = fig.text(0.5, 0.95, "", ha="center", va="top",
                    fontsize=14, color="black", fontweight="bold")

        # Set up the writer
        writer = FFMpegWriter(fps=fps, metadata=dict(artist="Me"), bitrate=1800)
        with writer.saving(fig, path, dpi):
            for t in tqdm(range(self.max_steps)):
                frame, adv_attention, good_attention, reward = next(data_collector)
                # Update environment frame
                ax_env.clear()
                ax_env.axis("off")
                ax_env.set_title(f"{self.scenario} Scenario", fontsize=14, fontweight='bold')
                ax_env.imshow(frame)

                # Update frame counter
                txt.set_text(f"Frame: {t+1}/{self.max_steps} - Reward: {reward[0]:.4f}")

                # Update adversary attention heatmaps
                for i, ax in enumerate(adv_axes):
                    ax.clear()
                    ax.axis("off")
                    heat = adv_attention[i]
                    
                    im = ax.imshow(
                        heat,
                        interpolation="nearest",
                        norm=Normalize(vmin=0, vmax=1),
                        cmap='Reds'
                    )

                    for (x, y), value in np.ndenumerate(heat):
                        ax.text(
                            y, x, f'{value:.2f}', ha='center', va='center', 
                            color='white' if value > 0.5 else 'black', fontsize=8
                        )
                    ax.set_title(f"Predator {i}", fontsize=10, color='red')
                    ax.set_xlabel("Adversaries", fontsize=8)
                    ax.set_ylabel("Prey Agents", fontsize=8)


                # Update good agent attention heatmaps
                for i, ax in enumerate(good_axes):
                    ax.clear()
                    ax.axis("off")
                    heat = good_attention[i]
                    
                    im = ax.imshow(
                        heat,
                        interpolation="nearest", 
                        norm=Normalize(vmin=0, vmax=1),
                        cmap='Blues'
                    )

                    # Add text annotations
                    for (x, y), value in np.ndenumerate(heat):
                        ax.text(
                            y, x, f'{value:.2f}', ha='center', va='center',
                            color='white' if value > 0.5 else 'black', fontsize=8
                        )
                    ax.set_title(f"Prey {i}", fontsize=10, color='blue')
                    ax.set_xlabel("Landmarks", fontsize=8)
                    ax.set_ylabel("Sequence Position", fontsize=8)

                # Grab and write frame
                writer.grab_frame()

        plt.close(fig)
            

if __name__ == "__main__":    
    animator = AttentionAnimator()
    animator.create_env(
        sim="vmas",
        env_idx=0,
        scenario="grassland",
        n_agents_good=5,
        n_agents_adversaries=3,
        num_envs=2,
        continuous_actions=True,
        max_steps=100,
        seed=0,
        device="cpu",
        terminated_truncated=False,
    )
    animator.attach_and_load_model(
        model_name="ppo",
        path="ppo_infomarl.zip",
        device="cpu",
    )
    animator.create_mp4("grassland_attention_animation.mp4")