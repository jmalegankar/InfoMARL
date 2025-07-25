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
        self.cross_attention_weights = []
        self.landmark_weights = None
        self.env_frames = []
        self.scenario = None
        self.max_steps = None
        self.n_agents = None
        self.env_idx = 0
    
    def create_env(self, sim, env_idx, **kwargs):
        if sim == "vmas":
            if kwargs["scenario"] == "food_collection":
                env = vmas.make_env(
                    scenario=kwargs["scenario"],
                    n_agents=kwargs["n_agents"],
                    n_food=kwargs["n_food"],
                    num_envs=kwargs["num_envs"],
                    continuous_actions=kwargs["continuous_actions"],
                    max_steps=kwargs["max_steps"],
                    seed=kwargs["seed"],
                    device=kwargs["device"],
                    terminated_truncated=kwargs["terminated_truncated"],
                )
            else:
                env = vmas.make_env(
                    scenario=kwargs["scenario"],
                    n_agents=kwargs["n_agents"],
                    num_envs=kwargs["num_envs"],
                    continuous_actions=kwargs["continuous_actions"],
                    max_steps=kwargs["max_steps"],
                    seed=kwargs["seed"],
                    device=kwargs["device"],
                    terminated_truncated=kwargs["terminated_truncated"],
                )
            env = wrapper.VMASVecEnv(env, rnd_nums=True)
            self.env = env
            self.env_idx = env_idx
            self.scenario = kwargs["scenario"]
            self.max_steps = kwargs["max_steps"]
            self.n_agents = kwargs["n_agents"]
            self.num_envs = kwargs["num_envs"]
            self.landmark_weights = []
            if self.scenario == "food_collection":
                self.n_food = kwargs["n_food"]
            
    
    def attach_and_load_model(self, model_name, path, **kwargs):
        if model_name == "ppo":
            self.model = PPO.load(path, **kwargs)
            if self.model.policy.observation_space != self.env.observation_space:
                self.model.policy.observation_space = self.env.observation_space
                self.model.policy.action_space = self.env.action_space
                self.model.policy.pi_features_extractor.actor.number_agents = self.env.num_agents
                self.model.policy.pi_features_extractor.actor.number_food = self.n_food

                with torch.no_grad():
                    self.model.policy.log_std = torch.nn.Parameter(
                        torch.zeros(np.prod(self.env.action_space.shape), dtype=torch.float32)
                    )
        else:
            raise ValueError(f"Model {model_name} not supported.")

    def collect_data(self):
        obs = self.env.reset()
        for _ in range(self.max_steps):
            action, _ = self.model.predict(obs, deterministic=True)
            actor = self.model.policy.features_extractor.actor
            cross_attention_weights = actor.cross_attention_weights
            cross_attention_weights = cross_attention_weights.view(self.num_envs, self.n_agents, self.n_food, self.n_agents)
            cross_attention_weights = cross_attention_weights[self.env_idx].cpu().numpy()

            frame = self.env.render(
                mode="rgb_array",
                agent_index_focus=None
            )
            
            # take a step in the environment
            obs, _, _, _ = self.env.step(action)


            yield frame, cross_attention_weights
            
    
    def create_mp4(self, path, fps=10, dpi=100):
        print(f"Creating mp4 file at {path}...")
        
        n_cols = int(np.ceil(np.sqrt(self.n_agents)))
        n_rows = int(np.ceil(self.n_agents / n_cols))

        # set up figure + GridSpec: left column for env, right columns for attention
        fig = plt.figure(figsize=(12, 8))
        gs = gridspec.GridSpec(n_rows, n_cols + 1,
                            width_ratios=[2] + [1] * n_cols,
                            height_ratios=[1] * n_rows,
                            wspace=0.3, hspace=0.3)

        ax_env = fig.add_subplot(gs[:, 0])
        ax_env.axis("off")
        ax_env.set_title(f"{self.scenario} Scenario")

        att_axes = []
        for idx in range(self.n_agents):
            row = idx // n_cols
            col = 1 + (idx % n_cols)
            ax = fig.add_subplot(gs[row, col])
            ax.set_title(f"Agent {idx} Attention")
            ax.axis("off")
            att_axes.append(ax)

        txt = fig.text(0.5, 0.95, "", ha="center", va="top",
                    fontsize=14, color="black", fontweight="bold")
        
        # Data collector
        collector = self.collect_data()
        n_frames = self.max_steps
        # set up the writer
        writer = FFMpegWriter(fps=fps, metadata=dict(artist="Me"), bitrate=1800)
        with writer.saving(fig, path, dpi):
            for t in tqdm(range(n_frames)):
                frame, cross_att_weights = next(collector)
                # update env frame
                ax_env.clear()
                ax_env.axis("off")
                ax_env.set_title(f"{self.scenario} Scenario")
                ax_env.imshow(frame)

                # update counter
                txt.set_text(f"Frame: {t+1}/{n_frames}")

                # update each attention heatmap
                for i, ax in enumerate(att_axes):
                    ax.clear()
                    ax.axis("off")
                    heat = cross_att_weights[i]
                    ax.imshow(
                        heat,
                        interpolation="nearest",
                        norm=Normalize(vmin=0, vmax=1)
                    )

                    for (x, y), value in np.ndenumerate(heat):
                        ax.text(
                            y, x, f'{value:.2f}', ha='center', va='center', color='black', fontsize=6
                        )
                    ax.set_title(f"Agent {i}")

                # grab and write
                writer.grab_frame()

        plt.close(fig)             
        
            

if __name__ == "__main__":    
    animator = AttentionAnimator()
    animator.create_env(
        sim="vmas",
        env_idx=0,
        scenario="food_collection",
        n_agents=4,
        n_food=5,
        num_envs=2,
        continuous_actions=True,
        max_steps=100,
        seed=0,
        device="cpu",
        terminated_truncated=False,
        respawn_food=True,
        collection_radius=0.15,
        
    )

    animator.attach_and_load_model(
        model_name="ppo",
        path="ppo_infomarl.zip",
        device="cpu",
    )
        
    animator.collect_data()
    animator.create_mp4("attention_animation.mp4")