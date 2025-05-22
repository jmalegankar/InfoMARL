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
            
            
    
    def attach_and_load_model(self, model_name, path, **kwargs):
        if model_name == "ppo":
            self.model = PPO(
                policy=kwargs["policy"],
                env=self.env,
                device=kwargs["device"],
                verbose=kwargs["verbose"],
                batch_size=kwargs["batch_size"],
                n_epochs=kwargs["n_epochs"],
                max_grad_norm=kwargs["max_grad_norm"],
                gamma=kwargs["gamma"],
                n_steps=kwargs["n_steps"],
            )
        self.model.load(path)


    def collect_data(self):
        obs = self.env.reset()
        for step in range(self.max_steps):
            action, _ = self.model.predict(obs, deterministic=True)
            actor = self.model.policy.features_extractor.actor
            cross_attention_weights = actor.cross_attention_weights
            cross_attention_weights = cross_attention_weights.view(self.num_envs, self.n_agents, self.n_agents, self.n_agents)
            cross_attention_weights = cross_attention_weights[self.env_idx].cpu().numpy()

            frame = self.env.render(
                mode="rgb_array",
                agent_index_focus=None
            )

            self.env_frames.append(frame)
            self.cross_attention_weights.append(cross_attention_weights)

            if self.scenario == "simple_spread":
                #do same for landmark weights(landmarks are same as number of agents)
                landmark_weights = actor.landmark_attention_weights
                landmark_weights = landmark_weights.view(self.num_envs, self.n_agents, 1, self.n_agents)
                landmark_weights = landmark_weights[self.env_idx].cpu().numpy()
                self.landmark_weights.append(landmark_weights)
            
            # take a step in the environment
            obs, rewards, dones, infos = self.env.step(action)
            
    
    def create_mp4(self, path, fps=30, dpi=100):
        # get att frames for heat map
        cross_att_frames = [[] for _ in range(self.n_agents)]
        for step_weights in self.cross_attention_weights:
            # step_weights shape = (n_agents, n_agents, n_agents)
            for i in range(self.n_agents):
                cross_att_frames[i].append(step_weights[i])

        # set up figure + GridSpec: left column for env, right columns for attention
        fig = plt.figure(figsize=(12, 8))
        gs = gridspec.GridSpec(2, 3,
                            width_ratios=[2, 1, 1],
                            height_ratios=[1, 1],
                            wspace=0.3, hspace=0.3)

        ax_env = fig.add_subplot(gs[:, 0])
        ax_env.axis("off")
        ax_env.set_title(f"{self.scenario} Scenario")

        att_axes = []
        for idx in range(self.n_agents):
            row = idx // 2
            col = 1 + (idx % 2)
            ax = fig.add_subplot(gs[row, col])
            ax.set_title(f"Agent {idx} Attention")
            ax.axis("off")
            att_axes.append(ax)

        txt = fig.text(0.5, 0.95, "", ha="center", va="top",
                    fontsize=14, color="white", fontweight="bold")

        # set up the writer
        writer = FFMpegWriter(fps=fps, metadata=dict(artist="Me"), bitrate=1800)
        with writer.saving(fig, path, dpi):
            n_frames = len(self.env_frames)
            for t in range(n_frames):
                # update env frame
                ax_env.clear()
                ax_env.axis("off")
                ax_env.set_title(f"{self.scenario} Scenario")
                ax_env.imshow(self.env_frames[t])

                # update counter
                txt.set_text(f"Frame: {t+1}/{n_frames}")

                # update each attention heatmap
                for i, ax in enumerate(att_axes):
                    ax.clear()
                    ax.axis("off")
                    heat = cross_att_frames[i][t]
                    ax.imshow(
                        heat,
                        interpolation="nearest",
                        norm=Normalize(vmin=0, vmax=1)
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
        scenario="simple_spread",
        n_agents=4,
        num_envs=40,
        continuous_actions=True,
        max_steps=100,
        seed=42,
        device="cuda" if torch.cuda.is_available() else "cpu",
        terminated_truncated=False,
    )
    animator.attach_and_load_model(
        model_name="ppo",
        path="/Users/jmalegaonkar/Desktop/InfoMARL-1/sb3/ppo_infomarl.zip",
        policy=policy.InfoMARLActorCriticPolicy,
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=1,
        batch_size=400,
        n_epochs=10,
        max_grad_norm=10,
        gamma=0.99,
        n_steps=100
    )
    animator.collect_data()
    animator.create_mp4("attention_animation.mp4")
