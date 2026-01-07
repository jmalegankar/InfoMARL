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
        self.goal_weights = None
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
                shared_rew=kwargs.get("shared_rew", True),
            )
            
            env = wrapper.VMASVecEnv(env, rnd_nums=True)
            self.env = env
            self.env_idx = env_idx
            self.scenario = kwargs["scenario"]
            self.max_steps = kwargs["max_steps"]
            self.n_agents = kwargs["n_agents"]
            self.num_envs = kwargs["num_envs"]
            self.goal_weights = []
    
    def attach_and_load_model(self, model_name, path, **kwargs):
        if model_name == "ppo":
            self.model = PPO.load(path, **kwargs)
            if self.model.policy.observation_space != self.env.observation_space:
                self.model.policy.observation_space = self.env.observation_space
                self.model.policy.action_space = self.env.action_space
                self.model.policy.pi_features_extractor.actor.number_agents = self.env.num_agents
                
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
            
            # Get goal attention weights
            if hasattr(actor, 'goal_attention_weights'):
                goal_attention_weights = actor.goal_attention_weights
                # Shape: (batch*agents, 1, 1)
                goal_attention_weights = goal_attention_weights.view(
                    self.num_envs, self.n_agents, 1, 1
                )
                goal_attention_weights = goal_attention_weights[self.env_idx].squeeze().cpu().numpy()
            else:
                goal_attention_weights = np.zeros((self.n_agents,))
            
            frame = self.env.render(
                mode="rgb_array",
                agent_index_focus=None
            )
            
            # Take a step in the environment
            obs, _, _, _ = self.env.step(action)
            
            yield frame, goal_attention_weights
    
    def create_mp4(self, path, fps=10, dpi=100):
        print(f"Creating mp4 file at {path}...")
        
        # Calculate grid layout
        n_cols = int(np.ceil(np.sqrt(self.n_agents)))
        n_rows = int(np.ceil(self.n_agents / n_cols))
        
        # Set up figure + GridSpec: left column for env, right columns for attention
        fig = plt.figure(figsize=(12, 8))
        gs = gridspec.GridSpec(
            n_rows, n_cols + 1,
            width_ratios=[2] + [1] * n_cols,
            height_ratios=[1] * n_rows,
            wspace=0.3, hspace=0.3
        )
        
        ax_env = fig.add_subplot(gs[:, 0])
        ax_env.axis("off")
        ax_env.set_title(f"{self.scenario} Scenario")
        
        # Create attention axes
        att_axes = []
        for idx in range(self.n_agents):
            row = idx // n_cols
            col = 1 + (idx % n_cols)
            ax = fig.add_subplot(gs[row, col])
            ax.set_title(f"Agent {idx} Goal Attn")
            ax.axis("off")
            att_axes.append(ax)
        
        txt = fig.text(0.5, 0.95, "", ha="center", va="top",
                      fontsize=14, color="black", fontweight="bold")
        
        # Data collector
        collector = self.collect_data()
        n_frames = self.max_steps
        
        # Set up the writer
        writer = FFMpegWriter(fps=fps, metadata=dict(artist="InfoMARL"), bitrate=1800)
        with writer.saving(fig, path, dpi):
            for t in tqdm(range(n_frames)):
                frame, goal_att_weights = next(collector)
                
                # Update env frame
                ax_env.clear()
                ax_env.axis("off")
                ax_env.set_title(f"{self.scenario} Scenario")
                ax_env.imshow(frame)
                
                # Update counter
                txt.set_text(f"Frame: {t+1}/{n_frames}")
                
                # Update attention heatmaps (shows hierarchy attention)
                for i, ax in enumerate(att_axes):
                    ax.clear()
                    ax.axis("off")
                    # Show as a single value per agent
                    value = goal_att_weights[i]
                    
                    # Create a simple bar
                    ax.barh([0], [value], color='blue', alpha=0.7)
                    ax.set_xlim(0, 1)
                    ax.set_ylim(-0.5, 0.5)
                    ax.text(value/2, 0, f'{value:.3f}', ha='center', va='center',
                           color='white', fontsize=10, fontweight='bold')
                    ax.set_title(f"Agent {i} Goal Attention")
                
                # Grab and write
                writer.grab_frame()
        
        plt.close(fig)
        print(f"Video saved to {path}")


if __name__ == "__main__":
    animator = AttentionAnimator()
    animator.create_env(
        sim="vmas",
        env_idx=0,
        scenario="multi_give_way",
        n_agents=4,
        num_envs=2,
        continuous_actions=True,
        max_steps=400,
        seed=1,
        device="cpu",
        terminated_truncated=False,
        shared_rew=True,
    )
    
    animator.attach_and_load_model(
        model_name="ppo",
        path="ppo_infomarl.zip",
        device="cpu",
    )
    
    animator.create_mp4("give_way_attention.mp4")