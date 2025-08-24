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
            else:
                self.n_food = self.n_agents
    
    def attach_and_load_model(self, model_name, path, **kwargs):
        if model_name == "ppo":
            self.model = PPO.load(path, **kwargs)
            if self.model.policy.observation_space != self.env.observation_space:
                self.model.policy.observation_space = self.env.observation_space
                self.model.policy.action_space = self.env.action_space
                self.model.policy.pi_features_extractor.actor.number_agents = self.env.num_agents
                self.model.policy.pi_features_extractor.actor.number_food = self.env.num_agents

                with torch.no_grad():
                    self.model.policy.log_std = torch.nn.Parameter(
                        torch.zeros(np.prod(self.env.action_space.shape), dtype=torch.float32)
                    )
        else:
            raise ValueError(f"Model {model_name} not supported.")

    def setup_corner_positions(self):
        """Set landmarks in four corners and agents in center with spacing"""
        # Define corner positions for landmarks
        corner_positions = [
            torch.tensor([-0.8, -0.8]),  # Bottom-left
            torch.tensor([0.8, -0.8]),   # Bottom-right
            torch.tensor([0.8, 0.8]),    # Top-right
            torch.tensor([-0.8, 0.8])    # Top-left
        ]
        
        # Place landmarks in corners
        for i, landmark in enumerate(self.env.env.world.landmarks):
            if i < len(corner_positions):
                landmark.set_pos(corner_positions[i], 0)
        
        # Place agents in center area with spacing
        if self.n_agents <= 4:
            # For few agents, arrange in a small square pattern
            agent_positions = [
                torch.tensor([-0.2, -0.2]),  # Bottom-left center
                torch.tensor([0.2, -0.2]),   # Bottom-right center
                torch.tensor([0.2, 0.2]),    # Top-right center
                torch.tensor([-0.2, 0.2])    # Top-left center
            ]
        else:
            # For more agents, arrange in a grid pattern in the center
            grid_size = int(np.ceil(np.sqrt(self.n_agents)))
            spacing = 0.3 / grid_size
            agent_positions = []
            
            for i in range(self.n_agents):
                row = i // grid_size
                col = i % grid_size
                
                # Center the grid around origin
                start_x = -(grid_size - 1) * spacing / 2
                start_y = -(grid_size - 1) * spacing / 2
                
                pos_x = start_x + col * spacing
                pos_y = start_y + row * spacing
                
                agent_positions.append(torch.tensor([pos_x, pos_y]))
        
        # Set agent positions
        for i, agent in enumerate(self.env.env.agents):
            if i < len(agent_positions):
                agent.set_pos(agent_positions[i], 0)

    def collect_data(self):
        obs = self.env.reset()
        
        # Set up corner positions
        self.setup_corner_positions()
        
        # Take a step to update observations with new positions
        actions = np.zeros_like(self.env.action_space.sample())
        obs, _, _, _ = self.env.step(np.array([actions] * self.num_envs))
        
        for _ in range(self.max_steps):
            action, _ = self.model.predict(obs, deterministic=True)
            actor = self.model.policy.features_extractor.actor
            
            # Get both attention weights
            cross_attention_weights = actor.cross_attention_weights
            cross_attention_weights = cross_attention_weights.view(self.num_envs, self.n_agents, self.n_food, self.n_agents)
            cross_attention_weights = cross_attention_weights[self.env_idx].cpu().numpy()
            
            # Get landmark attention weights
            landmark_attention_weights = actor.landmark_attention_weights
            landmark_attention_weights = landmark_attention_weights.view(self.num_envs, self.n_agents, self.n_food)
            landmark_attention_weights = landmark_attention_weights[self.env_idx].cpu().numpy()

            frame = self.env.render(
                mode="rgb_array",
                agent_index_focus=None
            )
            
            # Take a step in the environment
            obs, _, _, _ = self.env.step(action)

            yield frame, cross_attention_weights, landmark_attention_weights
    
    def create_mp4(self, path, fps=10, dpi=100):
        print(f"Creating mp4 file at {path}...")
        
        n_cols = int(np.ceil(np.sqrt(self.n_agents)))
        n_rows = int(np.ceil(self.n_agents / n_cols))

        # Set up figure with more complex GridSpec
        # Layout: [Environment] [Cross Attention Matrices] 
        #         [            ] [Landmark Attention Matrices]
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(n_rows * 2, n_cols + 1,
                            width_ratios=[2] + [1] * n_cols,
                            height_ratios=[1] * (n_rows * 2),
                            wspace=0.3, hspace=0.4)

        # Environment subplot spans the full left column
        ax_env = fig.add_subplot(gs[:, 0])
        ax_env.axis("off")
        ax_env.set_title(f"{self.scenario} Scenario - Corner Layout")

        # Cross attention matrices (top half of right side)
        cross_att_axes = []
        for idx in range(self.n_agents):
            row = idx // n_cols
            col = 1 + (idx % n_cols)
            ax = fig.add_subplot(gs[row, col])
            ax.set_title(f"Agent {idx} Cross Attention")
            ax.axis("off")
            cross_att_axes.append(ax)

        # Landmark attention matrices (bottom half of right side)
        landmark_att_axes = []
        for idx in range(self.n_agents):
            row = n_rows + (idx // n_cols)
            col = 1 + (idx % n_cols)
            ax = fig.add_subplot(gs[row, col])
            ax.set_title(f"Agent {idx} Landmark Attention")
            ax.axis("off")
            landmark_att_axes.append(ax)

        txt = fig.text(0.5, 0.95, "", ha="center", va="top",
                    fontsize=14, color="black", fontweight="bold")
        
        # Data collector
        collector = self.collect_data()
        n_frames = self.max_steps
        
        # Set up the writer
        writer = FFMpegWriter(fps=fps, metadata=dict(artist="Me"), bitrate=1800)
        with writer.saving(fig, path, dpi):
            for t in tqdm(range(n_frames)):
                frame, cross_att_weights, landmark_att_weights = next(collector)
                
                # Update env frame
                ax_env.clear()
                ax_env.axis("off")
                ax_env.set_title(f"{self.scenario} Scenario - Corner Layout")
                ax_env.imshow(frame)

                # Update counter
                txt.set_text(f"Frame: {t+1}/{n_frames} | Landmarks in Corners, Agents in Center")

                # Update each cross attention heatmap
                for i, ax in enumerate(cross_att_axes):
                    ax.clear()
                    ax.axis("off")
                    heat = cross_att_weights[i]
                    im = ax.imshow(
                        heat,
                        interpolation="nearest",
                        norm=Normalize(vmin=0, vmax=1),
                        cmap='Blues'
                    )

                    for (x, y), value in np.ndenumerate(heat):
                        ax.text(
                            y, x, f'{value:.2f}', ha='center', va='center', 
                            color='white' if value > 0.5 else 'black', fontsize=6
                        )
                    ax.set_title(f"Agent {i} Cross Att")
                    ax.set_xlabel("To Agents")
                    ax.set_ylabel("From Landmarks")

                # Update each landmark attention heatmap
                for i, ax in enumerate(landmark_att_axes):
                    ax.clear()
                    ax.axis("off")
                    heat = landmark_att_weights[i].reshape(1, -1)  # Reshape to row for better visualization
                    im = ax.imshow(
                        heat,
                        interpolation="nearest",
                        norm=Normalize(vmin=0, vmax=1),
                        cmap='Reds',
                        aspect='auto'
                    )

                    for (x, y), value in np.ndenumerate(heat):
                        ax.text(
                            y, x, f'{value:.2f}', ha='center', va='center', 
                            color='white' if value > 0.5 else 'black', fontsize=8
                        )
                    ax.set_title(f"Agent {i} Landmark Att")
                    ax.set_xlabel("Landmarks")

                # Grab and write
                writer.grab_frame()

        plt.close(fig)


if __name__ == "__main__":    
    animator = AttentionAnimator()
    animator.create_env(
        sim="vmas",
        env_idx=0,
        scenario="simple_spread",
        n_agents=4,
        num_envs=2,
        continuous_actions=True,
        max_steps=200,
        seed=2,
        device="cpu",
        terminated_truncated=False,        
    )

    animator.attach_and_load_model(
        model_name="ppo",
        path="ppo_infomarl.zip",
        device="cpu",
    )
        
    animator.collect_data()
    animator.create_mp4("corner_attention_animation.mp4")