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

def fix_attention_masks(model):
    policy = model.policy
    features_extractor = policy.features_extractor
    actor = features_extractor.actor
    
    orig_cross_attn = actor.cross_attention.forward
    orig_landmark_attn = actor.landmark_attention.forward
    
    actor.stored_cross_attn = None
    actor.stored_landmark_attn = None
    
    def cross_attn_hook(*args, **kwargs):
        output, attn_weights = orig_cross_attn(*args, **kwargs)
        actor.stored_cross_attn = attn_weights
        return output, attn_weights
    
    def landmark_attn_hook(*args, **kwargs):
        output, attn_weights = orig_landmark_attn(*args, **kwargs)
        actor.stored_landmark_attn = attn_weights
        return output, attn_weights
    
    actor.cross_attention.forward = cross_attn_hook
    actor.landmark_attention.forward = landmark_attn_hook
    
    return model

def get_attention_weights_via_hooks(model, observations):
    with torch.no_grad():
        obs_tensor = torch.as_tensor(observations, device=model.device)
        actions, _ = model.predict(obs_tensor, deterministic=True)
        actor = model.policy.features_extractor.actor
        cross_attn_weights = actor.stored_cross_attn
        landmark_attn_weights = actor.stored_landmark_attn
        
    return cross_attn_weights, landmark_attn_weights

class AttentionAnimator:
    def __init__(self, model, env, num_steps=400, env_idx=0, n_agents=4):
        self.model = model
        self.env = env
        self.num_steps = num_steps
        self.env_idx = env_idx
        self.n_agents = n_agents
        self.n_landmarks = n_agents
        
        self.agent_positions = []
        self.landmark_positions = None
        self.cross_attentions = []
        self.landmark_attentions = []
        
        self.agent_labels = [f"A{i}" for i in range(n_agents)]
        self.landmark_labels = [f"L{i}" for i in range(self.n_landmarks)]
        
    def create_unified_figure(self):
        self.fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 2, height_ratios=[3, 1, 1], width_ratios=[1.5, 2])
        
        self.ax_positions = plt.subplot(gs[0, 0])
        cross_gs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[0, 1], hspace=0.4, wspace=0.4)
        self.ax_cross_attns = []
        for i in range(self.n_agents):
            ax = plt.subplot(cross_gs[i//2, i%2])
            self.ax_cross_attns.append(ax)
        
        # Create a single subplot for all landmark attentions
        self.ax_landmark_attns = plt.subplot(gs[1:, 1])  # Takes up both bottom rows in right column
        
        self.ax_positions.set_title("Agent and Landmark Positions")
        self.ax_positions.set_xlim(-1.5, 1.5)
        self.ax_positions.set_ylim(-1.5, 1.5)
        self.ax_positions.grid(True)
        
        self.scatter_landmarks = self.ax_positions.scatter([], [], c='blue', s=100, marker='*', label='Landmarks')
        self.scatter_agents = self.ax_positions.scatter([], [], s=80, marker='o', label='Agents')
        self.agent_colors = ['red', 'green', 'purple', 'orange'][:self.n_agents]
        
        self.agent_annotations = []
        for i in range(self.n_agents):
            agent_ann = self.ax_positions.annotate(f"A{i}", (0, 0), fontsize=10, 
                                               xytext=(5, 5), textcoords='offset points')
            self.agent_annotations.append(agent_ann)
        
        self.landmark_annotations = []
        
        # Cross attention heatmaps
        self.cross_heatmaps = []
        for i, ax in enumerate(self.ax_cross_attns):
            ax.set_title(f"Agent {i} Cross Attention")
            dummy_cross = np.zeros((self.n_landmarks, self.n_agents))
            heatmap = ax.imshow(dummy_cross, cmap="viridis", vmin=0, vmax=1)
            self.cross_heatmaps.append(heatmap)
            
            # Add text annotations for values
            for j in range(self.n_landmarks):
                for k in range(self.n_agents):
                    text = ax.text(k, j, f"{0:.2f}",
                                 ha="center", va="center", color="w", fontsize=8)
            
            ax.set_xticks(np.arange(self.n_agents))
            ax.set_yticks(np.arange(self.n_landmarks))
            ax.set_xticklabels(self.agent_labels)
            ax.set_yticklabels(self.landmark_labels)
        
        # Landmark attention heatmap - now as a single plot
        self.ax_landmark_attns.set_title("Landmark Attention Across Agents")
        dummy_landmark = np.zeros((self.n_agents, self.n_landmarks))
        self.landmark_heatmap = self.ax_landmark_attns.imshow(dummy_landmark, 
                                                             cmap="viridis", 
                                                             vmin=0, vmax=1,
                                                             aspect='auto')
        
        # Add text annotations for values
        for i in range(self.n_agents):
            for j in range(self.n_landmarks):
                text = self.ax_landmark_attns.text(j, i, f"{0:.2f}",
                                                  ha="center", va="center", 
                                                  color="w", fontsize=8)
        
        self.ax_landmark_attns.set_xticks(np.arange(self.n_landmarks))
        self.ax_landmark_attns.set_yticks(np.arange(self.n_agents))
        self.ax_landmark_attns.set_xticklabels(self.landmark_labels)
        self.ax_landmark_attns.set_yticklabels(self.agent_labels)
        
        # Create colorbars
        # self.cross_cbar = self.fig.colorbar(self.cross_heatmaps[0], 
        #                                   ax=self.ax_cross_attns, 
        #                                   fraction=0.05, 
        #                                   pad=0.04)
        # self.landmark_cbar = self.fig.colorbar(self.landmark_heatmap,
        #                                      ax=self.ax_landmark_attns,
        #                                      fraction=0.05,
        #                                      pad=0.04)
        
        self.cross_cbar.set_label('Attention Weight')
        self.landmark_cbar.set_label('Attention Weight')
        
        self.step_title = self.fig.suptitle("Step: 0", fontsize=16)
        self.fig.tight_layout()
        self.fig.subplots_adjust(top=0.9)
    
    def collect_data(self):
        obs = self.env.reset()
        
        cur_pos_initial, _, landmarks_tensor, other_agents, _ = policy.env_parser_simple(
            torch.tensor(obs[self.env_idx]), self.n_agents
        )
        
        if landmarks_tensor.dim() > 2:
            self.landmark_positions = landmarks_tensor[0].cpu().numpy()
        else:
            self.landmark_positions = landmarks_tensor.view(-1, 2).cpu().numpy()
            
        if len(self.landmark_positions) != self.n_landmarks:
            print(f"Warning: Expected {self.n_landmarks} landmarks but found {len(self.landmark_positions)}")
            if len(self.landmark_positions) > self.n_landmarks:
                self.landmark_positions = self.landmark_positions[:self.n_landmarks]
        
        for step in range(self.num_steps):
            cross_attn, landmark_attn = get_attention_weights_via_hooks(self.model, obs)
            
            print(f"\nStep {step}:")
            print(f"Cross attention shape: {cross_attn.shape}")
            print(f"Landmark attention shape: {landmark_attn.shape}")

            cross_attn_np = cross_attn.view(40, 4, 4, 4)[self.env_idx].cpu().numpy()
            landmark_attn_np = landmark_attn.view(40, 4, 1, 4)[self.env_idx].cpu().numpy()

            self.cross_attentions.append(cross_attn_np)
            self.landmark_attentions.append(landmark_attn_np)
            
            cur_pos, _, _, _, _ = policy.env_parser_simple(
                torch.tensor(obs[self.env_idx]), self.n_agents
            )
            cur_pos_np = cur_pos.cpu().numpy()
            self.agent_positions.append(cur_pos_np)
            
            action, _ = self.model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = self.env.step(action)
            
            if step % 50 == 0:
                print(f"Data collection: Step {step}/{self.num_steps} completed")
    
    def update_animation(self, frame):
        self.step_title.set_text(f"Step: {frame}")
        
        # Update positions
        agents_pos = self.agent_positions[frame]
        self.scatter_agents.set_offsets(agents_pos)
        self.scatter_agents.set_color(self.agent_colors)
        
        for i, ann in enumerate(self.agent_annotations):
            ann.set_position(agents_pos[i])
        
        # Update cross attention heatmaps with values
        cross_attn_data = self.cross_attentions[frame]
        for i, heatmap in enumerate(self.cross_heatmaps):
            agent_data = cross_attn_data[i]
            heatmap.set_array(agent_data)
            
            # Update text annotations
            for j in range(self.n_landmarks):
                for k in range(self.n_agents):
                    self.ax_cross_attns[i].texts[j*self.n_agents + k].set_text(f"{agent_data[j,k]:.2f}")
        
        # Update landmark attention heatmap with values
        landmark_attn_data = self.landmark_attentions[frame]
        landmark_data = np.concatenate([x for x in landmark_attn_data])
        self.landmark_heatmap.set_array(landmark_data)
        
        # Update text annotations for landmark attention
        for i in range(self.n_agents):
            for j in range(self.n_landmarks):
                self.ax_landmark_attns.texts[i*self.n_landmarks + j].set_text(f"{landmark_data[i,j]:.2f}")
        
        return [self.scatter_agents, self.scatter_landmarks] + self.cross_heatmaps + [self.landmark_heatmap]
    
    def create_animation(self, save_path):
        print("Collecting data for animation...")
        self.collect_data()
        
        self.create_unified_figure()
        
        self.scatter_landmarks.set_offsets(self.landmark_positions)
        for i, pos in enumerate(self.landmark_positions):
            landmark_ann = self.ax_positions.annotate(f"L{i}", (pos[0], pos[1]), fontsize=10,
                                                    xytext=(5, 5), textcoords='offset points')
            self.landmark_annotations.append(landmark_ann)
        
        print("Creating unified animation...")
        animation = FuncAnimation(
            self.fig, self.update_animation,
            frames=self.num_steps, interval=50, blit=True
        )
        
        os.makedirs(save_path, exist_ok=True)
        animation.save(f"{save_path}/unified_attention_env{self.env_idx}.gif", 
                      writer='pillow', fps=10, dpi=100)
        
        print(f"Unified animation saved to {save_path}")
        return self.fig

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = vmas.make_env(
        scenario="simple_spread",
        n_agents=4,
        num_envs=40,
        continuous_actions=True,
        max_steps=400,
        seed=42,
        device=device,
        terminated_truncated=True,
    )

    env = wrapper.VMASVecEnv(env, rnd_nums=True)

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
    model_path = "/Users/jmalegaonkar/Desktop/InfoMARL-1/sb3/ppo_infomarl.zip"
    model.load(model_path)

    model = fix_attention_masks(model)

    animator = AttentionAnimator(
        model=model,
        env=env,
        num_steps=400,
        env_idx=0,
        n_agents=4
    )

    animator.create_animation(save_path="animations")