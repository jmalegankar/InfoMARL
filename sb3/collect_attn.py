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
    
    # store the original forward methods
    orig_cross_attn = actor.cross_attention.forward
    orig_landmark_attn = actor.landmark_attention.forward
    
    # store for the results
    actor.stored_cross_attn = None
    actor.stored_landmark_attn = None
    
    # define hook for cross attention
    def cross_attn_hook(*args, **kwargs):
        output, attn_weights = orig_cross_attn(*args, **kwargs)
        actor.stored_cross_attn = attn_weights
        return output, attn_weights
    
    # define hook for landmark attention
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
        # get the policy to run forward on the observations
        actions, _ = model.predict(obs_tensor, deterministic=True)
        
        # get the stored attention weights
        actor = model.policy.features_extractor.actor
        cross_attn_weights = actor.stored_cross_attn
        landmark_attn_weights = actor.stored_landmark_attn
        
    return cross_attn_weights, landmark_attn_weights

class AttentionAnimator:
    def __init__(self, model, env, num_steps=400, env_idx=0, n_agents=4):
        self.model = model
        self.env = env
        self.num_steps = num_steps
        self.env_idx = env_idx  # Environment index to visualize
        self.n_agents = n_agents
        self.n_landmarks = n_agents
        
        self.agent_positions = []
        self.landmark_positions = None  #store once - landmarks don't move
        self.cross_attentions = []
        self.landmark_attentions = []
        
        # Labels
        self.agent_labels = [f"A{i}" for i in range(n_agents)]
        self.landmark_labels = [f"L{i}" for i in range(self.n_landmarks)]
        
    def create_unified_figure(self):
        # Create a figure with custom grid layout
        self.fig = plt.figure(figsize=(14, 10))
        
        # create GridSpec for layout control
        gs = gridspec.GridSpec(3, 2, height_ratios=[3, 1, 1], width_ratios=[1.5, 2])
        
        self.ax_positions = plt.subplot(gs[0, 0])  # Position plot (larger, top-left)
        self.ax_cross_attn = plt.subplot(gs[0, 1])  # Cross attention matrix (top-right)
        
        # Create a row of agent-specific landmark attention visualizations
        self.ax_landmark_attns = []
        for i in range(self.n_agents):
            if i < 2:
                ax = plt.subplot(gs[1, i])
            else:
                ax = plt.subplot(gs[2, i-2])
            self.ax_landmark_attns.append(ax)
        
        self.ax_positions.set_title("Agent and Landmark Positions")
        self.ax_positions.set_xlim(-1.5, 1.5)
        self.ax_positions.set_ylim(-1.5, 1.5)
        self.ax_positions.grid(True)
        
        # Initialize landmarks and agents scatter plots with empty data
        self.scatter_landmarks = self.ax_positions.scatter([], [], c='blue', s=100, marker='*', label='Landmarks')
        
        # For agent scatter, we'll set colors when we first update with actual data
        # Just initialize with empty data for now
        self.scatter_agents = self.ax_positions.scatter([], [], s=80, marker='o', label='Agents')
        self.agent_colors = ['red', 'green', 'purple', 'orange'][:self.n_agents]
        
        # Create agent annotations
        self.agent_annotations = []
        for i in range(self.n_agents):
            agent_ann = self.ax_positions.annotate(f"A{i}", (0, 0), fontsize=10, 
                                               xytext=(5, 5), textcoords='offset points')
            self.agent_annotations.append(agent_ann)
        
        # Create landmark annotations
        self.landmark_annotations = []
        
        # Position legend
        self.ax_positions.legend(loc='upper right')
        
        # Set up cross attention heatmap
        self.ax_cross_attn.set_title("Cross Attention (Landmark to Agent)")
        dummy_cross = np.zeros((self.n_landmarks, self.n_agents))
        self.cross_heatmap = self.ax_cross_attn.imshow(
            dummy_cross, cmap="viridis", vmin=0, vmax=1, aspect='auto'
        )
        self.fig.colorbar(self.cross_heatmap, ax=self.ax_cross_attn)
        
        # Set tick labels for cross attention
        self.ax_cross_attn.set_xticks(np.arange(self.n_agents))
        self.ax_cross_attn.set_yticks(np.arange(self.n_landmarks))
        self.ax_cross_attn.set_xticklabels(self.agent_labels)
        self.ax_cross_attn.set_yticklabels(self.landmark_labels)
        
        # Set up landmark attention heatmaps for each agent
        self.landmark_heatmaps = []
        
        for i, ax in enumerate(self.ax_landmark_attns):
            ax.set_title(f"Agent {i} Landmark Attention")
            dummy_landmark = np.zeros((1, self.n_landmarks))
            # Use a fixed aspect ratio that's wider than tall for rectangluar shape
            heatmap = ax.imshow(dummy_landmark, cmap="viridis", vmin=0, vmax=1, aspect=3.0)
            self.landmark_heatmaps.append(heatmap)
            
            # Set up tick labels
            ax.set_xticks(np.arange(self.n_landmarks))
            ax.set_yticks([0])
            ax.set_xticklabels(self.landmark_labels)
            ax.set_yticklabels([f"A{i}"])
        
        # Add a colorbar for the landmark attention (just one for all agents)
        # Use the last heatmap for the colorbar reference
        self.fig.colorbar(self.landmark_heatmaps[-1], ax=self.ax_landmark_attns[-1])
        
        # Step title for animation
        self.step_title = self.fig.suptitle("Step: 0", fontsize=16)
        
        # Adjust layout for better spacing
        self.fig.tight_layout()
        self.fig.subplots_adjust(top=0.9)
    
    def collect_data(self):
        # Reset the environment
        obs = self.env.reset()
        
        # Extract landmark positions
        cur_pos_initial, _, landmarks_tensor, other_agents, _ = policy.env_parser(
            torch.tensor(obs[self.env_idx]), self.n_agents
        )
        
        if landmarks_tensor.dim() > 2:
            # If landmarks are repeated per agent, take the first agent's view
            self.landmark_positions = landmarks_tensor[0].cpu().numpy()
        else:
            # If landmarks are already flattened to [n_landmarks, 2]
            self.landmark_positions = landmarks_tensor.view(-1, 2).cpu().numpy()
            
        # Ensure we have exactly n_landmarks
        if len(self.landmark_positions) != self.n_landmarks:
            print(f"Warning: Expected {self.n_landmarks} landmarks but found {len(self.landmark_positions)}")
            if len(self.landmark_positions) > self.n_landmarks:
                self.landmark_positions = self.landmark_positions[:self.n_landmarks]
        
        # print(f"Fixed landmarks positions: {self.landmark_positions}")
        
        for step in range(self.num_steps):
            # Get attention weights
            cross_attn, landmark_attn = get_attention_weights_via_hooks(self.model, obs)
            
            # Extract weights for the specific environment
            if cross_attn.dim() == 4:  # For batch multi-head attention weights [batch, heads, src, tgt]
                cross_attn_np = cross_attn[self.env_idx, 0].cpu().numpy()  # Take the first head
            elif cross_attn.dim() == 3:  # For batch attention weights [batch, src, tgt]
                cross_attn_np = cross_attn[self.env_idx].cpu().numpy()
            
            if landmark_attn.dim() == 4:
                landmark_attn_np = landmark_attn[self.env_idx, 0].cpu().numpy()
            elif landmark_attn.dim() == 3:
                landmark_attn_np = landmark_attn[self.env_idx].cpu().numpy()
            
            # Store the attention weights
            self.cross_attentions.append(cross_attn_np)
            self.landmark_attentions.append(landmark_attn_np)
            
            # Parse observations to get agent positions
            cur_pos, _, _, _, _ = policy.env_parser(
                torch.tensor(obs[self.env_idx]), self.n_agents
            )
            
            # Convert to numpy and store agent positions
            cur_pos_np = cur_pos.cpu().numpy()
            self.agent_positions.append(cur_pos_np)
            
            # Get action and step the environment
            action, _ = self.model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = self.env.step(action)
            
            if step % 50 == 0:
                print(f"Data collection: Step {step}/{self.num_steps} completed")
    
    def update_animation(self, frame):
        """Update function for the unified animation"""
        # Update step title
        self.step_title.set_text(f"Step: {frame}")
        
        # 1. Update agent positions
        agents_pos = self.agent_positions[frame]
        
        # Set agent positions and colors
        self.scatter_agents.set_offsets(agents_pos)
        self.scatter_agents.set_color(self.agent_colors)
        
        # Update agent annotations
        for i, ann in enumerate(self.agent_annotations):
            ann.set_position(agents_pos[i])
        
        # Update cross attention heatmap
        cross_attn_data = self.cross_attentions[frame]
        
        # Ensure correct shape
        if cross_attn_data.shape[0] != self.n_landmarks or cross_attn_data.shape[1] != self.n_agents:
            print(f"Warning: Expected cross attention shape [{self.n_landmarks}, {self.n_agents}] but got {cross_attn_data.shape}")
            # Try to reshape or slice if possible
            if cross_attn_data.size >= self.n_landmarks * self.n_agents:
                cross_attn_data = cross_attn_data[:self.n_landmarks, :self.n_agents]
        
        self.cross_heatmap.set_array(cross_attn_data)
        
        # 3. Update landmark attention heatmaps for each agent
        landmark_attn_data = self.landmark_attentions[frame]
        
        # Update each agent's landmark attention visualization
        for i, heatmap in enumerate(self.landmark_heatmaps):
            # Extract the attention for this specific agent (reshape if needed)
            if landmark_attn_data.shape[0] == self.n_agents:
                # If we have separate attention per agent
                agent_landmark_attn = landmark_attn_data[i:i+1, :]
            else:
                # If we have a single attention matrix, use that for all agents
                agent_landmark_attn = landmark_attn_data[0:1, :]
            
            # Ensure correct shape
            if agent_landmark_attn.shape[1] != self.n_landmarks:
                if agent_landmark_attn.size >= self.n_landmarks:
                    agent_landmark_attn = agent_landmark_attn[:, :self.n_landmarks]
            
            heatmap.set_array(agent_landmark_attn)
        
        # Return all updated artists
        return [self.scatter_agents, self.scatter_landmarks, self.cross_heatmap] + self.landmark_heatmaps
    
    def create_animation(self, save_path):
        # Collect data first
        print("Collecting data for animation...")
        self.collect_data()
        
        # Create the figure
        self.create_unified_figure()
        
        # Set landmark positions and annotations - they're static!
        self.scatter_landmarks.set_offsets(self.landmark_positions)
        for i, pos in enumerate(self.landmark_positions):
            landmark_ann = self.ax_positions.annotate(f"L{i}", (pos[0], pos[1]), fontsize=10,
                                                    xytext=(5, 5), textcoords='offset points')
            self.landmark_annotations.append(landmark_ann)
        
        # Create animation
        print("Creating unified animation...")
        
        animation = FuncAnimation(
            self.fig, self.update_animation,
            frames=self.num_steps, interval=50, blit=True
        )
        
        # Save animation
        os.makedirs(save_path, exist_ok=True)
        animation.save(f"{save_path}/unified_attention_env{self.env_idx}.gif", 
                      writer='pillow', fps=10, dpi=100)
        
        print(f"Unified animation saved to {save_path}")
        return self.fig

# Main execution
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

    # Apply hooks for attention
    model = fix_attention_masks(model)

    # Create the improved animator
    animator = AttentionAnimator(
        model=model,
        env=env,
        num_steps=400,
        env_idx=0,
        n_agents=4
    )

    # Create and save the unified animation
    animator.create_animation(save_path="animations")