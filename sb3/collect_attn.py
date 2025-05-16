import vmas
import wrapper
import policy
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from stable_baselines3 import PPO
from matplotlib.animation import FuncAnimation
import os
from matplotlib import cm

def fix_attention_masks(model):
    policy = model.policy
    features_extractor = policy.features_extractor
    actor = features_extractor.actor
    
    # Store the original forward methods
    orig_cross_attn = actor.cross_attention.forward
    orig_landmark_attn = actor.landmark_attention.forward
    
    # Store for the results
    actor.stored_cross_attn = None
    actor.stored_landmark_attn = None
    
    # Define hook for cross attention
    def cross_attn_hook(*args, **kwargs):
        output, attn_weights = orig_cross_attn(*args, **kwargs)
        actor.stored_cross_attn = attn_weights
        return output, attn_weights
    
    # Define hook for landmark attention
    def landmark_attn_hook(*args, **kwargs):
        output, attn_weights = orig_landmark_attn(*args, **kwargs)
        actor.stored_landmark_attn = attn_weights
        return output, attn_weights
    
    # Replace the forward methods
    actor.cross_attention.forward = cross_attn_hook
    actor.landmark_attention.forward = landmark_attn_hook
    
    return model

def get_attention_weights_via_hooks(model, observations):
    with torch.no_grad():
        obs_tensor = torch.as_tensor(observations, device=model.device)
        # Get the policy to run forward on the observations
        actions, _ = model.predict(obs_tensor, deterministic=True)
        
        # Get the stored attention weights
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
        
        # Create storage for data
        self.agent_positions = []
        self.landmark_positions = []
        self.cross_attentions = []
        self.landmark_attentions = []
        
        # Labels
        self.agent_labels = [f"Agent {i}" for i in range(n_agents)]
        self.landmark_labels = [f"Landmark {i}" for i in range(n_agents)]
        self.all_agent_labels = [f"Agent {i}" for i in range(n_agents)]
        
        # Create figures
        self.create_figures()
    
    def create_figures(self):
        # Create figures and axes for animations
        self.fig_positions, self.ax_positions = plt.subplots(figsize=(8, 8))
        self.fig_cross_attn, self.ax_cross_attn = plt.subplots(figsize=(10, 8))
        self.fig_landmark_attn, self.ax_landmark_attn = plt.subplots(figsize=(10, 8))
        
        # Set titles
        self.ax_positions.set_title("Agent and Landmark Positions")
        self.ax_cross_attn.set_title("Cross Attention (Landmark to Agent)")
        self.ax_landmark_attn.set_title("Landmark Attention")
        
        # Initialize heatmaps with dummy data
        dummy_cross = np.zeros((self.n_agents, self.n_agents))
        dummy_landmark = np.zeros((1, self.n_agents))
        
        self.cross_heatmap = self.ax_cross_attn.imshow(
            dummy_cross, cmap="viridis", vmin=0, vmax=1, aspect='auto'
        )
        self.landmark_heatmap = self.ax_landmark_attn.imshow(
            dummy_landmark, cmap="viridis", vmin=0, vmax=1, aspect='auto'
        )
        
        # Add colorbars
        self.fig_cross_attn.colorbar(self.cross_heatmap, ax=self.ax_cross_attn)
        self.fig_landmark_attn.colorbar(self.landmark_heatmap, ax=self.ax_landmark_attn)
        
        # Set tick labels
        self.ax_cross_attn.set_xticks(np.arange(self.n_agents))
        self.ax_cross_attn.set_yticks(np.arange(self.n_agents))
        self.ax_cross_attn.set_xticklabels(self.all_agent_labels)
        self.ax_cross_attn.set_yticklabels(self.landmark_labels)
        
        self.ax_landmark_attn.set_xticks(np.arange(self.n_agents))
        self.ax_landmark_attn.set_yticks([0])
        self.ax_landmark_attn.set_xticklabels(self.landmark_labels)
        self.ax_landmark_attn.set_yticklabels(["Current Agent"])
        
        # Text annotations for heatmaps
        self.cross_attn_texts = []
        for i in range(self.n_agents):
            for j in range(self.n_agents):
                text = self.ax_cross_attn.text(j, i, "", 
                                              ha="center", va="center", color="black")
                self.cross_attn_texts.append(text)
        
        self.landmark_attn_texts = []
        for j in range(self.n_agents):
            text = self.ax_landmark_attn.text(j, 0, "", 
                                            ha="center", va="center", color="black")
            self.landmark_attn_texts.append(text)
        
        # Initialize position plots
        self.scatter_landmarks = self.ax_positions.scatter([], [], c='blue', s=100, marker='*', label='Landmarks')
        self.scatter_agents = self.ax_positions.scatter([], [], c='red', s=80, marker='o', label='Agents')
        
        # Labels for positions
        self.agent_annotations = []
        self.landmark_annotations = []
        for i in range(self.n_agents):
            agent_ann = self.ax_positions.annotate(f"Agent {i}", (0, 0), fontsize=10, 
                                                 xytext=(5, 5), textcoords='offset points')
            self.agent_annotations.append(agent_ann)
            
            landmark_ann = self.ax_positions.annotate(f"L{i}", (0, 0), fontsize=10,
                                                    xytext=(5, 5), textcoords='offset points')
            self.landmark_annotations.append(landmark_ann)
        
        # Set up position plot
        self.ax_positions.set_xlim(-1.5, 1.5)
        self.ax_positions.set_ylim(-1.5, 1.5)
        self.ax_positions.grid(True)
        self.ax_positions.legend()
        
        # Title will be updated with step information
        self.step_title = self.ax_positions.text(0.5, 1.02, "", transform=self.ax_positions.transAxes,
                                               ha="center", fontsize=12)
    
    def collect_data(self):
        """Run simulation and collect all the data for the animation"""
        # Reset the environment
        obs = self.env.reset()
        
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
            
            # Parse observations to get positions
            cur_pos, _, landmarks, other_agents, _ = policy.env_parser(
                torch.tensor(obs[self.env_idx]), self.n_agents
            )
            
            # Convert to numpy and store
            cur_pos_np = cur_pos.cpu().numpy()
            landmarks_np = landmarks.view(-1, 2).cpu().numpy()
            
            self.agent_positions.append(cur_pos_np)
            self.landmark_positions.append(landmarks_np)
            
            # Get action and step the environment
            action, _ = self.model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = self.env.step(action)
            
            if step % 50 == 0:
                print(f"Data collection: Step {step}/{self.num_steps} completed")
    
    def update_positions(self, frame):
        # Update step title
        self.step_title.set_text(f"Step: {frame}")
        
        # Update scatter plots
        agents_pos = self.agent_positions[frame]
        landmarks_pos = self.landmark_positions[frame]
        
        self.scatter_agents.set_offsets(agents_pos)
        self.scatter_landmarks.set_offsets(landmarks_pos)
        
        # Update annotations
        for i, ann in enumerate(self.agent_annotations):
            ann.set_position(agents_pos[i])
        
        for i, ann in enumerate(self.landmark_annotations):
            ann.set_position(landmarks_pos[i])
        
        return self.scatter_agents, self.scatter_landmarks, self.step_title
    
    def update_cross_attn(self, frame):
        # Update heatmap
        cross_attn_data = self.cross_attentions[frame]
        self.cross_heatmap.set_array(cross_attn_data)
        
        # Update text annotations
        for i in range(self.n_agents):
            for j in range(self.n_agents):
                idx = i * self.n_agents + j
                self.cross_attn_texts[idx].set_text(f"{cross_attn_data[i, j]:.2f}")
        
        self.ax_cross_attn.set_title(f"Cross Attention (Landmark to Agent) - Step {frame}")
        return self.cross_heatmap,
    
    def update_landmark_attn(self, frame):
        # Update heatmap
        landmark_attn_data = self.landmark_attentions[frame]
        self.landmark_heatmap.set_array(landmark_attn_data)
        
        # Update text annotations
        for j in range(self.n_agents):
            self.landmark_attn_texts[j].set_text(f"{landmark_attn_data[0, j]:.2f}")
        
        self.ax_landmark_attn.set_title(f"Landmark Attention - Step {frame}")
        return self.landmark_heatmap,
    
    def create_animations(self, save_path):
        # Collect data first
        print("Collecting data for animations...")
        self.collect_data()
        
        # Create animations
        print("Creating animations...")
        
        # Agent positions animation
        pos_anim = FuncAnimation(
            self.fig_positions, self.update_positions,
            frames=self.num_steps, interval=50, blit=True
        )
        
        # Cross attention animation
        cross_anim = FuncAnimation(
            self.fig_cross_attn, self.update_cross_attn,
            frames=self.num_steps, interval=50, blit=True
        )
        
        # Landmark attention animation
        landmark_anim = FuncAnimation(
            self.fig_landmark_attn, self.update_landmark_attn,
            frames=self.num_steps, interval=50, blit=True
        )
        
        # Save animations
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(f"{save_path}/pos", exist_ok=True)
        os.makedirs(f"{save_path}/cross", exist_ok=True)
        os.makedirs(f"{save_path}/landmarks", exist_ok=True)
        pos_anim.save(f"{save_path}/pos/agent_positions_env{self.env_idx}.gif", 
                    writer='pillow', fps=10, dpi=100)
        cross_anim.save(f"{save_path}/cross/cross_attention_env{self.env_idx}.gif", 
                      writer='pillow', fps=10, dpi=100)
        landmark_anim.save(f"{save_path}/landmarks/landmark_attention_env{self.env_idx}.gif", 
                         writer='pillow', fps=10, dpi=100)
        
        print(f"Animations saved to {save_path}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = vmas.make_env(
        scenario="simple_spread",
        n_agents=4,
        num_envs=64,
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
    )
    model_path = "/Users/jmalegaonkar/Desktop/InfoMARL-1/sb3/ppo_infomarl.zip"
    model.load(model_path)

    # Apply hooks for attention
    model = fix_attention_masks(model)

    # Create the animator
    animator = AttentionAnimator(
        model=model,
        env=env,
        num_steps=400,  # You can reduce this for faster testing
        env_idx=0,      # Which environment to visualize
        n_agents=4
    )

    # Create and save animations
    animator.create_animations(save_path="animations")