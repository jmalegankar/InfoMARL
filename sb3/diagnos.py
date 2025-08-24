import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import your modules
import vmas
import wrapper
import policy
from stable_baselines3 import PPO

class AttentionBehaviorDiagnostic:
    def __init__(self, n_agents=4, max_steps=100):
        self.n_agents = n_agents
        self.n_landmarks = n_agents  # In simple_spread
        self.max_steps = max_steps
        
    def setup_environment(self, model_path="ppo_infomarl.zip", seed=None):
        """Setup environment with optional random seed"""
        self.env = vmas.make_env(
            scenario="simple_spread",
            n_agents=self.n_agents,
            num_envs=1,
            continuous_actions=True,
            max_steps=self.max_steps,
            seed=seed if seed else np.random.randint(0, 10000),
            device="cpu",
            terminated_truncated=False,
        )
        self.env = wrapper.VMASVecEnv(self.env, rnd_nums=True)
        
        # Load model
        self.model = PPO.load(model_path, device="cpu")
        self.model.policy.observation_space = self.env.observation_space
        self.model.policy.action_space = self.env.action_space
        self.model.policy.pi_features_extractor.actor.number_agents = self.n_agents
        self.model.policy.pi_features_extractor.actor.number_food = self.n_landmarks
        
        # Important: Set model to eval mode to access attention weights
        self.model.policy.eval()
        self.model.policy.pi_features_extractor.actor.training = False
        
    def extract_all_attention_types(self, obs):
        """Extract ALL types of attention from the model"""
        with torch.no_grad():
            # Get action and trigger attention computation
            action, _ = self.model.predict(obs, deterministic=True)
            
            actor = self.model.policy.pi_features_extractor.actor
            
            # Extract all three types of attention
            attentions = {}
            
            # 1. Cross attention (agent_landmark) - shape: (n_landmarks, n_agents)
            if hasattr(actor, 'cross_attention_weights'):
                cross_attn = actor.cross_attention_weights
                cross_attn = cross_attn.view(1, self.n_agents, self.n_landmarks, self.n_agents)
                attentions['landmark_to_agent'] = cross_attn[0].cpu().numpy()
            
            # 2. Landmark attention (cur_landmark) - shape: (n_agents, 1, n_landmarks)
            if hasattr(actor, 'landmark_attention_weights'):
                landmark_attn = actor.landmark_attention_weights
                landmark_attn = landmark_attn.view(self.n_agents, 1, self.n_landmarks)
                attentions['agent_to_landmark'] = landmark_attn[:, 0, :].cpu().numpy()
            
            return attentions, action
    
    def compute_agent_landmark_distances(self):
        """Compute distances between all agents and landmarks"""
        distances = np.zeros((self.n_agents, self.n_landmarks))
        
        for i, agent in enumerate(self.env.env.agents):
            agent_pos = agent.state.pos[0].cpu().numpy()
            for j, landmark in enumerate(self.env.env.world.landmarks):
                landmark_pos = landmark.state.pos[0].cpu().numpy()
                distances[i, j] = np.linalg.norm(agent_pos - landmark_pos)
                
        return distances
    
    def identify_target_landmarks(self, distances):
        """Identify which landmark each agent is targeting (closest)"""
        return np.argmin(distances, axis=1)
    
    def run_diagnostic(self, n_episodes=100, use_random_positions=True):
        """Run diagnostic across multiple episodes"""
        
        all_results = {
            'attention_types': {},
            'behavior_correlation': [],
            'final_assignments': [],
            'attention_behavior_match': []
        }
        
        for episode in tqdm(range(n_episodes), desc="Running diagnostic episodes"):
            # Reset with new seed each time
            obs = self.env.reset()
            
            if not use_random_positions:
                # Use controlled positions
                pos = torch.tensor([0.0, 0.0])
                for idx, agent in enumerate(self.env.env.agents):
                    agent.set_pos(pos.clone(), 0)
                    
                pos = torch.tensor([-1.0, 1.0])
                for idx, landmark in enumerate(self.env.env.world.landmarks):
                    pos[0] = (idx % 2) * 2 - 1.0
                    pos[1] = (idx // 2) * 2 - 1.0
                    landmark.set_pos(pos.clone(), 0)
            
            episode_data = {
                'agent_to_landmark_attention': [],
                'landmark_to_agent_attention': [],
                'target_landmarks': [],
                'distances': []
            }
            
            # Run episode
            for t in range(self.max_steps):
                # Extract all attention types
                attentions, action = self.extract_all_attention_types(obs)
                
                # Get current distances and targets
                distances = self.compute_agent_landmark_distances()
                targets = self.identify_target_landmarks(distances)
                
                # Store data
                if 'agent_to_landmark' in attentions:
                    episode_data['agent_to_landmark_attention'].append(attentions['agent_to_landmark'])
                if 'landmark_to_agent' in attentions:
                    episode_data['landmark_to_agent_attention'].append(attentions['landmark_to_agent'])
                episode_data['target_landmarks'].append(targets)
                episode_data['distances'].append(distances)
                
                # Step environment
                obs, _, _, _ = self.env.step(action)
            
            # Analyze episode results
            if episode_data['agent_to_landmark_attention']:
                agent_to_landmark = np.array(episode_data['agent_to_landmark_attention'])
                targets = np.array(episode_data['target_landmarks'])
                
                # Compute correlation between attention and actual targets
                correlations = []
                for agent in range(self.n_agents):
                    # Get attention to each landmark over time
                    agent_attention = agent_to_landmark[:, agent, :]
                    agent_targets = targets[:, agent]
                    
                    # Create binary matrix of actual targets
                    target_matrix = np.zeros_like(agent_attention)
                    for t in range(len(agent_targets)):
                        target_matrix[t, agent_targets[t]] = 1
                    
                    # Compute correlation between attention and targets
                    corr_per_landmark = []
                    for landmark in range(self.n_landmarks):
                        if np.std(agent_attention[:, landmark]) > 0 and np.std(target_matrix[:, landmark]) > 0:
                            corr, _ = spearmanr(agent_attention[:, landmark], target_matrix[:, landmark])
                            corr_per_landmark.append(corr)
                    
                    if corr_per_landmark:
                        correlations.append(np.mean(corr_per_landmark))
                
                all_results['behavior_correlation'].append(correlations)
            
            # Check final configuration
            final_distances = episode_data['distances'][-1]
            final_assignment = self.identify_target_landmarks(final_distances)
            all_results['final_assignments'].append(final_assignment)
            
            # Check if attention matches behavior at the end
            if episode_data['agent_to_landmark_attention']:
                final_attention = episode_data['agent_to_landmark_attention'][-1]
                attention_assignment = np.argmax(final_attention, axis=1)
                match_rate = np.mean(attention_assignment == final_assignment)
                all_results['attention_behavior_match'].append(match_rate)
        
        return all_results, episode_data
    
    def visualize_diagnostic(self, results, last_episode_data):
        """Visualize the diagnostic results"""
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        
        # 1. Final attention vs actual targets (last episode)
        if last_episode_data['agent_to_landmark_attention']:
            final_attention = last_episode_data['agent_to_landmark_attention'][-1]
            final_distances = last_episode_data['distances'][-1]
            final_targets = self.identify_target_landmarks(final_distances)
            
            # Attention heatmap
            sns.heatmap(final_attention, annot=True, fmt='.2f', cmap='YlOrRd',
                       xticklabels=[f'L{i}' for i in range(self.n_landmarks)],
                       yticklabels=[f'A{i}' for i in range(self.n_agents)],
                       ax=axes[0, 0], vmin=0, vmax=1)
            
            # Mark actual targets
            for agent, target in enumerate(final_targets):
                axes[0, 0].add_patch(plt.Rectangle((target, agent), 1, 1, 
                                                   fill=False, edgecolor='blue', lw=3))
            axes[0, 0].set_title('Final Attention (boxes = actual targets)')
        
        # 2. Distance matrix at the end
        sns.heatmap(final_distances, annot=True, fmt='.2f', cmap='YlGn_r',
                   xticklabels=[f'L{i}' for i in range(self.n_landmarks)],
                   yticklabels=[f'A{i}' for i in range(self.n_agents)],
                   ax=axes[0, 1])
        axes[0, 1].set_title('Final Agent-Landmark Distances')
        
        # 3. Attention-behavior match rate across episodes
        if results['attention_behavior_match']:
            match_rates = results['attention_behavior_match']
            axes[0, 2].bar(range(len(match_rates)), match_rates)
            axes[0, 2].axhline(y=1.0, color='g', linestyle='--', label='Perfect match')
            axes[0, 2].axhline(y=1/self.n_landmarks, color='r', linestyle='--', label='Random')
            axes[0, 2].set_xlabel('Episode')
            axes[0, 2].set_ylabel('Match Rate')
            axes[0, 2].set_title('Attention-Behavior Agreement')
            axes[0, 2].legend()
        
        # 4. Attention evolution over time (last episode)
        if last_episode_data['agent_to_landmark_attention']:
            attention_over_time = np.array(last_episode_data['agent_to_landmark_attention'])
            
            for agent in range(min(2, self.n_agents)):  # Show first 2 agents
                ax = axes[1, agent]
                # Show how attention to each landmark changes over time
                for landmark in range(self.n_landmarks):
                    ax.plot(attention_over_time[:, agent, landmark], 
                           label=f'L{landmark}', alpha=0.7)
                
                # Mark when agent reached its target
                targets = np.array(last_episode_data['target_landmarks'])[:, agent]
                changes = np.where(np.diff(targets) != 0)[0]
                for change in changes:
                    ax.axvline(x=change, color='red', linestyle='--', alpha=0.5)
                
                ax.set_xlabel('Timestep')
                ax.set_ylabel('Attention Weight')
                ax.set_title(f'Agent {agent} Attention Evolution')
                ax.legend(loc='upper right', fontsize=8)
                ax.set_ylim([0, 1])
        
        # 5. Landmark-to-agent attention (if available)
        if last_episode_data['landmark_to_agent_attention']:
            final_landmark_to_agent = last_episode_data['landmark_to_agent_attention'][-1]
            
            # Show how landmarks "attend" to agents
            sns.heatmap(final_landmark_to_agent[:, :, 0], annot=True, fmt='.2f', 
                       cmap='Blues',
                       xticklabels=[f'A{i}' for i in range(self.n_agents)],
                       yticklabels=[f'L{i}' for i in range(self.n_landmarks)],
                       ax=axes[1, 2])
            axes[1, 2].set_title('Landmark→Agent Attention (Agent 0 perspective)')
        
        # 6. Average attention vs average proximity
        if last_episode_data['agent_to_landmark_attention']:
            attention_over_time = np.array(last_episode_data['agent_to_landmark_attention'])
            distances_over_time = np.array(last_episode_data['distances'])
            
            # Convert distances to proximity (inverse)
            proximity_over_time = 1 / (distances_over_time + 0.1)  # Add small value to avoid div by 0
            proximity_over_time = proximity_over_time / proximity_over_time.sum(axis=2, keepdims=True)
            
            # Average over time
            mean_attention = np.mean(attention_over_time, axis=0)
            mean_proximity = np.mean(proximity_over_time, axis=0)
            
            # Scatter plot comparing attention vs proximity
            ax = axes[2, 0]
            for agent in range(self.n_agents):
                ax.scatter(mean_proximity[agent], mean_attention[agent], 
                          label=f'Agent {agent}', s=100)
            
            # Add diagonal line
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
            ax.set_xlabel('Normalized Proximity')
            ax.set_ylabel('Attention Weight')
            ax.set_title('Attention vs Proximity Correlation')
            ax.legend()
        
        # 7. Assignment uniqueness check
        final_assignments = np.array(results['final_assignments'])
        uniqueness_scores = []
        for assignment in final_assignments:
            uniqueness_scores.append(len(np.unique(assignment)) / self.n_agents)
        
        axes[2, 1].hist(uniqueness_scores, bins=10, edgecolor='black')
        axes[2, 1].axvline(x=1.0, color='g', linestyle='--', label='Perfect coordination')
        axes[2, 1].set_xlabel('Uniqueness Score')
        axes[2, 1].set_ylabel('Frequency')
        axes[2, 1].set_title('Coordination Success Rate')
        axes[2, 1].legend()
        
        # 8. Summary statistics
        summary_text = "DIAGNOSTIC SUMMARY\n" + "="*30 + "\n\n"
        
        # Check if behavior works
        mean_uniqueness = np.mean(uniqueness_scores)
        summary_text += f"Behavioral Success Rate: {mean_uniqueness:.1%}\n"
        summary_text += f"(Agents reach unique landmarks)\n\n"
        
        # Check attention-behavior alignment
        if results['attention_behavior_match']:
            mean_match = np.mean(results['attention_behavior_match'])
            summary_text += f"Attention-Behavior Match: {mean_match:.1%}\n"
            summary_text += f"(Attention predicts actual target)\n\n"
        
        # Interpretation
        if mean_uniqueness > 0.9 and mean_match < 0.5:
            summary_text += "FINDING: Behavior succeeds but\n"
            summary_text += "attention doesn't directly\n"
            summary_text += "correspond to targets!\n\n"
            summary_text += "Likely causes:\n"
            summary_text += "• Attention is for feature\n"
            summary_text += "  extraction, not selection\n"
            summary_text += "• Action determined by MLPs\n"
            summary_text += "  after attention\n"
            summary_text += "• Multiple attention types\n"
            summary_text += "  working together"
        
        axes[2, 2].text(0.1, 0.5, summary_text, fontsize=10,
                       verticalalignment='center', family='monospace')
        axes[2, 2].axis('off')
        
        plt.tight_layout()
        return fig
    
    def run_complete_diagnostic(self):
        """Run the complete diagnostic analysis"""
        print("="*60)
        print("ATTENTION-BEHAVIOR DIAGNOSTIC")
        print("="*60)
        
        # Test with random positions
        print("\n1. Testing with RANDOM initial positions...")
        self.setup_environment(seed=42)
        random_results, random_episode = self.run_diagnostic(n_episodes=100, use_random_positions=True)
        
        # Test with controlled positions
        print("\n2. Testing with CONTROLLED linear positions...")
        self.setup_environment(seed=42)
        controlled_results, controlled_episode = self.run_diagnostic(n_episodes=100, use_random_positions=False)
        
        # Create visualizations
        print("\n3. Creating diagnostic visualizations...")
        
        # Random positions visualization
        fig1 = self.visualize_diagnostic(random_results, random_episode)
        fig1.suptitle("RANDOM Initial Positions", fontsize=16, y=1.02)
        plt.savefig('diagnostic_random.png', dpi=150, bbox_inches='tight')
        
        # Controlled positions visualization
        fig2 = self.visualize_diagnostic(controlled_results, controlled_episode)
        fig2.suptitle("CONTROLLED Linear Positions", fontsize=16, y=1.02)
        plt.savefig('diagnostic_controlled.png', dpi=150, bbox_inches='tight')
        
        # Print findings
        print("\n" + "="*60)
        print("KEY FINDINGS")
        print("="*60)
        
        # Check behavioral success
        random_success = np.mean([len(np.unique(a))/self.n_agents 
                                 for a in random_results['final_assignments']])
        controlled_success = np.mean([len(np.unique(a))/self.n_agents 
                                     for a in controlled_results['final_assignments']])
        
        print(f"\nBehavioral Coordination Success:")
        print(f"  Random positions: {random_success:.1%}")
        print(f"  Controlled positions: {controlled_success:.1%}")
        
        # Check attention-behavior alignment
        if random_results['attention_behavior_match']:
            random_match = np.mean(random_results['attention_behavior_match'])
            controlled_match = np.mean(controlled_results['attention_behavior_match'])
            
            print(f"\nAttention-Behavior Alignment:")
            print(f"  Random positions: {random_match:.1%}")
            print(f"  Controlled positions: {controlled_match:.1%}")
            
            if random_success > 0.9 and random_match < 0.5:
                print("\n⚠️  IMPORTANT DISCOVERY:")
                print("The agents successfully coordinate (reach unique landmarks)")
                print("but their attention weights DON'T directly indicate their targets!")
                print("\nThis suggests:")
                print("1. Attention is used for feature extraction, not direct selection")
                print("2. The MLP layers after attention determine final actions")
                print("3. The policy might use indirect cues (e.g., avoiding others)")
                print("4. Multiple attention mechanisms work together in complex ways")
        
        plt.show()
        
        return random_results, controlled_results

if __name__ == "__main__":
    diagnostic = AttentionBehaviorDiagnostic(n_agents=4, max_steps=400)
    random_results, controlled_results = diagnostic.run_complete_diagnostic()
    
    print("\n" + "="*60)
    print("Diagnostic complete! Check 'diagnostic_random.png' and")
    print("'diagnostic_controlled.png' for detailed visualizations.")
    print("="*60)