import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns
import vmas
import wrapper
import policy
from stable_baselines3 import PPO

class InfoMARLExplainabilityTester:
    def __init__(self, model_path, device="cpu"):
        self.device = device
        self.model = None
        self.env = None
        self.load_model(model_path)
        
    def load_model(self, model_path):
        """Load trained InfoMARL model"""
        env = vmas.make_env(
            scenario="simple_spread",
            n_agents=4,
            num_envs=1,
            continuous_actions=True,
            max_steps=100,
            device=self.device,
        )
        env = wrapper.VMASVecEnv(env, rnd_nums=True)
        
        self.model = PPO.load(model_path, env=env, device=self.device)
        self.env = env
        print(f"Loaded model for {self.env.num_agents} agents")
        
    def setup_scenario(self, agent_positions, landmark_positions):
        """Set up environment with specific positions"""
        self.env.reset()
        
        for i, pos in enumerate(agent_positions):
            self.env.env.agents[i].set_pos(torch.tensor(pos, device=self.device), 0)
            
        for i, pos in enumerate(landmark_positions):
            self.env.env.world.landmarks[i].set_pos(torch.tensor(pos, device=self.device), 0)
            
        obs = self.env.reset()
        return obs, agent_positions, landmark_positions
        
    def get_attention_data(self, obs):
        """Extract attention weights and understand their structure"""
        with torch.no_grad():
            self.model.policy.pi_features_extractor.actor.training = False
            action, _ = self.model.predict(obs, deterministic=True)
            actor = self.model.policy.pi_features_extractor.actor
            
            if not hasattr(actor, 'cross_attention_weights'):
                print("Warning: No attention weights found!")
                return None
            
            # Get raw attention weights
            cross_attention = actor.cross_attention_weights.cpu().numpy()
            print(f"Raw attention shape: {cross_attention.shape}")
            
            # Based on visualize.py, this should be reshaped to (agents, landmarks, agents)
            # But we're getting (4,4,4), so let's understand what each dimension means
            
            self.model.policy.pi_features_extractor.actor.training = True
            
        return cross_attention
        
    def test_attention_consistency(self):
        """Test 1: Does the same scenario produce consistent attention patterns?"""
        print("=== Test 1: Attention Consistency ===")
        
        # Fixed scenario
        agent_pos = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
        landmark_pos = np.array([[-0.8, 0.8], [0.8, 0.8], [0.8, -0.8], [-0.8, -0.8]])
        
        attention_matrices = []
        
        # Run same scenario multiple times
        for trial in range(5):
            obs, _, _ = self.setup_scenario(agent_pos, landmark_pos)
            attention = self.get_attention_data(obs)
            if attention is not None:
                attention_matrices.append(attention)
        
        if len(attention_matrices) < 2:
            print("Not enough attention data for consistency test")
            return None
            
        # Calculate consistency (correlation between trials)
        consistency_scores = []
        for i in range(len(attention_matrices)):
            for j in range(i+1, len(attention_matrices)):
                # Flatten matrices and correlate
                flat1 = attention_matrices[i].flatten()
                flat2 = attention_matrices[j].flatten()
                if len(set(flat1)) > 1 and len(set(flat2)) > 1:
                    corr, _ = pearsonr(flat1, flat2)
                    consistency_scores.append(corr)
        
        avg_consistency = np.mean(consistency_scores) if consistency_scores else 0
        print(f"Average attention consistency: {avg_consistency:.3f}")
        print(f"Good if > 0.8 (consistent), Bad if < 0.5 (random)")
        
        self.visualize_consistency(attention_matrices, consistency_scores)
        return {'consistency_scores': consistency_scores, 'avg_consistency': avg_consistency}
        
    def test_attention_diversity(self):
        """Test 2: Do agents have diverse attention patterns (not all the same)?"""
        print("\n=== Test 2: Attention Diversity ===")
        
        # Random scenario
        agent_pos = np.random.uniform(-1, 1, (4, 2))
        landmark_pos = np.random.uniform(-1, 1, (4, 2))
        
        obs, _, _ = self.setup_scenario(agent_pos, landmark_pos)
        attention = self.get_attention_data(obs)
        
        if attention is None:
            return None
            
        # Compare attention patterns between agents
        diversity_scores = []
        
        for i in range(4):
            for j in range(i+1, 4):
                # Get attention matrices for agents i and j
                agent_i_attention = attention[i].flatten()
                agent_j_attention = attention[j].flatten()
                
                # Calculate similarity (1 - correlation = diversity)
                if len(set(agent_i_attention)) > 1 and len(set(agent_j_attention)) > 1:
                    similarity, _ = pearsonr(agent_i_attention, agent_j_attention)
                    diversity = 1 - abs(similarity)  # Higher diversity = lower similarity
                    diversity_scores.append(diversity)
                    print(f"Agent {i} vs Agent {j}: diversity = {diversity:.3f}")
        
        avg_diversity = np.mean(diversity_scores) if diversity_scores else 0
        print(f"Average attention diversity: {avg_diversity:.3f}")
        print(f"Good if > 0.5 (agents attend to different things), Bad if < 0.2 (all same)")
        
        self.visualize_diversity(attention, diversity_scores)
        return {'diversity_scores': diversity_scores, 'avg_diversity': avg_diversity}
        
    def test_attention_sensitivity(self):
        """Test 3: Does attention change appropriately when environment changes?"""
        print("\n=== Test 3: Attention Sensitivity ===")
        
        # Scenario 1: Agents close to landmarks
        agent_pos_1 = np.array([[-0.9, -0.9], [0.9, -0.9], [0.9, 0.9], [-0.9, 0.9]])
        landmark_pos_1 = np.array([[-0.8, -0.8], [0.8, -0.8], [0.8, 0.8], [-0.8, 0.8]])
        
        # Scenario 2: Agents far from landmarks  
        agent_pos_2 = np.array([[-0.3, -0.3], [0.3, -0.3], [0.3, 0.3], [-0.3, 0.3]])
        landmark_pos_2 = np.array([[-0.8, -0.8], [0.8, -0.8], [0.8, 0.8], [-0.8, 0.8]])
        
        obs1, _, _ = self.setup_scenario(agent_pos_1, landmark_pos_1)
        attention1 = self.get_attention_data(obs1)
        
        obs2, _, _ = self.setup_scenario(agent_pos_2, landmark_pos_2)
        attention2 = self.get_attention_data(obs2)
        
        if attention1 is None or attention2 is None:
            return None
            
        # Calculate how much attention changed
        attention_change = np.abs(attention1 - attention2)
        avg_change = np.mean(attention_change)
        max_change = np.max(attention_change)
        
        print(f"Average attention change: {avg_change:.3f}")
        print(f"Maximum attention change: {max_change:.3f}")
        print(f"Good if avg > 0.1 (sensitive to environment), Bad if avg < 0.05 (ignores environment)")
        
        self.visualize_sensitivity(attention1, attention2, attention_change)
        return {'avg_change': avg_change, 'max_change': max_change}
        
    def test_task_relevance_simple_spread(self):
        """Test 4: In simple_spread, do agents coordinate to cover different landmarks?"""
        print("\n=== Test 4: Task Relevance (Simple Spread) ===")
        
        # Set up simple spread scenario - agents should spread to different landmarks
        agent_pos = np.array([[0, 0], [0.1, 0.1], [-0.1, 0.1], [0.1, -0.1]])  # Start clustered
        landmark_pos = np.array([[-0.8, -0.8], [0.8, -0.8], [0.8, 0.8], [-0.8, 0.8]])  # Spread out
        
        obs, _, _ = self.setup_scenario(agent_pos, landmark_pos)
        attention = self.get_attention_data(obs)
        
        if attention is None:
            return None
        
        # Analyze if agents are coordinating (attending to different landmarks)
        # For each landmark, which agents attend to it most?
        landmark_attention = np.zeros((4, 4))  # landmark x agent
        
        for agent_idx in range(4):
            agent_attention = attention[agent_idx]  # (4,4) matrix for this agent
            
            # Sum across the agent dimension to get attention to each landmark
            if len(agent_attention.shape) == 2:
                # Try different interpretations based on attention matrix structure
                landmark_attn_option1 = np.sum(agent_attention, axis=1)  # Sum across columns
                landmark_attn_option2 = np.sum(agent_attention, axis=0)  # Sum across rows
                landmark_attn_option3 = np.diag(agent_attention)  # Take diagonal
                
                # Use the option with most variation (most informative)
                options = [landmark_attn_option1, landmark_attn_option2, landmark_attn_option3]
                variances = [np.var(opt) for opt in options]
                best_option = options[np.argmax(variances)]
                
                landmark_attention[:, agent_idx] = best_option
        
        # Check coordination: ideally each landmark should have a different primary agent
        primary_agents = np.argmax(landmark_attention, axis=1)  # Which agent attends most to each landmark
        unique_assignments = len(set(primary_agents))
        
        coordination_score = unique_assignments / 4.0  # Perfect = 1.0 (all landmarks have different primary agents)
        
        print(f"Landmark primary agents: {primary_agents}")
        print(f"Unique assignments: {unique_assignments}/4")
        print(f"Coordination score: {coordination_score:.3f}")
        print(f"Good if > 0.75 (good coordination), Bad if < 0.5 (poor coordination)")
        
        self.visualize_task_relevance(agent_pos, landmark_pos, landmark_attention, primary_agents)
        
        return {
            'primary_agents': primary_agents, 
            'coordination_score': coordination_score,
            'landmark_attention': landmark_attention
        }
        
    def visualize_consistency(self, attention_matrices, consistency_scores):
        """Visualize attention consistency across trials - individual agent heatmaps"""
        n_trials = min(3, len(attention_matrices))
        fig, axes = plt.subplots(n_trials, 4, figsize=(16, 4*n_trials))
        
        if n_trials == 1:
            axes = axes.reshape(1, -1)
        
        for trial in range(n_trials):
            attention = attention_matrices[trial]
            for agent_idx in range(4):
                ax = axes[trial, agent_idx]
                
                # Each agent gets their own heatmap
                agent_attention = attention[agent_idx]  # (4,4) matrix for this agent
                
                im = ax.imshow(agent_attention, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
                ax.set_title(f'Trial {trial+1}: Agent {agent_idx}')
                
                # Add value annotations
                for i in range(agent_attention.shape[0]):
                    for j in range(agent_attention.shape[1]):
                        ax.text(j, i, f'{agent_attention[i,j]:.2f}', 
                               ha='center', va='center', fontsize=8)
                
                ax.set_xlabel('Attention Targets')
                ax.set_ylabel('Attention Sources')
                
                if trial == 0:  # Add colorbar to top row
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.suptitle('Individual Agent Attention Matrices Across Trials', fontsize=16)
        plt.tight_layout()
        plt.savefig('attention_consistency_test.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def visualize_diversity(self, attention, diversity_scores):
        """Visualize attention diversity between agents - individual heatmaps"""
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        # Top row: Individual agent attention heatmaps
        for agent_idx in range(4):
            ax = axes[0, agent_idx]
            agent_attention = attention[agent_idx]  # (4,4) matrix for this agent
            
            im = ax.imshow(agent_attention, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
            ax.set_title(f'Agent {agent_idx} Attention Matrix')
            
            # Add value annotations
            for i in range(agent_attention.shape[0]):
                for j in range(agent_attention.shape[1]):
                    ax.text(j, i, f'{agent_attention[i,j]:.2f}', 
                           ha='center', va='center', fontsize=8)
            
            ax.set_xlabel('Attention Targets')
            ax.set_ylabel('Attention Sources')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Bottom row: Diversity analysis
        # Plot diversity scores
        if diversity_scores:
            ax = axes[1, 0]
            pair_labels = [f'A{i}-A{j}' for i in range(4) for j in range(i+1, 4)]
            bars = ax.bar(range(len(diversity_scores)), diversity_scores, color='orange', alpha=0.7)
            ax.set_xlabel('Agent Pairs')
            ax.set_ylabel('Diversity Score')
            ax.set_title('Attention Diversity Between Agent Pairs')
            ax.set_xticks(range(len(diversity_scores)))
            ax.set_xticklabels(pair_labels, rotation=45)
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, val in zip(bars, diversity_scores):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom')
        
        # Agent attention pattern comparison (line plots)
        ax = axes[1, 1]
        colors = ['red', 'blue', 'green', 'orange']
        for agent_idx in range(4):
            agent_attention = attention[agent_idx].flatten()
            ax.plot(agent_attention, color=colors[agent_idx], alpha=0.8, 
                   label=f'Agent {agent_idx}', linewidth=2)
        
        ax.set_xlabel('Attention Weight Index')
        ax.set_ylabel('Attention Value')
        ax.set_title('Agent Attention Patterns Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Clear unused subplots
        for idx in [2, 3]:
            axes[1, idx].axis('off')
        
        plt.tight_layout()
        plt.savefig('attention_diversity_test.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def visualize_sensitivity(self, attention1, attention2, attention_change):
        """Visualize attention sensitivity to environment changes - individual agent heatmaps"""
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        
        # Row 1: Scenario 1 attention matrices
        for agent_idx in range(4):
            ax = axes[0, agent_idx]
            agent_attention = attention1[agent_idx]
            
            im = ax.imshow(agent_attention, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
            ax.set_title(f'Scenario 1: Agent {agent_idx}')
            
            # Add value annotations
            for i in range(agent_attention.shape[0]):
                for j in range(agent_attention.shape[1]):
                    ax.text(j, i, f'{agent_attention[i,j]:.2f}', 
                           ha='center', va='center', fontsize=8)
            
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Row 2: Scenario 2 attention matrices
        for agent_idx in range(4):
            ax = axes[1, agent_idx]
            agent_attention = attention2[agent_idx]
            
            im = ax.imshow(agent_attention, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
            ax.set_title(f'Scenario 2: Agent {agent_idx}')
            
            # Add value annotations
            for i in range(agent_attention.shape[0]):
                for j in range(agent_attention.shape[1]):
                    ax.text(j, i, f'{agent_attention[i,j]:.2f}', 
                           ha='center', va='center', fontsize=8)
            
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Row 3: Attention change matrices
        for agent_idx in range(4):
            ax = axes[2, agent_idx]
            agent_change = attention_change[agent_idx]
            
            im = ax.imshow(agent_change, cmap='Reds', aspect='auto', vmin=0)
            ax.set_title(f'Change: Agent {agent_idx}')
            
            # Add value annotations
            for i in range(agent_change.shape[0]):
                for j in range(agent_change.shape[1]):
                    ax.text(j, i, f'{agent_change[i,j]:.2f}', 
                           ha='center', va='center', fontsize=8)
            
            ax.set_xlabel('Attention Targets')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Add row labels
        for row, label in enumerate(['Scenario 1 (Close)', 'Scenario 2 (Far)', 'Attention Change']):
            axes[row, 0].set_ylabel(f'{label}\nAttention Sources', fontsize=12, fontweight='bold')
        
        plt.suptitle('Individual Agent Attention Sensitivity to Environment Changes', fontsize=16)
        plt.tight_layout()
        plt.savefig('attention_sensitivity_test.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def visualize_task_relevance(self, agent_pos, landmark_pos, landmark_attention, primary_agents):
        """Visualize task relevance for simple spread - individual agent analysis"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Top row: Individual agent attention heatmaps
        for agent_idx in range(4):
            if agent_idx < 3:  # First 3 agents in top row
                ax = axes[0, agent_idx]
            else:  # 4th agent in bottom left
                ax = axes[1, 0]
                
            # Get this agent's attention pattern
            agent_attention = landmark_attention[:, agent_idx].reshape(4, 1)  # Make it 2D for heatmap
            
            im = ax.imshow(agent_attention, cmap='YlOrRd', aspect='auto')
            ax.set_title(f'Agent {agent_idx} Attention to Landmarks')
            ax.set_ylabel('Landmarks')
            ax.set_xlabel('Agent Focus')
            ax.set_yticks(range(4))
            ax.set_yticklabels([f'L{i}' for i in range(4)])
            ax.set_xticks([0])
            ax.set_xticklabels([f'A{agent_idx}'])
            
            # Add value annotations
            for i in range(4):
                ax.text(0, i, f'{landmark_attention[i, agent_idx]:.2f}', 
                       ha='center', va='center', fontsize=10, fontweight='bold')
            
            plt.colorbar(im, ax=ax, fraction=0.3, pad=0.1)
        
        # Bottom middle: Environment with coordination assignments
        ax = axes[1, 1]
        colors = ['red', 'blue', 'green', 'orange']
        
        # Plot agents
        for i, pos in enumerate(agent_pos):
            ax.scatter(*pos, c=colors[i], s=200, marker='o', alpha=0.8, label=f'Agent {i}')
        
        # Plot landmarks with their primary agent color
        for i, pos in enumerate(landmark_pos):
            primary_agent = primary_agents[i]
            ax.scatter(*pos, c=colors[primary_agent], s=300, marker='s', alpha=0.6, edgecolor='black', linewidth=2)
            ax.annotate(f'L{i}\n(A{primary_agent})', pos, xytext=(5, 5), textcoords='offset points', fontweight='bold')
        
        # Draw assignment lines
        for i, primary_agent in enumerate(primary_agents):
            ax.plot([agent_pos[primary_agent, 0], landmark_pos[i, 0]], 
                    [agent_pos[primary_agent, 1], landmark_pos[i, 1]], 
                    color=colors[primary_agent], linestyle='--', alpha=0.8, linewidth=2)
        
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('Agent-Landmark Coordination\n(Lines show primary assignments)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Bottom right: Overall coordination heatmap
        ax = axes[1, 2]
        im = ax.imshow(landmark_attention, cmap='YlOrRd', aspect='auto')
        ax.set_xlabel('Agents')
        ax.set_ylabel('Landmarks')
        ax.set_title('Full Landmark-Agent\nAttention Matrix')
        ax.set_xticks(range(4))
        ax.set_xticklabels([f'A{i}' for i in range(4)])
        ax.set_yticks(range(4))
        ax.set_yticklabels([f'L{i}' for i in range(4)])
        
        # Add text annotations
        for i in range(4):
            for j in range(4):
                text_color = "white" if landmark_attention[i, j] > 0.5 else "black"
                ax.text(j, i, f'{landmark_attention[i, j]:.2f}',
                       ha="center", va="center", color=text_color, fontweight='bold')
        
        # Highlight primary assignments with boxes
        for i, primary_agent in enumerate(primary_agents):
            rect = plt.Rectangle((primary_agent-0.4, i-0.4), 0.8, 0.8, 
                               fill=False, edgecolor='red', linewidth=3)
            ax.add_patch(rect)
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.suptitle(f'Task Coordination Analysis (Coordination Score: {len(set(primary_agents))}/4)', fontsize=16)
        plt.tight_layout()
        plt.savefig('attention_task_relevance_test.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def run_comprehensive_tests(self):
        """Run all explainability tests"""
        print("=== InfoMARL Explainability Test Suite ===")
        print("Testing attention mechanism for meaningful patterns...\n")
        
        results = {}
        
        # Test 1: Consistency
        results['consistency'] = self.test_attention_consistency()
        
        # Test 2: Diversity  
        results['diversity'] = self.test_attention_diversity()
        
        # Test 3: Sensitivity
        results['sensitivity'] = self.test_attention_sensitivity()
        
        # Test 4: Task Relevance
        results['task_relevance'] = self.test_task_relevance_simple_spread()
        
        # Summary Report
        print("\n" + "="*50)
        print("SUMMARY REPORT")
        print("="*50)
        
        if results['consistency']:
            consistency = results['consistency']['avg_consistency']
            print(f"✓ Consistency: {consistency:.3f} {'GOOD' if consistency > 0.8 else 'NEEDS IMPROVEMENT' if consistency > 0.5 else 'POOR'}")
        
        if results['diversity']:
            diversity = results['diversity']['avg_diversity']
            print(f"✓ Diversity: {diversity:.3f} {'GOOD' if diversity > 0.5 else 'NEEDS IMPROVEMENT' if diversity > 0.2 else 'POOR'}")
        
        if results['sensitivity']:
            sensitivity = results['sensitivity']['avg_change']
            print(f"✓ Sensitivity: {sensitivity:.3f} {'GOOD' if sensitivity > 0.1 else 'NEEDS IMPROVEMENT' if sensitivity > 0.05 else 'POOR'}")
        
        if results['task_relevance']:
            coordination = results['task_relevance']['coordination_score']
            print(f"✓ Task Relevance: {coordination:.3f} {'GOOD' if coordination > 0.75 else 'NEEDS IMPROVEMENT' if coordination > 0.5 else 'POOR'}")
        
        print("\nVisualization files created:")
        print("- attention_consistency_test.png")
        print("- attention_diversity_test.png") 
        print("- attention_sensitivity_test.png")
        print("- attention_task_relevance_test.png")
        
        return results

if __name__ == "__main__":
    import os
    
    model_path = "ppo_infomarl.zip"
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        exit(1)
    
    tester = InfoMARLExplainabilityTester(model_path)
    results = tester.run_comprehensive_tests()