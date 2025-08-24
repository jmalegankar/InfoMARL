import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy, spearmanr, kendalltau
import seaborn as sns
from matplotlib.animation import FFMpegWriter
from tqdm import tqdm
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# Import your existing modules
import vmas
import wrapper
import policy
from stable_baselines3 import PPO

@dataclass
class AttentionAnalysisResults:
    """Store all analysis results"""
    specialization_indices: np.ndarray
    entropies: np.ndarray
    temporal_consistency: Dict
    inter_agent_diversity: np.ndarray
    learning_progression: Dict
    phase_transitions: List[int]
    statistical_tests: Dict

class LandmarkAttentionAnalyzer:
    def __init__(self, n_agents=4, n_landmarks=4, max_steps=200, num_trials=10):
        self.n_agents = n_agents
        self.n_landmarks = n_landmarks
        self.max_steps = max_steps
        self.num_trials = num_trials
        self.attention_history = []
        
    def setup_controlled_environment(self, model_path="ppo_infomarl.zip"):
        """Setup the controlled linear movement environment"""
        self.env = vmas.make_env(
            scenario="simple_spread",
            n_agents=self.n_agents,
            num_envs=1,
            continuous_actions=True,
            max_steps=self.max_steps,
            seed=0,
            device="cpu",
            terminated_truncated=False,
        )
        self.env = wrapper.VMASVecEnv(self.env, rnd_nums=True)
        
        # Load trained model
        self.model = PPO.load(model_path, device="cpu")
        self.model.policy.observation_space = self.env.observation_space
        self.model.policy.action_space = self.env.action_space
        self.model.policy.pi_features_extractor.actor.number_agents = self.n_agents
        self.model.policy.pi_features_extractor.actor.number_food = self.n_landmarks
        
    def collect_attention_data(self, trial_idx=0):
        """Collect attention weights for one trial"""
        obs = self.env.reset()
        
        # Set controlled initial positions
        pos = torch.tensor([0.0, 0.0])
        for idx, agent in enumerate(self.env.env.agents):
            agent.set_pos(pos.clone(), 0)
            
        pos = torch.tensor([-1.0, 1.0])
        for idx, landmark in enumerate(self.env.env.world.landmarks):
            pos[0] = (idx % 2) * 2 - 1.0
            pos[1] = (idx // 2) * 2 - 1.0
            landmark.set_pos(pos.clone(), 0)
            
            
        
        trial_attention = np.zeros((self.max_steps, self.n_agents, self.n_landmarks))
        
        for t in range(self.max_steps):
            action, _ = self.model.predict(obs, deterministic=True)
            
            # Extract attention weights
            actor = self.model.policy.features_extractor.actor
            landmark_weights = actor.landmark_attention_weights
            landmark_weights = landmark_weights.view(1, self.n_agents, 1, self.n_landmarks)
            landmark_weights = landmark_weights[0, :, 0, :].cpu().numpy()
            
            trial_attention[t] = landmark_weights
            obs, _, _, _ = self.env.step(action)
            
        return trial_attention
    
    def calculate_specialization_index(self, attention_weights):
        """Calculate how specialized each agent's attention is"""
        # Shape: (timesteps, agents, landmarks)
        max_attention = np.max(attention_weights, axis=2)
        mean_attention = np.mean(attention_weights, axis=2)
        specialization = max_attention - mean_attention
        return specialization
    
    def calculate_entropy(self, attention_weights):
        """Calculate entropy of attention distribution"""
        eps = 1e-10
        attention_weights = np.clip(attention_weights, eps, 1.0)
        entropies = np.zeros((attention_weights.shape[0], attention_weights.shape[1]))
        
        for t in range(attention_weights.shape[0]):
            for a in range(attention_weights.shape[1]):
                entropies[t, a] = entropy(attention_weights[t, a])
        return entropies
    
    def calculate_temporal_consistency(self, attention_weights, lag=5):
        """Measure how consistent preferences are across time"""
        consistency_scores = []
        
        for agent in range(self.n_agents):
            agent_attention = attention_weights[:, agent, :]
            
            # Calculate autocorrelation for preferred landmark
            preferred_landmarks = np.argmax(agent_attention, axis=1)
            
            # Compute consistency as probability of maintaining preference
            consistency = 0
            for t in range(len(preferred_landmarks) - lag):
                if preferred_landmarks[t] == preferred_landmarks[t + lag]:
                    consistency += 1
            consistency /= (len(preferred_landmarks) - lag)
            consistency_scores.append(consistency)
            
        return np.array(consistency_scores)
    
    def calculate_inter_agent_diversity(self, attention_weights):
        """Measure how different agents' attention patterns are"""
        # Calculate JS divergence between all agent pairs at each timestep
        diversity_scores = []
        
        for t in range(attention_weights.shape[0]):
            js_distances = []
            for i in range(self.n_agents):
                for j in range(i+1, self.n_agents):
                    dist = jensenshannon(attention_weights[t, i], attention_weights[t, j])
                    js_distances.append(dist)
            diversity_scores.append(np.mean(js_distances))
            
        return np.array(diversity_scores)
    
    def mann_kendall_trend_test(self, data):
        """Perform Mann-Kendall trend test for monotonic trend"""
        n = len(data)
        s = 0
        for i in range(n-1):
            for j in range(i+1, n):
                s += np.sign(data[j] - data[i])
        
        # Calculate variance
        var_s = n * (n - 1) * (2 * n + 5) / 18
        
        # Calculate z-score
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0
            
        # Calculate p-value
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        return {'statistic': s, 'z_score': z, 'p_value': p_value, 'trend': np.sign(s)}
    
    def detect_phase_transitions(self, attention_weights):
        """Detect when agents transition from exploration to exploitation"""
        specialization = self.calculate_specialization_index(attention_weights)
        
        # Use rolling variance to detect stability changes
        window_size = 10
        phase_transitions = []
        
        for agent in range(self.n_agents):
            agent_spec = specialization[:, agent]
            
            # Calculate rolling variance
            rolling_var = pd.Series(agent_spec).rolling(window_size).var().values
            
            # Find point of maximum variance reduction
            if len(rolling_var) > window_size * 2:
                var_diff = np.diff(rolling_var[window_size:])
                if len(var_diff) > 0:
                    transition_point = np.argmin(var_diff) + window_size
                    phase_transitions.append(transition_point)
                    
        return phase_transitions
    
    def run_statistical_tests(self, attention_data):
        """Run comprehensive statistical tests"""
        results = {}
        
        # 1. Test for significant specialization (vs uniform attention)
        specialization = self.calculate_specialization_index(attention_data)
        uniform_expected = 0  # Under uniform attention, max - mean = 0
        
        # One-sample t-test for each agent
        for agent in range(self.n_agents):
            t_stat, p_val = stats.ttest_1samp(specialization[:, agent], uniform_expected)
            results[f'agent_{agent}_specialization_test'] = {
                't_statistic': t_stat,
                'p_value': p_val,
                'mean_specialization': np.mean(specialization[:, agent]),
                'effect_size': np.mean(specialization[:, agent]) / np.std(specialization[:, agent])
            }
        
        # 2. Test for learning progression
        mean_specialization = np.mean(specialization, axis=1)
        mk_test = self.mann_kendall_trend_test(mean_specialization)
        results['learning_trend'] = mk_test
        
        # 3. Test for entropy reduction over time
        entropies = self.calculate_entropy(attention_data)
        mean_entropy = np.mean(entropies, axis=1)
        
        # Split into early and late phases
        midpoint = len(mean_entropy) // 2
        early_entropy = mean_entropy[:midpoint]
        late_entropy = mean_entropy[midpoint:]
        
        t_stat, p_val = stats.ttest_ind(early_entropy, late_entropy)
        results['entropy_reduction_test'] = {
            't_statistic': t_stat,
            'p_value': p_val,
            'early_mean': np.mean(early_entropy),
            'late_mean': np.mean(late_entropy),
            'reduction_percentage': (np.mean(early_entropy) - np.mean(late_entropy)) / np.mean(early_entropy) * 100
        }
        
        # 4. Bootstrap confidence intervals for specialization
        n_bootstrap = 1000
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            sample_idx = np.random.choice(len(mean_specialization), len(mean_specialization), replace=True)
            bootstrap_means.append(np.mean(mean_specialization[sample_idx]))
        
        results['bootstrap_ci'] = {
            'mean': np.mean(bootstrap_means),
            'ci_lower': np.percentile(bootstrap_means, 2.5),
            'ci_upper': np.percentile(bootstrap_means, 97.5)
        }
        
        return results
    
    def visualize_results(self, attention_data, results):
        """Create comprehensive visualization of results"""
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        
        # 1. Attention heatmap over time
        mean_attention = np.mean(attention_data, axis=0)  # Average over time
        sns.heatmap(mean_attention, ax=axes[0, 0], cmap='YlOrRd', 
                   xticklabels=[f'L{i}' for i in range(self.n_landmarks)],
                   yticklabels=[f'A{i}' for i in range(self.n_agents)])
        axes[0, 0].set_title('Mean Attention Weights')
        
        # 2. Specialization over time
        specialization = self.calculate_specialization_index(attention_data)
        for agent in range(self.n_agents):
            axes[0, 1].plot(specialization[:, agent], label=f'Agent {agent}', alpha=0.7)
        axes[0, 1].set_xlabel('Timestep')
        axes[0, 1].set_ylabel('Specialization Index')
        axes[0, 1].set_title('Specialization Evolution')
        axes[0, 1].legend()
        
        # 3. Entropy evolution
        entropies = self.calculate_entropy(attention_data)
        mean_entropy = np.mean(entropies, axis=1)
        axes[0, 2].plot(mean_entropy, color='blue', linewidth=2)
        axes[0, 2].fill_between(range(len(mean_entropy)), 
                               mean_entropy - np.std(entropies, axis=1),
                               mean_entropy + np.std(entropies, axis=1),
                               alpha=0.3)
        axes[0, 2].set_xlabel('Timestep')
        axes[0, 2].set_ylabel('Entropy')
        axes[0, 2].set_title('Attention Entropy Over Time')
        
        # 4. Inter-agent diversity
        diversity = self.calculate_inter_agent_diversity(attention_data)
        axes[1, 0].plot(diversity, color='green', linewidth=2)
        axes[1, 0].set_xlabel('Timestep')
        axes[1, 0].set_ylabel('JS Divergence')
        axes[1, 0].set_title('Inter-Agent Attention Diversity')
        
        # 5. Preferred landmark distribution
        preferred_landmarks = np.argmax(attention_data, axis=2)
        landmark_counts = np.zeros((self.n_agents, self.n_landmarks))
        for agent in range(self.n_agents):
            for landmark in range(self.n_landmarks):
                landmark_counts[agent, landmark] = np.sum(preferred_landmarks[:, agent] == landmark)
        
        landmark_counts = landmark_counts / self.max_steps * 100
        sns.heatmap(landmark_counts, ax=axes[1, 1], cmap='Blues', annot=True, fmt='.1f',
                   xticklabels=[f'L{i}' for i in range(self.n_landmarks)],
                   yticklabels=[f'A{i}' for i in range(self.n_agents)])
        axes[1, 1].set_title('Landmark Preference % Over Episode')
        
        # 6. Statistical test results
        test_results_text = "Statistical Test Results:\n\n"
        
        # Specialization tests
        for agent in range(self.n_agents):
            agent_test = results.statistical_tests[f'agent_{agent}_specialization_test']
            test_results_text += f"Agent {agent}: p={agent_test['p_value']:.4f}, "
            test_results_text += f"ES={agent_test['effect_size']:.2f}\n"
        
        # Learning trend
        trend = results.statistical_tests['learning_trend']
        test_results_text += f"\nLearning Trend: p={trend['p_value']:.4f}, "
        test_results_text += f"direction={'increasing' if trend['trend'] > 0 else 'decreasing'}\n"
        
        # Entropy reduction
        entropy_test = results.statistical_tests['entropy_reduction_test']
        test_results_text += f"\nEntropy Reduction: {entropy_test['reduction_percentage']:.1f}%, "
        test_results_text += f"p={entropy_test['p_value']:.4f}"
        
        axes[1, 2].text(0.1, 0.5, test_results_text, fontsize=10, 
                       verticalalignment='center', family='monospace')
        axes[1, 2].axis('off')
        axes[1, 2].set_title('Statistical Significance')
        
        # 7. Temporal consistency
        consistency = results.temporal_consistency
        axes[2, 0].bar(range(self.n_agents), consistency)
        axes[2, 0].set_xlabel('Agent')
        axes[2, 0].set_ylabel('Consistency Score')
        axes[2, 0].set_title('Temporal Consistency of Preferences')
        axes[2, 0].set_xticks(range(self.n_agents))
        axes[2, 0].set_xticklabels([f'A{i}' for i in range(self.n_agents)])
        
        # 8. Phase transitions
        if results.phase_transitions:
            axes[2, 1].hist(results.phase_transitions, bins=20, color='purple', alpha=0.7)
            axes[2, 1].axvline(np.mean(results.phase_transitions), color='red', 
                              linestyle='--', label=f'Mean: {np.mean(results.phase_transitions):.1f}')
            axes[2, 1].set_xlabel('Timestep')
            axes[2, 1].set_ylabel('Frequency')
            axes[2, 1].set_title('Phase Transition Points')
            axes[2, 1].legend()
        
        # 9. Bootstrap confidence intervals
        ci = results.statistical_tests['bootstrap_ci']
        axes[2, 2].barh([0], [ci['mean']], xerr=[[ci['mean']-ci['ci_lower']], 
                                                  [ci['ci_upper']-ci['mean']]], 
                       color='teal', alpha=0.7)
        axes[2, 2].set_yticks([0])
        axes[2, 2].set_yticklabels(['Specialization'])
        axes[2, 2].set_xlabel('Mean Value')
        axes[2, 2].set_title(f"95% CI: [{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}]")
        
        plt.tight_layout()
        return fig
    
    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        print("Setting up controlled environment...")
        self.setup_controlled_environment()
        
        print(f"Running {self.num_trials} trials...")
        all_attention_data = []
        
        for trial in tqdm(range(self.num_trials)):
            trial_data = self.collect_attention_data(trial)
            all_attention_data.append(trial_data)
        
        # Aggregate data across trials
        all_attention_data = np.stack(all_attention_data)
        mean_attention_data = np.mean(all_attention_data, axis=0)
        
        print("Analyzing attention patterns...")
        
        # Calculate all metrics
        results = AttentionAnalysisResults(
            specialization_indices=self.calculate_specialization_index(mean_attention_data),
            entropies=self.calculate_entropy(mean_attention_data),
            temporal_consistency=self.calculate_temporal_consistency(mean_attention_data),
            inter_agent_diversity=self.calculate_inter_agent_diversity(mean_attention_data),
            learning_progression=self.mann_kendall_trend_test(
                np.mean(self.calculate_specialization_index(mean_attention_data), axis=1)
            ),
            phase_transitions=self.detect_phase_transitions(mean_attention_data),
            statistical_tests=self.run_statistical_tests(mean_attention_data)
        )
        
        # Print summary
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)
        
        print("\n1. SPECIALIZATION SIGNIFICANCE:")
        for agent in range(self.n_agents):
            test = results.statistical_tests[f'agent_{agent}_specialization_test']
            sig = "***" if test['p_value'] < 0.001 else "**" if test['p_value'] < 0.01 else "*" if test['p_value'] < 0.05 else "ns"
            print(f"   Agent {agent}: ES={test['effect_size']:.2f}, p={test['p_value']:.4f} {sig}")
        
        print("\n2. LEARNING PROGRESSION:")
        trend = results.statistical_tests['learning_trend']
        print(f"   Trend: {'Increasing' if trend['trend'] > 0 else 'Decreasing'}")
        print(f"   Significance: p={trend['p_value']:.4f}")
        
        print("\n3. ENTROPY REDUCTION:")
        entropy_test = results.statistical_tests['entropy_reduction_test']
        print(f"   Early phase: {entropy_test['early_mean']:.3f}")
        print(f"   Late phase: {entropy_test['late_mean']:.3f}")
        print(f"   Reduction: {entropy_test['reduction_percentage']:.1f}%")
        print(f"   Significance: p={entropy_test['p_value']:.4f}")
        
        print("\n4. TEMPORAL CONSISTENCY:")
        print(f"   Mean consistency: {np.mean(results.temporal_consistency):.3f}")
        print(f"   Range: [{np.min(results.temporal_consistency):.3f}, {np.max(results.temporal_consistency):.3f}]")
        
        print("\n5. PHASE TRANSITIONS:")
        if results.phase_transitions:
            print(f"   Mean transition point: {np.mean(results.phase_transitions):.1f} steps")
            print(f"   Std deviation: {np.std(results.phase_transitions):.1f} steps")
        
        # Create visualization
        print("\nGenerating visualizations...")
        fig = self.visualize_results(mean_attention_data, results)
        plt.savefig('landmark_attention_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return results, mean_attention_data

if __name__ == "__main__":
    # Run the complete analysis
    analyzer = LandmarkAttentionAnalyzer(
        n_agents=4,
        n_landmarks=4,
        max_steps=400,
        num_trials=50
    )
    
    results, attention_data = analyzer.run_full_analysis()
    
    print("\n" + "="*60)
    print("Analysis complete! Results saved to 'landmark_attention_analysis.png'")
    print("="*60)