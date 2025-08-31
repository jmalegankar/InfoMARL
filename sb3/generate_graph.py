import cbor2
import numpy as np
import torch
import vmas
import wrapper
import json
from typing import Dict, List, Tuple
import os
import matplotlib.pyplot as plt
from collections import defaultdict

class VMASRewardComputer:
    def __init__(self, device="cpu"):
        """VMAS-based reward computer - simplified version"""
        self.device = device
        self.envs = {}  # Cache environments
        
    def get_or_create_env(self, scenario: str, n_agents: int):
        """Get or create VMAS environment"""
        env_key = f"{scenario}_{n_agents}"
        
        if env_key not in self.envs:
            print(f"Creating VMAS environment: {scenario} with {n_agents} agents")
            
            # Map filename scenarios to VMAS scenarios
            scenario_mapping = {
                'spread': 'simple_spread',
                'collection': 'food_collection', 
                'transport': 'transport',
                'sampling': 'sampling'
            }
            
            vmas_scenario = scenario_mapping.get(scenario, scenario)
            
            try:
                if vmas_scenario == 'food_collection':
                    try:
                        import food_collection
                        env = vmas.make_env(
                            scenario=food_collection.Scenario(),
                            n_agents=n_agents,
                            n_food=n_agents,
                            num_envs=1,
                            continuous_actions=True,
                            max_steps=400,
                            seed=42,
                            device=self.device,
                            terminated_truncated=False,
                        )
                    except ImportError:
                        env = vmas.make_env(
                            scenario="food_collection",
                            n_agents=n_agents,
                            n_food=n_agents,
                            num_envs=1,
                            continuous_actions=True,
                            max_steps=400,
                            seed=42,
                            device=self.device,
                            terminated_truncated=False,
                        )
                else:
                    env = vmas.make_env(
                        scenario=vmas_scenario,
                        n_agents=n_agents,
                        num_envs=1,
                        continuous_actions=True,
                        max_steps=400,
                        seed=42,
                        device=self.device,
                        terminated_truncated=False,
                    )
                
                wrapped_env = wrapper.VMASVecEnv(env, rnd_nums=True)
                self.envs[env_key] = wrapped_env
                print(f"Successfully created {vmas_scenario} environment")
                
            except Exception as e:
                print(f"Error creating environment: {e}")
                return None

        return self.envs[env_key]
    
    def get_reward_from_positions(self, wrapped_env, agent_positions: List[List[float]], 
                                landmark_positions: List[List[float]]) -> float:
        """Set positions and get reward from VMAS environment"""
        vmas_env = wrapped_env.env
        
        # Convert to tensors
        agent_pos = torch.tensor(agent_positions, device=self.device, dtype=torch.float32)
        landmark_pos = torch.tensor(landmark_positions, device=self.device, dtype=torch.float32)
        
        # Set positions
        for i, agent in enumerate(vmas_env.agents):
            if i < len(agent_pos):
                pos = agent_pos[i].unsqueeze(0)
                agent.set_pos(pos, batch_index=0)
        
        for i, landmark in enumerate(vmas_env.world.landmarks):
            if i < len(landmark_pos):
                pos = landmark_pos[i].unsqueeze(0)
                landmark.set_pos(pos, batch_index=0)
        
        # Get reward
        reward_tensor = vmas_env.scenario.reward(vmas_env.agents[0])
        
        if hasattr(reward_tensor, 'item'):
            return reward_tensor.item()
        elif isinstance(reward_tensor, torch.Tensor):
            return reward_tensor[0].item() if reward_tensor.numel() > 0 else 0.0
        else:
            return float(reward_tensor)
    
    def process_file(self, filepath: str, max_chunks: int = None) -> Dict:
        """Process CBOR file and return episode rewards for all 64 environments"""
        print(f"Processing {filepath}...")
        
        # Read config
        try:
            with open(filepath, 'rb') as f:
                config = cbor2.load(f)
        except Exception as e:
            print(f"Error reading config: {e}")
            return {}
        
        scenario = config.get('scenario', 'simple_spread')
        n_agents = config.get('n_agents', 4)
        num_envs = config.get('num_envs', 64)  # Should be 64
        
        print(f"  Scenario: {scenario}, Agents: {n_agents}, Environments: {num_envs}")
        
        # Get environment
        env = self.get_or_create_env(scenario, n_agents)
        if env is None:
            return {}
        
        # Store trajectories for each environment separately
        # env_trajectories[env_idx] = [(agent_positions, landmark_positions), ...]
        env_trajectories = [[] for _ in range(num_envs)]
        
        try:
            with open(filepath, 'rb') as f:
                chunk_count = 0
                trajectory_count = 0
                
                while True:
                    if max_chunks and trajectory_count >= max_chunks:
                        break
                        
                    try:
                        chunk = cbor2.load(f)
                        chunk_count += 1
                        
                        # Skip config/metadata chunks
                        if chunk_count <= 2:
                            continue
                        
                        # Extract positions for this timestep across all environments
                        step = chunk.get('step', 0)
                        agent_data = chunk.get('agent_data', {})
                        landmark_data = chunk.get('landmarks', {})
                        
                        # For each environment, extract positions at this timestep
                        for env_idx in range(num_envs):
                            # Agent positions for this environment at this timestep
                            agent_positions = []
                            for agent_name in sorted(agent_data.keys()):
                                trajectory = agent_data[agent_name]
                                if env_idx < len(trajectory):
                                    agent_positions.append(trajectory[env_idx])
                                else:
                                    # Fallback if data is missing
                                    agent_positions.append([0.0, 0.0])
                            
                            # Landmark positions for this environment at this timestep
                            landmark_positions = []
                            for landmark_name in sorted(landmark_data.keys()):
                                trajectory = landmark_data[landmark_name]
                                if env_idx < len(trajectory):
                                    landmark_positions.append(trajectory[env_idx])
                                else:
                                    landmark_positions.append([0.0, 0.0])
                            
                            # Store this timestep for this environment
                            env_trajectories[env_idx].append((agent_positions, landmark_positions))
                        
                        trajectory_count += 1
                        
                        if trajectory_count % 50 == 0:
                            print(f"  Processed {trajectory_count} timesteps...")
                            
                    except EOFError:
                        break
                    except Exception as e:
                        print(f"  Error at chunk {chunk_count}: {e}")
                        break
        
        except Exception as e:
            print(f"Error processing file: {e}")
            return {}
        
        if not env_trajectories[0]:  # Check if we got any data
            return {}
        
        print(f"  Collected {len(env_trajectories[0])} timesteps for {num_envs} environments")
        
        # Compute episode reward for each environment
        episode_rewards = []
        
        for env_idx in range(num_envs):
            if env_idx % 16 == 0:  # Progress update every 16 environments
                print(f"  Computing rewards for environments {env_idx}-{min(env_idx+15, num_envs-1)}")
            
            env_trajectory = env_trajectories[env_idx]
            step_rewards = []
            
            # Compute reward for each timestep in this environment
            for agent_positions, landmark_positions in env_trajectory:
                try:
                    reward = self.get_reward_from_positions(env, agent_positions, landmark_positions)
                    step_rewards.append(reward)
                except Exception as e:
                    print(f"    Error computing reward for env {env_idx}: {e}")
                    step_rewards.append(0.0)  # Default reward on error
            
            # Sum rewards for this episode
            episode_reward = sum(step_rewards)
            episode_rewards.append(episode_reward)
        
        print(f"  Computed {len(episode_rewards)} episode rewards")
        print(f"  Mean episode reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        
        return {
            'scenario': scenario,
            'n_agents': n_agents,
            'num_envs': num_envs,
            'episode_rewards': episode_rewards,  # List of 64 episode rewards
            'episode_length': len(env_trajectories[0]),
            'mean_reward': float(np.mean(episode_rewards)),
            'std_reward': float(np.std(episode_rewards))
        }

def extract_file_info(filename: str) -> Tuple[str, str, int]:
    """Extract method, scenario, agent count from filename"""
    basename = os.path.basename(filename).replace('.dat', '')
    parts = basename.split('_')
    
    if len(parts) >= 3:
        return parts[0], parts[1], int(parts[2])
    return 'unknown', 'unknown', 0

def process_all_files_and_visualize(file_pattern: str, max_chunks_per_file: int = None):
    """Process all files, create JSON summary, and generate visualization"""
    import glob
    
    files = glob.glob(file_pattern)
    files = [f for f in files if not f.endswith('_results.dat')]
    
    print(f"Found {len(files)} files to process")
    
    reward_computer = VMASRewardComputer()
    
    # Simple structure: {method: {scenario: {n_agents: [rewards]}}}
    all_results = {}
    
    for filepath in sorted(files):
        method, scenario, n_agents = extract_file_info(filepath)
        
        print(f"\nProcessing: {method} - {scenario} - {n_agents} agents")
        
        result = reward_computer.process_file(filepath, max_chunks=max_chunks_per_file)
        
        if result and 'episode_rewards' in result:
            # Initialize nested structure
            if method not in all_results:
                all_results[method] = {}
            if scenario not in all_results[method]:
                all_results[method][scenario] = {}
            if n_agents not in all_results[method][scenario]:
                all_results[method][scenario][n_agents] = []
            
            # Add all episode rewards
            all_results[method][scenario][n_agents].extend(result['episode_rewards'])
            print(f"  Added {len(result['episode_rewards'])} episodes")
    
    # Convert to final format with statistics
    final_results = {}
    
    for method, method_data in all_results.items():
        final_results[method] = {}
        for scenario, scenario_data in method_data.items():
            final_results[method][scenario] = {}
            for n_agents, rewards_list in scenario_data.items():
                if rewards_list:
                    rewards_array = np.array(rewards_list)
                    final_results[method][scenario][n_agents] = {
                        'mean': float(np.mean(rewards_array)),
                        'std': float(np.std(rewards_array)),
                        'n_episodes': len(rewards_array),
                        'min': float(np.min(rewards_array)),
                        'max': float(np.max(rewards_array))
                    }
                    print(f"  {method}-{scenario}-{n_agents}: {len(rewards_array)} episodes, mean={np.mean(rewards_array):.1f}±{np.std(rewards_array):.1f}")
    
    # Save JSON
    with open('marl_results_summary.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f"\nResults saved to: marl_results_summary.json")
    
    # Create visualization
    create_paper_style_visualization(final_results)

def create_paper_style_visualization(results_data):
    """Create visualization with 3 methods and error bars"""
    
    if not results_data:
        print("No results to visualize")
        return
    
    # Method name mapping for display
    method_display_names = {
        'infomarl': 'Ours (InfoMARL)',
        'phmarl': 'phMARL', 
        'gsa': 'GSA'
    }
    
    colors = {
        'Ours (InfoMARL)': '#1f77b4',   # Blue
        'phMARL': '#ff7f0e',            # Orange  
        'GSA': '#d62728'                # Red
    }
    
    # Get all scenarios and agent counts
    all_scenarios = set()
    all_agent_counts = set()
    
    for method_data in results_data.values():
        for scenario, scenario_data in method_data.items():
            all_scenarios.add(scenario)
            all_agent_counts.update(scenario_data.keys())
    
    scenarios = sorted(all_scenarios)
    agent_counts = sorted(all_agent_counts)
    methods = ['infomarl', 'phmarl', 'gsa']  # Use file method names
    
    print(f"Visualizing scenarios: {scenarios}")
    print(f"Agent counts: {agent_counts}")
    print(f"Available methods: {list(results_data.keys())}")
    
    # Create subplots
    n_scenarios = len(scenarios)
    if n_scenarios == 1:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        axes = [ax]
    else:
        fig, axes = plt.subplots(1, n_scenarios, figsize=(6 * n_scenarios, 5))
    
    for idx, scenario in enumerate(scenarios):
        ax = axes[idx]
        
        bar_width = 0.25
        x = np.array(range(len(agent_counts)))
        
        for i, method in enumerate(methods):
            means = []
            stds = []
            
            for n_agents in agent_counts:
                if (method in results_data and 
                    scenario in results_data[method] and 
                    n_agents in results_data[method][scenario]):
                    
                    data = results_data[method][scenario][n_agents]
                    means.append(data['mean'])
                    stds.append(data['std'])
                else:
                    means.append(0)
                    stds.append(0)
            
            # Get display name and color
            display_name = method_display_names.get(method, method)
            color = colors.get(display_name, 'gray')
            
            # Calculate x positions
            x_pos = x + i * bar_width - bar_width
            
            # Plot bars with error bars
            bars = ax.bar(x_pos, means, bar_width,
                         yerr=stds,
                         capsize=3,
                         label=display_name, 
                         color=color,
                         alpha=0.8, 
                         edgecolor='black', 
                         linewidth=0.7)
        
        # Customize subplot
        ax.set_xlabel('number of robots')
        ax.set_ylabel('R̄')
        ax.set_title(f'({chr(97+idx)}) {scenario.replace("_", " ").title()}')
        ax.set_xticks(x)
        ax.set_xticklabels(agent_counts)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Legend on first subplot only
        if idx == 0:
            ax.legend(loc='upper right')
        
        # Clean styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('marl_performance_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Visualization saved as: marl_performance_comparison.png")
    plt.show()

def main():
    """Main function"""
    print("VMAS Reward Computer - Simplified")
    print("=" * 50)
    print("Processing trajectory files and creating paper-style visualization...")
    
    # Process all files
    process_all_files_and_visualize(
        file_pattern="/Users/jmalegaonkar/Desktop/InfoMARL-1/eval_data/*.dat",
        max_chunks_per_file=400  # Adjust as needed
    )

if __name__ == "__main__":
    main()