"""eval_fixed.py - Working evaluation with proper termination handling"""
import argparse
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from smaclite_torchrl_wrapper import SMACliteTask, SMACliteTorchRLWrapper
from torchrl.envs import step_mdp
from torchrl.envs.utils import ExplorationType, set_exploration_type

from benchmarl.algorithms import MappoConfig, IppoConfig, QmixConfig, MasacConfig
from benchmarl.models.mlp import MlpConfig
from benchmarl.experiment import Experiment, ExperimentConfig

import smaclite

class BenchMARLModel:
    """Wrapper to load and run BenchMARL policies."""
    
    def __init__(self, checkpoint_path, map_name, algorithm, device="cuda"):
        self.device = device
        self.algorithm = algorithm
        
        # Create task
        task_map = {
            "2s3z": SMACliteTask.TWO_S_THREE_Z,
            "3s5z": SMACliteTask.THREE_S_FIVE_Z,
            "MMM": SMACliteTask.MMM,
            "MMM2": SMACliteTask.MMM2,
            "3m": SMACliteTask.THREE_M,
            "8m": SMACliteTask.EIGHT_M,
            "25m": SMACliteTask.TWENTY_FIVE_M,
        }
        task = task_map[map_name].get_from_yaml()
        
        # Algorithm config
        algo_map = {
            "mappo": MappoConfig,
            "ippo": IppoConfig,
            "qmix": QmixConfig,
            "masac": MasacConfig,
        }
        algo_config = algo_map[algorithm].get_from_yaml()
        
        # Model configs
        model_config = MlpConfig.get_from_yaml()
        model_config.num_cells = [256, 256]
        critic_config = MlpConfig.get_from_yaml()
        critic_config.num_cells = [256, 256]
        
        # Experiment config with checkpoint
        temp_dir = Path("/tmp/benchmarl_eval_temp")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        exp_config = ExperimentConfig.get_from_yaml()
        exp_config.restore_file = str(checkpoint_path)
        exp_config.train_device = device
        exp_config.sampling_device = device
        exp_config.loggers = []
        exp_config.save_folder = str(temp_dir)
        exp_config.checkpoint_interval = 0
        exp_config.evaluation = False
        exp_config.create_json = False
        
        print(f"Loading checkpoint: {checkpoint_path}")
        self.experiment = Experiment(
            task=task,
            algorithm_config=algo_config,
            model_config=model_config,
            critic_model_config=critic_config,
            seed=0,
            config=exp_config,
        )
        
        self.policy = self.experiment.policy
        self.policy.eval()
        print("✓ Model loaded\n")
    
    def __call__(self, td):
        """Run policy on tensordict."""
        with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
            td = self.policy(td)
            
            # Extract actions
            if "action" in td["agents"].keys():
                actions = td["agents", "action"]
            else:
                actions = td["agents", "loc"]
            
            return actions


def evaluate(checkpoint_path, algorithm, map_name, n_episodes=100, device="cuda"):
    """Evaluate BenchMARL checkpoint - FIXED VERSION."""
    
    print(f"\n{'='*60}")
    print(f"EVALUATION: {algorithm.upper()} on {map_name}")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Episodes: {n_episodes}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # Load model
    model = BenchMARLModel(checkpoint_path, map_name, algorithm, device)
    
    # Create environment
    print("Creating environment...")
    env_name = f"smaclite/{map_name}-v0"
    env = SMACliteTorchRLWrapper(
        env_name=env_name,
        device=device,
        categorical_actions=True,
    )
    print(f"✓ Environment created")
    print(f"  Episode limit: {env.episode_limit}\n")
    
    # Evaluation loop
    print("Starting evaluation...")
    wins = 0
    episode_returns = []
    episode_lengths = []
    
    pbar = tqdm(total=n_episodes, desc="Evaluating")
    
    for ep in range(n_episodes):
        # Reset environment
        td = env.reset()
        episode_reward = 0.0
        step_count = 0
        
        # Run episode
        while True:
            # Get actions from policy
            actions = model(td)
            
            # Set actions in tensordict
            td.set(("agents", "action"), actions)
            
            # Step environment - returns TED format with "next" key
            td_out = env.step(td)
            
            # Check termination from the OUTPUT tensordict (not "next")
            # The wrapper's _step returns done/terminated/truncated at the top level
            done = td_out["done"].item()
            
            # Accumulate reward from output
            if "agents" in td_out.keys() and "reward" in td_out["agents"].keys():
                step_reward = td_out["agents", "reward"].sum().item()
                episode_reward += step_reward
            
            step_count += 1
            
            # Break if episode is done or exceeds limit
            if done or step_count >= env.episode_limit:
                if step_count >= env.episode_limit and not done:
                    # Episode hit limit without terminating - treat as truncation
                    episode_reward += 0  # No win reward
                break
            
            # Advance to next step (reset td with current observations)
            # For next iteration, we need the current state (not "next")
            td = td_out.clone()
            # Remove the "next" key if it exists
            if "next" in td.keys():
                del td["next"]
        
        # Episode statistics
        is_win = episode_reward > 10  # SMAC win gives +20 reward
        if is_win:
            wins += 1
        
        episode_returns.append(episode_reward)
        episode_lengths.append(step_count)
        
        # Update progress bar
        win_rate = wins / (ep + 1)
        pbar.update(1)
        pbar.set_postfix({
            "WR": f"{win_rate:.1%}",
            "AvgR": f"{np.mean(episode_returns):.1f}",
            "AvgLen": f"{np.mean(episode_lengths):.0f}"
        })
    
    pbar.close()
    
    # Final results
    win_rate = wins / n_episodes
    avg_return = np.mean(episode_returns)
    std_return = np.std(episode_returns)
    avg_length = np.mean(episode_lengths)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Algorithm: {algorithm.upper()}")
    print(f"Map: {map_name}")
    print(f"Episodes: {n_episodes}")
    print(f"Wins: {wins}")
    print(f"Losses: {n_episodes - wins}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Average Return: {avg_return:.2f} ± {std_return:.2f}")
    print(f"Average Episode Length: {avg_length:.1f} steps")
    print(f"Min Return: {min(episode_returns):.2f}")
    print(f"Max Return: {max(episode_returns):.2f}")
    print("="*60)
    
    env.close()
    
    return {
        "win_rate": win_rate,
        "avg_return": avg_return,
        "std_return": std_return,
        "avg_length": avg_length,
        "episodes": n_episodes,
        "wins": wins,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate BenchMARL SMACLite models")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--algo", choices=["mappo", "ippo", "qmix", "masac"], required=True)
    parser.add_argument("--map", type=str, default="2s3z",
                        choices=["2s3z", "3s5z", "MMM", "MMM2", "3m", "8m", "25m"])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    
    args = parser.parse_args()
    
    evaluate(
        checkpoint_path=args.checkpoint,
        algorithm=args.algo,
        map_name=args.map,
        n_episodes=args.episodes,
        device=args.device,
    )