import argparse
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from smaclite_torchrl_wrapper import SMACliteTask
from torchrl.envs import ParallelEnv, step_mdp
from torchrl.envs.utils import ExplorationType, set_exploration_type
from tensordict import TensorDict
import smaclite

# BenchMARL imports
from benchmarl.algorithms import MappoConfig, IppoConfig, QmixConfig, MasacConfig
from benchmarl.models.mlp import MlpConfig
from benchmarl.experiment import Experiment, ExperimentConfig


class BenchMARLModel:
    """Wrapper to load and run BenchMARL policies."""
    
    def __init__(self, checkpoint_path, map_name, algorithm, device="cuda"):
        self.device = device
        self.algorithm = algorithm
        
        # 1. Create task
        task_map = {
            "2s3z": SMACliteTask.TWO_S_THREE_Z,
            "3s5z": SMACliteTask.THREE_S_FIVE_Z,
            "MMM": SMACliteTask.MMM,
            "MMM2": SMACliteTask.MMM2,
            "3m": SMACliteTask.THREE_M,
            "8m": SMACliteTask.EIGHT_M,
            "25m": SMACliteTask.TWENTY_FIVE_M,
        }
        
        if map_name not in task_map:
            raise ValueError(f"Unknown map: {map_name}")
        
        task = task_map[map_name].get_from_yaml()
        
        # 2. Algorithm config
        algo_map = {
            "mappo": MappoConfig.get_from_yaml(),
            "ippo": IppoConfig.get_from_yaml(),
            "qmix": QmixConfig.get_from_yaml(),
            "masac": MasacConfig.get_from_yaml(),
        }
        
        if algorithm not in algo_map:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        algo_config = algo_map[algorithm]
        
        # 3. Model config (match training)
        model_config = MlpConfig.get_from_yaml()
        model_config.num_cells = [256, 256]
        
        critic_config = MlpConfig.get_from_yaml()
        critic_config.num_cells = [256, 256]
        
        # 4. Experiment config with checkpoint restoration
        temp_dir = Path("/tmp/benchmarl_eval_temp")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        exp_config = ExperimentConfig.get_from_yaml()
        # exp_config.restore_file = str(checkpoint_path)
        exp_config.train_device = device
        exp_config.sampling_device = device
        exp_config.loggers = []
        exp_config.save_folder = str(temp_dir)
        
        # 5. Create experiment - automatically loads checkpoint
        print(f"Loading checkpoint from {checkpoint_path}...")
        self.experiment = Experiment(
            task=task,
            algorithm_config=algo_config,
            model_config=model_config,
            critic_model_config=critic_config,
            seed=0,
            config=exp_config,
        )
        state_dict = torch.load(checkpoint_path, map_location=device)
        self.experiment.policy.load_state_dict(state_dict)
        
        self.policy = self.experiment.policy
        self.policy.eval()
        # self.policy = self.experiment.policy
        # self.policy.eval()
        print("✓ Model loaded successfully")
    
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


def evaluate(checkpoint_path, algorithm, map_name, n_episodes=1000, num_envs=1):
    """Evaluate BenchMARL checkpoint on SMACLite.
    

    """
    
    # Device configuration
    if torch.cuda.is_available():
        model_device = "cuda"
        # Use CPU for parallel envs due to CUDA multiprocessing issues
        env_device = "cpu" if num_envs > 1 else "cuda"
    else:
        model_device = "cpu"
        env_device = "cpu"
    
    print(f"\n{'='*60}")
    print(f"EVALUATION: {algorithm.upper()} on {map_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Episodes: {n_episodes} | Parallel Envs: {num_envs}")
    print(f"Model Device: {model_device} | Env Device: {env_device}")
    print(f"{'='*60}\n")
    
    # Load model
    model = BenchMARLModel(checkpoint_path, map_name, algorithm, model_device)
    
    # Create environments
    env_name = f"smaclite/{map_name}-v0"
    
    if num_envs > 1:
        print(f"Creating {num_envs} parallel environments on {env_device}...")
        from smaclite_torchrl_wrapper import SMACliteTorchRLWrapper
        env = ParallelEnv(
            num_workers=num_envs,
            create_env_fn=lambda: SMACliteTorchRLWrapper(
                env_name=env_name,
                device=env_device,
                categorical_actions=True,
            ),
            device=env_device,
        )
    else:
        from smaclite_torchrl_wrapper import SMACliteTorchRLWrapper
        env = SMACliteTorchRLWrapper(
            env_name=env_name, 
            device=env_device, 
            categorical_actions=True
        )
    
    # Evaluation loop
    print(f"\nStarting evaluation...")
    
    wins = np.zeros(num_envs if num_envs > 1 else 1, dtype=int)
    episodes_completed = np.zeros(num_envs if num_envs > 1 else 1, dtype=int)
    episode_returns = []
    
    td = env.reset()
    
    pbar = tqdm(total=n_episodes, desc="Evaluating")
    total_episodes = 0
    
    while total_episodes < n_episodes:
        # Move data to model device if different from env device
        if model_device != env_device:
            td_model = td.to(model_device)
        else:
            td_model = td
        
        # Get actions from policy
        actions = model(td_model)
        
        # Move actions back to env device if needed
        if model_device != env_device:
            actions = actions.to(env_device)
        
        # Update tensordict with actions
        td.set(("agents", "action"), actions)
        
        # Step environment
        td_next = env.step(td)
        
        # Check for episode completion
        dones = td_next["next", "done"]
        rewards = td_next["next", "agents", "reward"]
        
        if num_envs > 1:
            for env_idx in range(num_envs):
                if dones[env_idx].item():
                    # Win heuristic: SMAC gives +20 for win, 0 for loss
                    episode_reward = rewards[env_idx].sum().item()
                    is_win = episode_reward > 10  # Threshold to detect win
                    
                    if is_win:
                        wins[env_idx] += 1
                    
                    episodes_completed[env_idx] += 1
                    episode_returns.append(episode_reward)
                    total_episodes += 1
                    
                    pbar.update(1)
                    
                    current_wr = wins.sum() / episodes_completed.sum()
                    pbar.set_postfix({
                        "WR": f"{current_wr:.1%}",
                        "AvgR": f"{np.mean(episode_returns[-100:]):.1f}"
                    })
                    
                    if total_episodes >= n_episodes:
                        break
        else:
            if dones.item():
                episode_reward = rewards.sum().item()
                is_win = episode_reward > 10
                
                if is_win:
                    wins[0] += 1
                
                episodes_completed[0] += 1
                episode_returns.append(episode_reward)
                total_episodes += 1
                
                pbar.update(1)
                
                current_wr = wins[0] / episodes_completed[0]
                pbar.set_postfix({
                    "WR": f"{current_wr:.1%}",
                    "AvgR": f"{np.mean(episode_returns[-100:]):.1f}"
                })
        
        # Advance to next step
        td = step_mdp(td_next)
        
        if total_episodes >= n_episodes:
            break
    
    pbar.close()
    
    # Final results
    total_wins = int(wins.sum())
    total_completed = int(episodes_completed.sum())
    final_wr = total_wins / total_completed if total_completed > 0 else 0.0
    avg_return = np.mean(episode_returns)
    
    print("\n" + "="*60)
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Algorithm: {algorithm.upper()}")
    print(f"Map: {map_name}")
    print(f"Episodes: {total_completed}")
    print(f"Wins: {total_wins}")
    print(f"Losses: {total_completed - total_wins}")
    print(f"Win Rate: {final_wr:.2%}")
    print(f"Average Return: {avg_return:.2f} ± {np.std(episode_returns):.2f}")
    print(f"{'='*60}")
    
    env.close()
    
    return {
        "win_rate": final_wr,
        "avg_return": avg_return,
        "std_return": np.std(episode_returns),
        "episodes": total_completed,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate BenchMARL SMACLite models")
    parser.add_argument("--checkpoint", type=str, required=True, 
                        help="Path to model checkpoint (.pt file)")
    parser.add_argument("--algo", choices=["mappo", "ippo", "qmix", "masac"], required=True,
                        help="Algorithm used for training")
    parser.add_argument("--map", type=str, default="2s3z", 
                        choices=["2s3z", "3s5z", "MMM", "MMM2", "3m", "8m", "25m"],
                        help="SMACLite map name")
    parser.add_argument("--episodes", type=int, default=100, 
                        help="Number of episodes to evaluate")
    parser.add_argument("--num_envs", type=int, default=1,
                        help="Number of parallel environments (use 1 for fastest CUDA evaluation)")
    
    args = parser.parse_args()
    
    # Warn about multiprocessing
    if args.num_envs > 1:
        print("\n⚠️  WARNING: Using multiple environments requires CPU device for environments.")
        print("    For fastest evaluation with CUDA, use --num_envs 1\n")
    
    evaluate(
        checkpoint_path=args.checkpoint,
        algorithm=args.algo,
        map_name=args.map,
        n_episodes=args.episodes,
        num_envs=args.num_envs,
    )