import argparse
import torch
import gc
import traceback
import os
import heapq
import numpy as np
from dataclasses import dataclass
from enum import Enum
from tensordict import TensorDict

from benchmarl.algorithms import MappoConfig, MasacConfig, QmixConfig, IppoConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig, Callback
from benchmarl.models.mlp import MlpConfig

# ============================================================================
# CALLBACK: ROBUST SAVE TOP-K (Safe Counter Version)
# ============================================================================

class SaveTopK(Callback):
    def __init__(self, k=3, save_dir="best_models"):
        super().__init__()
        self.k = k
        self.save_dir = save_dir
        self.top_k_heap = [] 
        self.full_save_dir = None
        self.eval_counter = 0
        
    def on_setup(self):
        self.full_save_dir = os.path.join(self.experiment.folder_name, self.save_dir)
        os.makedirs(self.full_save_dir, exist_ok=True)
        print(f"   [SaveTopK] Tracking Top {self.k} models in: {self.full_save_dir}")

    def on_evaluation_end(self, rollouts: list):
        if not rollouts: return
        self.eval_counter += 1
        
        stacked = torch.stack(rollouts)
        
        # 1. Robust Reward Retrieval
        try:
            rewards = stacked[("next", "agents", "reward")]
        except KeyError:
            try:
                rewards = stacked[("next", "reward")]
            except KeyError:
                return 

        mean_reward = rewards.mean().item()

        # 2. Heap Logic
        frames = getattr(self.experiment, "n_frames_collected", self.eval_counter)
        
        filename = f"ckpt_{frames}_reward_{mean_reward:.4f}.pt"
        file_path = os.path.join(self.full_save_dir, filename)

        if len(self.top_k_heap) < self.k:
            heapq.heappush(self.top_k_heap, (mean_reward, file_path))
            self._save(file_path)
            print(f"   [SaveTopK] ⭐ New Best: {mean_reward:.4f}")
        else:
            worst_reward, worst_path = self.top_k_heap[0]
            if mean_reward > worst_reward:
                heapq.heappop(self.top_k_heap)
                if os.path.exists(worst_path): os.remove(worst_path)
                
                heapq.heappush(self.top_k_heap, (mean_reward, file_path))
                self._save(file_path)
                print(f"   [SaveTopK] 🏆 Top 3 Updated: {mean_reward:.4f} (Beat {worst_reward:.4f})")

    def _save(self, path):
        torch.save(self.experiment.policy.state_dict(), path)


# ============================================================================
# CALLBACK: ENHANCED EARLY STOPPING (With Performance + Loss + Entropy)
# ============================================================================
class RobustMARLStopping(Callback):
    def __init__(self, 
                 window_size=40, 
                 relative_delta=0.005, 
                 entropy_floor=0.01, 
                 entropy_patience=5,
                 plateau_patience=3,
                 reward_patience=10,
                 reward_threshold=0.002):
        super().__init__()
        self.window_size = window_size
        self.relative_delta = relative_delta
        self.entropy_floor = entropy_floor
        self.entropy_patience = entropy_patience
        self.plateau_patience = plateau_patience
        self.reward_patience = reward_patience
        self.reward_threshold = reward_threshold
        
        # Counters
        self.entropy_hits = 0
        self.plateau_hits = 0
        self.reward_plateau_hits = 0
        
        # History
        self.loss_history = []
        self.reward_history = []
        self.warmup_steps = 10  # Don't check until we have some history

    def on_train_end(self, training_td: TensorDict, group: str):
        keys = training_td.keys()
        
        # --- 1. Loss Plateau Check (WITH PATIENCE) ---
        current_loss = None
        if "loss_objective" in keys:      # MAPPO / IPPO
            current_loss = abs(training_td["loss_objective"].mean().item())
        elif "loss_qvalue" in keys:       # MASAC
            current_loss = abs(training_td["loss_qvalue"].mean().item())
        elif "loss" in keys:              # QMIX
            current_loss = abs(training_td["loss"].mean().item())

        if current_loss is not None and not (np.isnan(current_loss) or np.isinf(current_loss)):
            self.loss_history.append(current_loss)
            
            if len(self.loss_history) > self.window_size:
                past_avg = np.mean(self.loss_history[-self.window_size:-1])
                change = abs(current_loss - past_avg) / (past_avg + 1e-6)

                if change < self.relative_delta:
                    self.plateau_hits += 1
                else:
                    self.plateau_hits = 0  # Reset streak

                if self.plateau_hits >= self.plateau_patience:
                    print(f"   [STOP] Loss Plateau Detected ({self.plateau_hits} consecutive, Δ: {change:.4%})")
                    raise StopIteration("Loss stabilized.")
                
                self.loss_history.pop(0)

        # --- 2. Entropy Streak Check (Skip MASAC, require warmup) ---
        is_masac = "loss_alpha" in keys
        if not is_masac and "entropy" in keys and len(self.loss_history) > self.warmup_steps:
            ent = training_td["entropy"].mean().item()
            
            if ent < self.entropy_floor:
                self.entropy_hits += 1
            else:
                self.entropy_hits = 0  # Reset streak
            
            if self.entropy_hits >= self.entropy_patience:
                print(f"   [STOP] Policy Converged (Entropy < {self.entropy_floor} for {self.entropy_patience} consecutive steps)")
                raise StopIteration("Entropy floor reached.")

    def on_evaluation_end(self, rollouts: list):
        """Track reward plateau for additional convergence signal"""
        if not rollouts or len(self.loss_history) < self.warmup_steps:
            return
            
        stacked = torch.stack(rollouts)
        try:
            rewards = stacked[("next", "agents", "reward")]
        except KeyError:
            try:
                rewards = stacked[("next", "reward")]
            except KeyError:
                return
        
        mean_reward = rewards.mean().item()
        self.reward_history.append(mean_reward)
        
        # Check reward plateau
        if len(self.reward_history) > self.reward_patience:
            recent_avg = np.mean(self.reward_history[-self.reward_patience:])
            older_avg = np.mean(self.reward_history[-2*self.reward_patience:-self.reward_patience])
            
            if abs(recent_avg - older_avg) / (abs(older_avg) + 1e-6) < self.reward_threshold:
                self.reward_plateau_hits += 1
                
                if self.reward_plateau_hits >= 3:
                    print(f"   [STOP] Performance Converged (Reward plateau: {recent_avg:.4f})")
                    raise StopIteration("Reward plateau reached.")
            else:
                self.reward_plateau_hits = 0


# ============================================================================
# CONFIGURATION & RUNNER
# ============================================================================

class AlgorithmType(Enum):
    ON_POLICY = "on_policy"
    OFF_POLICY_CONTINUOUS = "off_policy_continuous"
    OFF_POLICY_DISCRETE = "off_policy_discrete"

@dataclass
class AlgorithmInfo:
    config_class: type
    algo_type: AlgorithmType

ALGORITHMS = {
    "mappo": AlgorithmInfo(MappoConfig, AlgorithmType.ON_POLICY),
    "ippo": AlgorithmInfo(IppoConfig, AlgorithmType.ON_POLICY),
    "masac": AlgorithmInfo(MasacConfig, AlgorithmType.OFF_POLICY_CONTINUOUS),
    "qmix": AlgorithmInfo(QmixConfig, AlgorithmType.OFF_POLICY_DISCRETE),
}

VMAS_TASKS = {
    "simple_spread": VmasTask.SIMPLE_SPREAD,
    "food_collection": VmasTask.FOOD_COLLECTION,
}

def get_exp_config(args, algo_type: AlgorithmType) -> ExperimentConfig:
    config = ExperimentConfig.get_from_yaml()
    config.max_n_frames = args.max_n_frames
    config.train_device = args.device
    config.sampling_device = args.device
    config.loggers = args.loggers
    config.save_folder = args.save_folder
    config.checkpoint_at_end = True

    if args.resume:
        config.restore_file = True
    
    # OPTIMIZED FOR RTX 3070 Ti (8GB)
    if algo_type == AlgorithmType.ON_POLICY:
        n_envs = args.n_envs_on_policy  # 800-1200 recommended
        batch_size = max(args.on_policy_frames_per_batch, n_envs * args.task_max_steps)
        config.on_policy_n_envs_per_worker = n_envs
        config.on_policy_collected_frames_per_batch = batch_size
    else:
        n_envs = args.n_envs_off_policy  # 64-96 recommended
        batch_size = n_envs * args.task_max_steps
        config.off_policy_n_envs_per_worker = n_envs
        config.off_policy_collected_frames_per_batch = batch_size
        config.off_policy_train_batch_size = args.train_batch_size
        config.off_policy_memory_size = args.replay_buffer_size
        
    config.evaluation_interval = batch_size * 5

    return config

def run_benchmark(args):
    for task_name in args.tasks:
        for algo_name in args.algorithms:
            info = ALGORITHMS[algo_name]
            
            exp_config = get_exp_config(args, info.algo_type)
            algo_cfg = info.config_class.get_from_yaml()
            
            task = VMAS_TASKS[task_name].get_from_yaml()
            task.config["n_agents"] = args.n_agents
            task.config["max_steps"] = args.task_max_steps 

            model_cfg = MlpConfig.get_from_yaml()
            model_cfg.num_cells = [args.hidden_dim, args.hidden_dim]

            for seed in args.seeds:
                print(f"\n{'='*70}")
                print(f">>> RUNNING: {algo_name.upper()} on {task_name} | Seed {seed}")
                print(f"    Envs: {exp_config.on_policy_n_envs_per_worker if info.algo_type == AlgorithmType.ON_POLICY else exp_config.off_policy_n_envs_per_worker}")
                print(f"{'='*70}\n")
                
                stopper = RobustMARLStopping(
                    window_size=args.plateau_window,
                    relative_delta=args.plateau_threshold,
                    entropy_floor=args.entropy_stop,
                    entropy_patience=args.entropy_patience,
                    plateau_patience=args.plateau_patience,
                    reward_patience=args.reward_patience,
                    reward_threshold=args.reward_threshold
                )
                saver = SaveTopK(k=3, save_dir="best_models")

                try:
                    gc.collect()
                    if torch.cuda.is_available(): 
                        torch.cuda.empty_cache()

                    experiment = Experiment(
                        task=task,
                        algorithm_config=algo_cfg,
                        model_config=model_cfg,
                        seed=seed,
                        config=exp_config,
                        callbacks=[stopper, saver]
                    )
                    experiment.run()
                    
                except StopIteration as e:
                    print(f"\n{'='*70}")
                    print(f"✅ Experiment Finished Early: {e}")
                    print(f"{'='*70}\n")
                    
                except Exception as e:
                    print(f"\n{'='*70}")
                    print(f"❌ CRITICAL ERROR: {e}")
                    traceback.print_exc()
                    print(f"{'='*70}\n")
                    if args.fail_fast: return
                    
                finally:
                    # Aggressive cleanup
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

def parse_args():
    parser = argparse.ArgumentParser(description="Optimized MARL Benchmarking with Early Stopping")
    
    # Experiment Settings
    parser.add_argument("--algorithms", "-a", nargs="+", default=["mappo", "ippo", "qmix", "masac"])
    parser.add_argument("--tasks", "-t", nargs="+", default=["simple_spread", "food_collection"])
    parser.add_argument("--seeds", "-s", type=int, nargs="+", default=[42])
    parser.add_argument("--n_agents", type=int, default=4)
    parser.add_argument("--task_max_steps", type=int, default=200)
    parser.add_argument("--max_n_frames", type=int, default=2_000_000)
    
    # Parallelism (Optimized for RTX 3070 Ti 8GB)
    parser.add_argument("--n_envs_on_policy", type=int, default=1000, 
                        help="Parallel environments for on-policy (MAPPO/IPPO). Try 800-1200")
    parser.add_argument("--n_envs_off_policy", type=int, default=80,
                        help="Parallel environments for off-policy (QMIX/MASAC). Try 64-96")
    
    # Batch Sizes
    parser.add_argument("--on_policy_frames_per_batch", type=int, default=100_000,
                        help="Frames per batch for on-policy algorithms")
    parser.add_argument("--train_batch_size", type=int, default=512,
                        help="Training batch size for off-policy algorithms")
    parser.add_argument("--replay_buffer_size", type=int, default=100_000)
    
    # Model
    parser.add_argument("--hidden_dim", type=int, default=256)
    
    # Hardware
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # Logging
    parser.add_argument("--loggers", nargs="+", default=["csv", "tensorboard"])
    parser.add_argument("--save_folder", type=str, default="results/marl_benchmark")
    parser.add_argument("--fail_fast", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
    
    # Early Stopping - Loss Plateau
    parser.add_argument("--plateau_window", type=int, default=40,
                        help="Window size for loss plateau detection")
    parser.add_argument("--plateau_threshold", type=float, default=0.005,
                        help="Relative change threshold for loss plateau (0.5%)")
    parser.add_argument("--plateau_patience", type=int, default=3,
                        help="How many consecutive plateau detections before stopping")
    
    # Early Stopping - Entropy
    parser.add_argument("--entropy_stop", type=float, default=0.01,
                        help="Entropy threshold for convergence (higher = earlier stop)")
    parser.add_argument("--entropy_patience", type=int, default=5,
                        help="Consecutive low entropy steps before stopping")
    
    # Early Stopping - Reward Plateau
    parser.add_argument("--reward_patience", type=int, default=10,
                        help="Evaluations to consider for reward plateau")
    parser.add_argument("--reward_threshold", type=float, default=0.002,
                        help="Relative reward change threshold (0.2%)")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    print("\n" + "="*70)
    print("MARL BENCHMARK - OPTIMIZED FOR RTX 3070 Ti")
    print("="*70)
    print(f"Device: {args.device}")
    print(f"On-Policy Envs: {args.n_envs_on_policy}")
    print(f"Off-Policy Envs: {args.n_envs_off_policy}")
    print(f"Early Stopping Enabled:")
    print(f"  - Loss Plateau: {args.plateau_patience} hits @ Δ<{args.plateau_threshold:.2%}")
    print(f"  - Entropy Floor: {args.entropy_patience} hits @ <{args.entropy_stop}")
    print(f"  - Reward Plateau: @ Δ<{args.reward_threshold:.2%}")
    print("="*70 + "\n")
    
    run_benchmark(args)