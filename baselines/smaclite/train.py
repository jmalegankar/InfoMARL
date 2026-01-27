import warnings
import torch
import argparse
import os
import heapq
import numpy as np
import traceback
import gc
from tensordict import TensorDict

# Import your wrapper
from smaclite_torchrl_wrapper import SMACliteTask
import smaclite

from benchmarl.algorithms import MappoConfig, IppoConfig, QmixConfig, MasacConfig
from benchmarl.experiment import Experiment, ExperimentConfig, Callback
from benchmarl.models.mlp import MlpConfig

# Filter warnings for clean output
warnings.filterwarnings("ignore")

# ============================================================================
# 1. CALLBACK: ROBUST SAVE TOP-K
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
        
        # Robust Reward Retrieval
        try:
            rewards = stacked[("next", "agents", "reward")]
        except KeyError:
            try:
                rewards = stacked[("next", "reward")]
            except KeyError:
                return

        mean_reward = rewards.mean().item()
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
# 2. CALLBACK: ENHANCED EARLY STOPPING (FIXED - With All 3 Convergence Signals)
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
        self.warmup_steps = 10

    def on_train_end(self, training_td: TensorDict, group: str):
        keys = training_td.keys()
        
        # --- 1. Loss Plateau Check ---
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

        # --- 2. Entropy Streak Check ---
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
# 3. EXPERIMENT RUNNER (FIXED)
# ============================================================================

def run_experiment(args):
    algo_name = args.algo
    task_name = args.map
    
    print(f"\n{'='*70}")
    print(f"STARTING {algo_name.upper()} on SMACLite {task_name} | Seed {args.seed}")
    print(f"Max Frames: {args.max_frames:,} | Device: {args.device}")
    print(f"{'='*70}")

    # 1. Task
    if task_name == "2s3z":
        task = SMACliteTask.TWO_S_THREE_Z.get_from_yaml()
    elif task_name == "3s5z":
        task = SMACliteTask.THREE_S_FIVE_Z.get_from_yaml()
    elif task_name == "3m":
        task = SMACliteTask.THREE_M.get_from_yaml()
    elif task_name == "8m":
        task = SMACliteTask.EIGHT_M.get_from_yaml()
    else:
        raise ValueError(f"Invalid map name: {task_name}")

    # 2. Experiment Config
    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.train_device = args.device
    experiment_config.sampling_device = args.device
    experiment_config.max_n_frames = args.max_frames
    experiment_config.loggers = args.loggers
    experiment_config.save_folder = f"results/{algo_name}_{task_name}"
    os.makedirs(experiment_config.save_folder, exist_ok=True)
    experiment_config.checkpoint_at_end = True
    experiment_config.clip_grad_norm = True
    experiment_config.clip_grad_val = 1.0       
        
    # 3. Algorithm-Specific Config
    if algo_name == "mappo":
        algorithm_config = MappoConfig.get_from_yaml()
        algorithm_config.clip_epsilon = 0.2
        algorithm_config.entropy_coef = 0.01

        
        # OPTIMIZED: More parallel envs
        N_ENVS = args.n_envs  # 32-64 for SMACLite
        experiment_config.on_policy_n_envs_per_worker = N_ENVS
        experiment_config.on_policy_collected_frames_per_batch = N_ENVS * 200  # batch = n_envs * episode_len
        experiment_config.on_policy_n_minibatch_iters = 10
        
        experiment_config.checkpoint_interval = experiment_config.on_policy_collected_frames_per_batch * 5
        experiment_config.evaluation_interval = experiment_config.on_policy_collected_frames_per_batch * 5

    elif algo_name == "ippo":
        algorithm_config = IppoConfig.get_from_yaml()
        algorithm_config.clip_epsilon = 0.2
        algorithm_config.entropy_coef = 0.01
        
        N_ENVS = args.n_envs
        experiment_config.on_policy_n_envs_per_worker = N_ENVS
        experiment_config.on_policy_collected_frames_per_batch = N_ENVS * 200
        experiment_config.on_policy_n_minibatch_iters = 10
        
        experiment_config.checkpoint_interval = experiment_config.on_policy_collected_frames_per_batch * 5
        experiment_config.evaluation_interval = experiment_config.on_policy_collected_frames_per_batch * 5

    elif algo_name == "qmix":
        algorithm_config = QmixConfig.get_from_yaml()
        
        N_ENVS = args.n_envs
        BATCH_SIZE = N_ENVS * 100  
        
        experiment_config.off_policy_n_envs_per_worker = N_ENVS
        experiment_config.off_policy_collected_frames_per_batch = BATCH_SIZE
        experiment_config.off_policy_train_batch_size = 256
        experiment_config.off_policy_memory_size = 100_000
        experiment_config.off_policy_n_optimizer_steps = 10
        experiment_config.off_policy_init_random_frames = 10_000
        
        # Target network update
        experiment_config.target_update_interval_or_tau = 200
        
        experiment_config.checkpoint_interval = BATCH_SIZE * 5
        experiment_config.evaluation_interval = BATCH_SIZE * 5

    elif algo_name == "masac":
        algorithm_config = MasacConfig.get_from_yaml()
        algorithm_config.target_entropy = "auto"
        
        N_ENVS = args.n_envs
        BATCH_SIZE = N_ENVS * 100
        
        experiment_config.off_policy_n_envs_per_worker = N_ENVS
        experiment_config.off_policy_collected_frames_per_batch = BATCH_SIZE
        experiment_config.off_policy_train_batch_size = 256
        experiment_config.off_policy_memory_size = 100_000
        experiment_config.off_policy_init_random_frames = 10_000
        experiment_config.off_policy_n_optimizer_steps = 10
        
        # Soft target update
        experiment_config.target_update_interval_or_tau = 0.005
        
        experiment_config.checkpoint_interval = BATCH_SIZE * 10
        experiment_config.evaluation_interval = BATCH_SIZE * 10

    # 4. Model Architecture
    model_config = MlpConfig.get_from_yaml()
    model_config.num_cells = [args.hidden_dim, args.hidden_dim]
    model_config.activation_class = torch.nn.ReLU
    model_config.layer_class = torch.nn.Linear
    
    critic_config = MlpConfig.get_from_yaml()
    critic_config.num_cells = [args.hidden_dim, args.hidden_dim]
    critic_config.activation_class = torch.nn.ReLU

    # 5. Callbacks (ENHANCED VERSION)
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

    print(f"   Parallel Envs: {N_ENVS}")
    print(f"   Early Stopping: Loss Plateau={args.plateau_patience}, Entropy={args.entropy_patience}")
    print(f"{'='*70}\n")

    try:
        gc.collect()
        if torch.cuda.is_available(): 
            torch.cuda.empty_cache()

        experiment = Experiment(
            task=task,
            algorithm_config=algorithm_config,
            model_config=model_config,
            critic_model_config=critic_config,
            seed=args.seed,
            config=experiment_config,
            callbacks=[stopper, saver]
        )
        experiment.run()
        
        print(f"\n{'='*70}")
        print(f"✅ {algo_name.upper()} on {task_name} COMPLETE")
        print(f"{'='*70}\n")
        
    except StopIteration as e:
        print(f"\n{'='*70}")
        print(f"✅ Experiment Converged Early: {e}")
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"❌ CRITICAL ERROR: {e}")
        traceback.print_exc()
        print(f"{'='*70}\n")
        
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SMACLite MARL Training with Enhanced Early Stopping")
    
    # Experiment
    parser.add_argument("--algo", choices=["mappo", "ippo", "qmix", "masac"], required=True)
    parser.add_argument("--map", default="2s3z", choices=["2s3z", "3s5z", "3m", "8m"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_frames", type=int, default=10_000_000)
    
    # Parallelism
    parser.add_argument("--n_envs", type=int, default=48,
                        help="Parallel environments. Try 32-64 for SMACLite on RTX 3070 Ti")
    
    # Model
    parser.add_argument("--hidden_dim", type=int, default=256)
    
    # Hardware
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # Logging
    parser.add_argument("--loggers", nargs="+", default=["csv", "tensorboard"])
    
    # Early Stopping - Loss Plateau
    parser.add_argument("--plateau_window", type=int, default=40)
    parser.add_argument("--plateau_threshold", type=float, default=0.005)
    parser.add_argument("--plateau_patience", type=int, default=3)
    
    # Early Stopping - Entropy
    parser.add_argument("--entropy_stop", type=float, default=0.01)
    parser.add_argument("--entropy_patience", type=int, default=5)
    
    # Early Stopping - Reward Plateau
    parser.add_argument("--reward_patience", type=int, default=10)
    parser.add_argument("--reward_threshold", type=float, default=0.002)

    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("SMACLITE BENCHMARK ")
    print("="*70)
    print(f"Algorithm: {args.algo.upper()}")
    print(f"Map: {args.map}")
    print(f"Device: {args.device}")
    print(f"Parallel Envs: {args.n_envs}")
    print(f"Max Frames: {args.max_frames:,}")
    print(f"\nEarly Stopping:")
    print(f"  - Loss Plateau: {args.plateau_patience} hits @ Δ<{args.plateau_threshold:.2%}")
    print(f"  - Entropy Floor: {args.entropy_patience} hits @ <{args.entropy_stop}")
    print(f"  - Reward Plateau: @ Δ<{args.reward_threshold:.2%}")
    print("="*70 + "\n")
    
    run_experiment(args)