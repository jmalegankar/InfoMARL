import warnings
import torch
import argparse
import math
import os
import heapq
import numpy as np
import traceback
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
# 1. CALLBACK: ROBUST SAVE TOP-K (SMACLite Ready)
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
        
        # Robust Reward Retrieval (Handles SMAC vs VMAS differences)
        try:
            # VMAS often puts it here
            rewards = stacked[("next", "agents", "reward")]
        except KeyError:
            try:
                # SMAC/Global rewards often live here
                rewards = stacked[("next", "reward")]
            except KeyError:
                return # Can't save if we can't find reward

        mean_reward = rewards.mean().item()

        # Try to get experiment frames, fallback to counter
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
# 2. CALLBACK: VERIFIED EARLY STOPPING
# ============================================================================
class RobustMARLStopping(Callback):
    def __init__(self, window_size=40, relative_delta=0.005, entropy_floor=0.01):
        super().__init__()
        self.window_size = window_size
        self.relative_delta = relative_delta
        self.entropy_floor = entropy_floor
        self.loss_history = []

    def on_train_end(self, training_td: TensorDict, group: str):
        keys = training_td.keys()
        
        # Determine Loss Key
        current_loss = None
        if "loss_objective" in keys:      # MAPPO / IPPO
            current_loss = abs(training_td["loss_objective"].mean().item())
        elif "loss_qvalue" in keys:       # MASAC
            current_loss = abs(training_td["loss_qvalue"].mean().item())
        elif "loss" in keys:              # QMIX
            current_loss = abs(training_td["loss"].mean().item())

        # Loss Plateau Check
        if current_loss is not None:
            self.loss_history.append(current_loss)
            
            # Check plateau only after buffer fills (implicit warmup)
            if len(self.loss_history) > self.window_size:
                past_avg = np.mean(self.loss_history[-self.window_size:-1])
                change = abs(current_loss - past_avg) / (past_avg + 1e-6)

                if change < self.relative_delta:
                    print(f"   [STOP] Loss Plateau Detected (Delta: {change:.4%})")
                    raise StopIteration("Loss stabilized.")
                
                self.loss_history.pop(0)

        # Entropy Floor Check (Skip MASAC)
        is_masac = "loss_alpha" in keys
        if not is_masac and "entropy" in keys:
            ent = training_td["entropy"].mean().item()
            if ent < self.entropy_floor and len(self.loss_history) > 10:
                print(f"   [STOP] Policy Converged (Entropy: {ent:.4f})")
                raise StopIteration("Entropy floor reached.")

# ============================================================================
# 3. EXPERIMENT RUNNER
# ============================================================================

def run_experiment(args):
    algo_name = args.algo
    task_name = args.map
    
    print(f"\n{'='*60}")
    print(f"STARTING {algo_name.upper()} on SMACLite {task_name}")
    print(f"Frames: {args.max_frames} | Loggers: {args.loggers}")
    print(f"{'='*60}")

    # 1. Task
    if task_name == "2s3z":
        task = SMACliteTask.TWO_S_THREE_Z.get_from_yaml()
    elif task_name == "3s5z":
        task = SMACliteTask.THREE_S_FIVE_Z.get_from_yaml()
    else:
        raise ValueError("Invalid map name")

    # 2. Experiment Config
    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.train_device = "cuda" if torch.cuda.is_available() else "cpu"
    experiment_config.max_n_frames = args.max_frames
    experiment_config.loggers = args.loggers
    experiment_config.save_folder = f"results/{algo_name}_{task_name}"
    experiment_config.checkpoint_at_end = True
    
    # Optimization
    experiment_config.clip_grad_norm = True
    experiment_config.clip_grad_val = 1.0  
    experiment_config.lr = 5e-4
    experiment_config.gamma = 0.99
    experiment_config.gae_lambda = 0.95
    experiment_config.epsilon_anneal_frames = 100_000
    experiment_config.epsilon_beg = 1.0
    experiment_config.epsilon_end = 0.05

    # 3. Algorithm-Specific Config
    N_ENVS = 8 

    if algo_name == "mappo":
        algorithm_config = MappoConfig.get_from_yaml()
        algorithm_config.clip_epsilon = 0.2
        algorithm_config.entropy_coef = 0.01
        
        experiment_config.on_policy_n_envs_per_worker = N_ENVS
        experiment_config.on_policy_collected_frames_per_batch = 3200 
        experiment_config.on_policy_n_minibatch_iters = 10
        
        experiment_config.checkpoint_interval = experiment_config.on_policy_collected_frames_per_batch
        experiment_config.evaluation_interval = experiment_config.on_policy_collected_frames_per_batch

    elif algo_name == "ippo":
        algorithm_config = IppoConfig.get_from_yaml()
        algorithm_config.clip_epsilon = 0.2
        algorithm_config.entropy_coef = 0.01
        
        experiment_config.on_policy_n_envs_per_worker = N_ENVS
        experiment_config.on_policy_collected_frames_per_batch = 3200
        experiment_config.on_policy_n_minibatch_iters = 10
        
        experiment_config.checkpoint_interval = experiment_config.on_policy_collected_frames_per_batch
        experiment_config.evaluation_interval = experiment_config.on_policy_collected_frames_per_batch

    elif algo_name == "qmix":
        algorithm_config = QmixConfig.get_from_yaml()
        experiment_config.target_update_interval_or_tau = 200
        
        SAFE_BATCH = 2500
        
        experiment_config.off_policy_n_envs_per_worker = N_ENVS
        experiment_config.off_policy_collected_frames_per_batch = SAFE_BATCH
        experiment_config.off_policy_train_batch_size = 256
        experiment_config.off_policy_memory_size = 100_000
        experiment_config.off_policy_n_optimizer_steps = 5
        experiment_config.off_policy_init_random_frames = 10_000
        
        experiment_config.checkpoint_interval = experiment_config.off_policy_collected_frames_per_batch
        experiment_config.evaluation_interval = experiment_config.off_policy_collected_frames_per_batch

    elif algo_name == "masac":
        algorithm_config = MasacConfig.get_from_yaml()
        algorithm_config.target_entropy = "auto"
        experiment_config.target_update_interval_or_tau = 0.005
        
        SAFE_BATCH = 2500
        
        experiment_config.off_policy_n_envs_per_worker = N_ENVS
        experiment_config.off_policy_collected_frames_per_batch = SAFE_BATCH
        experiment_config.off_policy_train_batch_size = 256
        experiment_config.off_policy_memory_size = 100_000
        experiment_config.off_policy_init_random_frames = 10_000
        experiment_config.off_policy_n_optimizer_steps = 5
        
        experiment_config.checkpoint_interval = experiment_config.off_policy_collected_frames_per_batch * 10
        experiment_config.evaluation_interval = experiment_config.off_policy_collected_frames_per_batch * 10

    # 4. Model Architecture
    model_config = MlpConfig.get_from_yaml()
    model_config.num_cells = [256, 256] 
    
    critic_config = MlpConfig.get_from_yaml()
    critic_config.num_cells = [256, 256]

    # 5. Run with Callbacks
    stopper = RobustMARLStopping(
        window_size=args.plateau_patience,
        entropy_floor=args.entropy_stop
    )
    saver = SaveTopK(k=3, save_dir="best_models")

    try:
        experiment = Experiment(
            task=task,
            algorithm_config=algorithm_config,
            model_config=model_config,
            critic_model_config=critic_config,
            seed=args.seed,
            config=experiment_config,
            callbacks=[stopper, saver] # <--- Added Callbacks Here
        )
        experiment.run()
        print(f">>> {algo_name.upper()} COMPLETE")
        
    except StopIteration as e:
        print(f"--- Experiment Converged: {e} ---")
    except Exception as e:
        print(f"!!! CRITICAL ERROR: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["mappo", "ippo", "qmix", "masac"], required=True, help="Algorithm to train")
    parser.add_argument("--map", default="2s3z", choices=["2s3z", "3s5z"], help="SMACLite Map")
    
    # New Arguments for Convergence
    parser.add_argument("--max_frames", type=int, default=10_000_000, help="Total training frames")
    parser.add_argument("--loggers", nargs="+", default=["csv", "tensorboard"], help="Logging backends")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Stopper arguments
    parser.add_argument("--plateau_patience", type=int, default=40)
    parser.add_argument("--entropy_stop", type=float, default=0.01)

    args = parser.parse_args()
    
    run_experiment(args)