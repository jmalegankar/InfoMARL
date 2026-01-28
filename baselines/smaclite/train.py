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
# CALLBACK: ROBUST SAVE TOP-K (FULL STATE + WARMUP)
# ============================================================================
class SaveTopK(Callback):
    def __init__(self, k=3, save_dir="best_models", warmup_frames=1_000_000):
        super().__init__()
        self.k = k
        self.save_dir = save_dir
        self.warmup_frames = warmup_frames
        self.top_k_heap = [] 
        self.full_save_dir = None
        self.eval_counter = 0 
        
    def on_setup(self):
        self.full_save_dir = os.path.join(self.experiment.folder_name, self.save_dir)
        os.makedirs(self.full_save_dir, exist_ok=True)
        print(f"   [SaveTopK] Initialized. Warmup: {self.warmup_frames:,} frames. Dir: {self.full_save_dir}")

    def on_evaluation_end(self, rollouts: list):
        if not rollouts: return
        self.eval_counter += 1
        frames = getattr(self.experiment, "n_frames_collected", self.eval_counter)

        if frames < self.warmup_frames:
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
        filename = f"ckpt_f{frames}_r{mean_reward:.4f}.pt"
        file_path = os.path.join(self.full_save_dir, filename)

        if len(self.top_k_heap) < self.k:
            heapq.heappush(self.top_k_heap, (mean_reward, file_path))
            self._save(file_path)
            print(f"   [SaveTopK] ⭐ New Best: {mean_reward:.4f} at {frames:,} frames")
        else:
            worst_reward, worst_path = self.top_k_heap[0]
            if mean_reward > worst_reward:
                heapq.heappop(self.top_k_heap)
                if os.path.exists(worst_path): os.remove(worst_path)
                heapq.heappush(self.top_k_heap, (mean_reward, file_path))
                self._save(file_path)
                print(f"   [SaveTopK] 🏆 Top 3 Updated: {mean_reward:.4f} (Replacing {worst_reward:.4f})")

    def _save(self, path):
        # Save full experiment state (Actor, Critic, Target Nets, Optimizers)
        torch.save(self.experiment.state_dict(), path)

# ============================================================================
# CALLBACK: ROBUST EARLY STOPPING (WARMUP AWARE)
# ============================================================================
class RobustMARLStopping(Callback):
    def __init__(self, warmup_frames=1_000_000, **kwargs):
        super().__init__()
        self.warmup_frames = warmup_frames
        self.window_size = kwargs.get('window_size', 60)
        self.relative_delta = kwargs.get('relative_delta', 0.001)
        self.entropy_floor = kwargs.get('entropy_floor', 0.01)
        self.entropy_patience = kwargs.get('entropy_patience', 10)
        self.plateau_patience = kwargs.get('plateau_patience', 5)
        self.reward_patience = kwargs.get('reward_patience', 15)
        self.reward_threshold = kwargs.get('reward_threshold', 0.001)
        
        self.entropy_hits = 0
        self.plateau_hits = 0
        self.reward_plateau_hits = 0
        self.loss_history = []
        self.reward_history = []

    def on_train_end(self, training_td: TensorDict, group: str):
        frames = getattr(self.experiment, "n_frames_collected", 0)
        if frames < self.warmup_frames: return

        keys = training_td.keys()
        current_loss = None
        if "loss_objective" in keys: current_loss = abs(training_td["loss_objective"].mean().item())
        elif "loss_qvalue" in keys: current_loss = abs(training_td["loss_qvalue"].mean().item())
        elif "loss" in keys: current_loss = abs(training_td["loss"].mean().item())

        if current_loss is not None and not (np.isnan(current_loss) or np.isinf(current_loss)):
            self.loss_history.append(current_loss)
            if len(self.loss_history) > self.window_size:
                past_avg = np.mean(self.loss_history[-self.window_size:-1])
                change = abs(current_loss - past_avg) / (past_avg + 1e-6)
                if change < self.relative_delta: self.plateau_hits += 1
                else: self.plateau_hits = 0
                if self.plateau_hits >= self.plateau_patience:
                    print(f"   [STOP] Loss Plateau: {self.plateau_hits} consecutive hits (Δ: {change:.4%})")
                    raise StopIteration("Loss stabilized.")
                self.loss_history.pop(0)

        # Entropy Check
        if "entropy" in keys:
            ent = training_td["entropy"].mean().item()
            if ent < self.entropy_floor: self.entropy_hits += 1
            else: self.entropy_hits = 0
            if self.entropy_hits >= self.entropy_patience:
                print(f"   [STOP] Entropy Floor reached ({ent:.4f})")
                raise StopIteration("Policy converged.")

    def on_evaluation_end(self, rollouts: list):
        frames = getattr(self.experiment, "n_frames_collected", 0)
        if not rollouts or frames < self.warmup_frames: return
        
        stacked = torch.stack(rollouts)
        try: rewards = stacked[("next", "agents", "reward")]
        except: return
        
        mean_reward = rewards.mean().item()
        self.reward_history.append(mean_reward)
        if len(self.reward_history) > self.reward_patience:
            recent_avg = np.mean(self.reward_history[-self.reward_patience:])
            older_avg = np.mean(self.reward_history[-2*self.reward_patience:-self.reward_patience])
            if abs(recent_avg - older_avg) / (abs(older_avg) + 1e-6) < self.reward_threshold:
                self.reward_plateau_hits += 1
                if self.reward_plateau_hits >= 3:
                    print(f"   [STOP] Reward Plateau: {recent_avg:.4f}")
                    raise StopIteration("Performance stabilized.")
            else: self.reward_plateau_hits = 0

# ============================================================================
# RUNNER
# ============================================================================
def run_experiment(args):
    # 1. Task Initialization
    task_map = {"2s3z": SMACliteTask.TWO_S_THREE_Z, "3s5z": SMACliteTask.THREE_S_FIVE_Z, "3m": SMACliteTask.THREE_M, "8m": SMACliteTask.EIGHT_M}
    task = task_map[args.map].get_from_yaml()

    # 2. Configs
    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.train_device = args.device
    experiment_config.sampling_device = args.device
    experiment_config.max_n_frames = args.max_frames
    experiment_config.loggers = args.loggers
    experiment_config.save_folder = f"results_final/{args.algo}_{args.map}"
    os.makedirs(experiment_config.save_folder, exist_ok=True)
    experiment_config.checkpoint_at_end = True
    experiment_config.clip_grad_norm = True
    experiment_config.clip_grad_val = 1.0
    if args.algo in ["mappo", "ippo"]:
        algorithm_config = MappoConfig.get_from_yaml() if args.algo == "mappo" else IppoConfig.get_from_yaml()
        experiment_config.on_policy_n_envs_per_worker = args.n_envs
        experiment_config.on_policy_collected_frames_per_batch = args.n_envs * 200 # Approx 1 episode per env
    else:
        algorithm_config = QmixConfig.get_from_yaml() if args.algo == "qmix" else MasacConfig.get_from_yaml()
        experiment_config.off_policy_n_envs_per_worker = args.n_envs
        experiment_config.off_policy_collected_frames_per_batch = args.n_envs * 100
        experiment_config.off_policy_train_batch_size = 128

    model_config = MlpConfig.get_from_yaml()
    model_config.num_cells = [args.hidden_dim, args.hidden_dim]
    critic_config = MlpConfig.get_from_yaml()
    critic_config.num_cells = [args.hidden_dim, args.hidden_dim]

    # 3. Execution
    try:
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        experiment = Experiment(
            task=task, algorithm_config=algorithm_config, model_config=model_config,
            critic_model_config=critic_config, seed=args.seed, config=experiment_config,
            callbacks=[RobustMARLStopping(warmup_frames=args.warmup_frames), SaveTopK(warmup_frames=args.warmup_frames)]
        )
        experiment.run()
    except StopIteration as e: print(f"\n>>> ✅ EARLY CONVERGENCE: {e}")
    except Exception as e: traceback.print_exc()
    finally:
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["mappo", "ippo", "qmix", "masac"], required=True)
    parser.add_argument("--map", default="2s3z", choices=["2s3z", "3s5z", "3m", "8m"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_frames", type=int, default=10_000_000)
    parser.add_argument("--n_envs", type=int, default=32)
    parser.add_argument("--warmup_frames", type=int, default=1_000_000)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--loggers", nargs="+", default=["csv", "tensorboard"])
    run_experiment(parser.parse_args())