import argparse
import torch
import gc
import traceback
import math
from dataclasses import dataclass
from enum import Enum
from typing import List

from benchmarl.algorithms import MappoConfig, MasacConfig, QmixConfig, IppoConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig


class AlgorithmType(Enum):
    ON_POLICY = "on_policy"
    OFF_POLICY_CONTINUOUS = "off_policy_continuous"
    OFF_POLICY_DISCRETE = "off_policy_discrete"

@dataclass
class AlgorithmInfo:
    config_class: type
    algo_type: AlgorithmType
    name: str

ALGORITHMS = {
    "mappo": AlgorithmInfo(MappoConfig, AlgorithmType.ON_POLICY, "MAPPO"),
    "masac": AlgorithmInfo(MasacConfig, AlgorithmType.OFF_POLICY_CONTINUOUS, "MASAC"),
    "qmix": AlgorithmInfo(QmixConfig, AlgorithmType.OFF_POLICY_DISCRETE, "QMIX"),
    "ippo": AlgorithmInfo(IppoConfig, AlgorithmType.ON_POLICY, "IPPO"),
}

# ============================================================================
# DYNAMIC CONFIGURATION
# ============================================================================

def configure_experiment(args, algo_type: AlgorithmType) -> ExperimentConfig:
    config = ExperimentConfig.get_from_yaml()
    
    # Global
    config.max_n_frames = args.max_n_frames
    config.evaluation = True
    config.loggers = args.loggers
    config.save_folder = args.save_folder
    config.train_device = args.device
    config.sampling_device = args.device
    
    # Optimization
    config.gamma = 0.99
    config.lr = args.lr_on_policy if algo_type == AlgorithmType.ON_POLICY else args.lr_off_policy
    config.clip_grad_norm = True
    config.clip_grad_val = args.clip_grad
    config.adam_eps = 1e-5
    config.gae_lambda = 0.95 
    
    # Algo Specifics
    config.epsilon_beg = 1.0
    config.epsilon_end = 0.05
    config.epsilon_anneal_frames = 100_000
    config.target_update_interval_or_tau = 0.005 if algo_type == AlgorithmType.OFF_POLICY_CONTINUOUS else 200

    # Evaluation
    config.evaluation_interval = args.eval_interval
    config.evaluation_episodes = 10
    config.evaluation_deterministic_actions = True
    config.checkpoint_interval = args.checkpoint_interval
    config.checkpoint_at_end = True
    
    # ========================================================================
    # OPTIMAL PARALLELISM LOGIC
    # ========================================================================
    if algo_type == AlgorithmType.ON_POLICY:
        # PPO: MAX PARALLELISM
        # 600 envs is ideal for RTX 3060. 
        # Batch size must be >= 600 * max_steps to avoid partial episodes.
        
        n_envs = 600
        min_batch = n_envs * args.task_max_steps
        
        # Round up user batch request to ensure full episodes
        batch_size = max(args.on_policy_frames_per_batch, min_batch)
        
        print(f"   [PPO Optimization] Using {n_envs} Envs | Batch Size: {batch_size}")
        
        config.on_policy_n_envs_per_worker = n_envs
        config.on_policy_collected_frames_per_batch = batch_size
        config.on_policy_minibatch_size = 4096
        config.on_policy_n_minibatch_iters = args.n_epochs
        config.evaluation_interval = config.on_policy_collected_frames_per_batch
        config.checkpoint_interval = config.on_policy_collected_frames_per_batch
    else:
        # QMIX/MASAC: BALANCED PARALLELISM
        # We need frequent updates, so we can't wait for 600 envs to finish.
        # But 5 envs is too slow. 
        # Ideal: 32 envs. This allows faster collection but keeps batches reasonable.
        
        n_envs = 32
        
        # Ensure batch captures full episodes: 32 * max_steps
        # e.g. 32 * 200 = 6400 frames per batch.
        batch_size = n_envs * args.task_max_steps
        
        print(f"   [QMIX Optimization] Using {n_envs} Envs | Batch Size: {batch_size}")

        config.off_policy_n_envs_per_worker = n_envs
        config.off_policy_collected_frames_per_batch = batch_size
        
        config.off_policy_train_batch_size = args.train_batch_size
        config.off_policy_n_optimizer_steps = args.n_optimizer_steps
        config.off_policy_memory_size = args.replay_buffer_size
        config.off_policy_init_random_frames = batch_size * 2 # Warmup
        config.evaluation_interval = config.off_policy_collected_frames_per_batch * 10  # Less frequent evals for off-policy
        config.checkpoint_interval = config.off_policy_collected_frames_per_batch * 10
        
        
    return config


def configure_algorithm(algo_name: str, args):
    algo_config = ALGORITHMS[algo_name].config_class.get_from_yaml()
    if algo_name in ["mappo", "ippo"]:
        algo_config.clip_epsilon = 0.2
        algo_config.entropy_coef = 0.01 
        algo_config.critic_coef = 1.0
    elif algo_name == "masac":
        algo_config.target_entropy = "auto"
    return algo_config

VMAS_TASKS = {
    "simple_spread": VmasTask.SIMPLE_SPREAD,
    "food_collection": VmasTask.FOOD_COLLECTION,
}

def run_benchmark(args):
    tasks = [VMAS_TASKS[t].get_from_yaml() for t in args.tasks]
    
    for task in tasks:
        if hasattr(task, 'config'):
            task.config["n_agents"] = args.n_agents
            task.config["max_steps"] = args.task_max_steps 

    for algo_name in args.algorithms:
        algo_info = ALGORITHMS[algo_name]
        exp_config = configure_experiment(args, algo_info.algo_type)
        algo_config = configure_algorithm(algo_name, args)
        
        model_config = MlpConfig.get_from_yaml()
        critic_config = MlpConfig.get_from_yaml()
        model_config.num_cells = [args.hidden_dim, args.hidden_dim]
        critic_config.num_cells = [args.hidden_dim, args.hidden_dim]

        for task in tasks:
            for seed in args.seeds:
                print(f"\n>>> Running {algo_name.upper()} | Task: {task.name} | Seed: {seed}")
                try:
                    gc.collect()
                    if torch.cuda.is_available(): torch.cuda.empty_cache()

                    experiment = Experiment(
                        task=task,
                        algorithm_config=algo_config,
                        model_config=model_config,
                        critic_model_config=critic_config,
                        seed=seed,
                        config=exp_config,
                    )
                    experiment.run()
                except Exception as e:
                    print(f"FAILED: {e}")
                    traceback.print_exc()
                    if args.fail_fast: return

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithms", "-a", nargs="+", default=["mappo"], choices=list(ALGORITHMS.keys()))
    parser.add_argument("--tasks", "-t", nargs="+", default=["simple_spread"], choices=list(VMAS_TASKS.keys()))
    parser.add_argument("--seeds", "-s", type=int, nargs="+", default=[0])
    parser.add_argument("--n_agents", type=int, default=4)
    parser.add_argument("--task_max_steps", type=int, default=100)
    parser.add_argument("--max_n_frames", type=int, default=3_000_000)
    parser.add_argument("--on_policy_frames_per_batch", type=int, default=60_000)
    parser.add_argument("--minibatch_size", type=int, default=4096)
    parser.add_argument("--n_epochs", type=int, default=15)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--n_optimizer_steps", type=int, default=1)
    parser.add_argument("--replay_buffer_size", type=int, default=100_000)
    parser.add_argument("--lr_on_policy", type=float, default=5e-4)
    parser.add_argument("--lr_off_policy", type=float, default=5e-4)
    parser.add_argument("--clip_grad", type=float, default=10.0)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--eval_interval", type=int, default=60_000)
    parser.add_argument("--checkpoint_interval", type=int, default=60_000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--loggers", nargs="+", default=["csv", "tensorboard"])
    parser.add_argument("--save_folder", type=str, default="results/simple_spread")
    parser.add_argument("--fail_fast", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    run_benchmark(parse_args())