"""
Unified BenchMARL Training Script
Supports: MAPPO (on-policy), MASAC (off-policy), QMIX (off-policy, discrete)
Properly handles custom VMAS scenarios and algorithm-specific configurations.
"""

import argparse
import copy
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Type

import torch

from benchmarl.algorithms import MappoConfig, MasacConfig, QmixConfig
from benchmarl.environments import VmasTask
from benchmarl.environments.common import TaskClass
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig
from benchmarl.benchmark import Benchmark

# ============================================================================
# ALGORITHM CLASSIFICATION
# ============================================================================

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
}


# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

def configure_experiment(
    args,
    algo_type: AlgorithmType
) -> ExperimentConfig:
    """
    Configure experiment with algorithm-appropriate settings.
    On-policy and off-policy algorithms have different batch/buffer configs.
    """
    config = ExperimentConfig.get_from_yaml()
    
    # Common settings
    config.max_n_frames = args.max_n_frames
    config.evaluation = args.evaluation
    config.loggers = args.loggers
    config.save_folder = args.save_folder
    config.train_device = args.device
    config.sampling_device = args.device
    
    # Numerical stability (helps prevent NaN)
    config.clip_grad_norm = True
    config.clip_grad_val = args.clip_grad
    config.adam_eps = 1e-5
    
    # Evaluation settings
    config.evaluation_deterministic_actions = True
    config.evaluation_interval = args.eval_interval
    
    # Algorithm-type-specific settings
    if algo_type == AlgorithmType.ON_POLICY:
        config.lr = args.lr_on_policy
        config.on_policy_collected_frames_per_batch = args.frames_per_batch
        config.on_policy_minibatch_size = args.minibatch_size
        config.on_policy_n_minibatch_iters = 4
        
    elif algo_type == AlgorithmType.OFF_POLICY_CONTINUOUS:
        config.lr = args.lr_off_policy
        config.off_policy_collected_frames_per_batch = args.frames_per_batch
        config.off_policy_n_envs_per_worker = 60
        config.off_policy_train_batch_size = args.train_batch_size
        config.off_policy_n_optimizer_steps = args.n_optimizer_steps
        config.off_policy_memory_size = args.replay_buffer_size
        config.off_policy_init_random_frames = args.frames_per_batch
        
    elif algo_type == AlgorithmType.OFF_POLICY_DISCRETE:
        config.lr = args.lr_off_policy
        config.prefer_continuous_actions = False  # QMIX needs discrete
        config.off_policy_collected_frames_per_batch = args.frames_per_batch
        config.off_policy_n_envs_per_worker = 60
        config.off_policy_train_batch_size = args.train_batch_size
        config.off_policy_n_optimizer_steps = args.n_optimizer_steps
        config.off_policy_memory_size = args.replay_buffer_size
        config.off_policy_init_random_frames = args.frames_per_batch
    
    return config


# ============================================================================
# TASK CREATION
# ============================================================================

# Map of built-in VMAS tasks
VMAS_TASKS = {
    "simple_spread": VmasTask.SIMPLE_SPREAD,
    "reverse_transport": VmasTask.REVERSE_TRANSPORT,
    "navigation": VmasTask.NAVIGATION,
    "sampling": VmasTask.SAMPLING,
    "balance": VmasTask.BALANCE,
    "transport": VmasTask.TRANSPORT,
    "discovery": VmasTask.DISCOVERY,
    "flocking": VmasTask.FLOCKING,
    "dispersion": VmasTask.DISPERSION,

}


def create_tasks(task_names: List[str], n_agents: int) -> List[TaskClass]:
    """Create task configurations, modifying n_agents where applicable."""
    tasks = []
    
    for task_name in task_names:
        if task_name not in VMAS_TASKS:
            raise ValueError(
                f"Unknown task: {task_name}. "
                f"Available: {list(VMAS_TASKS.keys())}"
            )
        
        task = VMAS_TASKS[task_name].get_from_yaml()
        
        # Override n_agents if the task supports it
        if "n_agents" in task.config:
            task.config["n_agents"] = n_agents
        
        tasks.append(task)
    
    return tasks


# ============================================================================
# MAIN BENCHMARK RUNNER
# ============================================================================

def run_single_algorithm_benchmark(
    algo_name: str,
    tasks: List[TaskClass],
    seeds: set,
    args,
) -> None:
    """Run benchmark for a single algorithm across all tasks."""
    
    algo_info = ALGORITHMS[algo_name]
    print(f"\n{'='*70}")
    print(f"Running {algo_info.name} ({algo_info.algo_type.value})")
    print(f"{'='*70}")
    
    # Get algorithm-appropriate experiment config
    exp_config = configure_experiment(args, algo_info.algo_type)
    
    # Get algorithm config
    algo_config = algo_info.config_class.get_from_yaml()
    
    # Model configs
    model_config = MlpConfig.get_from_yaml()
    critic_model_config = MlpConfig.get_from_yaml()
    
    # For discrete algorithms, we need to run separately 
    # since they require different action space handling
    if algo_info.algo_type == AlgorithmType.OFF_POLICY_DISCRETE:
        # QMIX needs discrete actions - run experiments individually
        for task in tasks:
            for seed in seeds:
                print(f"\n  → Task: {task.name}, Seed: {seed}")
                try:
                    experiment = Experiment(
                        task=task,
                        algorithm_config=algo_config,
                        model_config=model_config,
                        critic_model_config=critic_model_config,
                        seed=seed,
                        config=exp_config,
                    )
                    experiment.run()
                except Exception as e:
                    print(f"    ✗ Failed: {e}")
                    if args.fail_fast:
                        raise
    else:
        # Continuous algorithms can use Benchmark class
        benchmark = Benchmark(
            algorithm_configs=[algo_config],
            tasks=tasks,
            seeds=seeds,
            experiment_config=exp_config,
            model_config=model_config,
            critic_model_config=critic_model_config,
        )
        benchmark.run_sequential()


def run_benchmark(args):
    """Main benchmark execution."""
    
    print("\n" + "="*70)
    print("BenchMARL Unified Training Script")
    print("="*70)
    print(f"Algorithms: {args.algorithms}")
    print(f"Tasks: {args.tasks}")
    print(f"Agents: {args.n_agents}")
    print(f"Seeds: {args.seeds}")
    print(f"Max frames: {args.max_n_frames:,}")
    print(f"Device: {args.device}")
    print("="*70)
    
    # Create tasks
    tasks = create_tasks(args.tasks, args.n_agents)
    seeds = set(args.seeds)
    
    # Run each algorithm
    for algo_name in args.algorithms:
        if algo_name not in ALGORITHMS:
            print(f"Unknown algorithm: {algo_name}, skipping...")
            continue
        
        try:
            run_single_algorithm_benchmark(algo_name, tasks, seeds, args)
            print(f"\n{algo_name.upper()} completed successfully!")
        except Exception as e:
            print(f"\n{algo_name.upper()} failed: {e}")
            if args.fail_fast:
                raise
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("BENCHMARK COMPLETE")
    print(f"Results saved to: {args.save_folder}/")
    print("="*70)


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified BenchMARL training for MAPPO, MASAC, and QMIX",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # What to run
    parser.add_argument(
        "--algorithms", "-a",
        nargs="+",
        default=["mappo"],
        choices=list(ALGORITHMS.keys()),
        help="Algorithms to benchmark",
    )
    parser.add_argument(
        "--tasks", "-t",
        nargs="+",
        default=["simple_spread"],
        choices=list(VMAS_TASKS.keys()),
        help="VMAS tasks to run",
    )
    parser.add_argument(
        "--seeds", "-s",
        type=int,
        nargs="+",
        default=[0],
        help="Random seeds",
    )
    parser.add_argument(
        "--n_agents",
        type=int,
        default=4,
        help="Number of agents (for tasks that support it)",
    )
    
    # Training settings
    parser.add_argument(
        "--max_n_frames",
        type=int,
        default=1_000_000,
        help="Maximum training frames",
    )
    parser.add_argument(
        "--frames_per_batch",
        type=int,
        default=6000,
        help="Frames collected per batch",
    )
    parser.add_argument(
        "--minibatch_size",
        type=int,
        default=4096,
        help="Minibatch size (on-policy)",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=256,
        help="Training batch size (off-policy)",
    )
    parser.add_argument(
        "--n_optimizer_steps",
        type=int,
        default=100,
        help="Optimizer steps per collection (off-policy)",
    )
    parser.add_argument(
        "--replay_buffer_size",
        type=int,
        default=100_000,
        help="Replay buffer size (off-policy)",
    )
    
    # Learning rates
    parser.add_argument(
        "--lr_on_policy",
        type=float,
        default=3e-4,
        help="Learning rate for on-policy algorithms",
    )
    parser.add_argument(
        "--lr_off_policy",
        type=float,
        default=3e-4,
        help="Learning rate for off-policy algorithms",
    )
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=5.0,
        help="Gradient clipping value",
    )
    
    # Evaluation
    parser.add_argument(
        "--no_evaluation",
        action="store_true",
        help="Disable evaluation during training",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=12_000,
        help="Evaluation interval (frames)",
    )
    
    # Infrastructure
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for training",
    )
    parser.add_argument(
        "--loggers",
        nargs="+",
        default=["csv", "tensorboard"],
        help="Loggers to use",
    )
    parser.add_argument(
        "--save_folder",
        type=str,
        default="results",
        help="Folder to save results",
    )
    parser.add_argument(
        "--fail_fast",
        action="store_true",
        help="Stop on first error",
    )
    
    args = parser.parse_args()
    args.evaluation = not args.no_evaluation
    return args


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Reproducibility settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    args = parse_args()
    run_benchmark(args)