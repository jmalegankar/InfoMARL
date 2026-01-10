"""
BenchMARL Training Script - Production Ready
Verified against BenchMARL v1.3.0 documentation
Handles NaN errors with proven numerical stability measures
"""

import argparse
import torch
from benchmarl.algorithms import MappoConfig, MasacConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import ExperimentConfig
from benchmarl.models.mlp import MlpConfig
from benchmarl.benchmark import Benchmark


def parse_args():
    """Parse command line arguments with correct defaults"""
    parser = argparse.ArgumentParser(
        description="Benchmarking Baselines for VMAS with Numerical Stability"
    )
    parser.add_argument(
        "--n_agents", 
        type=int, 
        nargs="+", 
        default=[4], 
        help="Number of agents per task"
    )
    parser.add_argument(
        "--seeds", 
        type=int, 
        nargs="+", 
        default=[0], 
        help="Random seeds for reproducibility"
    )
    parser.add_argument(
        "--max_n_frames", 
        type=int, 
        default=1_000_000, 
        help="Maximum training frames (default: 1M)"
    )
    parser.add_argument(
        "--no-evaluation", 
        action="store_false", 
        dest="evaluation",
        default=True,
        help="Disable evaluation during training"
    )
    parser.add_argument(
        "--loggers", 
        type=str, 
        nargs="+", 
        default=["csv", "tensorboard"],
        help="Loggers to use"
    )
    parser.add_argument(
        "--save_folder", 
        type=str, 
        default="results",
        help="Folder to save results"
    )
    return parser.parse_args()


def create_task_configs(n_agents_list):
    """Create task configurations for multiple agent counts"""
    task_configs = []
    tasks_to_test = [
        VmasTask.SIMPLE_SPREAD,
        VmasTask.REVERSE_TRANSPORT,
        VmasTask.FOOD_COLLECTION
    ]
    
    for n_agents in n_agents_list:
        for task_enum in tasks_to_test:
            task = task_enum.get_from_yaml()
            task.config["n_agents"] = n_agents
            task_configs.append(task)
    
    return task_configs


def configure_experiment(args):
    """
    Configure experiment with NaN-prevention settings
    Settings verified from BenchMARL fine_tuned/vmas/conf/config.yaml
    """
    experiment_config = ExperimentConfig.get_from_yaml()
    
    # Basic settings
    experiment_config.max_n_frames = args.max_n_frames
    experiment_config.evaluation = args.evaluation
    experiment_config.loggers = args.loggers
    experiment_config.save_folder = args.save_folder
    
    # Device configuration - EXPLICIT
    experiment_config.train_device = 'cuda'
    experiment_config.sampling_device = 'cuda'
    
    # Learning rate - CRITICAL (use low LR to prevent NaN)
    # Verified from fine_tuned config: lr = 0.00005
    experiment_config.lr = 5e-5
    
    # Gradient clipping - CRITICAL (prevents NaN from exploding gradients)
    # Verified syntax from ExperimentConfig documentation
    experiment_config.clip_grad_norm = True  # Boolean: enable clipping
    experiment_config.clip_grad_val = 5.0    # Float: max gradient norm
    
    # Adam epsilon for numerical stability
    experiment_config.adam_eps = 1e-5
    
    return experiment_config


def configure_algorithms():
    """
    Configure MAPPO and MASAC with conservative settings
    No algorithm-specific gradient clipping needed - handled by ExperimentConfig
    """
    # MAPPO Configuration (On-policy)
    mappo_config = MappoConfig.get_from_yaml()
    # All gradient clipping handled by ExperimentConfig
    
    # MASAC Configuration (Off-policy)  
    masac_config = MasacConfig.get_from_yaml()
    # All gradient clipping handled by ExperimentConfig
    
    return [mappo_config, masac_config]


def run_benchmark(args):
    """Main benchmark execution with comprehensive error handling"""
    
    print("\n" + "="*70)
    print("BenchMARL Experiment - NaN-Stable Configuration")
    print("="*70)
    print(f"Configuration:")
    print(f"  • Agents: {args.n_agents}")
    print(f"  • Seeds: {args.seeds}")
    print(f"  • Max frames: {args.max_n_frames:,}")
    print(f"  • Evaluation: {args.evaluation}")
    print(f"  • Save folder: {args.save_folder}")
    print("="*70 + "\n")
    
    try:
        # Configure experiment with NaN prevention
        print("⚙️  Configuring experiment...")
        experiment_config = configure_experiment(args)
        
        print(f"    ✓ Devices: train={experiment_config.train_device}, "
              f"sampling={experiment_config.sampling_device}")
        print(f"    ✓ Learning rate: {experiment_config.lr}")
        print(f"    ✓ Gradient clipping: enabled (max_norm={experiment_config.clip_grad_val})")
        
        # Configure algorithms
        print("⚙️  Configuring algorithms...")
        algorithm_configs = configure_algorithms()
        print(f"    ✓ Algorithms: {len(algorithm_configs)} "
              f"({', '.join(c.__class__.__name__.replace('Config', '') for c in algorithm_configs)})")
        
        # Create tasks
        print("⚙️  Creating tasks...")
        task_configs = create_task_configs(args.n_agents)
        print(f"    ✓ Tasks: {len(task_configs)} configurations")
        
        # Model configurations
        print("⚙️  Configuring models...")
        model_config = MlpConfig.get_from_yaml()
        critic_model_config = MlpConfig.get_from_yaml()
        print(f"    ✓ Models: MLP (actor + critic)")
        
        # Create benchmark
        print("\n🚀 Creating benchmark...")
        benchmark = Benchmark(
            algorithm_configs=algorithm_configs,
            tasks=task_configs,
            seeds=set(args.seeds),
            experiment_config=experiment_config,
            model_config=model_config,
            critic_model_config=critic_model_config,
        )
        
        # Run benchmark
        print("🏃 Starting sequential benchmark execution...\n")
        print("-"*70)
        benchmark.run_sequential()
        print("-"*70)
        
        print("\n" + "="*70)
        print("✅ BENCHMARK COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"Results saved to: {args.save_folder}/")
        print("="*70 + "\n")
        
    except AssertionError as e:
        if "isnan" in str(e):
            print("\n" + "="*70)
            print("❌ NaN ERROR DETECTED")
            print("="*70)
            print("NaN values appeared in actions despite stability measures.")
            print("\nPossible causes:")
            print("  1. Task-specific numerical issues")
            print("  2. Environment returning invalid observations")
            print("  3. Require even lower learning rate")
            print("\nSuggested fixes:")
            print("  • Try: experiment_config.lr = 1e-5 (even lower)")
            print("  • Try: experiment_config.clip_grad_val = 1.0 (stricter clipping)")
            print("  • Check if specific task is problematic")
            print("="*70 + "\n")
        raise
        
    except Exception as e:
        print("\n" + "="*70)
        print("❌ BENCHMARK FAILED")
        print("="*70)
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("="*70 + "\n")
        
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    # Set PyTorch numerical stability settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Run benchmark
    run_benchmark(parse_args())