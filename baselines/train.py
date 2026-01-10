"""
BenchMARL Training Script - EVALUATION NaN FIX (CORRECTED)
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
    Configure experiment with EVALUATION NaN prevention
    Key insight: NaN happens during evaluation rollouts, not training
    """
    experiment_config = ExperimentConfig.get_from_yaml()
    
    # Basic settings
    experiment_config.max_n_frames = args.max_n_frames
    experiment_config.evaluation = args.evaluation
    experiment_config.loggers = args.loggers
    experiment_config.save_folder = args.save_folder
    
    # Device configuration
    experiment_config.train_device = 'cuda'
    experiment_config.sampling_device = 'cuda'
    
    # CRITICAL: Even lower learning rate (1e-5 instead of 5e-5)
    experiment_config.lr = 1e-5
    
    # CRITICAL: Stricter gradient clipping (1.0 instead of 5.0)
    experiment_config.clip_grad_norm = True
    experiment_config.clip_grad_val = 1.0
    
    # Adam epsilon for numerical stability
    experiment_config.adam_eps = 1e-5
    
    # CRITICAL: Evaluation uses deterministic actions
    experiment_config.evaluation_deterministic_actions = True
    
    # CRITICAL: Evaluation interval must be multiple of batch size (6000)
    # Valid values: 6000, 12000, 18000, 24000, 30000, etc.
    experiment_config.evaluation_interval = 12_000  # Every 12k frames (2 batches)
    
    # Training batch sizes
    experiment_config.on_policy_collected_frames_per_batch = 6000
    experiment_config.on_policy_minibatch_size = 4096
    
    return experiment_config


def configure_algorithms():
    """Configure MAPPO and MASAC"""
    mappo_config = MappoConfig.get_from_yaml()
    masac_config = MasacConfig.get_from_yaml()
    return [mappo_config, masac_config]


def configure_models():
    """Configure models"""
    model_config = MlpConfig.get_from_yaml()
    critic_model_config = MlpConfig.get_from_yaml()
    return model_config, critic_model_config


def run_benchmark(args):
    """Main benchmark execution with comprehensive error handling"""
    
    print("\n" + "="*70)
    print("BenchMARL Experiment - EVALUATION NaN FIX")
    print("="*70)
    print(f"Configuration:")
    print(f"  • Agents: {args.n_agents}")
    print(f"  • Seeds: {args.seeds}")
    print(f"  • Max frames: {args.max_n_frames:,}")
    print(f"  • Evaluation: {args.evaluation}")
    print(f"  • Save folder: {args.save_folder}")
    print("="*70 + "\n")
    
    try:
        # Configure experiment
        print("⚙️  Configuring experiment...")
        experiment_config = configure_experiment(args)
        
        print(f"    ✓ Devices: train={experiment_config.train_device}, "
              f"sampling={experiment_config.sampling_device}")
        print(f"    ✓ Learning rate: {experiment_config.lr} (ultra-low)")
        print(f"    ✓ Gradient clipping: max_norm={experiment_config.clip_grad_val} (strict)")
        print(f"    ✓ Evaluation: deterministic={experiment_config.evaluation_deterministic_actions}")
        print(f"    ✓ Evaluation interval: {experiment_config.evaluation_interval:,} frames")
        print(f"    ✓ Batch size: {experiment_config.on_policy_collected_frames_per_batch:,} frames")
        
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
        model_config, critic_model_config = configure_models()
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
            print("❌ NaN ERROR DURING EVALUATION")
            print("="*70)
            print("The policy network is outputting NaN actions during evaluation.")
            print("\nThis means weights diverged during training despite clipping.")
            print("\n🔧 NUCLEAR OPTIONS TO TRY:")
            print("  1. Reduce LR to 5e-6: experiment_config.lr = 5e-6")
            print("  2. Reduce grad clip to 0.5: experiment_config.clip_grad_val = 0.5")
            print("  3. Use CPU: experiment_config.train_device = 'cpu'")
            print("  4. Disable evaluation: python3 ./train.py --no-evaluation")
            print("  5. Test MAPPO only (comment out MASAC)")
            print("  6. Test SIMPLE_SPREAD only (comment out other tasks)")
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
    # Set PyTorch for maximum numerical stability
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Run benchmark
    run_benchmark(parse_args())