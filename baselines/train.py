import argparse
from benchmarl.algorithms import MappoConfig, MasacConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig
from benchmarl.benchmark import Benchmark


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark MAPPO and MASAC on VMAS tasks"
    )
    parser.add_argument(
        "--n_agents",
        type=int,
        nargs="+",
        default=[3, 5],
        help="Number of agents to test (can specify multiple values)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Random seeds for experiments",
    )
    parser.add_argument(
        "--max_n_frames",
        type=int,
        default=1_000,
        help="Maximum number of frames to collect",
    )
    parser.add_argument(
        "--evaluation",
        action="store_true",
        help="Enable evaluation during training",
    )
    parser.add_argument(
        "--loggers",
        type=str,
        nargs="+",
        default=["csv"],
        help="Loggers to use (e.g., wandb, csv, tensorboard)",
    )
    parser.add_argument(
        "--save_folder",
        type=str,
        default=None,
        help="Folder to save results (default: current directory)",
    )
    return parser.parse_args()


def create_task_configs(n_agents_list):
    """Create task configurations for different agent counts."""
    task_configs = []
    
    for n_agents in n_agents_list:
        # # Simple Spread task
        # simple_spread = VmasTask.SIMPLE_SPREAD.get_from_yaml()
        # simple_spread.config["n_agents"] = n_agents
        # task_configs.append(simple_spread)
        
        # # Reverse Transport task
        # reverse_transport = VmasTask.REVERSE_TRANSPORT.get_from_yaml()
        # reverse_transport.config["n_agents"] = n_agents
        # task_configs.append(reverse_transport)

        food_collection = VmasTask.FOOD_COLLECTION.get_from_yaml()
        food_collection.config["n_agents"] = n_agents
        task_configs.append(food_collection)
    
    return task_configs


def run_benchmark(args):
    """Run the benchmark with specified configuration."""
    
    # Create experiment configuration
    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.max_n_frames = args.max_n_frames
    experiment_config.evaluation = args.evaluation
    experiment_config.loggers = args.loggers
    
    if args.save_folder:
        experiment_config.save_folder = args.save_folder
    
    # Create algorithm configurations
    algorithm_configs = [
        MappoConfig.get_from_yaml(),
        MasacConfig.get_from_yaml(),
    ]
    
    # Create task configurations with different agent counts
    task_configs = create_task_configs(args.n_agents)
    
    # Create model configurations
    model_config = MlpConfig.get_from_yaml()
    critic_model_config = MlpConfig.get_from_yaml()
    
    # Create and run benchmark
    print(f"\n{'='*80}")
    print("Starting BenchMARL Benchmark")
    print(f"{'='*80}")
    print(f"Algorithms: MAPPO, MASAC")
    print(f"Tasks: simple_spread, reverse_transport, food_collection")
    print(f"Agent counts: {args.n_agents}")
    print(f"Seeds: {args.seeds}")
    print(f"Max frames: {args.max_n_frames:,}")
    print(f"Loggers: {args.loggers}")
    print(f"{'='*80}\n")
    
    benchmark = Benchmark(
        algorithm_configs=algorithm_configs,
        tasks=task_configs,
        seeds=set(args.seeds),
        experiment_config=experiment_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
    )
    
    # Run experiments sequentially
    benchmark.run_sequential()
    
    print(f"\n{'='*80}")
    print("Benchmark completed!")
    print(f"{'='*80}\n")


def main():
    """Main entry point."""
    args = parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()