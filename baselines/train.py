import argparse
from benchmarl.algorithms import MappoConfig, MasacConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import ExperimentConfig
from benchmarl.models.mlp import MlpConfig
from benchmarl.benchmark import Benchmark

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmarking Baselines for VMAS")
    parser.add_argument("--n_agents", type=int, nargs="+", default=[4], help="Agents count")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0], help="Random seeds")
    parser.add_argument("--max_n_frames", type=int, default=1_000_000, help="Paper uses ~1M+ frames")
    parser.add_argument("--evaluation", action="store_true", default=True)
    parser.add_argument("--loggers", type=str, nargs="+", default=["csv", "tensorboard"])
    parser.add_argument("--save_folder", type=str, default="results")
    return parser.parse_args()

def create_task_configs(n_agents_list):
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

def run_benchmark(args):
    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.max_n_frames = args.max_n_frames
    experiment_config.evaluation = args.evaluation
    experiment_config.loggers = args.loggers
    experiment_config.save_folder = args.save_folder
    # experiment_config.train_device = 'mps'      # Optimization for network updates
    # experiment_config.sampling_device = 'cpu'   # Optimization for env simulation
        
    # baselines: MAPPO (on-policy), MASAC (off-policy), QMIX (value-based)
    algorithm_configs = [
        MappoConfig.get_from_yaml(),
        MasacConfig.get_from_yaml(),
        # QmixConfig.get_from_yaml(),
    ]
    
    task_configs = create_task_configs(args.n_agents)
    model_config = MlpConfig.get_from_yaml()
    critic_model_config = MlpConfig.get_from_yaml()
    
    benchmark = Benchmark(
        algorithm_configs=algorithm_configs,
        tasks=task_configs,
        seeds=set(args.seeds),
        experiment_config=experiment_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
    )
    
    benchmark.run_sequential()

if __name__ == "__main__":
    run_benchmark(parse_args())