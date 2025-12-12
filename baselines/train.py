#!/usr/bin/env python3
import argparse
from benchmarl.algorithms import MappoConfig, MasacConfig, QmixConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig

ALGO_MAP = {
    "mappo": MappoConfig,
    "masac": MasacConfig,
    "qmix": QmixConfig,
}

VMAS_TASKS = [
    "SIMPLE_SPREAD", "BALANCE", "NAVIGATION", "SAMPLING",
    "TRANSPORT", "WHEEL", "DISCOVERY", "FLOCKING"
]

def train_baseline(task_name, algo_name, seed=0):
    # 1. Get configurations from YAML
    task = getattr(VmasTask, task_name).get_from_yaml()
    algo_config = ALGO_MAP[algo_name].get_from_yaml()
    model_config = MlpConfig.get_from_yaml()
    experiment_config = ExperimentConfig.get_from_yaml()
    
    # 2. Configure EXPERIMENT parameters as needed
    experiment_config.max_n_iters = 500
    experiment_config.lr = 3e-4
    experiment_config.gamma = 0.99
    experiment_config.logger = "tensorboard"
    experiment_config.logger_config = {
        "project_name": "benchmarl_vmas",  
        "log_dir": "results/tensorboard_logs",
        "create_unique_dir": True,
    }
    
    # 3. Algorithm-specific experiment settings
    if algo_name in ["mappo"]:  # On-policy
        experiment_config.on_policy_collected_frames_per_batch = 6000
        experiment_config.on_policy_n_minibatch_iters = 15
    else:  # Off-policy (MASAC, QMIX)
        experiment_config.off_policy_collected_frames_per_batch = 1000
        experiment_config.off_policy_n_optimizer_steps = 100
    
    # 4. Create and run experiment
    experiment = Experiment(
        task=task,
        algorithm_config=algo_config,
        model_config=model_config,
        critic_model_config=MlpConfig.get_from_yaml() if algo_name in ["mappo", "masac"] else None,
        seed=seed,
        config=experiment_config,
    )
    
    experiment.run()
    print(f"Completed {algo_name.upper()} on {task_name} (seed={seed})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="SIMPLE_SPREAD", choices=VMAS_TASKS)
    parser.add_argument("--algo", type=str, default="mappo", choices=list(ALGO_MAP.keys()))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    
    train_baseline(args.task, args.algo, args.seed)