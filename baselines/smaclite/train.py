import warnings
import torch
import argparse
import math
from smaclite_torchrl_wrapper import SMACliteTask

# Filter warnings for clean output
warnings.filterwarnings("ignore")

def run_experiment(algo_name, task_name="2s3z"):
    print(f"\n{'='*60}")
    print(f"STARTING {algo_name.upper()} on SMACLite {task_name}")
    print(f"Applying VMAS lessons: Clip=1.0, Safe Batch Sizing")
    print(f"{'='*60}")

    # BenchMARL Imports
    from benchmarl.algorithms import MappoConfig, IppoConfig, QmixConfig
    from benchmarl.experiment import Experiment, ExperimentConfig
    from benchmarl.models.mlp import MlpConfig

    # 1. Select Task & Constants
    if task_name == "2s3z":
        task = SMACliteTask.TWO_S_THREE_Z.get_from_yaml()
        MAX_STEPS = 120 # Approx limit for 2s3z
    elif task_name == "3s5z":
        task = SMACliteTask.THREE_S_FIVE_Z.get_from_yaml()
        MAX_STEPS = 160
    else:
        raise ValueError("Invalid map name")

    # 2. Configure Experiment (Global Training Params)
    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.train_device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # SMAC is harder, so we usually train longer, but 3M is good for comparison
    experiment_config.max_n_frames = 3_000_000  
    experiment_config.save_folder = f"results_smaclite/{algo_name}_{task_name}"
    experiment_config.checkpoint_at_end = True
    experiment_config.checkpoint_interval = 200_000 # Save less frequently to save disk
    experiment_config.evaluation_interval = 50_000
    
    # === LESSON 1: STABILITY (Prevent Crashes) ===
    # SMAC has spikey rewards (unit deaths). 
    # High clip (10.0) causes NaNs/Crashes. Low clip (1.0) is safe.
    experiment_config.clip_grad_norm = True
    experiment_config.clip_grad_val = 1.0  
    
    experiment_config.lr = 5e-4
    experiment_config.gamma = 0.99
    experiment_config.gae_lambda = 0.95

    # === LESSON 2: API COMPATIBILITY (Prevent TypeErrors) ===
    # Move annealing/target updates here (BenchMARL best practice)
    experiment_config.epsilon_anneal_frames = 100_000
    experiment_config.epsilon_beg = 1.0
    experiment_config.epsilon_end = 0.05
    experiment_config.target_update_interval_or_tau = 200

    # 3. Algorithm Specifics & Batching
    
    # CPU Parallelism Limit (Don't kill the PC)
    N_ENVS = 8 

    if algo_name == "mappo":
        algorithm_config = MappoConfig.get_from_yaml()
        algorithm_config.clip_epsilon = 0.2
        algorithm_config.entropy_coef = 0.01
        
        # On-Policy: Larger batches are fine
        experiment_config.on_policy_n_envs_per_worker = N_ENVS
        experiment_config.on_policy_collected_frames_per_batch = 3200 
        experiment_config.on_policy_n_minibatch_iters = 10

    elif algo_name == "ippo":
        algorithm_config = IppoConfig.get_from_yaml()
        algorithm_config.clip_epsilon = 0.2
        algorithm_config.entropy_coef = 0.01
        
        experiment_config.on_policy_n_envs_per_worker = N_ENVS
        experiment_config.on_policy_collected_frames_per_batch = 3200
        experiment_config.on_policy_n_minibatch_iters = 10

    elif algo_name == "qmix":
        algorithm_config = QmixConfig.get_from_yaml()
        # Note: Epsilon/Target params moved to experiment_config above
        
        # === LESSON 3: LOGGING (Prevent NaNs) ===
        # Batch size must be > N_ENVS * MAX_STEPS
        # 8 envs * 120 steps = 960 frames minimum.
        # We set 2000 to be safe.
        SAFE_BATCH = 2000 
        
        experiment_config.off_policy_n_envs_per_worker = N_ENVS
        experiment_config.off_policy_collected_frames_per_batch = SAFE_BATCH
        experiment_config.off_policy_train_batch_size = 32
        experiment_config.off_policy_memory_size = 5000 
        experiment_config.off_policy_n_optimizer_steps = 1

    # 4. Model Architecture
    model_config = MlpConfig.get_from_yaml()
    model_config.num_cells = [256, 256] 
    
    critic_config = MlpConfig.get_from_yaml()
    critic_config.num_cells = [256, 256]

    # 5. Run
    experiment = Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=model_config,
        critic_model_config=critic_config,
        seed=42,
        config=experiment_config
    )
    experiment.run()
    print(f">>> {algo_name.upper()} COMPLETE")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["mappo", "ippo", "qmix"], required=True, help="Algorithm to train")
    parser.add_argument("--map", default="2s3z", choices=["2s3z", "3s5z"], help="SMACLite Map")
    args = parser.parse_args()
    
    run_experiment(args.algo, args.map)