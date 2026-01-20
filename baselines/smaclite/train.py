
import warnings
# Suppress PyTorch deprecation warnings from TorchRL internals
warnings.filterwarnings("ignore", message="size_average and reduce args will be deprecated")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn._reduction")
# Suppress BenchMARL episode termination warnings (expected when horizon > batch)
warnings.filterwarnings("ignore", message="No episode terminated this iteration")

import torch
from torchrl.envs import step_mdp

# Import our custom SMACLite integration
from smaclite_torchrl_wrapper import SMACliteTask, SMACliteTorchRLWrapper
import smaclite  # noqa: F401

def run_mappo_experiment():
    """Run MAPPO on SMACLite 2s3z."""
    from benchmarl.algorithms import MappoConfig
    from benchmarl.experiment import Experiment, ExperimentConfig
    from benchmarl.models.mlp import MlpConfig
    
    task = SMACliteTask.TWO_S_THREE_Z.get_from_yaml()
    
    # Algorithm config
    algorithm_config = MappoConfig.get_from_yaml()
    algorithm_config.clip_epsilon = 0.2
    algorithm_config.entropy_coef = 0.01
    algorithm_config.critic_coef = 0.5
    
    # Model config
    model_config = MlpConfig.get_from_yaml()
    model_config.num_cells = [256, 256]  # Larger networks for SMAC
    critic_model_config = MlpConfig.get_from_yaml()
    critic_model_config.num_cells = [256, 256]
    
    # Experiment config
    experiment_config = ExperimentConfig.get_from_yaml()
    
    # Device - force GPU
    experiment_config.train_device = "cuda"
    experiment_config.sampling_device = "cuda"
    # Training settings
    experiment_config.max_n_frames = 2_000_000  # ~30-45 min on GPU
    experiment_config.on_policy_collected_frames_per_batch = 3200  # 800 steps * 4 envs
    experiment_config.on_policy_n_envs_per_worker = 8
    experiment_config.on_policy_minibatch_size = 1600
    experiment_config.on_policy_n_minibatch_iters = 4
    
    # Stability
    experiment_config.lr = 5e-4
    experiment_config.clip_grad_norm = True
    experiment_config.clip_grad_val = 10.0  # SMAC is more stable than VMAS
    experiment_config.adam_eps = 1e-5
    
    # Evaluation & logging
    experiment_config.evaluation = True
    experiment_config.evaluation_interval = 32_000  # Every 10 batches
    experiment_config.evaluation_episodes = 10
    experiment_config.evaluation_deterministic_actions = True
    
    # Checkpointing
    experiment_config.checkpoint_interval = 96_000
    experiment_config.checkpoint_at_end = True
    experiment_config.keep_checkpoints_num = 3
    
    experiment_config.render = False
    experiment_config.loggers = ["csv", "tensorboard"]
    experiment_config.save_folder = "./outputs/smaclite_mappo"
    
    experiment = Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=42,
        config=experiment_config,
    )
    
    experiment.run()


def run_qmix_experiment():
    """Run QMIX on SMACLite 2s3z."""
    from benchmarl.algorithms import QmixConfig
    from benchmarl.experiment import Experiment, ExperimentConfig
    from benchmarl.models.mlp import MlpConfig
    
    task = SMACliteTask.TWO_S_THREE_Z.get_from_yaml()
    
    # Algorithm config
    algorithm_config = QmixConfig.get_from_yaml()
    
    # Model config
    model_config = MlpConfig.get_from_yaml()
    model_config.num_cells = [256, 256]
    
    # Experiment config
    experiment_config = ExperimentConfig.get_from_yaml()
    
    # Device - force GPU
    experiment_config.train_device = "cuda"
    experiment_config.sampling_device = "cuda"
    
    # Training settings
    experiment_config.max_n_frames = 2_000_000  # ~45-60 min on GPU
    experiment_config.off_policy_collected_frames_per_batch = 800
    experiment_config.off_policy_n_envs_per_worker = 8
    experiment_config.off_policy_train_batch_size = 32
    experiment_config.off_policy_n_optimizer_steps = 1  # QMIX typically 1 update per batch
    experiment_config.off_policy_memory_size = 5000  # In episodes, not frames
    experiment_config.off_policy_init_random_frames = 1600  # 2 batches of exploration
    
    # Stability
    experiment_config.lr = 5e-4
    experiment_config.clip_grad_norm = True
    experiment_config.clip_grad_val = 10.0
    experiment_config.adam_eps = 1e-5
    
    # Evaluation & logging
    experiment_config.evaluation = True
    experiment_config.evaluation_interval = 8_000  # Every 10 batches
    experiment_config.evaluation_episodes = 10
    experiment_config.evaluation_deterministic_actions = True
    
    # Checkpointing
    experiment_config.checkpoint_interval = 100_000
    experiment_config.checkpoint_at_end = True
    experiment_config.keep_checkpoints_num = 3
    
    experiment_config.render = False
    experiment_config.loggers = ["csv", "tensorboard"]
    experiment_config.save_folder = "./outputs/smaclite_qmix"
    
    experiment = Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=model_config,
        seed=42,
        config=experiment_config,
    )
    
    experiment.run()

def run_both_experiments():
    """Run both MAPPO and QMIX sequentially."""
    print("\n" + "=" * 60)
    print("STARTING SMACLITE BENCHMARK: MAPPO + QMIX on 2s3z")
    print("=" * 60 + "\n")
    
    run_mappo_experiment()
    print("\n" + "-" * 60 + "\n")
    run_qmix_experiment()
    
    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE!")
    print("Results saved to ./outputs/")
    print("=" * 60)


def quick_test():
    print("Running quick integration test...")
    
    # Create environment directly
    env = SMACliteTorchRLWrapper(env_name="smaclite/2s3z-v0", seed=0)
    
    # Do a short rollout
    print(f"Environment: {env._env_name}")
    print(f"Agents: {env.n_agents}, Obs: {env._obs_dim}, Actions: {env._n_actions}")
    
    # Option 1: Use rollout with auto_reset=True (default)
    td = env.rollout(max_steps=10)
    print(f"Rollout shape: {td.shape}")
    print(f"Total reward: {td['next', 'agents', 'reward'].sum().item():.3f}")
    
    # Option 2: Manual rollout loop (for demonstration)
    print("\nManual rollout loop:")
    td = env.reset()
    total_reward = 0.0
    for step in range(10):
        action_td = env.sample_action(td)
        td = td.update(action_td)
        td_step = env.step(td)
        
        reward = td_step["next", "agents", "reward"].sum().item()
        done = td_step["next", "done"].item()
        total_reward += reward
        
        if done:
            print(f"  Episode ended at step {step+1}")
            break
        
        td = step_mdp(td_step)
    
    print(f"  Total reward (manual): {total_reward:.3f}")
    
    env.close()
    print("\nQuick test passed!")


def test_benchmarl_task():
    print("\nTesting BenchMARL Task integration...")
    
    task = SMACliteTask.TWO_S_THREE_Z.get_from_yaml()
    print(f"Task: {task}")
    print(f"Config: {task.config}")
    
    # Get environment function
    env_fn = task.get_env_fun(
        num_envs=1,
        continuous_actions=False,
        seed=42,
        device="cpu"
    )
    
    # Create environment
    env = env_fn()
    
    print(f"\nEnvironment created successfully!")
    print(f"  group_map: {task.group_map(env)}")
    print(f"  max_steps: {task.max_steps(env)}")
    print(f"  supports_discrete: {task.supports_discrete_actions()}")
    print(f"  supports_continuous: {task.supports_continuous_actions()}")
    
    # Quick rollout
    td = env.rollout(max_steps=5)
    print(f"  Rollout reward: {td['next', 'agents', 'reward'].sum().item():.3f}")
    
    env.close()
    print("\nBenchMARL Task test passed!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SMACLite + BenchMARL experiments")
    parser.add_argument(
        "--mode", 
        choices=["test", "task_test", "mappo", "qmix", "both"],
        default="test", 
        help="Experiment mode: test (quick check), task_test (BenchMARL task), mappo, qmix, or both"
    )
    args = parser.parse_args()
    
    if args.mode == "test":
        quick_test()
    elif args.mode == "task_test":
        test_benchmarl_task()
    elif args.mode == "mappo":
        run_mappo_experiment()
    elif args.mode == "qmix":
        run_qmix_experiment()
    elif args.mode == "both":
        run_both_experiments()