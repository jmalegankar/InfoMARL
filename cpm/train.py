"""
CPM Training Script for VMAS
Fixed version with all critical issues addressed
"""

import argparse
import torch
import os
import numpy as np
from pathlib import Path
from torch.autograd import Variable
import gymnasium as gym
from tensorboardX import SummaryWriter
from gymnasium.spaces import Discrete

from algorithms.attention_sac import AttentionSAC
from utils.buffer import ReplayBuffer
from vmas_cpm_wrapper import make_parallel_env_for_cpm


def evaluate(model, env, n_eval_episodes=10, max_steps=400):
    """
    Evaluate the model without exploration noise.
    Returns mean episode reward across all agents.
    """
    total_rewards = []
    total_envs = env.num_envs * env.num_vmas_envs_per_wrapper

    model.prep_rollouts(device='cpu')

    for _ in range(n_eval_episodes):
        obs = env.reset()
        episode_rewards = np.zeros((total_envs, env.n))

        # Initialize previous actions
        previous_actions = []
        for ac_sp in env.action_space:
            if isinstance(ac_sp, Discrete):
                prev_act = torch.zeros((total_envs, ac_sp.n))
            else:
                prev_act = torch.zeros((total_envs, ac_sp.shape[0]))
            previous_actions.append(prev_act)

        for _ in range(max_steps):
            # Convert observations to torch
            torch_obs = []
            for i in range(model.nagents):
                agent_obs = obs[:, i, :]
                torch_obs.append(Variable(torch.Tensor(agent_obs), requires_grad=False))

            # Get predicted actions
            predict_acs = model.pre_step(torch_obs, previous_actions)
            torch_predict_actions = [
                torch.stack(predict_ac, dim=1)
                for predict_ac in predict_acs
            ]

            # Get actions WITHOUT exploration
            torch_agent_outs = model.step(
                torch_obs,
                torch_predict_actions,
                explore=False,  # No exploration during eval
                return_all_probs=True
            )
            torch_agent_actions = [out[0] for out in torch_agent_outs]
            torch_agent_probs = [out[1] for out in torch_agent_outs]

            # Convert actions to numpy
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            actions = np.stack([ac for ac in agent_actions], axis=1)

            # Step environment
            obs, rewards, dones, _ = env.step(actions)
            episode_rewards += rewards

            # Update previous actions
            previous_actions = torch_agent_probs.copy()

            if dones.all():
                break

        total_rewards.append(episode_rewards.mean(axis=0))

    # Return mean reward per agent
    mean_rewards = np.array(total_rewards).mean(axis=0)
    return mean_rewards


def run(config):
    """Main training loop for CPM on VMAS scenarios."""
    
    # Set up directories
    model_dir = Path('./models_cpm') / config.scenario / f'cpm_{config.scenario}'
    if not model_dir.exists():
        run_num = 1
    else:
        exst_run_nums = [
            int(str(folder.name).split('run')[1]) 
            for folder in model_dir.iterdir() 
            if str(folder.name).startswith('run')
        ]
        if len(exst_run_nums) == 0:
            run_num = 1
        else:
            run_num = max(exst_run_nums) + 1
    
    curr_run = f'run{run_num}'
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir, exist_ok=True)
    logger = SummaryWriter(str(log_dir))
    
    # Set random seeds
    torch.manual_seed(run_num)
    np.random.seed(run_num)
    
    # Create VMAS environment wrapped for CPM
    print("Creating environment...")
    env_kwargs = {}
    if config.scenario == "food_collection":
        env_kwargs["n_food"] = config.n_food if config.n_food is not None else config.n_agents
    
    env = make_parallel_env_for_cpm(
        scenario=config.scenario,
        n_agents=config.n_agents,
        n_rollout_threads=config.n_rollout_threads,
        seed=run_num,
        max_steps=config.max_steps,
        device=config.device,
        **env_kwargs
    )
    
    # Calculate total environments
    total_envs = env.num_envs * env.num_vmas_envs_per_wrapper
    
    print(f"Environment created: {config.scenario}")
    print(f"  Agents: {env.n}")
    print(f"  Total envs: {total_envs}")
    print(f"  Observation space: {[obs.shape for obs in env.observation_space]}")
    print(f"  Action space: {[act.n for act in env.action_space]}")
    
    # Initialize CPM model with error handling
    print("Initializing CPM model...")
    try:
        model = AttentionSAC.init_from_env(
            env,
            tau=config.tau,
            pi_lr=config.pi_lr,
            q_lr=config.q_lr,
            pre_lr=config.pre_lr,
            gamma=config.gamma,
            pol_hidden_dim=config.pol_hidden_dim,
            critic_hidden_dim=config.critic_hidden_dim,
            attend_heads=config.attend_heads,
            reward_scale=config.reward_scale,
        )
        print("  ✓ Model initialized successfully")
    except Exception as e:
        print(f"  ✗ Failed to create model: {e}")
        print("  Check that environment spaces are compatible")
        raise
    
    # Initialize replay buffer
    replay_buffer = ReplayBuffer(
        config.buffer_length, 
        model.nagents,
        [obsp.shape[0] for obsp in env.observation_space],
        [acsp.n for acsp in env.action_space]
    )
    
    t = 0
    best_reward = float('-inf')
    num_updates_performed = 0

    print(f"\nStarting training for {config.n_episodes} episodes...")
    print("="*60)

    # Training loop
    for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
        print(f"\nEpisode {ep_i + 1}/{config.n_episodes} (batch {ep_i//config.n_rollout_threads + 1})")
        
        obs = env.reset()  # Shape: (total_envs, n_agents, obs_dim)
        
        # Initialize previous actions (FIXED: handle total_envs correctly)
        previous_actions = []
        for ac_sp in env.action_space:
            if isinstance(ac_sp, Discrete):
                prev_act = torch.zeros((total_envs, ac_sp.n))
            else:
                prev_act = torch.zeros((total_envs, ac_sp.shape[0]))
            previous_actions.append(prev_act)
        
        model.prep_rollouts(device='cpu')
        
        for et_i in range(config.max_steps):
            # Convert observations to torch (FIXED: proper indexing)
            torch_obs = []
            for i in range(model.nagents):
                agent_obs = obs[:, i, :]  # (total_envs, obs_dim)
                torch_obs.append(Variable(torch.Tensor(agent_obs), requires_grad=False))
            
            # Get predicted actions of other agents (CPM's peer modeling)
            predict_acs = model.pre_step(torch_obs, previous_actions)
            torch_predict_actions = [
                torch.stack(predict_ac, dim=1) 
                for predict_ac in predict_acs
            ]
            predict_actions = [preac.data.numpy() for preac in torch_predict_actions]
            
            # Get actions from policy
            torch_agent_outs = model.step(
                torch_obs, 
                torch_predict_actions,
                explore=True, 
                return_all_probs=True
            )
            torch_agent_actions = [out[0] for out in torch_agent_outs]
            torch_agent_probs = [out[1] for out in torch_agent_outs]
            agent_probs = [ac.data.numpy() for ac in torch_agent_probs]
            
            # Convert actions to numpy
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            
            # Rearrange actions to be per environment
            # CPM expects actions as (total_envs, n_agents)
            actions = np.stack([ac for ac in agent_actions], axis=1)
            
            # Step environment
            next_obs, rewards, dones, infos = env.step(actions)
            
            # Get next predicted actions
            next_torch_obs = []
            for i in range(model.nagents):
                agent_obs = next_obs[:, i, :]
                next_torch_obs.append(Variable(torch.Tensor(agent_obs), requires_grad=False))
            
            previous_actions = torch_agent_probs.copy()
            next_predict_acs = model.pre_step(next_torch_obs, previous_actions)
            next_torch_predict_actions = [
                torch.stack(predict_ac, dim=1) 
                for predict_ac in next_predict_acs
            ]
            next_predict_actions = [
                preac.data.numpy() 
                for preac in next_torch_predict_actions
            ]
            
            # Store in replay buffer
            replay_buffer.push(
                obs, predict_actions, agent_actions, agent_probs, 
                rewards, next_obs, next_predict_actions, dones
            )
            
            obs = next_obs
            t += total_envs
            
            # Update model
            if (len(replay_buffer) >= config.batch_size and
                (t % config.steps_per_update) < total_envs):
                
                if config.use_gpu:
                    model.prep_training(device='gpu')
                else:
                    model.prep_training(device='cpu')
                
                for u_i in range(config.num_updates):
                    sample = replay_buffer.sample(
                        config.batch_size,
                        to_gpu=config.use_gpu,
                        norm_rews=config.normalize_rewards
                    )
                    model.update_critic(sample, logger=logger)
                    model.update_policies(sample, logger=logger)
                    model.update_predict(sample, logger=logger)
                    model.update_all_targets()
                    num_updates_performed += 1

                logger.add_scalar('training/buffer_size', len(replay_buffer), ep_i)
                logger.add_scalar('training/num_updates', num_updates_performed, ep_i)
                model.prep_rollouts(device='cpu')
        
        # Log episode statistics
        ep_rews = replay_buffer.get_average_rewards(
            config.max_steps * total_envs
        )
        
        mean_episode_reward = sum(ep_rews) * config.max_steps
        
        for a_i, a_ep_rew in enumerate(ep_rews):
            logger.add_scalar(
                f'agent{a_i}/mean_episode_rewards',
                a_ep_rew * config.max_steps, 
                ep_i
            )
        
        logger.add_scalar('sum_episode_rewards', mean_episode_reward, ep_i)
        print(f"  Mean episode reward: {mean_episode_reward:.2f}")

        # Periodic evaluation
        if ep_i % config.eval_interval < config.n_rollout_threads:
            print(f"  Running evaluation...")
            eval_rewards = evaluate(model, env, n_eval_episodes=5, max_steps=config.max_steps)
            eval_mean = eval_rewards.mean()
            logger.add_scalar('eval/mean_reward', eval_mean, ep_i)
            for a_i, a_rew in enumerate(eval_rewards):
                logger.add_scalar(f'eval/agent{a_i}_reward', a_rew, ep_i)
            print(f"  Eval mean reward: {eval_mean:.2f}")

            # Save best model based on evaluation performance
            if eval_mean > best_reward:
                best_reward = eval_mean
                model.prep_rollouts(device='cpu')
                model.save(run_dir / 'model_best.pt')
                print(f"  ✓ New best! Saved to {run_dir / 'model_best.pt'}")
        else:
            # Fallback: save if training reward improves (less reliable)
            if mean_episode_reward > best_reward and ep_i % config.eval_interval >= config.n_rollout_threads:
                model.prep_rollouts(device='cpu')
                model.save(run_dir / 'model_checkpoint.pt')
        
        # Periodic saves
        if ep_i % config.save_interval < config.n_rollout_threads:
            model.prep_rollouts(device='cpu')
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            model.save(run_dir / 'incremental' / f'model_ep{ep_i + 1}.pt')
            model.save(run_dir / 'model.pt')
    
    # Final save
    model.save(run_dir / 'model.pt')
    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best reward: {best_reward:.2f}")
    print(f"Model saved to: {run_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train CPM on VMAS scenarios")
    
    # Environment
    parser.add_argument("--scenario", type=str, default="simple_spread",
                       help="VMAS scenario (simple_spread, food_collection)")
    parser.add_argument("--n_agents", type=int, default=4,
                       help="Number of agents")
    parser.add_argument("--n_food", type=int, default=None,
                       help="Number of food items (food_collection only, defaults to n_agents)")
    parser.add_argument("--max_steps", type=int, default=400,
                       help="Maximum steps per episode")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device for VMAS (cpu/cuda)")
    
    # Training
    parser.add_argument("--n_rollout_threads", type=int, default=8,
                       help="Number of parallel environments")
    parser.add_argument("--n_episodes", type=int, default=50000,
                       help="Total training episodes")
    parser.add_argument("--buffer_length", type=int, default=int(1e6),
                       help="Replay buffer size")
    parser.add_argument("--steps_per_update", type=int, default=100,
                       help="Steps between updates")
    parser.add_argument("--num_updates", type=int, default=4,
                       help="Number of updates per cycle")
    parser.add_argument("--batch_size", type=int, default=1024,
                       help="Batch size for training")
    parser.add_argument("--save_interval", type=int, default=1000,
                       help="Episodes between saves")
    parser.add_argument("--eval_interval", type=int, default=500,
                       help="Episodes between evaluations")
    
    # Model hyperparameters
    parser.add_argument("--pol_hidden_dim", type=int, default=128,
                       help="Policy hidden dimension")
    parser.add_argument("--critic_hidden_dim", type=int, default=128,
                       help="Critic hidden dimension")
    parser.add_argument("--attend_heads", type=int, default=4,
                       help="Number of attention heads")
    parser.add_argument("--pi_lr", type=float, default=0.001,
                       help="Policy learning rate")
    parser.add_argument("--q_lr", type=float, default=0.001,
                       help="Critic learning rate")
    parser.add_argument("--pre_lr", type=float, default=0.01,
                       help="Prediction network learning rate")
    parser.add_argument("--tau", type=float, default=0.001,
                       help="Target network update rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                       help="Discount factor")
    parser.add_argument("--reward_scale", type=float, default=100.0,
                       help="Reward scaling factor")
    parser.add_argument("--use_gpu", action='store_true',
                       help="Use GPU for training")
    parser.add_argument("--normalize_rewards", action='store_true',
                       help="Normalize rewards in replay buffer (can be unstable early in training)")

    args = parser.parse_args()
    
    print("="*60)
    print("CPM Training on VMAS")
    print("="*60)
    print(f"Scenario: {args.scenario}")
    print(f"Agents: {args.n_agents}")
    if args.scenario == "food_collection":
        n_food_display = args.n_food if args.n_food is not None else args.n_agents
        print(f"Food items: {n_food_display}")
    print(f"Episodes: {args.n_episodes}")
    print(f"Parallel envs: {args.n_rollout_threads}")
    print(f"Device: {args.device}")
    print("="*60)
    
    run(args)