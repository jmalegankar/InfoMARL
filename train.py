import torch as th
import numpy as np
import argparse
import os
from tqdm import tqdm
import datetime

# Local imports
from agents.randomized_attention_policy import RandomizedAttentionAgent
from buffers.buffer import ReplayBuffer, StateBuffer
from utils.env_wrappers import make_env_for_training, make_env_for_evaluation
from utils.training_utils import update_target_networks, save_model, load_model
from utils.evaluation import evaluate_policy

def parse_args():
    parser = argparse.ArgumentParser("Randomized Attention MARL Training")
    
    # Environment
    parser.add_argument("--scenario", type=str, default="simple_spread", help="name of the scenario")
    parser.add_argument("--n-agents", type=int, default=5, help="number of agents")
    parser.add_argument("--max-episode-len", type=int, default=100, help="maximum episode length")
    parser.add_argument("--num-envs", type=int, default=8, help="number of parallel environments")
    
    # Core training parameters
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.005, help="soft target update parameter")
    parser.add_argument("--batch-size", type=int, default=128, help="batch size")
    parser.add_argument("--hidden-dim", type=int, default=64, help="hidden dimension")
    parser.add_argument("--total-frames", type=int, default=1000000, help="total frames to train")
    parser.add_argument("--frames-per-batch", type=int, default=1000, help="frames per batch")
    parser.add_argument("--optim-steps-per-batch", type=int, default=10, help="optimization steps per batch")
    parser.add_argument("--buffer-size", type=int, default=100000, help="replay buffer size")
    parser.add_argument("--start-training-after", type=int, default=10000, help="start training after this many frames")
    
    # SAC specific parameters
    parser.add_argument("--alpha-init", type=float, default=0.2, help="initial alpha value")
    parser.add_argument("--min-alpha", type=float, default=0.01, help="minimum alpha value")
    parser.add_argument("--max-alpha", type=float, default=1.0, help="maximum alpha value")
    parser.add_argument("--lr-alpha", type=float, default=3e-4, help="learning rate for alpha")
    parser.add_argument("--reward-scaling", type=float, default=1.0, help="reward scaling factor")
    
    # Saving and evaluation
    parser.add_argument("--save-dir", type=str, default="./results", help="directory to save agent")
    parser.add_argument("--save-interval", type=int, default=10000, help="save interval")
    parser.add_argument("--eval-interval", type=int, default=10000, help="evaluation interval")
    parser.add_argument("--num-eval-episodes", type=int, default=10, help="number of episodes for evaluation")
    parser.add_argument("--save-gifs", action="store_true", default=False, help="save GIFs during evaluation")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--cuda", action="store_true", default=False, help="use CUDA")
    parser.add_argument("--exp-name", type=str, default=None, help="experiment name")
    
    return parser.parse_args()

def create_models(observation_dim, action_dim, n_agents, hidden_dim, device):
    """Create policy and critic models"""
    # Create RandomizedAttention agent with proper dimensions
    agent = RandomizedAttentionAgent(
        observation_dim=observation_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        device=device
    ).to(device)
    
    
    class MLPQFunction(th.nn.Module):
        def __init__(self, obs_dim, act_dim, hidden_dim=64):
            super().__init__()
            self.q = th.nn.Sequential(
                th.nn.Linear(obs_dim + act_dim, hidden_dim),
                th.nn.ReLU(),
                th.nn.Linear(hidden_dim, hidden_dim),
                th.nn.ReLU(),
                th.nn.Linear(hidden_dim, 1)
            )
        
        def forward(self, obs_dict, act):
            # Process dictionary observations
            obs = obs_dict['obs']
            batch_size = obs.shape[0]
            obs_flat = obs.reshape(batch_size, -1)
            act_flat = act.reshape(batch_size, -1)
            q_input = th.cat([obs_flat, act_flat], dim=1)
            return self.q(q_input)
    
    # Create critic networks
    critic1 = MLPQFunction(observation_dim, action_dim, hidden_dim).to(device)
    critic2 = MLPQFunction(observation_dim, action_dim, hidden_dim).to(device)
    target_critic1 = MLPQFunction(observation_dim, action_dim, hidden_dim).to(device)
    target_critic2 = MLPQFunction(observation_dim, action_dim, hidden_dim).to(device)
    
    # Copy parameters to target networks
    target_critic1.load_state_dict(critic1.state_dict())
    target_critic2.load_state_dict(critic2.state_dict())
    
    return agent, critic1, critic2, target_critic1, target_critic2

def compute_critic_loss(critic1, critic2, target_critic1, target_critic2, agent, 
                         state_buffer, gamma, alpha, device):
    """Compute critic loss for SAC"""
    with th.no_grad():
        # Sample actions from policy for next observations
        next_action_dist = agent(state_buffer.next_obs)
        next_actions = next_action_dist.sample()
        next_log_probs = next_action_dist.log_prob(next_actions).sum(dim=-1)
        
        # Compute target Q-value
        next_q1 = target_critic1(state_buffer.next_obs, next_actions)
        next_q2 = target_critic2(state_buffer.next_obs, next_actions)
        next_q = th.min(next_q1, next_q2) - alpha * next_log_probs.unsqueeze(-1)
        target_q = state_buffer.reward.unsqueeze(-1) + gamma * (1 - state_buffer.done.unsqueeze(-1)) * next_q
    
    # Compute current Q-values
    current_q1 = critic1(state_buffer.obs, state_buffer.action)
    current_q2 = critic2(state_buffer.obs, state_buffer.action)
    
    # Compute critic losses (MSE)
    critic1_loss = th.nn.functional.mse_loss(current_q1, target_q)
    critic2_loss = th.nn.functional.mse_loss(current_q2, target_q)
    
    return critic1_loss + critic2_loss

def compute_actor_loss(critic1, critic2, agent, state_buffer, alpha, device):
    """Compute actor loss for SAC"""
    # Sample actions from policy
    action_dist = agent(state_buffer.obs)
    actions = action_dist.sample()
    log_probs = action_dist.log_prob(actions).sum(dim=-1, keepdim=True)
    
    # Compute min Q-value
    q1 = critic1(state_buffer.obs, actions)
    q2 = critic2(state_buffer.obs, actions)
    min_q = th.min(q1, q2)
    
    # Compute actor loss
    actor_loss = (alpha * log_probs - min_q).mean()
    
    return actor_loss, log_probs

def main():
    args = parse_args()
    
    # Set seed for reproducibility
    th.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda and th.cuda.is_available():
        device = th.device("cuda")
        th.cuda.manual_seed_all(args.seed)
    else:
        device = th.device("cpu")
    
    # Create experiment directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = args.exp_name or f"{args.scenario}_{args.n_agents}_{timestamp}"
    save_dir = os.path.join(args.save_dir, exp_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # Create directories for artifacts
    model_dir = os.path.join(save_dir, "models")
    gif_dir = os.path.join(save_dir, "gifs")
    log_dir = os.path.join(save_dir, "logs")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(gif_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Create environments
    envs = make_env_for_training(args, device)
    
    # Get observation and action dimensions
    observation_dim = envs.observation_dim
    action_dim = envs.action_dim
    
    # Create agent and critic networks
    agent, critic1, critic2, target_critic1, target_critic2 = create_models(
        observation_dim=observation_dim * args.n_agents, 
        action_dim=action_dim * args.n_agents,
        n_agents=args.n_agents,
        hidden_dim=args.hidden_dim, 
        device=device
    )
    
    # Initialize replay buffer
    buffer = ReplayBuffer(args.buffer_size, device)
    
    # Setup optimizers
    policy_optimizer = th.optim.Adam(agent.parameters(), lr=args.lr)
    critic_optimizer = th.optim.Adam(
        list(critic1.parameters()) + list(critic2.parameters()), 
        lr=args.lr
    )
    
    # Setup temperature (alpha) optimization
    log_alpha = th.tensor(np.log(args.alpha_init), requires_grad=True, device=device)
    alpha_optimizer = th.optim.Adam([log_alpha], lr=args.lr_alpha)
    target_entropy = -action_dim  # Heuristic
    
    # Training loop variables
    total_frames = 0
    episode_rewards = []
    current_episode_rewards = th.zeros(args.num_envs, device=device)
    training_stats = {"rewards": [], "critic_losses": [], "actor_losses": [], "alpha_losses": []}
    
    # Reset environments
    obs = envs.reset()
    
    # Main training loop
    progress_bar = tqdm(total=args.total_frames)
    
    while total_frames < args.total_frames:
        # Sample actions
        with th.no_grad():
            obs_dict = agent.prepare_obs_dict(obs)
            action_dist = agent(obs_dict)
            actions = action_dist.sample()
        
        # Execute actions
        next_obs, rewards, dones, _ = envs.step(actions)
        
        # Track rewards
        current_episode_rewards += rewards.sum(dim=1)
        
        # Store transition in replay buffer
        next_obs_dict = agent.prepare_obs_dict(next_obs)
        
        buffer.add(
            obs=obs_dict,
            action=actions, 
            reward=rewards.sum(dim=1),  # Sum rewards across agents
            next_obs=next_obs_dict,
            done=dones
        )
        
        # Update observation
        obs = next_obs
        
        # Handle episode termination
        for i, done in enumerate(dones):
            if done:
                episode_rewards.append(current_episode_rewards[i].item())
                training_stats["rewards"].append(current_episode_rewards[i].item())
                current_episode_rewards[i] = 0
                
                # Reset environment if needed
                if hasattr(envs, 'reset_at'):
                    obs_i = envs.reset_at(i)
                    obs[i] = obs_i
        
        # Increment frame counter
        batch_frames = args.num_envs
        total_frames += batch_frames
        progress_bar.update(batch_frames)
        
        # Perform optimization
        if len(buffer) >= args.batch_size and total_frames >= args.start_training_after:
            if total_frames % args.frames_per_batch == 0:
                for _ in range(args.optim_steps_per_batch):
                    # Sample from replay buffer
                    batch = buffer.sample(args.batch_size)
                    
                    # Update critics
                    critic_loss = compute_critic_loss(
                        critic1, critic2, target_critic1, target_critic2,
                        agent, batch, args.gamma, th.exp(log_alpha).detach(), device
                    )
                    
                    critic_optimizer.zero_grad()
                    critic_loss.backward()
                    critic_optimizer.step()
                    
                    # Update actor
                    actor_loss, log_probs = compute_actor_loss(
                        critic1, critic2, agent, batch, th.exp(log_alpha).detach(), device
                    )
                    
                    policy_optimizer.zero_grad()
                    actor_loss.backward()
                    policy_optimizer.step()
                    
                    # Update temperature (alpha)
                    alpha_loss = -(log_alpha * (log_probs.detach() + target_entropy)).mean()
                    
                    alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    alpha_optimizer.step()
                    
                    # Clamp alpha
                    with th.no_grad():
                        log_alpha.clamp_(np.log(args.min_alpha), np.log(args.max_alpha))
                    
                    # Update target networks
                    update_target_networks(target_critic1, critic1, args.tau)
                    update_target_networks(target_critic2, critic2, args.tau)
                    
                    # Track statistics
                    training_stats["critic_losses"].append(critic_loss.item())
                    training_stats["actor_losses"].append(actor_loss.item())
                    training_stats["alpha_losses"].append(alpha_loss.item())
        
        # Save checkpoint
        if total_frames % args.save_interval == 0 and total_frames > 0:
            save_path = os.path.join(model_dir, f"checkpoint_{total_frames}.pt")
            save_model({
                "agent": agent.state_dict(),
                "critic1": critic1.state_dict(),
                "critic2": critic2.state_dict(),
                "target_critic1": target_critic1.state_dict(),
                "target_critic2": target_critic2.state_dict(),
                "log_alpha": log_alpha.item(),
                "frames": total_frames
            }, save_path)
            
            # Log training progress
            tqdm.write(f"\nFrame {total_frames}: Saving model to {save_path}")
            tqdm.write(f"Recent rewards: {np.mean(episode_rewards[-10:]):.2f}")
            tqdm.write(f"Alpha: {th.exp(log_alpha).item():.4f}")
        
        # Evaluation
        if total_frames % args.eval_interval == 0 and total_frames > 0:
            tqdm.write(f"\nFrame {total_frames}: Evaluating policy...")
            
            # Create evaluation environment
            eval_env = make_env_for_evaluation(args, device)
            
            # Evaluate policy
            eval_rewards, eval_frames = evaluate_policy(
                eval_env, agent, args.num_eval_episodes, args.max_episode_len, 
                device, args.save_gifs, os.path.join(gif_dir, f"eval_{total_frames}")
            )
            
            mean_reward = np.mean(eval_rewards)
            std_reward = np.std(eval_rewards)
            
            tqdm.write(f"Evaluation over {args.num_eval_episodes} episodes: {mean_reward:.2f} ± {std_reward:.2f}")
    
    # Save final model
    final_path = os.path.join(model_dir, "final_model.pt")
    save_model({
        "agent": agent.state_dict(),
        "critic1": critic1.state_dict(),
        "critic2": critic2.state_dict(),
        "target_critic1": target_critic1.state_dict(),
        "target_critic2": target_critic2.state_dict(),
        "log_alpha": log_alpha.item(),
        "frames": total_frames
    }, final_path)
    
    tqdm.write(f"\nTraining completed. Final model saved to {final_path}")
    
    # Final evaluation
    eval_env = make_env_for_evaluation(args, device)
    eval_rewards, _ = evaluate_policy(
        eval_env, agent, args.num_eval_episodes, args.max_episode_len, 
        device, True, os.path.join(gif_dir, "final_eval")
    )
    
    mean_reward = np.mean(eval_rewards)
    std_reward = np.std(eval_rewards)
    
    tqdm.write(f"Final evaluation: {mean_reward:.2f} ± {std_reward:.2f}")

if __name__ == "__main__":
    main()