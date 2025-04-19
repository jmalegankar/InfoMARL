import torch
import torch.nn.functional as F
import torch.optim as optim
import vmas
from actor import RandomAgentPolicy
from critic import RAP_qvalue
from buffer import ReplayBuffer

from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

import imageio
import numpy as np
import random


# TensorBoard init
current_time = datetime.now().strftime('%b%d_%H-%M-%S')
log_dir = os.path.join('runs', f'vmas_simple_spread_{current_time}')
writer = SummaryWriter(log_dir=log_dir)

# HYPERPARAMETERS
num_envs = 32
number_agents = 2
total_steps = 400000000
checkpoint_interval = 50000
gif_save_interval = 5000
batch_size = 1024
gamma = 0.99
tau = 0.005
actor_lr = 1e-4
critic_lr = 1e-4
alpha_lr = 1e-5
update_every = 96*5
num_updates = 1
initial_alpha = 0.5
alpha_min = 0.1
alpha_max = 1.0
target_entropy = -2.0
max_steps_per_episode = 400
critic_only_steps = 10

checkpoint_dir = os.path.join(log_dir, 'checkpoints')
os.makedirs(checkpoint_dir, exist_ok=True)
resume_training = False 
checkpoint_path = None

# Log hyperparameters to tensorboard
writer.add_text('Hyperparameters/num_envs', str(num_envs))
writer.add_text('Hyperparameters/number_agents', str(number_agents))
writer.add_text('Hyperparameters/total_steps', str(total_steps))
writer.add_text('Hyperparameters/max_steps_per_episode', str(400))
writer.add_text('Hyperparameters/batch_size', str(batch_size))
writer.add_text('Hyperparameters/gamma', str(gamma))
writer.add_text('Hyperparameters/tau', str(tau))
writer.add_text('Hyperparameters/actor_lr', str(actor_lr))
writer.add_text('Hyperparameters/critic_lr', str(critic_lr))
writer.add_text('Hyperparameters/alpha_lr', str(alpha_lr))
writer.add_text('Hyperparameters/initial_alpha', str(initial_alpha))
writer.add_text('Hyperparameters/target_entropy', str(target_entropy))
writer.add_text('Hyperparameters/alpha_min', str(alpha_min))
writer.add_text('Hyperparameters/alpha_max', str(alpha_max))

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set seed for reproducibility
seed = 42
set_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer.add_text('System/device', str(device))

gifs_dir = os.path.join(log_dir, 'gifs')
os.makedirs(gifs_dir, exist_ok=True)

# Environment
env = vmas.make_env(
    scenario="simple_spread",
    num_envs=num_envs,
    n_agents=number_agents,
    continuous_actions=True,
    max_steps=max_steps_per_episode,
    seed=seed,
)

# Initialize environment
all_obs = env.reset()
obs_dim = all_obs[0][0].shape[0]
action_dim = 2
agent_dim = 4
hidden_dim = 32
landmark_dim = 2 * number_agents
other_agent_dim = 2 * (number_agents - 1)

# Networks
actor = RandomAgentPolicy(
    number_agents, 
    agent_dim=4, 
    landmark_dim=landmark_dim, 
    other_agent_dim=other_agent_dim,
    hidden_dim=hidden_dim
).to(device)

qvalue_config = {
    "device": device,
    "n_agents": number_agents,
    "observation_dim_per_agent": obs_dim,
    "action_dim_per_agent": action_dim
}
critic = RAP_qvalue(qvalue_config).to(device)

target_critic = RAP_qvalue(qvalue_config).to(device)
target_critic.load_state_dict(critic.state_dict())
target_critic.eval()

# Optimizers
actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr)
critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr)

# Temperature parameter alpha
alpha = torch.tensor(initial_alpha, device=device, requires_grad=True, dtype=torch.float32) 
alpha_optimizer = optim.Adam([alpha], lr=alpha_lr)

# Replay buffer
buffer_capacity = 2000000
replay_buffer = ReplayBuffer(buffer_capacity, obs_dim, action_dim, number_agents, device)

# Skip adding actor graph to tensorboard to avoid TracerWarning errors
# The warnings are caused by non-deterministic operations in the actor network
# Commented out code:
# dummy_obs = torch.zeros(1, obs_dim).to(device)
# dummy_rand = torch.zeros(1, number_agents).to(device)
# writer.add_graph(actor, [dummy_obs, dummy_rand])

def get_permuted_env_random_numbers(env_random_numbers, number_agents, num_envs):
    """
    Permute the random numbers for each agent in the environment.
    """
    permutation_indices = torch.zeros(number_agents, number_agents, dtype=torch.long, device=device)
    for i in range(number_agents):
        other_agents = sorted([j for j in range(number_agents) if j != i])
        permutation_indices[i] = torch.tensor([i] + other_agents)
    expanded_rand = env_random_numbers.unsqueeze(1).expand(-1, number_agents, -1)
    permuted_rand = torch.gather(expanded_rand, dim=2, index=permutation_indices.unsqueeze(0).expand(num_envs, -1, -1))
    return permuted_rand

def save_checkpoint(actor, critic, target_critic, actor_optimizer, critic_optimizer, 
                   alpha, alpha_optimizer, replay_buffer, global_step, 
                   update_step, best_reward=None):
    """
    Save a checkpoint of the training state.
    """
    checkpoint = {
        'actor_state_dict': actor.state_dict(),
        'critic_state_dict': critic.state_dict(),
        'target_critic_state_dict': target_critic.state_dict(),
        'actor_optimizer_state_dict': actor_optimizer.state_dict(),
        'critic_optimizer_state_dict': critic_optimizer.state_dict(),
        'alpha': alpha.item(),
        'alpha_optimizer_state_dict': alpha_optimizer.state_dict(),
        'replay_buffer': replay_buffer.get_save_state(),
        'global_step': global_step,
        'update_step': update_step,
        'seed': seed,
        'best_reward': best_reward
    }
    
    # Regular checkpoint
    checkpoint_filename = os.path.join(checkpoint_dir, f'checkpoint_step_{global_step}.pt')
    torch.save(checkpoint, checkpoint_filename)
    print(f"Checkpoint saved to {checkpoint_filename}")
    
    # Always save the latest checkpoint for easy resume
    latest_checkpoint = os.path.join(checkpoint_dir, 'checkpoint_latest.pt')
    torch.save(checkpoint, latest_checkpoint)
    
    # If this is a best reward checkpoint, save it separately
    if best_reward is not None:
        best_checkpoint = os.path.join(checkpoint_dir, 'checkpoint_best.pt')
        torch.save(checkpoint, best_checkpoint)
        print(f"New best reward {best_reward:.2f} - saved best checkpoint")

def load_checkpoint(checkpoint_path):
    """
    Load a checkpoint and return the saved states.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    return checkpoint

# Initialize or load from checkpoint
global_step = 0
update_step = 0
best_reward = float('-inf')
render_env = None

if resume_training:
    if checkpoint_path is None:
        checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pt')
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = load_checkpoint(checkpoint_path)
        
        actor.load_state_dict(checkpoint['actor_state_dict'])
        critic.load_state_dict(checkpoint['critic_state_dict'])
        target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        # Load alpha related parameters
        loaded_alpha = torch.tensor(checkpoint['alpha'], device=device)
        loaded_alpha = torch.clamp(loaded_alpha, min=alpha_min, max=alpha_max)
        alpha = loaded_alpha
        alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        
        # Load replay buffer if it exists in the checkpoint
        if 'replay_buffer' in checkpoint:
            replay_buffer.load_save_state(checkpoint['replay_buffer'])
        
        global_step = checkpoint['global_step']
        update_step = checkpoint['update_step']
        
        if 'best_reward' in checkpoint:
            best_reward = checkpoint['best_reward']
        
        print(f"Resuming from global step {global_step}, update step {update_step}")
        print(f"Current best reward: {best_reward:.2f}")
        print(f"Current alpha value: {alpha.item():.6f}")
    else:
        print(f"No checkpoint found at {checkpoint_path}, starting from scratch.")

# Metrics for tracking
running_reward = 0.0
reward_window_size = 100
reward_window = []
update_metrics = {
    'actor_losses': [],
    'critic_losses': [],
    'q_values': [],
    'log_probs': [],
    'alpha_losses': [],
    'alpha_values': []
}

# Initialize observation
all_obs = env.reset()
obs_batched = torch.stack(all_obs, dim=1).view(-1, obs_dim).to(device)

# Main training loop
while global_step < total_steps:
    # Collect a step from the environment
    actor.eval()
    critic.eval()
    with torch.no_grad():
        env_random_numbers = torch.rand(num_envs, number_agents, device=device)
        permuted_rand = get_permuted_env_random_numbers(env_random_numbers, number_agents, num_envs)
        rand_batched = permuted_rand.view(-1, number_agents)
        actions_batched, log_probs = actor(obs_batched, rand_batched)
        
        # Track log probabilities
        avg_log_prob = log_probs.mean().item()
        update_metrics['log_probs'].append(avg_log_prob)
        
        actions_batched = actions_batched.view(number_agents, num_envs, -1)
        all_obs_next, rewards, dones, infos = env.step(actions_batched)

        # Create and update render environment for visualization at intervals
        if global_step % gif_save_interval == 0 and render_env is None:
            render_env = vmas.make_env(
                scenario="simple_spread",
                num_envs=1,
                n_agents=number_agents,
                continuous_actions=True,
                max_steps=max_steps_per_episode,
                device=device
            )
            render_obs = render_env.reset()
            frames = []

        # Render a frame if it's time to save a GIF
        if render_env is not None:
            try:
                with torch.no_grad():
                    # Get actions for rendering env
                    render_obs_batched = torch.stack(render_obs, dim=1).view(-1, obs_dim).to(device)
                    render_rand = torch.rand(1, number_agents, device=device)
                    render_permuted_rand = get_permuted_env_random_numbers(render_rand, number_agents, 1)
                    render_rand_batched = render_permuted_rand.view(-1, number_agents)
                    render_actions, _ = actor(render_obs_batched, render_rand_batched)
                    render_actions = render_actions.view(number_agents, 1, -1)
                    
                    # Step the rendering environment
                    render_obs_next, _, _, _ = render_env.step(render_actions)
                    
                    # Render and save frame
                    frame = render_env.render(
                        mode="rgb_array",
                        agent_index_focus=None,
                        visualize_when_rgb=False
                    )
                    if frame is not None:
                        frames.append(frame)
                    
                    render_obs = render_obs_next
            except Exception as e:
                print(f"Rendering failed: {e}")

        # Complete the GIF and reset render environment if we've recorded a full episode
        if render_env is not None and len(frames) >= max_steps_per_episode:  # Collect frames for the full episode length
            try:
                # Save the GIF
                gif_path = os.path.join(gifs_dir, f'step_{global_step}.gif')
                processed_frames = []
                for frame in frames:
                    if frame.dtype != np.uint8:
                        frame = (frame * 255).astype(np.uint8)
                    processed_frames.append(frame)
                
                imageio.mimsave(gif_path, processed_frames, fps=10)
                print(f"Successfully saved GIF to {gif_path}")
                
                # Save last frame
                last_frame_path = os.path.join(gifs_dir, f'step_{global_step}_last_frame.png')
                imageio.imwrite(last_frame_path, processed_frames[-1])
                
                # Log to TensorBoard as video
                video_tensor = np.array(processed_frames)
                video_tensor = video_tensor.transpose(0, 3, 1, 2)  # Change shape to (T, C, H, W)
                writer.add_video(f'Training/agent_behavior_step_{global_step}', video_tensor[None], global_step, fps=10)
                
                writer.add_image(f'Training/last_frame', processed_frames[-1], global_step, dataformats='HWC')
                
                # Calculate and log metrics about the visualization
                avg_intensity = np.mean([np.mean(frame) for frame in processed_frames])
                writer.add_scalar('Visualization/avg_pixel_intensity', avg_intensity, global_step)
                
                # Calculate and log agent spread (approximated by standard deviation of pixel values)
                last_frame = processed_frames[-1]
                std_pixel = np.std(last_frame)
                writer.add_scalar('Visualization/agent_spread_metric', std_pixel, global_step)
                
                # Reset for next time
                render_env = None
                frames = []
            except Exception as e:
                print(f"Failed to save or log GIF: {e}")
                render_env = None
                frames = []

        # Process and store transition in replay buffer
        obs_next_batched = torch.stack(all_obs_next, dim=1).view(number_agents, num_envs, -1).transpose(1,0).to(device)
        obs_batched = obs_batched.view(number_agents, num_envs, -1).transpose(1,0)
        actions_batched = actions_batched.transpose(1,0)
        rewards_batched = rewards[0]  # Shape: (num_envs, number_agents) -> (num_envs)
        rand_batched = rand_batched.view(number_agents, num_envs, -1).transpose(1,0)
        dones_batched = dones
        
        for i in range(num_envs):
            replay_buffer.add(
                obs_batched[i],
                actions_batched[i],
                rewards_batched[i],
                obs_next_batched[i],
                dones_batched[i],
                rand_batched[i]
            )
        
        # Update observation for next step
        obs_batched = obs_next_batched.transpose(1,0).view(-1, obs_dim)
        
        # Track rewards
        step_reward = rewards_batched.mean().item()
        running_reward += step_reward
        reward_window.append(step_reward)
        if len(reward_window) > reward_window_size:
            reward_window.pop(0)
            
        # Track episode boundaries for more informative logging
        if dones.any():
            episode_steps = global_step % 400
            if episode_steps == 0:
                episode_steps = 400  # If exactly divisible, it's a complete episode
            writer.add_scalar('Metrics/episode_length', episode_steps, global_step)
        
        writer.add_scalar('Rewards/step_reward', step_reward, global_step)
        writer.add_scalar('Rewards/running_avg_reward', np.mean(reward_window), global_step)
        
        global_step += num_envs  # Increment by number of environments
    
    # Perform update if we have enough samples and it's time to update
    if replay_buffer.size >= 10 * batch_size and global_step % update_every == 0:
        actor.train()
        critic.train()
        
        for _ in range(num_updates):
            # Sample from replay buffer
            obs_batch, actions_batch, reward_batch, next_obs_batch, dones_batch, rand_batch = replay_buffer.sample(batch_size)
            
            # Reshape batch data
            obs_batch = obs_batch.view(-1, obs_dim)
            actions_batch = actions_batch.view(-1, action_dim)
            next_obs_batch = next_obs_batch.view(-1, obs_dim)
            rand_batch = rand_batch.view(-1, number_agents)
            
            # Get actions and log probs for next state
            next_actions, next_log_probs = actor(next_obs_batch, rand_batch)
            next_log_probs = next_log_probs.view(-1, number_agents*action_dim).sum(dim=-1)
            
            # Reshape for critic input
            next_obs_batch = next_obs_batch.view(-1, number_agents, obs_dim)
            next_actions = next_actions.view(-1, number_agents, action_dim)
            
            # Compute target Q values
            with torch.no_grad():
                target_q1, target_q2 = target_critic(next_obs_batch, next_actions)
                target_q = torch.min(target_q1, target_q2).view(-1)
                target_value = reward_batch + gamma * (1 - dones_batch) * (target_q - alpha * next_log_probs)
            
            # Compute current Q values
            obs_batch = obs_batch.view(-1, number_agents, obs_dim)
            actions_batch = actions_batch.view(-1, number_agents, action_dim)
            current_q1, current_q2 = critic(obs_batch, actions_batch)
            current_q1 = current_q1.view(-1)
            current_q2 = current_q2.view(-1)
            
            # Compute critic loss
            critic_loss = F.mse_loss(current_q1, target_value) + F.mse_loss(current_q2, target_value)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()
            
            # Log critic metrics
            critic_loss_value = critic_loss.item()
            mean_q_value = (current_q1.mean().item() + current_q2.mean().item()) / 2
            update_metrics['critic_losses'].append(critic_loss_value)
            update_metrics['q_values'].append(mean_q_value)
            
            # Compute actor loss
            obs_batch = obs_batch.view(-1, obs_dim)
            rand_batch = rand_batch.view(-1, number_agents)
            new_actions, new_log_probs = actor(obs_batch, rand_batch)
            obs_batch = obs_batch.view(-1, number_agents, obs_dim)
            new_actions = new_actions.view(-1, number_agents, action_dim)
            new_log_probs = new_log_probs.view(-1, number_agents*action_dim).sum(dim=-1)
            
            if update_step > critic_only_steps:
                actor_q1, actor_q2 = target_critic(obs_batch, new_actions)
                actor_q = torch.min(actor_q1, actor_q2)
                actor_loss = (alpha * new_log_probs - actor_q).mean()
                
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()
                
                # Log actor metrics
                actor_loss_value = actor_loss.item()
                update_metrics['actor_losses'].append(actor_loss_value)
                writer.add_scalar('Training/actor_loss', actor_loss_value, update_step)
            
            # Update alpha
            with torch.no_grad():
                _, log_probs_for_alpha = actor(obs_batch.view(-1, obs_dim), rand_batch.view(-1, number_agents))
                log_probs_for_alpha = log_probs_for_alpha.view(-1, number_agents).sum(dim=-1)
            
            alpha_loss = -(alpha.log() * (log_probs_for_alpha + target_entropy).detach()).mean()
            alpha_optimizer.zero_grad()
            alpha_loss.backward()
            alpha_optimizer.step()
            
            with torch.no_grad():
                alpha.clamp_(alpha_min, alpha_max)
            
            # Log alpha metrics
            alpha_loss_value = alpha_loss.item()
            alpha_value = alpha.item()
            update_metrics['alpha_losses'].append(alpha_loss_value)
            update_metrics['alpha_values'].append(alpha_value)
            
            # Log training metrics to TensorBoard
            writer.add_scalar('Training/critic_loss', critic_loss_value, update_step)
            writer.add_scalar('Training/q_value', mean_q_value, update_step)
            writer.add_scalar('Training/target_q', target_q.mean().item(), update_step)
            writer.add_scalar('Training/alpha_loss', alpha_loss_value, update_step)
            writer.add_scalar('Training/alpha_value', alpha_value, update_step)
            writer.add_scalar('Training/log_prob', new_log_probs.mean().item(), update_step)
            
            # Log gradient norms
            with torch.no_grad():
                actor_grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in actor.parameters() if p.grad is not None) ** 0.5
                critic_grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in critic.parameters() if p.grad is not None) ** 0.5
                writer.add_scalar('Gradients/actor_grad_norm', actor_grad_norm, update_step)
                writer.add_scalar('Gradients/critic_grad_norm', critic_grad_norm, update_step)
            
            # Update target critic network
            for param, target_param in zip(critic.parameters(), target_critic.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            
            update_step += 1
    
    # Save checkpoint at intervals
    if global_step % checkpoint_interval == 0:
        # Calculate average reward
        avg_reward = np.mean(reward_window) if reward_window else 0
        
        # Check if this is the best reward
        if avg_reward > best_reward:
            best_reward = avg_reward
            save_checkpoint(actor, critic, target_critic, 
                          actor_optimizer, critic_optimizer,
                          alpha, alpha_optimizer, 
                          replay_buffer, global_step, update_step, best_reward)
        else:
            save_checkpoint(actor, critic, target_critic, 
                          actor_optimizer, critic_optimizer,
                          alpha, alpha_optimizer, 
                          replay_buffer, global_step, update_step)
        
        # Log summary metrics
        if update_metrics['actor_losses']:
            writer.add_scalar('Summary/avg_actor_loss', 
                             np.mean(update_metrics['actor_losses']), global_step)
            update_metrics['actor_losses'] = []
        
        if update_metrics['critic_losses']:
            writer.add_scalar('Summary/avg_critic_loss', 
                             np.mean(update_metrics['critic_losses']), global_step)
            update_metrics['critic_losses'] = []
        
        if update_metrics['q_values']:
            writer.add_scalar('Summary/avg_q_value', 
                             np.mean(update_metrics['q_values']), global_step)
            update_metrics['q_values'] = []
        
        if update_metrics['log_probs']:
            writer.add_scalar('Summary/avg_log_prob', 
                             np.mean(update_metrics['log_probs']), global_step)
            update_metrics['log_probs'] = []
        
        if update_metrics['alpha_losses']:
            writer.add_scalar('Summary/avg_alpha_loss', 
                             np.mean(update_metrics['alpha_losses']), global_step)
            update_metrics['alpha_losses'] = []
        
        if update_metrics['alpha_values']:
            writer.add_scalar('Summary/avg_alpha', 
                             np.mean(update_metrics['alpha_values']), global_step)
            update_metrics['alpha_values'] = []
        
        # Log replay buffer size
        writer.add_scalar('System/replay_buffer_size', replay_buffer.size, global_step)
        
        # Log weight histograms occasionally
        if global_step % (checkpoint_interval * 10) == 0:
            for name, param in actor.named_parameters():
                writer.add_histogram(f'Weights/actor_{name}', param.data, global_step)
            for name, param in critic.named_parameters():
                writer.add_histogram(f'Weights/critic_{name}', param.data, global_step)
        
        print(f"Step {global_step}/{total_steps} - Avg Reward: {avg_reward:.2f}, Alpha: {alpha.item():.6f}")

# Final checkpoint
save_checkpoint(actor, critic, target_critic, 
               actor_optimizer, critic_optimizer,
               alpha, alpha_optimizer, 
               replay_buffer, global_step, update_step)

print(f"Training completed after {global_step} steps.")
writer.close()