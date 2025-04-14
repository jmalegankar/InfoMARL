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

# TensorBoard init
current_time = datetime.now().strftime('%b%d_%H-%M-%S')
log_dir = os.path.join('runs', f'vmas_simple_spread_{current_time}')
writer = SummaryWriter(log_dir=log_dir)

# HYPERPARAMETERS
num_envs = 64
number_agents = 4
episode_length = 200
num_episodes = 1000000
batch_size = 32
gamma = 0.95
tau = 0.005
actor_lr = 1e-4
critic_lr = 1e-4
update_every = 64
num_updates = 2
alpha = 0.2

writer.add_text('Hyperparameters/num_envs', str(num_envs))
writer.add_text('Hyperparameters/number_agents', str(number_agents))
writer.add_text('Hyperparameters/episode_length', str(episode_length))
writer.add_text('Hyperparameters/num_episodes', str(num_episodes))
writer.add_text('Hyperparameters/batch_size', str(batch_size))
writer.add_text('Hyperparameters/gamma', str(gamma))
writer.add_text('Hyperparameters/tau', str(tau))
writer.add_text('Hyperparameters/actor_lr', str(actor_lr))
writer.add_text('Hyperparameters/critic_lr', str(critic_lr))
writer.add_text('Hyperparameters/alpha', str(alpha))


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
    max_steps=episode_length,
)

#Dims 
all_obs = env.reset()
obs_dim = all_obs[0][0].shape[0]
action_dim = 2
agent_dim = 4
landmark_dim = 2 * number_agents
other_agent_dim = 2 * (number_agents - 1)

#
actor = RandomAgentPolicy(
    number_agents, 
    agent_dim=4, 
    landmark_dim=landmark_dim, 
    other_agent_dim=other_agent_dim
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

actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr)
critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr)

buffer_capacity = 1000000
replay_buffer = ReplayBuffer(buffer_capacity, obs_dim, action_dim, number_agents, device)

dummy_obs = torch.zeros(1, obs_dim).to(device)
dummy_rand = torch.zeros(1, number_agents).to(device)
writer.add_graph(actor, [dummy_obs, dummy_rand])

def get_permuted_env_random_numbers(env_random_numbers, number_agents, num_envs):
    """
    Permute the random numbers for each agent in the environment.
    """
    permutation_indices = torch.zeros(number_agents, number_agents, dtype=torch.long)
    for i in range(number_agents):
        other_agents = sorted([j for j in range(number_agents) if j != i])
        permutation_indices[i] = torch.tensor([i] + other_agents)
    expanded_rand = env_random_numbers.unsqueeze(1).expand(-1, number_agents, -1)
    permuted_rand = torch.gather(expanded_rand, dim=2, index=permutation_indices.unsqueeze(0).expand(num_envs, -1, -1))
    return permuted_rand

global_step = 0
update_step = 0
gif_save_interval = 100

for episode in range(num_episodes):
    all_obs = env.reset()
    obs_batched = torch.stack(all_obs, dim=1).view(-1, obs_dim).to(device)
    episode_reward = 0.0
    
    # Initialize episode metrics for tensorboard
    episode_actor_losses = []
    episode_critic_losses = []
    episode_q_values = []
    episode_log_probs = []

    # For GIF saving
    if episode % gif_save_interval == 0:
            frames = []
            # Create a separate env for rendering
            render_env = vmas.make_env(
                scenario="simple_spread",
                num_envs=1,
                n_agents=number_agents,
                continuous_actions=True,
                max_steps=episode_length,
                device=device
            )
            render_obs = render_env.reset()

    
    for t in range(episode_length):
        with torch.no_grad():
            env_random_numbers = torch.rand(num_envs, number_agents, device=device) #shape: (num_envs, number_agents)
            permuted_rand = get_permuted_env_random_numbers(env_random_numbers, number_agents, num_envs) #shape: (num_envs, number_agents, number_agents)
            rand_batched = permuted_rand.view(-1, number_agents) #shape: (num_envs * number_agents, number_agents)
            actions_batched, log_probs = actor(obs_batched, rand_batched) #shape: (num_envs * number_agents, action_dim),  (num_envs * number_agents, 1)
            
            # Track log probabilities
            avg_log_prob = log_probs.mean().item()
            episode_log_probs.append(avg_log_prob)
            
            
            actions_batched = actions_batched.view(number_agents, num_envs, -1) #shape: (number_agents, num_envs, action_dim)
            all_obs_next, rewards, dones, infos = env.step(actions_batched) #shape  3, 3, (num_envs), [{}, {}, {}]

            if episode % gif_save_interval == 0:
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
                    try:
                        frame = render_env.render(
                            mode="rgb_array",
                            agent_index_focus=None,
                            visualize_when_rgb=False
                        )
                        if frame is not None:
                            frames.append(frame)
                        else:
                            print(f"Warning: Got None frame at episode {episode}, step {t}")
                    except Exception as e:
                        print(f"Rendering failed: {e}")
                    
                    render_obs = render_obs_next


            obs_next_batched = torch.stack(all_obs_next, dim=1).view(number_agents, num_envs, -1).transpose(1,0).to(device) #shape: (num_envs, number_agents, obs_dim)
            obs_batched = obs_batched.view(number_agents, num_envs, -1).transpose(1,0) #shape: (num_envs, number_agents, obs_dim)
            actions_batched = actions_batched.transpose(1,0) #shape: (num_envs, number_agents, action_dim)
            rewards_batched = rewards[0] #shape: (num_envs, number_agents) -> (num_envs)
            rand_batched = rand_batched.view(number_agents, num_envs, -1).transpose(1,0) #shape: (num_envs, number_agents, number_agents)
            dones_batched = dones #shape:(num_envs)
            for i in range(num_envs):
                replay_buffer.add(
                    obs_batched[i],
                    actions_batched[i],
                    rewards_batched[i],
                    obs_next_batched[i],
                    dones_batched[i],
                    rand_batched[i]
                )
            
            obs_batched = obs_next_batched.transpose(1,0).view(-1, obs_dim) #shape: (num_envs * number_agents, obs_dim)
            step_reward = rewards_batched.mean().item()
            episode_reward += rewards_batched.mean().item()

            writer.add_scalar('Rewards/step_reward', step_reward, global_step)
            global_step += 1
            
        
        if not (replay_buffer.size >= 10*batch_size):
            continue
        if (global_step * num_envs) % update_every:
            continue
        for _ in range(num_updates):
            obs_batch, actions_batch, reward_batch, next_obs_batch, dones_batch, rand_batch = replay_buffer.sample(batch_size) #shape (num_envs * number_agents, obs_dim), (num_envs * number_agents, action_dim), (num_envs * number_agents), (num_envs * number_agents, obs_dim), (num_envs * number_agents), (num_envs * number_agents, number_agents)
            obs_batch = obs_batch.view(-1, obs_dim) #shape: (num_envs * number_agents, obs_dim)
            actions_batch = actions_batch.view(-1, action_dim) #shape: (num_envs * number_agents, action_dim)
            next_obs_batch = next_obs_batch.view(-1, obs_dim) #shape: (num_envs * number_agents, obs_dim)
            rand_batch = rand_batch.view(-1, number_agents) #shape: (num_envs * number_agents, number_agents)
            next_actions, next_log_probs = actor(next_obs_batch, rand_batch) #shape: (num_envs * number_agents, action_dim), (num_envs * number_agents, 1)
            next_log_probs = next_log_probs.view(-1, number_agents).sum(dim=-1) #shape: (num_envs * number_agents)
            next_obs_batch = next_obs_batch.view(-1, number_agents, obs_dim) #shape: (num_envs, number_agents, obs_dim)
            next_actions = next_actions.view(-1, number_agents, action_dim) #shape: (num_envs, number_agents, action_dim)
            with torch.no_grad():
                target_q1, target_q2 = target_critic(next_obs_batch, next_actions) #shape(num_envs, 1), (num_envs, 1)  
                target_q = torch.min(target_q1, target_q2).view(-1) #shape (num_envs)
                target_value = reward_batch + gamma * (1 - dones_batch) * (target_q - alpha * next_log_probs) #shape (num_envs)
            
            obs_batch = obs_batch.view(-1, number_agents, obs_dim) #shape: (num_envs, number_agents, obs_dim)
            actions_batch = actions_batch.view(-1, number_agents, action_dim) #shape: (num_envs, number_agents, action_dim)
            current_q1, current_q2 = critic(obs_batch, actions_batch) #shape (num_envs, 1), (num_envs, 1)  
            current_q1 = current_q1.view(-1) #shape (num_envs)
            current_q2 = current_q2.view(-1) #shape (num_envs)

            critic_loss = F.mse_loss(current_q1, target_value) + F.mse_loss(current_q2, target_value) 
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # Log critic loss
            critic_loss_value = critic_loss.item()
            episode_critic_losses.append(critic_loss_value)

            # Log Q-values
            mean_q_value = (current_q1.mean().item() + current_q2.mean().item()) / 2
            episode_q_values.append(mean_q_value)

            obs_batch = obs_batch.view(-1, obs_dim) #shape: (num_envs * number_agents, obs_dim)
            rand_batch = rand_batch.view(-1, number_agents) #shape: (num_envs * number_agents, number_agents)
            new_actions, new_log_probs = actor(obs_batch, rand_batch) #shape: (num_envs * number_agents, action_dim), (num_envs * number_agents, 1)
            obs_batch = obs_batch.view(-1, number_agents, obs_dim) #shape: (num_envs, number_agents, obs_dim)
            new_actions = new_actions.view(-1, number_agents, action_dim) #shape: (num_envs, number_agents, action_dim)
            new_log_probs = new_log_probs.view(-1, number_agents).sum(dim=-1) #shape: (num_envs * number_agents)
            actor_q1, actor_q2 = critic(obs_batch, new_actions) #shape(num_envs, 1), (num_envs, 1)  
            actor_q = torch.min(actor_q1, actor_q2) #shape (num_envs)
            actor_loss = (alpha * new_log_probs - actor_q).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()


            # Save actor loss for this batch
            actor_loss_value = actor_loss.item()
            episode_actor_losses.append(actor_loss_value)
            
            # Log training metrics to TensorBoard
            writer.add_scalar('Training/actor_loss', actor_loss_value, update_step)
            writer.add_scalar('Training/critic_loss', critic_loss_value, update_step)
            writer.add_scalar('Training/q_value', mean_q_value, update_step)
            writer.add_scalar('Training/target_q', target_q.mean().item(), update_step)
            
            # Log gradient norms
            actor_grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in actor.parameters() if p.grad is not None) ** 0.5
            critic_grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in critic.parameters() if p.grad is not None) ** 0.5
            writer.add_scalar('Gradients/actor_grad_norm', actor_grad_norm, update_step)
            writer.add_scalar('Gradients/critic_grad_norm', critic_grad_norm, update_step)
            
            update_step += 1
            
            for param, target_param in zip(critic.parameters(), target_critic.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    
    if episode % gif_save_interval == 0 and frames:
        try:
            gif_path = os.path.join(gifs_dir, f'episode_{episode}.gif')
            processed_frames = []
            for frame in frames:
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)
                processed_frames.append(frame)
            
            imageio.mimsave(gif_path, processed_frames, fps=10)
            print(f"Successfully saved GIF to {gif_path}")
            
            last_frame_path = os.path.join(gifs_dir, f'episode_{episode}_last_frame.png')
            imageio.imwrite(last_frame_path, processed_frames[-1])
        except Exception as e:
            print(f"Failed to save GIF: {e}")
    
    avg_episode_reward = episode_reward / episode_length
    print(f"Episode {episode+1}/{num_episodes} - Avg Reward: {avg_episode_reward:.2f}")
    writer.add_scalar('Rewards/episode_reward', episode_reward, episode)
    writer.add_scalar('Rewards/avg_episode_reward', avg_episode_reward, episode)
    
    if episode_actor_losses:
        writer.add_scalar('Training/episode_avg_actor_loss', sum(episode_actor_losses) / len(episode_actor_losses), episode)
    if episode_critic_losses:
        writer.add_scalar('Training/episode_avg_critic_loss', sum(episode_critic_losses) / len(episode_critic_losses), episode)
    if episode_q_values:
        writer.add_scalar('Training/episode_avg_q_value', sum(episode_q_values) / len(episode_q_values), episode)
    if episode_log_probs:
        writer.add_scalar('Training/episode_avg_log_prob', sum(episode_log_probs) / len(episode_log_probs), episode)
    
    # Log replay buffer size
    writer.add_scalar('System/replay_buffer_size', replay_buffer.size, episode)
    
    # Every 10 episodes, log weight histograms
    if episode % 10 == 0:
        for name, param in actor.named_parameters():
            writer.add_histogram(f'Weights/actor_{name}', param.data, episode)
        for name, param in critic.named_parameters():
            writer.add_histogram(f'Weights/critic_{name}', param.data, episode)

# Close the TensorBoard writer when training is done
writer.close()
