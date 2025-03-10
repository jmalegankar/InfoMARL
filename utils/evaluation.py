import torch as th
import numpy as np
import os
import imageio
from typing import List, Tuple, Dict, Any, Optional, Union

def evaluate_policy(
    env, agent, num_episodes, max_steps, device, save_gifs=False, gif_path=None
):
    """
    Evaluate policy over multiple episodes
    
    Args:
        env: Environment to evaluate in
        agent: Agent to evaluate
        num_episodes: Number of episodes to evaluate
        max_steps: Maximum number of steps per episode
        device: Device to place tensors on
        save_gifs: Whether to save GIFs of episodes
        gif_path: Path to save GIFs to
        
    Returns:
        episode_rewards: List of episode rewards
        frames: List of frames from last episode (if save_gifs is True)
    """
    episode_rewards = []
    all_frames = []
    
    for episode in range(num_episodes):
        # Reset environment
        obs = env.reset()
        episode_reward = 0
        frames = []
        
        # Capture initial frame if saving GIFs
        if save_gifs:
            frame = env.render(mode='rgb_array')
            if frame is not None:
                frames.append(frame)
        
        for step in range(max_steps):
            # Sample action
            with th.no_grad():
                obs_dict = agent.prepare_obs_dict(obs)
                action_dist = agent(obs_dict)
                action = action_dist.mean  # Use mean for evaluation
            
            # Take step in environment
            next_obs, rewards, dones, _ = env.step(action)
            
            # Capture frame if saving GIFs
            if save_gifs:
                frame = env.render(mode='rgb_array')
                if frame is not None:
                    frames.append(frame)
            
            # Update rewards and observation
            episode_reward += rewards.sum().item()
            obs = next_obs
            
            # Check if episode is done
            if dones.any():
                break
        
        # Record episode results
        episode_rewards.append(episode_reward)
        
        # Save GIF if requested
        if save_gifs and len(frames) > 0:
            if gif_path is not None:
                os.makedirs(os.path.dirname(gif_path), exist_ok=True)
                episode_gif_path = f"{gif_path}_episode_{episode}.gif"
                imageio.mimsave(episode_gif_path, frames, fps=10)
                print(f"Saved GIF to {episode_gif_path}")
            
            # Save frames from last episode
            if episode == num_episodes - 1:
                all_frames = frames
    
    return episode_rewards, all_frames

def visualize_agent_behavior(
    env, agent, episode_length, device, save_path=None
):
    """
    Visualize agent behavior and save as GIF
    
    Args:
        env: Environment to visualize in
        agent: Agent to visualize
        episode_length: Length of episode to visualize
        device: Device to place tensors on
        save_path: Path to save GIF to
        
    Returns:
        frames: List of frames from episode
    """
    # Reset environment
    obs = env.reset()
    frames = []
    
    # Capture initial frame
    frame = env.render(mode='rgb_array')
    if frame is not None:
        frames.append(frame)
    
    # Simulate episode
    for step in range(episode_length):
        # Sample action
        with th.no_grad():
            obs_dict = agent.prepare_obs_dict(obs)
            action_dist = agent(obs_dict)
            action = action_dist.mean  # Use mean for visualization
        
        # Take step in environment
        next_obs, rewards, dones, _ = env.step(action)
        
        # Capture frame
        frame = env.render(mode='rgb_array')
        if frame is not None:
            frames.append(frame)
        
        # Update observation
        obs = next_obs
        
        # Check if episode is done
        if dones.any():
            break
    
    # Save GIF if requested
    if save_path is not None and len(frames) > 0:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        imageio.mimsave(save_path, frames, fps=10)
        print(f"Saved GIF to {save_path}")
    
    return frames