import torch
import vmas
import actor
import utils
import argparse
import os
import numpy as np
from tqdm import tqdm
import imageio

from train import get_permuted_env_random_numbers
from actor import env_parser

def evaluate_and_record(actor_model, env, number_agents, num_episodes=5, episode_length=400, device="cuda"):
    """
    Evaluates the actor model and records GIFs for visualization.
    
    Args:
        actor_model: The trained actor model
        env: VMAS environment
        number_agents: Number of agents in the environment
        num_episodes: Number of episodes to record
        episode_length: Maximum length of an episode
        device: Device to run evaluation on
    
    Returns:
        List of image frames for creating GIFs
    """
    frames_list = []
    
    for episode in range(num_episodes):
        print(f"Recording episode {episode+1}/{num_episodes}...")
        frames = []
        obs = env.reset()
        obs = torch.stack(obs, dim=1).view(
            number_agents, env.num_envs, -1
        ).transpose(1, 0).to(device).contiguous()
        
        for step in tqdm(range(episode_length)):
            # Render the environment
            frame = env.render(mode='rgb_array')
            frames.append(frame)
            
            # Generate random numbers for the environment
            rand_nums = torch.rand(
                env.num_envs, number_agents, device=device
            )
            rand_nums = get_permuted_env_random_numbers(rand_nums, number_agents, env.num_envs, device)
            rand_nums = rand_nums.view(-1, number_agents)
            
            # Get actions from the actor model
            obs_flat = obs.view(-1, obs.shape[-1])
            with torch.no_grad():
                actions, _ = actor_model(obs_flat, rand_nums)
            
            # Reshape actions for environment step
            actions = actions.view(number_agents, env.num_envs, -1)
            
            # Take a step in the environment
            next_obs, rewards, terminated, truncated, _ = env.step(actions)
            
            # Process next observations
            next_obs = torch.stack(next_obs, dim=1).view(
                number_agents, env.num_envs, -1
            ).transpose(1, 0).to(device).contiguous()
            
            # Check if episode is done
            dones = torch.logical_or(terminated, truncated).float().to(device)
            if dones.any():
                break
                
            obs = next_obs
        
        frames_list.append(frames)
    
    return frames_list

def save_gif(frames, filepath, fps=15):

    print(f"Saving GIF to {filepath}")
    imageio.mimsave(filepath, frames, fps=fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained policy and create GIFs")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/simple_spread/", 
                        help="Directory containing checkpoints")
    parser.add_argument("--agent_numbers", type=int, nargs="+", default=[1,3,4,5,7],
                        help="List of agent numbers to evaluate")
    parser.add_argument("--episodes_per_config", type=int, default=1, 
                        help="Number of episodes to record per configuration")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, 
                        help="Device to run evaluation on (default: auto-detect)")
    parser.add_argument("--output_dir", type=str, default="evaluation_gifs/",
                        help="Directory to save GIFs")
    parser.add_argument("--hidden_dim", type=int, default=64,
                        help="Hidden dimension size for the actor model")
    
    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    utils.set_seed(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    checkpoint_files = [f for f in os.listdir(args.checkpoint_dir) if f.endswith('.pt')]
    if not checkpoint_files:
        raise FileNotFoundError("No checkpoint files found in the directory.")
    
    latest_checkpoint = sorted(checkpoint_files, key=lambda x: int(x.split('_')[2].split('.')[0]), reverse=True)[0]
    checkpoint_path = os.path.join(args.checkpoint_dir, latest_checkpoint)
    print(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    for num_agents in args.agent_numbers:
        print(f"\nEvaluating with {num_agents} agents...")

        env = vmas.make_env(
            scenario="simple_spread",
            n_agents=num_agents,
            num_envs=1,
            continuous_actions=True,
            max_steps=400,
            seed=args.seed,
            device=device,
            terminated_truncated=True,
        )

        actor_model = actor.RandomAgentPolicy(
            number_agents=num_agents,
            agent_dim=2,  # Action dimension
            landmark_dim=2,  # Landmark dimension
            hidden_dim=args.hidden_dim
        ).to(device)

        if num_agents != 4:
            print(f"Note: Original model was trained with 4 agents, adapting to {num_agents} agents.")
            try:
                actor_model.load_state_dict(checkpoint['actor_state_dict'], strict=False)
            except:
                print("Strict loading failed. This is expected when changing the number of agents.")
                # Only load layers that are compatible
                pretrained_dict = checkpoint['actor_state_dict']
                model_dict = actor_model.state_dict()
                
                # Filter out incompatible layers
                pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                                  if k in model_dict and v.shape == model_dict[k].shape}
                
                # Update the model with compatible layers
                model_dict.update(pretrained_dict)
                actor_model.load_state_dict(model_dict)
        else:
            actor_model.load_state_dict(checkpoint['actor_state_dict'])
        
        actor_model.eval()

        frames_list = evaluate_and_record(
            actor_model=actor_model,
            env=env,
            number_agents=num_agents,
            num_episodes=args.episodes_per_config,
            device=device
        )

        for i, frames in enumerate(frames_list):
            gif_path = os.path.join(args.output_dir, f"agents_{num_agents}_episode_{i+1}.gif")
            save_gif(frames, gif_path)
        