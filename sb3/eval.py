import argparse
import torch
import numpy as np
from tqdm import tqdm
from stable_baselines3 import PPO

import wrapper
import policy
import smaclite2

def evaluate(model_path, map_name, n_episodes=1000, num_envs=20):
    env_name = f"smaclite/{map_name}-v0"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Initializing {num_envs} parallel environments for {env_name}...")
    
    env = wrapper.SMACVecEnv(
        env_name=env_name,
        num_envs=num_envs,
        max_steps=500,  
        rnd_nums=True, 
    )
    
    print(f"Loading model from {model_path}...")
    try:
        model = PPO(
            policy=policy.InfoMARLActorCriticPolicy,
            env=env,
            device=device,
            verbose=1,
            batch_size=320,
            n_epochs=10,
            gamma=0.99,
            n_steps=160,
            vf_coef=0.5,
            ent_coef=0.001,
            target_kl=0.25,
            max_grad_norm=10.0,
            learning_rate=1e-4,
        )
        del model.policy.vf_features_extractor
        del model.policy.value_net
        model.policy.load_state_dict(PPO.load("ppo_infomarl.zip", device=device).policy.state_dict(), strict=False)
        model.policy.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Ensure 'policy.py' and 'wrapper.py' are in the same directory.")
        return

    print(f"Starting evaluation over {n_episodes} episodes...")
    
    wins = 0
    episodes_completed = 0
    obs = env.reset()
    
    pbar = tqdm(total=n_episodes, desc="Battling")
    with torch.no_grad():
        while episodes_completed < n_episodes:
            action, _ = model.predict(obs, deterministic=True)
            
            obs, rewards, dones, infos = env.step(action)
            
            for i, done in enumerate(dones):
                if done:
                    is_win = infos[i].get("battle_won", False)
                    
                    if is_win:
                        wins += 1
                        
                    episodes_completed += 1
                    pbar.update(1)
                    
                    current_wr = wins / episodes_completed
                    pbar.set_postfix({"Win Rate": f"{current_wr:.1%}"})

                    if episodes_completed >= n_episodes:
                        break
        
        pbar.close()
    
    final_wr = wins / episodes_completed
    print("\n" + "="*40)
    print(f"RESULTS: {model_path}")
    print(f"Map: {map_name}")
    print(f"Episodes: {episodes_completed}")
    print(f"Wins: {wins}")
    print(f"Win Rate: {final_wr:.2%}")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to .zip model file")
    parser.add_argument("--map", type=str, default="2s3z", help="Map name (e.g., 2s3z, 3s5z)")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes to evaluate")
    args = parser.parse_args()
    
    evaluate(args.model, args.map, args.episodes)