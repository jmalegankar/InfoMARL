import argparse
import os
import numpy as np

def main():
    """
    Entry point for training and evaluation
    """
    parser = argparse.ArgumentParser("Randomized Attention MARL")
    
    # High-level commands
    parser.add_argument("--mode", type=str, choices=["train", "eval", "visualize"], 
                        default="train", help="Operation mode")
    
    # Add arguments from train.py
    try:
        from train import parse_args as train_parse_args
        train_args = train_parse_args()
        for action in train_args._actions:
            if action.dest != 'help':
                parser.add_argument(f"--{action.dest}", **{k: v for k, v in action.__dict__.items() 
                                                         if k not in ['container', 'dest', 'option_strings']})
    except Exception as e:
        print(f"Warning: Could not import arguments from train.py: {e}")
        # Add basic arguments
        parser.add_argument("--scenario", type=str, default="simple_spread", help="Scenario name")
        parser.add_argument("--n-agents", type=int, default=5, help="Number of agents")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute requested mode
    if args.mode == "train":
        from train import main as train_main
        train_main()
    elif args.mode == "eval":
        from utils.evaluation import evaluate_policy
        from agents.randomized_attention_policy import RandomizedAttentionAgent
        from utils.env_wrappers import make_env_for_evaluation
        from utils.training_utils import load_model
        import torch as th
        
        # Set up device
        device = th.device("cuda" if args.cuda else "cpu")
        
        # Create environment
        env = make_env_for_evaluation(args, device)
        
        # Create agent
        agent = RandomizedAttentionAgent(
            observation_dim=env.observation_dim,
            action_dim=env.action_dim,
            hidden_dim=args.hidden_dim,
            device=device
        ).to(device)
        
        # Load agent parameters
        state_dict = load_model(args.model_path, device)
        agent.load_state_dict(state_dict["agent"])
        
        # Evaluate agent
        rewards, frames = evaluate_policy(
            env, agent, args.num_eval_episodes, args.max_episode_len,
            device, args.save_gifs, os.path.join(args.save_dir, "eval_gifs/eval")
        )
        
        # Print results
        print(f"Evaluation results ({args.num_eval_episodes} episodes):")
        print(f"Mean reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        print(f"Min reward: {np.min(rewards):.2f}")
        print(f"Max reward: {np.max(rewards):.2f}")
        
    elif args.mode == "visualize":
        from utils.evaluation import visualize_agent_behavior
        from agents.randomized_attention_policy import RandomizedAttentionAgent
        from utils.env_wrappers import make_env_for_evaluation
        from utils.training_utils import load_model
        import torch as th
        
        # Set up device
        device = th.device("cuda" if args.cuda else "cpu")
        
        # Create environment
        env = make_env_for_evaluation(args, device)
        
        # Create agent
        agent = RandomizedAttentionAgent(
            observation_dim=env.observation_dim,
            action_dim=env.action_dim,
            hidden_dim=args.hidden_dim,
            device=device
        ).to(device)
        
        # Load agent parameters
        state_dict = load_model(args.model_path, device)
        agent.load_state_dict(state_dict["agent"])
        
        # Visualize agent behavior
        visualize_agent_behavior(
            env, agent, args.max_episode_len, device,
            os.path.join(args.save_dir, "visualizations/behavior.gif")
        )

if __name__ == "__main__":
    main()