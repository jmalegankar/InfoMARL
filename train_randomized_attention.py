"""
Training script for Randomized Attention MARL
"""

import torch
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable

# Import components from our module files
from randomized_replay_buffer import RandomizedReplayBuffer
from randomized_attention_actor import RandomizedAttention_actor
from randomized_attention_qvalue import RandomizedAttention_qvalue
from observation_parser import parse_observation

# Import environment and utilities
from vmas import make_env
from reinforcement_functions import SquashedNormal
from parse_args import parse_args

def train_randomized_attention(args):
    """
    Main training function for randomized attention MARL
    
    Args:
        args: Command line arguments parsed via parse_args
    
    Returns:
        mean_rewards: List of mean rewards during evaluation
        std_rewards: List of standard deviations during evaluation
    """
    # Set number of agents to 3 for this experiment
    args.n_agents = 3
    
    # Initial setup
    cuda_name = "cuda:0"
    preprocessor = True
    folder = "data"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wrapper = None
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Training Randomized Attention MARL with {args.n_agents} agents on {args.scenario_name}")
    print(f"Using device: {device}")

    # Create environments
    env = make_env(
        scenario=args.scenario_name,
        num_envs=args.num_envs,
        device=device,
        continuous_actions=True,
        wrapper=wrapper,
        seed=args.seed,
        max_steps=args.max_steps,
        # Environment specific variables
        n_agents=args.n_agents,
        n_agents_good=args.n_agents_good,
        n_agents_adversaries=args.n_agents_adversaries,
        n_packages=1,
        ratio=args.ratio,
    )

    evaluation_env = make_env(
        scenario=args.scenario_name,
        num_envs=1,
        device=device,
        continuous_actions=True,
        wrapper=wrapper,
        seed=args.seed,
        max_steps=args.max_steps,
        # Environment specific variables
        n_agents=args.n_agents,
        n_agents_good=args.n_agents_good,
        n_agents_adversaries=args.n_agents_adversaries,
        n_packages=1,
        ratio=args.ratio,
    )

    # Setup actor network
    actor_config = {
        "device": device,
        "n_agents": args.n_agents,
        "observation_dim_per_agent": env.observation_space[0].shape[0],
        "action_dim_per_agent": env.action_space[0].shape[0],
        "r_communication": args.r_communication,
        "batch_size": args.batch_size,
        "num_envs": args.num_envs,
        "scenario_name": args.scenario_name,
        "preprocessor": preprocessor,
        "ratio": args.ratio,
        "ratio_eval": args.ratio_eval
    }

    # Setup Q value networks
    qvalue_config = {
        "device": device,
        "n_agents": args.n_agents,
        "observation_dim_per_agent": env.observation_space[0].shape[0],
        "action_dim_per_agent": env.action_space[0].shape[0],
        "scenario_name": args.scenario_name
    }

    # Create networks
    actor_net = RandomizedAttention_actor(actor_config)
    q_value_net_1 = RandomizedAttention_qvalue(qvalue_config)
    q_value_net_2 = RandomizedAttention_qvalue(qvalue_config)
    q_value_target_net_1 = RandomizedAttention_qvalue(qvalue_config)
    q_value_target_net_2 = RandomizedAttention_qvalue(qvalue_config)

    # Copy initial weights to target networks
    q_value_net_1_weights = q_value_net_1.state_dict()
    q_value_net_2_weights = q_value_net_2.state_dict()

    q_value_target_net_1.load_state_dict(q_value_net_1_weights)
    q_value_target_net_2.load_state_dict(q_value_net_2_weights)

    # Setup collector config
    collector_config = {
        "total_frames": args.total_frames,
        "frames_per_batch": args.frames_per_batch,
        "init_random_frames": args.init_random_frames,
        "device": device,
        "seed": args.seed,
    }

    # Setup Loss Module
    sac_config = {
        "gamma": args.gamma,
        "alpha_init": args.alpha_init,
        "alpha_min": torch.Tensor([args.min_alpha]).to(device),
        "alpha_max": torch.Tensor([args.max_alpha]).to(device),
        "alpha": torch.Tensor([args.alpha_init]).to(device),
        "lr_alpha": args.lr_alpha,
        "target_entropy": -env.action_space[0].shape[0],
        "tau": args.tau,
        "reward_scaling": args.reward_scaling
    }
    sac_config["alpha"].requires_grad = True

    # Setup optimizers
    optimizer_policy = torch.optim.Adam(actor_net.parameters(), args.lr)
    optimizer_q_value = torch.optim.Adam(
        list(q_value_net_1.parameters()) + list(q_value_net_2.parameters()), 
        args.lr
    )
    optimizer_alpha = torch.optim.Adam([sac_config["alpha"]], sac_config["lr_alpha"])

    # Setup replay buffer
    replay_buffer = RandomizedReplayBuffer(
        env.observation_space[0].shape[0],  # Observation shape per agent
        env.action_space[0].shape[0],       # Action shape per agent
        args.num_envs,
        args.n_agents,
        args.max_size,
        device
    )

    # Setup trainer config
    trainer_config = {
        "total_frames": args.total_frames,
        "optim_steps_per_batch": args.optim_steps_per_batch,
        "clip_grad_norm": args.clip_grad_norm,
        "clip_norm": args.clip_norm,
        "progress_bar": True,
        "seed": args.seed,
        "save_trainer_interval": args.save_trainer_interval,
    }

    # Training loop
    actor_net.eval()
    q_value_net_1.eval()
    q_value_net_2.eval()
    obs = [None for i in range(args.num_envs)]
    dones = [False]
    MEAN_CUM_REWS = []
    STD_CUM_REWS = []

    # Main training loop
    for current_frame in tqdm(range(trainer_config["total_frames"]), desc="Training"):
        # Interaction with the environment
        with torch.no_grad():
            # Preprocess obs to a list of tensors for a proper reset
            obs = list(obs)

            # Reset environment if needed
            if current_frame == 0:
                for environment in range(args.num_envs):
                    obs[environment] = torch.stack(env.reset_at(environment)).squeeze(1)
            if any(dones):
                environments = [i for i, x in enumerate(dones) if x]
                for environment in environments:
                    obs[environment] = torch.stack(env.reset_at(environment)).squeeze(1)

            # Post process observation list to obtain a Tensor
            obs = torch.stack(obs)

            # If not enough data in the replay buffer, use random actions
            if current_frame < collector_config["init_random_frames"]:
                actions = (torch.rand(args.num_envs,
                                      args.n_agents,
                                      env.action_space[0].shape[0]) - 0.5) * 2.0
                random_nums = torch.rand(args.num_envs, args.n_agents)
            else:
                # Select action using the policy
                action_distribution, random_nums = actor_net(obs)
                actions = action_distribution.rsample()

            # Execute action in the environment
            next_obs, rews, dones, info = env.step(list(actions.transpose(0, 1)))

            # Process observations and rewards for the buffer
            next_obs_tensor = torch.stack(next_obs).transpose(0, 1)
            rewards_tensor = torch.stack(rews).transpose(0, 1)

            parsed_obs_dim_per_agent = 10  
            obs_shape = parsed_obs_dim_per_agent * args.n_agents

            # Update replay buffer with the new experiences
            replay_buffer = RandomizedReplayBuffer(
                obs_shape,  # Now using parsed dimension
                env.action_space[0].shape[0] * args.n_agents,
                args.num_envs,
                args.n_agents,
                args.max_size,
                device
            )

            # Update obs
            obs = next_obs_tensor.clone()

        # If it is time to update
        if current_frame % args.frames_per_batch == 0 and current_frame >= collector_config["init_random_frames"]:
            # Switch networks to training mode
            actor_net.train()
            q_value_net_1.train()
            q_value_net_2.train()

            # Store rewards for checking purposes
            rb_rewards = []
            
            for iteration in range(trainer_config["optim_steps_per_batch"]):
                # Randomly sample a batch of transitions
                b_obs, b_actions, b_rews, b_next_obs, b_dones, b_random_nums = replay_buffer.sample(args.batch_size)

                # Compute targets for the Q functions
                action_distribution, _ = actor_net(b_next_obs)
                next_actions = action_distribution.rsample()
                log_probs = action_distribution.log_prob(next_actions).sum(dim=-1).sum(dim=-1)
                
                q_value_target_1 = q_value_target_net_1(b_next_obs, next_actions).flatten()
                q_value_target_2 = q_value_target_net_2(b_next_obs, next_actions).flatten()

                b_target = sac_config["reward_scaling"] * b_rews.sum(dim=1) + \
                           sac_config["gamma"] * (1 - b_dones.flatten()) * \
                           (torch.minimum(q_value_target_1, q_value_target_2) - sac_config["alpha"] * log_probs)

                # Update Q-functions by one step gradient descent
                q_value_1 = q_value_net_1(b_obs, b_actions).flatten()
                q_value_2 = q_value_net_2(b_obs, b_actions).flatten()
                loss_q_value_1 = (q_value_1 - b_target).pow(2).mean()
                loss_q_value_2 = (q_value_2 - b_target).pow(2).mean()
                loss_q_value = loss_q_value_1 + loss_q_value_2

                optimizer_q_value.zero_grad()
                loss_q_value.backward()

                if trainer_config["clip_grad_norm"]:
                    torch.nn.utils.clip_grad_norm_(q_value_net_1.parameters(), trainer_config["clip_norm"])
                    torch.nn.utils.clip_grad_norm_(q_value_net_2.parameters(), trainer_config["clip_norm"])

                optimizer_q_value.step()

                # Update policy by one step gradient descent
                action_distribution, _ = actor_net(b_obs)
                actions = action_distribution.rsample()
                log_probs = action_distribution.log_prob(actions).sum(dim=-1).sum(dim=-1)
                
                q_value_1 = q_value_net_1(b_obs, actions).flatten()
                q_value_2 = q_value_net_2(b_obs, actions).flatten()
                loss_policy = (sac_config["alpha"] * log_probs - torch.minimum(q_value_1, q_value_2)).mean()

                optimizer_policy.zero_grad()
                loss_policy.backward()

                if trainer_config["clip_grad_norm"]:
                    torch.nn.utils.clip_grad_norm_(actor_net.parameters(), trainer_config["clip_norm"])

                optimizer_policy.step()

                # Update SAC temperature
                log_probs_alpha = Variable(log_probs.data, requires_grad=True)
                alpha_loss = (sac_config["alpha"] * (-log_probs_alpha - sac_config["target_entropy"])).mean()

                optimizer_alpha.zero_grad()
                alpha_loss.backward()

                if trainer_config["clip_grad_norm"]:
                    torch.nn.utils.clip_grad_norm_(sac_config["alpha"], trainer_config["clip_norm"])

                optimizer_alpha.step()

                # Apply alpha constraints
                with torch.no_grad():
                    sac_config["alpha"].clamp_(sac_config["alpha_min"], sac_config["alpha_max"])

                # Update target networks
                for param, target_param in zip(q_value_net_1.parameters(), q_value_target_net_1.parameters()):
                    target_param.data.copy_(
                        sac_config["tau"] * param.data + 
                        (1 - sac_config["tau"]) * target_param.data
                    )

                for param, target_param in zip(q_value_net_2.parameters(), q_value_target_net_2.parameters()):
                    target_param.data.copy_(
                        sac_config["tau"] * param.data + 
                        (1 - sac_config["tau"]) * target_param.data
                    )

                # Store rewards
                rb_rewards.append(sac_config["reward_scaling"] * b_rews.sum(dim=1).mean())

            # Switch networks back to evaluation mode
            actor_net.eval()
            q_value_net_1.eval()
            q_value_net_2.eval()

        # Save models periodically
        if current_frame % args.save_trainer_interval == 0 and current_frame >= collector_config["init_random_frames"]:
            model_suffix = f"{args.scenario_name}_{current_frame}_RandomizedAttention"
            
            torch.save(q_value_target_net_1, f'{folder}/q_value_target_net_1_{model_suffix}.pth')
            torch.save(q_value_target_net_2, f'{folder}/q_value_target_net_2_{model_suffix}.pth')
            torch.save(q_value_net_1, f'{folder}/q_value_net_1_{model_suffix}.pth')
            torch.save(q_value_net_2, f'{folder}/q_value_net_2_{model_suffix}.pth')
            torch.save(actor_net, f'{folder}/actor_net_{model_suffix}.pth')

            # Display metrics
            print("\n-------------------------------------------\n")
            print(f"Training instance {int(current_frame / args.frames_per_batch)}")
            print("")
            print(f"Policy Loss: {loss_policy.item()}")
            print(f"Q1 Loss: {loss_q_value_1.item()}")
            print(f"Q2 Loss: {loss_q_value_2.item()}")
            print(f"Reward: {(sac_config['reward_scaling'] * rb_rewards[-1]).item()}")
            print(f"Alpha: {sac_config['alpha'].item()}")
            print("\n-------------------------------------------\n")

        # Evaluate periodically
        if current_frame % args.evaluation_interval == 0 and current_frame >= collector_config["init_random_frames"]:
            eval_steps = args.max_frames_eval
            cum_rew = np.zeros([args.num_test_episodes])
            
            with torch.no_grad():
                for episode in range(args.num_test_episodes):
                    time_step = 0
                    eval_dones = [False]
                    eval_obs = evaluation_env.reset(seed=None)
                    eval_cum_rew = 0
                    
                    while not any(eval_dones) and time_step < eval_steps:
                        eval_obs = torch.stack(eval_obs)
                        if len(eval_obs.shape) == 3:
                            eval_obs = eval_obs.squeeze(1)
                        eval_obs = eval_obs.unsqueeze(0)
                        
                        eval_action_distribution, _ = actor_net(eval_obs)
                        eval_actions = eval_action_distribution.mean
                        
                        eval_obs, eval_rews, eval_dones, eval_info = evaluation_env.step(
                            list(eval_actions.transpose(0, 1))
                        )
                        
                        time_step += 1
                        eval_cum_rew += (sac_config["reward_scaling"] * torch.stack(eval_rews).sum()).item()
                        cum_rew[episode] = eval_cum_rew

            mean_cum_rew = np.mean(cum_rew)
            std_cum_rew = np.std(cum_rew)
            
            print("\n-------------------------------------------\n")
            print(f"Evaluation instance {int(current_frame / args.evaluation_interval)}")
            print("")
            print(f"Cumulative Reward: mean is {mean_cum_rew} and std is {std_cum_rew}")
            print("\n-------------------------------------------\n")
            
            MEAN_CUM_REWS.append(mean_cum_rew)
            STD_CUM_REWS.append(std_cum_rew)
            
            stats_suffix = f"{args.scenario_name}_{current_frame}_RandomizedAttention"
            np.save(f'{folder}/MEAN_CUM_REWS_{stats_suffix}.npy', np.array(MEAN_CUM_REWS))
            np.save(f'{folder}/STD_CUM_REWS_{stats_suffix}.npy', np.array(STD_CUM_REWS))

    return MEAN_CUM_REWS, STD_CUM_REWS

if __name__ == "__main__":
    args = parse_args()
    
    # Train the model
    mean_rewards, std_rewards = train_randomized_attention(args)
    
    # Print final results
    print("\n===========================================")
    print("Training complete!")
    print(f"Final mean reward: {mean_rewards[-1]}")
    print(f"Final std reward: {std_rewards[-1]}")
    print("===========================================\n")