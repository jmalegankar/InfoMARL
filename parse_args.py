"""
This file contains all the hyperparameters for training and evaluation.
Please, read the description of each argument.
"""

import argparse


def none_or_str(value):
    if value == 'None':
        return None
    return value


def parse_args():
    parser = argparse.ArgumentParser("SAC training with VMAS environment")

    """ What do I want to train """
    parser.add_argument(
        "--neural_network_name",
        type=str,
        default="ph-MARL",
        choices=['pH-MARL', 'MLP', 'MSA', 'GSA'],
        help="available neural networks"
    )

    """ Setup Environment """
    # NOTE: removed `choices=` so we can pass either a built-in scenario name (e.g., 'food_collection')
    # or a filesystem path to a custom scenario file (e.g., '/abs/path/food_collection.py').
    parser.add_argument(
        "--scenario_name",
        type=str,
        default="sampling",
        help="Scenario name (e.g., 'simple_spread', 'food_collection') or path to a scenario .py file"
    )

    parser.add_argument(
        "--num_envs",
        type=int,
        default=96,
        help="number of vectorized environments"
    )

    parser.add_argument(
        "--continuous_actions",
        type=bool,
        default=True,
        help="whether the action spaces of the agents is continuous or discrete"
    )

    parser.add_argument(
        "--share_reward",
        type=bool,
        default=True,
        help="whether the reward is shared across agents or not"
    )

    parser.add_argument(
        "--max_steps",
        type=int,
        default=100,
        help="maximum number of steps per episode"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="seed"
    )

    parser.add_argument(
        "--n_agents",
        type=int,
        default=4,
        help="number of agents"
    )

    # Kept legacy knobs used by some scenarios
    parser.add_argument(
        "--n_agents_good",
        type=int,
        default=4,
        help="number of good agents"
    )
    parser.add_argument(
        "--n_agents_adversaries",
        type=int,
        default=4,
        help="number of adversarial agents"
    )

    # ---------- New: food_collection-specific (and generally useful) knobs ----------
    parser.add_argument(
        "--n_food",
        type=int,
        default=5,
        help="[food_collection] number of food items in the arena"
    )
    parser.add_argument(
        "--respawn_food",
        type=bool,
        default=True,
        help="[food_collection] whether collected food respawns"
    )
    parser.add_argument(
        "--collection_radius",
        type=float,
        default=0.05,
        help="[food_collection] radius within which agents collect food"
    )
    parser.add_argument(
        "--obs_agents",
        type=bool,
        default=True,
        help="If supported by the scenario, include other agents in the observation"
    )
    # -------------------------------------------------------------------------------

    """ Setup actor network """
    parser.add_argument(
        "--r_communication",
        type=float,
        default=0.45,
        help="communication radius of the agents"
    )

    parser.add_argument(
        "--ratio",
        type=float,
        default=4.0,
        help="length of the size of the squared arena"
    )

    """ Setup collector """
    parser.add_argument(
        "--total_frames",
        type=int,
        default=4000000,
        help="total number of frames"
    )

    parser.add_argument(
        "--frames_per_batch",
        type=int,
        default=1,
        help="Time-length of a batch"
    )

    parser.add_argument(
        "--init_random_frames",
        type=int,
        default=1000,  # 1000
        help="Number of frames for which the policy is ignored before it is called."
    )

    """ Setup Loss Module """
    parser.add_argument(
        "--num_qvalue_nets",
        type=int,
        default=2,
        help="Number of Q value networks. The minimum predicted value will then be used for inference"
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="gamma value for the discounted reward"
    )

    parser.add_argument(
        "--alpha_init",
        type=float,
        default=5.0,
        help="initial alpha, weights the entropy regularization"
    )

    parser.add_argument(
        "--min_alpha",
        type=float,
        default=0.1,
        help="min alpha, weights the entropy regularization"
    )

    parser.add_argument(
        "--max_alpha",
        type=float,
        default=10.0,
        help="max alpha, weights the entropy regularization"
    )

    parser.add_argument(
        "--lr_alpha",
        type=float,
        default=1e-5,
        help="step size alpha adjustment"
    )

    parser.add_argument(
        "--tau",
        type=float,
        default=0.005,
        help="tau"
    )

    """ Setup optimizer """
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="learning rate for Adam optimizer"
    )

    """ Setup replay buffer """
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="batch size"
    )

    parser.add_argument(
        "--max_size",
        type=int,
        default=2000000,
        help="max size of the replay buffer"
    )

    """ Setup trainer """
    parser.add_argument(
        "--optim_steps_per_batch",
        type=int,
        default=1,
        help="Number of optimization steps per collection of data."
    )

    parser.add_argument(
        "--clip_grad_norm",
        type=bool,
        default=False,
        help=("If True, the gradients will be clipped based on the total norm of the model parameters."
              " If False, all the partial derivatives will be clamped to (-clip_norm, clip_norm).")
    )

    parser.add_argument(
        "--clip_norm",
        type=float,
        default=0.1,
        help="clips the gradients for a stable learning"
    )

    parser.add_argument(
        "--save_trainer_interval",
        type=int,
        default=10000,
        help="How often the trainer should be saved to disk."
    )

    parser.add_argument(
        "--reward_scaling",
        type=float,
        default=1,
        help="it is usually recommended to scale up the reward"
    )

    """ Others """
    parser.add_argument(
        "--num_test_episodes",
        type=int,
        default=10,
        help="Number of tests in evaluation"
    )

    parser.add_argument(
        "--evaluation_interval",
        type=int,
        default=10000,
        help="Total number of frames between two testing runs."
    )

    """ Evaluation """
    parser.add_argument(
        "--num_agents_eval",
        type=int,
        default=4,
        help="Number of agents for evaluation"
    )

    parser.add_argument(
        "--n_agents_good_eval",
        type=int,
        default=4,
        help="number of good agents for evaluation"
    )

    parser.add_argument(
        "--n_agents_adversaries_eval",
        type=int,
        default=4,
        help="number of adversarial agents for evaluation"
    )

    parser.add_argument(
        "--desired_frame",
        type=int,
        default=1900000,
        help="Frame from which restore the actor network for evaluation"
    )

    parser.add_argument(
        "--max_frames_eval",
        type=int,
        default=400,
        help="Max number of steps for the evaluation run"
    )

    parser.add_argument(
        "--ratio_eval",
        type=float,
        default=2,
        help="size of the stage during evaluation"
    )

    return parser.parse_args()
