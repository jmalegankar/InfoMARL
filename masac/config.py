SEED = 42

########################################################################
# Environment Configuration

NUM_ENVS = 3
NUMBER_AGENTS = 4
MAX_EPISODE_LENGTH = 2

ENV_NAME = "simple_spread"

ENV_CONTINUOUS_ACTION = True

DEVICE = None

#########################################################################
# Model Hyperparameters

ACTOR_MODULE = "RandomAgentPolicy"


ACTOR_LR = 1e-5
PPO_CLIP_PARAM = 0.2
PPO_EPOCHS = 10
PPO_VALUE_COEF = 0.5
PPO_ENTROPY_COEF = 0.01
PPO_MAX_GRAD_NORM = 0.5
PPO_USE_CLIP_VALUE = True

# GAE parameters
GAMMA = 0.99
GAE_LAMBDA = 0.95

ACTOR_HIDDEN_DIM = 64

# Training parameters

RESUME_FROM_CHECKPOINT = True

ROLLOUT_STEPS = 2048  # Steps to collect before updating
BATCH_SIZE = 256

BUFFER_SIZE = 2048000
BUFFER_DEVICE = None
BATCH_SIZE = 1024


ACTOR_UPDATE_START = 30000

NUM_TIMESTEPS = 80000000

ACTOR_OPTIMIZER = "Adam"
ACTOR_OPTIMIZER_PARAMS = {
    "eps": 1e-5
}

LR_SCHEDULER = True


REWARD_SCALE = 1.0

#########################################################################
# Logging configuration

CHECKPOINT_DIR = f"checkpoints/{ENV_NAME}/"

CHECKPOINT_INTERVAL_UPDATE = 10000

MAX_CHECKPOINTS = 2

SAVE_GIF_INTERVAL = 10000

LOG_DIR = f"logs/{ENV_NAME}/"

BASIC_CONFIG = {
    'level': 20,
}