SEED = 42

########################################################################
# Environment Configuration

NUM_ENVS = 64
NUMBER_AGENTS = 4
MAX_EPISODE_LENGTH = 400

ENV_NAME = "simple_spread"

ENV_CONTINUOUS_ACTION = True

DEVICE = None

#########################################################################
# Model Hyperparameters

ACTOR_MODULE = "RandomAgentPolicy"
CRITIC_MODULE = "RAP_qvalue"

ACTOR_LR = 1e-4
CRITIC_LR = 1e-4
ALPHA_LR = 1e-5
TAU = 0.005
GAMMA = 0.99

INITIAL_ALPHA = 3.0
ALPHA_MIN = 0.0001
ALPHA_MAX = 10.0

TARGET_ENTROPY = -2.0 * NUMBER_AGENTS

ACTOR_HIDDEN_DIM = 64

# Training parameters

RESUME_FROM_CHECKPOINT = False

BUFFER_SIZE = 2048000
BUFFER_DEVICE = None
BATCH_SIZE = 1024

UPDATE_EVERY = NUM_ENVS
UPDATE_TIMES = 1
UPDATE_START = NUM_ENVS * MAX_EPISODE_LENGTH

UPDATE_ACTOR_EVERY_CRITIC = 1

ACTOR_UPDATE_START = 0

NUM_TIMESTEPS = 80000000

ACTOR_OPTIMIZER = "Adam"
ACTOR_OPTIMIZER_PARAMS = {}
CRITIC_OPTIMIZER_PARAMS = {}
CRITIC_OPTIMIZER = "Adam"

ALPHA_OPTIMIZER = "Adam"
ALPHA_OPTIMIZER_PARAMS = {}

REWARD_SCALE = 1.0 / NUMBER_AGENTS

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