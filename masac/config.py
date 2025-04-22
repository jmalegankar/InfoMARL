SEED = 42

########################################################################
# Environment Configuration

NUM_ENVS = 96
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

INITIAL_ALPHA = 5.0
ALPHA_MIN = 0.1
ALPHA_MAX = 10.0

TARGET_ENTROPY = -2.0

ACTOR_HIDDEN_DIM = 64

# Training parameters

RESUME_FROM_CHECKPOINT = False

BUFFER_SIZE = 2000000
BUFFER_DEVICE = None
BATCH_SIZE = 1024

UPDATE_EVERY = NUM_ENVS*5
UPDATE_TIMES = 1
UPDATE_START = BATCH_SIZE * 100

UPDATE_ACTOR_EVERY_CRITIC = 10

NUM_TIMESTEPS = 40000000

ACTOR_OPTIMIZER = "Adam"
ACTOR_OPTIMIZER_PARAMS = {}
CRITIC_OPTIMIZER_PARAMS = {}
CRITIC_OPTIMIZER = "Adam"

ALPHA_OPTIMIZER = "Adam"
ALPHA_OPTIMIZER_PARAMS = {}

#########################################################################
# Logging configuration

CHECKPOINT_DIR = "checkpoints"

CHECKPOINT_INTERVAL_UPDATE = 1000

SAVE_GIF_INTERVAL = 10000

LOG_DIR = "logs"

BASIC_CONFIG = {}