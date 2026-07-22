"""
Central config — every hyperparameter and path used by train.py / evaluate.py
lives here. Nothing should be hardcoded anywhere else, so a full run is
always reproducible from this one file.
"""

import os

# ------------------------------------
# Environment
# ------------------------------------
ENV_NAME = "LunarLander-v3"
SEED = 42
N_ENVS = 8                    # parallel envs used to collect rollouts during training

# ------------------------------------
# PPO hyperparameters (Days 13-14)
# ------------------------------------
TOTAL_TIMESTEPS = 1_000_000
LEARNING_RATE = 3e-4
GAMMA = 0.999                  # discount factor
GAE_LAMBDA = 0.98              # lambda in Generalized Advantage Estimation
N_STEPS = 1024                  # steps collected per env before each update
BATCH_SIZE = 64
N_EPOCHS = 4                    # passes over each batch per update
CLIP_RANGE = 0.2                # epsilon in the clipped objective
ENT_COEF = 0.01                 # entropy bonus — keeps exploration alive
VF_COEF = 0.5                   # weight of the Critic's value loss
MAX_GRAD_NORM = 0.5             # gradient clipping, separate from PPO's own clipping

# ------------------------------------
# Checkpointing
# ------------------------------------
CHECKPOINT_FREQ = 10_000        # env steps between checkpoint saves
CHECKPOINT_PATH = "./checkpoints"
KEEP_LAST_N_CHECKPOINTS = 5

# ------------------------------------
# Evaluation (run periodically during training, and standalone in evaluate.py)
# ------------------------------------
EVAL_FREQ = 25_000              # env steps between in-training evaluations
N_EVAL_EPISODES = 20
SOLVED_REWARD_THRESHOLD = 200   # LunarLander-v2 is considered solved at ~200

# ------------------------------------
# Logging
# ------------------------------------
LOG_DIR = "./logs"
TENSORBOARD_LOG = os.path.join(LOG_DIR, "tensorboard")
WANDB_DIR = os.path.join(LOG_DIR, "wandb")
USE_WANDB = True
WANDB_PROJECT = "PPO-LunarLander"

# ------------------------------------
# Model / video output
# ------------------------------------
MODEL_DIR = "./models"
MODEL_PATH = os.path.join(MODEL_DIR, "final_model")   # SB3 appends .zip automatically
BEST_MODEL_DIR = os.path.join(CHECKPOINT_PATH, "best_model")
VIDEO_DIR = "./videos"
