"""
Shared helpers used by both train.py and evaluate.py.
"""

import os
import random

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor


def set_seed(seed: int):
    """
    Seed every RNG that RL reproducibility actually depends on.
    Python's `random` and NumPy are used inside many env/wrapper internals,
    PyTorch controls the policy/value network initialization and sampling.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(env_name: str, seed: int, rank: int = 0, log_dir: str = None):
    """
    Factory that returns a function creating one wrapped, seeded env
    instance. Used by SB3's vectorized env constructors, which need a
    *callable* per sub-environment, not an already-built env.

    `rank` offsets the seed per parallel env so each one explores a
    different part of the environment's randomness instead of all 8
    copies producing identical trajectories.
    """
    def _init():
        env = gym.make(env_name)
        env.reset(seed=seed + rank)
        env.action_space.seed(seed + rank)
        monitor_path = os.path.join(log_dir, str(rank)) if log_dir else None
        env = Monitor(env, filename=monitor_path)
        return env
    return _init


def ensure_dirs(*paths):
    """Create any output directories that don't exist yet."""
    for path in paths:
        os.makedirs(path, exist_ok=True)
