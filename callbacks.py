"""
All training callbacks in one place. Kept separate from train.py so the
training script stays readable, and so callbacks can be reused or tested
independently.

Three callbacks, each doing one job (Day 17 habits, turned into code):
  - checkpoint_callback : periodic saves, so a crash mid-run doesn't lose progress
  - eval_callback       : separate, deterministic evaluation during training,
                           and automatically keeps the best-performing model
  - wandb_callback       : optional cloud logging (only created if enabled)
"""

from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

import config


def get_checkpoint_callback():
    return CheckpointCallback(
        save_freq=config.CHECKPOINT_FREQ,
        save_path=config.CHECKPOINT_PATH,
        name_prefix="ppo_lander",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )


def get_eval_callback(eval_env):
    """
    eval_env should be a SEPARATE environment instance from the training
    envs (never reuse a training env for evaluation — Day 17 debugging
    note: never judge a policy using data it was trained on).
    """
    return EvalCallback(
        eval_env,
        best_model_save_path=config.BEST_MODEL_DIR,
        log_path=config.TENSORBOARD_LOG,
        eval_freq=config.EVAL_FREQ,
        n_eval_episodes=config.N_EVAL_EPISODES,
        deterministic=True,
        render=False,
    )


def get_wandb_callback():
    """
    Returns None if wandb isn't installed or is disabled in config, so
    train.py can safely append this to its callback list without extra
    branching logic.
    """
    if not config.USE_WANDB:
        return None
    try:
        from wandb.integration.sb3 import WandbCallback
        return WandbCallback(
            gradient_save_freq=100,
            model_save_path=config.BEST_MODEL_DIR,
            verbose=2,
        )
    except ImportError:
        print("[callbacks] wandb not installed — skipping WandB logging. "
              "Run `pip install wandb` to enable it.")
        return None
