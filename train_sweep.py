"""
Sweep-compatible training entry point. This is what sweep.yaml's
`program:` field points to — a normal `python train.py` won't work with
`wandb agent`, because train.py reads hyperparameters from config.py,
not from wandb.config.

This wrapper: initializes wandb first (which lets the sweep agent inject
this run's sampled hyperparameters into wandb.config), overrides the
relevant config.py values with those sampled values, then reuses train.py's
existing build_vec_env / build_eval_env / build_model functions so there's
no duplicated training logic.
"""

import wandb
from stable_baselines3.common.callbacks import CallbackList

import config
from utils import set_seed, ensure_dirs
from callbacks import get_checkpoint_callback, get_eval_callback
from train import build_vec_env, build_eval_env, build_model


def sweep_train():
    wandb.init(dir=config.LOG_DIR, sync_tensorboard=True, monitor_gym=True)
    cfg = wandb.config

    # Override config.py defaults with this run's sampled hyperparameters
    config.LEARNING_RATE = cfg.learning_rate
    config.CLIP_RANGE = cfg.clip_range
    config.GAE_LAMBDA = cfg.gae_lambda
    config.GAMMA = cfg.gamma
    config.ENT_COEF = cfg.ent_coef
    config.N_STEPS = cfg.n_steps
    config.BATCH_SIZE = cfg.batch_size
    config.N_EPOCHS = cfg.n_epochs

    # Shorter budget per sweep run than a full production run — the point
    # here is to compare configs quickly, not to fully train each one.
    config.TOTAL_TIMESTEPS = 300_000

    ensure_dirs(config.CHECKPOINT_PATH, config.TENSORBOARD_LOG, config.MODEL_DIR)
    set_seed(config.SEED)

    vec_env = build_vec_env()
    eval_env = build_eval_env()
    model = build_model(vec_env)

    callbacks = CallbackList([
        get_checkpoint_callback(),
        get_eval_callback(eval_env),
    ])

    model.learn(total_timesteps=config.TOTAL_TIMESTEPS, callback=callbacks)

    vec_env.close()
    eval_env.close()
    wandb.finish()


if __name__ == "__main__":
    sweep_train()