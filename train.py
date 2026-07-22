"""
Training entry point.

    python train.py

Everything (hyperparameters, paths, wandb toggle) comes from config.py.
While this runs, in a separate terminal:

    tensorboard --logdir ./logs/tensorboard

Watch rollout/ep_rew_mean, train/approx_kl, and train/clip_fraction to
sanity-check training health, not just "is it not crashing."
"""

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor

import config
from utils import set_seed, make_env, ensure_dirs
from callbacks import get_checkpoint_callback, get_eval_callback, get_wandb_callback


def build_vec_env():
    """N_ENVS parallel, independently-seeded copies of the environment,
    used to collect a bigger, more varied batch of rollouts per update."""
    env_fns = [
        make_env(config.ENV_NAME, config.SEED, rank=i, log_dir=config.TENSORBOARD_LOG)
        for i in range(config.N_ENVS)
    ]
    return SubprocVecEnv(env_fns)


def build_eval_env():
    """A single, separate environment reserved only for evaluation —
    deliberately never one of the training envs above."""
    env = gym.make(config.ENV_NAME)
    env.reset(seed=config.SEED + 1000)  # different seed range than training envs
    return Monitor(env)


def build_model(vec_env):
    return PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=config.LEARNING_RATE,
        n_steps=config.N_STEPS,
        batch_size=config.BATCH_SIZE,
        n_epochs=config.N_EPOCHS,
        gamma=config.GAMMA,
        gae_lambda=config.GAE_LAMBDA,
        clip_range=config.CLIP_RANGE,
        ent_coef=config.ENT_COEF,
        vf_coef=config.VF_COEF,
        max_grad_norm=config.MAX_GRAD_NORM,
        seed=config.SEED,
        verbose=1,
        tensorboard_log=config.TENSORBOARD_LOG,
    )


def main():
    ensure_dirs(
        config.CHECKPOINT_PATH,
        config.TENSORBOARD_LOG,
        config.MODEL_DIR,
        config.VIDEO_DIR,
    )
    set_seed(config.SEED)

    wandb_run = None
    if config.USE_WANDB:
        try:
            import wandb
            wandb_run = wandb.init(
                project=config.WANDB_PROJECT,
                dir=config.LOG_DIR,
                config={
                    "env": config.ENV_NAME,
                    "total_timesteps": config.TOTAL_TIMESTEPS,
                    "learning_rate": config.LEARNING_RATE,
                    "gamma": config.GAMMA,
                    "gae_lambda": config.GAE_LAMBDA,
                    "n_steps": config.N_STEPS,
                    "batch_size": config.BATCH_SIZE,
                    "n_epochs": config.N_EPOCHS,
                    "clip_range": config.CLIP_RANGE,
                    "ent_coef": config.ENT_COEF,
                    "seed": config.SEED,
                },
                sync_tensorboard=True,
                monitor_gym=True,
            )
        except ImportError:
            print("[train] wandb not installed — continuing with TensorBoard only.")

    vec_env = build_vec_env()
    eval_env = build_eval_env()
    model = build_model(vec_env)

    callbacks = [get_checkpoint_callback(), get_eval_callback(eval_env)]
    wandb_cb = get_wandb_callback()
    if wandb_cb is not None:
        callbacks.append(wandb_cb)

    model.learn(
        total_timesteps=config.TOTAL_TIMESTEPS,
        callback=CallbackList(callbacks),
        progress_bar=True,
    )

    model.save(config.MODEL_PATH)
    print(f"[train] Final model saved to {config.MODEL_PATH}.zip")
    print(f"[train] Best model (by eval reward) saved to {config.BEST_MODEL_DIR}")

    vec_env.close()
    eval_env.close()
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
