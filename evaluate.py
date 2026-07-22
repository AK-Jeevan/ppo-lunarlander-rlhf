"""
Standalone evaluation entry point — deliberately separate from train.py.
Never trust numbers computed during training itself; this loads a saved
checkpoint fresh, in a clean environment, and evaluates it in isolation.

    python evaluate.py                          # evaluate final_model, no video
    python evaluate.py --model checkpoints/best_model/best_model.zip
    python evaluate.py --render                 # watch live (needs a display)
    python evaluate.py --record_video            # save an mp4 to ./videos
"""

import argparse
import os

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

import config


def evaluate(model_path: str, episodes: int, render: bool, record_video: bool):
    render_mode = "human" if render else ("rgb_array" if record_video else None)
    env = gym.make(config.ENV_NAME, render_mode=render_mode)

    if record_video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=config.VIDEO_DIR,
            episode_trigger=lambda ep: True,
            name_prefix="ppo_lander_eval",
        )

    env = Monitor(env)
    model = PPO.load(model_path)

    rewards = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=config.SEED + 2000 + episode)
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        rewards.append(total_reward)
        print(f"Episode {episode + 1:2d} | Reward = {total_reward:.2f}")

    env.close()

    mean_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))
    print("-" * 40)
    print(f"Mean reward over {episodes} episodes: {mean_reward:.2f} +/- {std_reward:.2f}")

    status = "SOLVED" if mean_reward >= config.SOLVED_REWARD_THRESHOLD else "NOT SOLVED YET"
    print(f"Solved threshold ({config.SOLVED_REWARD_THRESHOLD}): {status}")

    if record_video:
        print(f"Videos saved to {os.path.abspath(config.VIDEO_DIR)}")

    return mean_reward, std_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=f"{config.MODEL_PATH}.zip",
                         help="Path to a saved .zip checkpoint")
    parser.add_argument("--episodes", type=int, default=config.N_EVAL_EPISODES)
    parser.add_argument("--render", action="store_true",
                         help="Render episodes live (requires a local display)")
    parser.add_argument("--record_video", action="store_true",
                         help="Save evaluation episodes as mp4 to ./videos")
    args = parser.parse_args()

    evaluate(args.model, args.episodes, args.render, args.record_video)
