# ppo-lunarlander-rlhf
PPO agent trained on LunarLander-v3 using Stable-Baselines3 + PyTorch, built as a production-style RL pipeline: config-driven hyperparameters, parallel environment rollouts, checkpointing, WandB tracking, hyperparameter sweeps, and standalone deterministic evaluation. Achieves a mean reward of ~252 over 20 episodes (solved threshold: 200).
