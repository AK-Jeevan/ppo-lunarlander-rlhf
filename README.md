# PPO LunarLander Project

This project trains a Proximal Policy Optimization (PPO) agent to solve the Gymnasium environment LunarLander-v3 using Stable Baselines 3.

It includes:
- PPO training with parallel vectorized environments
- Periodic checkpointing
- Online evaluation during training
- TensorBoard logging
- Optional Weights & Biases logging
- Standalone evaluation and optional video recording

## Project Overview

The training loop is implemented in [train.py](train.py), evaluation is handled by [evaluate.py](evaluate.py), and all configurable hyperparameters and paths live in [config.py](config.py).

The project is designed to be reproducible: you can change training settings in one place and rerun the pipeline without editing multiple scripts.

## Requirements

Make sure you have Python installed, then install the dependencies:

```bash
pip install -r requirements.txt
```

If you want Weights & Biases logging enabled, install:

```bash
pip install wandb
```

## Project Structure

- [config.py](config.py) — all hyperparameters, environment settings, and output paths
- [train.py](train.py) — training entry point
- [evaluate.py](evaluate.py) — standalone evaluation entry point
- [utils.py](utils.py) — seeding, environment creation, and directory helpers
- [callbacks.py](callbacks.py) — checkpointing, evaluation, and logging callbacks
- [requirements.txt](requirements.txt) — Python dependencies
- [checkpoints/](checkpoints) — saved model checkpoints
- [logs/](logs) — TensorBoard and WandB logs
- [models/](models) — final model artifacts
- [videos/](videos) — evaluation videos when recording is enabled

## Training

Run training with:

```bash
python train.py
```

This will:
1. Create the required output folders
2. Start training a PPO policy in LunarLander-v3
3. Save checkpoints during training
4. Save the best model found during evaluation
5. Write logs to TensorBoard and optionally Weights & Biases

To monitor training progress, open a second terminal and run:

```bash
tensorboard --logdir ./logs/tensorboard
```

## Evaluation

Evaluate the final trained model:

```bash
python evaluate.py
```

Evaluate the best saved checkpoint:

```bash
python evaluate.py --model checkpoints/best_model/best_model.zip
```

Render the evaluation live:

```bash
python evaluate.py --render
```

Record evaluation episodes as videos:

```bash
python evaluate.py --record_video
```

## Configuration

Adjust training behavior in [config.py](config.py), including:
- environment name
- total training timesteps
- PPO hyperparameters
- evaluation frequency
- logging settings
- checkpoint and model paths

## Notes

The default success threshold is set in [config.py](config.py) as 200 reward for LunarLander-v3. If the average evaluation reward meets or exceeds that value, the run is marked as solved.

## Example Workflow

```bash
pip install -r requirements.txt
python train.py
python evaluate.py --model checkpoints/best_model/best_model.zip --record_video
```
