"""
Main Training Script for Racing Game ML

CLI interface for training PPO models on the racing environment.
Supports configurable hyperparameters, checkpoint saving, and
logging to TensorBoard.
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.environment.racing_env import RacingEnv
from src.training.config import (
    TrainingConfig,
    get_quick_test_config,
    get_development_config,
    get_production_config,
)
from src.training.callbacks import create_training_callbacks, RacingMetricsCallback


def make_env(rank: int, seed: int = 0, **env_kwargs):
    """
    Create a function that returns a new environment instance.

    Args:
        rank: Environment rank (for seeding)
        seed: Base random seed
        **env_kwargs: Environment arguments

    Returns:
        Function that creates the environment
    """
    def _init():
        env = RacingEnv(**env_kwargs)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


def create_vec_env(
    n_envs: int,
    seed: Optional[int] = None,
    use_subprocess: bool = True,
    **env_kwargs,
):
    """
    Create a vectorized environment for training.

    Args:
        n_envs: Number of parallel environments
        seed: Random seed
        use_subprocess: Use SubprocVecEnv (faster but more memory)
        **env_kwargs: Environment arguments

    Returns:
        Vectorized environment wrapped with VecMonitor
    """
    seed = seed or int(time.time())

    env_fns = [make_env(i, seed, **env_kwargs) for i in range(n_envs)]

    if use_subprocess and n_envs > 1:
        # SubprocVecEnv is faster for CPU-bound environments
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)

    # Wrap with VecMonitor for logging
    vec_env = VecMonitor(vec_env)

    return vec_env


def get_policy_kwargs(config: TrainingConfig) -> dict:
    """
    Get policy network configuration.

    Args:
        config: Training configuration

    Returns:
        Policy kwargs dictionary
    """
    # Map activation function names to torch modules
    activation_fns = {
        "tanh": torch.nn.Tanh,
        "relu": torch.nn.ReLU,
        "elu": torch.nn.ELU,
        "leaky_relu": torch.nn.LeakyReLU,
    }

    activation_fn = activation_fns.get(config.activation_fn, torch.nn.Tanh)

    return {
        "net_arch": dict(
            pi=config.net_arch,  # Policy network
            vf=config.net_arch,  # Value network
        ),
        "activation_fn": activation_fn,
    }


def train(
    config: TrainingConfig,
    experiment_name: Optional[str] = None,
    resume_from: Optional[str] = None,
) -> PPO:
    """
    Train a PPO model on the racing environment.

    Args:
        config: Training configuration
        experiment_name: Name for this experiment (for logging)
        resume_from: Path to model to resume training from

    Returns:
        Trained PPO model
    """
    # Create experiment name if not provided
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"racing_ppo_{timestamp}"

    # Set up paths
    base_path = Path(config.model_save_path) / experiment_name
    base_path.mkdir(parents=True, exist_ok=True)

    tensorboard_path = Path(config.tensorboard_log) / experiment_name

    print("=" * 60)
    print("RACING GAME ML - TRAINING")
    print("=" * 60)
    print(f"Experiment: {experiment_name}")
    print(f"Model path: {base_path}")
    print(f"TensorBoard: {tensorboard_path}")
    print()

    # Save configuration
    config.save(str(base_path / "config.json"))
    print("Configuration saved.")

    # Create training environment
    print(f"\nCreating {config.n_envs} parallel environments...")
    env_kwargs = config.get_env_params()
    train_env = create_vec_env(
        n_envs=config.n_envs,
        seed=config.seed,
        use_subprocess=config.n_envs > 1,
        **env_kwargs,
    )
    print(f"Observation space: {train_env.observation_space}")
    print(f"Action space: {train_env.action_space}")

    # Create evaluation environment (single env, no subprocess)
    eval_env = create_vec_env(
        n_envs=1,
        seed=(config.seed or 0) + 1000,
        use_subprocess=False,
        **env_kwargs,
    )

    # Create or load model
    if resume_from:
        print(f"\nResuming training from: {resume_from}")
        model = PPO.load(
            resume_from,
            env=train_env,
            device=config.device,
            tensorboard_log=str(tensorboard_path),
        )
    else:
        print("\nCreating new PPO model...")
        ppo_params = config.get_ppo_params()
        ppo_params["tensorboard_log"] = str(tensorboard_path)

        policy_kwargs = get_policy_kwargs(config)

        model = PPO(
            policy=config.policy_type,
            env=train_env,
            policy_kwargs=policy_kwargs,
            **ppo_params,
        )

    # Print model summary
    print(f"\nModel architecture:")
    print(f"  Policy: {config.policy_type}")
    print(f"  Network: {config.net_arch}")
    print(f"  Activation: {config.activation_fn}")
    print(f"  Device: {model.device}")
    print()

    # Create callbacks
    callbacks = create_training_callbacks(
        model_save_path=str(base_path),
        eval_env=eval_env,
        save_freq=config.save_freq // config.n_envs,  # Adjust for vectorized env
        eval_freq=config.eval_freq // config.n_envs,
        n_eval_episodes=config.n_eval_episodes,
        log_freq=1000,
        total_timesteps=config.total_timesteps,
        verbose=config.verbose,
    )

    # Print training configuration
    print("Training configuration:")
    print(f"  Total timesteps: {config.total_timesteps:,}")
    print(f"  Parallel envs: {config.n_envs}")
    print(f"  Steps per update: {config.n_steps * config.n_envs:,}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Gamma: {config.gamma}")
    print(f"  Entropy coef: {config.ent_coef}")
    print()

    # Estimate training time
    steps_per_second = 2000  # Rough estimate
    estimated_time = config.total_timesteps / steps_per_second
    print(f"Estimated training time: {estimated_time / 60:.0f} minutes")
    print()

    # Start training
    print("Starting training...")
    print("-" * 60)
    start_time = time.time()

    try:
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=callbacks,
            log_interval=config.log_interval,
            progress_bar=config.verbose == 0,
            reset_num_timesteps=resume_from is None,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    finally:
        # Save final model
        final_model_path = base_path / "final_model"
        model.save(str(final_model_path))
        print(f"\nFinal model saved to: {final_model_path}")

        # Clean up
        train_env.close()
        eval_env.close()

    # Print final statistics
    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed / 60:.1f} minutes")
    print(f"Timesteps per second: {config.total_timesteps / elapsed:.0f}")

    return model


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train a PPO model for the Racing Game",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Preset configurations
    parser.add_argument(
        "--preset",
        type=str,
        choices=["quick", "dev", "prod"],
        help="Use a preset configuration (quick=10k, dev=100k, prod=1M timesteps)",
    )

    # Basic training parameters
    parser.add_argument(
        "--timesteps", "-t",
        type=int,
        help="Total training timesteps",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        help="Number of parallel environments",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed",
    )

    # PPO hyperparameters
    parser.add_argument(
        "--learning-rate", "--lr",
        type=float,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size for PPO updates",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        help="Number of epochs per PPO update",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        help="Discount factor",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        help="Entropy coefficient",
    )

    # Network architecture
    parser.add_argument(
        "--net-arch",
        type=str,
        help="Network architecture (e.g., '64,64' or '128,128,64')",
    )

    # Paths
    parser.add_argument(
        "--model-path",
        type=str,
        help="Directory to save models",
    )
    parser.add_argument(
        "--tensorboard-log",
        type=str,
        help="TensorBoard log directory",
    )
    parser.add_argument(
        "--experiment-name", "-n",
        type=str,
        help="Experiment name (for organizing outputs)",
    )

    # Resume training
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to model to resume training from",
    )

    # Load config file
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration JSON file",
    )

    # Output options
    parser.add_argument(
        "--verbose", "-v",
        type=int,
        choices=[0, 1, 2],
        default=1,
        help="Verbosity level",
    )

    return parser.parse_args()


def main():
    """Main entry point for training"""
    args = parse_args()

    # Start with default config or preset
    if args.config:
        print(f"Loading config from: {args.config}")
        config = TrainingConfig.load(args.config)
    elif args.preset:
        presets = {
            "quick": get_quick_test_config,
            "dev": get_development_config,
            "prod": get_production_config,
        }
        print(f"Using preset configuration: {args.preset}")
        config = presets[args.preset]()
    else:
        config = TrainingConfig()

    # Override with command line arguments
    if args.timesteps is not None:
        config.total_timesteps = args.timesteps
    if args.n_envs is not None:
        config.n_envs = args.n_envs
    if args.seed is not None:
        config.seed = args.seed
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.n_epochs is not None:
        config.n_epochs = args.n_epochs
    if args.gamma is not None:
        config.gamma = args.gamma
    if args.ent_coef is not None:
        config.ent_coef = args.ent_coef
    if args.net_arch is not None:
        config.net_arch = [int(x) for x in args.net_arch.split(",")]
    if args.model_path is not None:
        config.model_save_path = args.model_path
    if args.tensorboard_log is not None:
        config.tensorboard_log = args.tensorboard_log
    if args.verbose is not None:
        config.verbose = args.verbose

    # Train the model
    model = train(
        config=config,
        experiment_name=args.experiment_name,
        resume_from=args.resume,
    )

    print("\nTraining complete!")
    print(f"To evaluate: python -m src.training.evaluate models/<experiment>/final_model")
    print(f"To view TensorBoard: tensorboard --logdir {config.tensorboard_log}")


if __name__ == "__main__":
    main()
