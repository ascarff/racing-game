"""
Training Configuration for Racing Game ML

This module provides a dataclass-based configuration system for PPO training.
All hyperparameters can be customized through code or CLI arguments.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import json
from pathlib import Path


@dataclass
class TrainingConfig:
    """
    Configuration for PPO training on the Racing Environment

    Attributes:
        total_timesteps: Total number of environment steps for training
        n_envs: Number of parallel environments for training
        learning_rate: Learning rate for the optimizer
        n_steps: Number of steps per environment per update
        batch_size: Minibatch size for PPO updates
        n_epochs: Number of epochs for each PPO update
        gamma: Discount factor for rewards
        gae_lambda: GAE lambda for advantage estimation
        clip_range: PPO clipping parameter
        clip_range_vf: Value function clipping parameter (None = no clipping)
        ent_coef: Entropy coefficient for exploration
        vf_coef: Value function coefficient in loss
        max_grad_norm: Maximum gradient norm for clipping
        target_kl: Target KL divergence for early stopping (None = disabled)

        # Network architecture
        policy_type: Policy type (MlpPolicy for this task)
        net_arch: Network architecture for policy and value networks
        activation_fn: Activation function name ('tanh', 'relu', 'elu')

        # Environment settings
        max_episode_steps: Maximum steps per episode
        reward_progress_weight: Weight for progress reward
        reward_speed_weight: Weight for speed reward
        reward_collision_penalty: Penalty for wall collisions
        reward_lap_complete: Reward for completing a lap
        reward_time_penalty: Time penalty per step

        # Saving and logging
        save_freq: Save checkpoint every N timesteps
        eval_freq: Evaluate model every N timesteps
        n_eval_episodes: Number of episodes for evaluation
        log_interval: Log metrics every N updates
        tensorboard_log: Directory for TensorBoard logs
        model_save_path: Directory for model checkpoints

        # Experiment settings
        seed: Random seed for reproducibility
        device: Device for training ('auto', 'cpu', 'cuda')
        verbose: Verbosity level (0=none, 1=info, 2=debug)
    """
    # PPO hyperparameters
    total_timesteps: int = 500_000
    n_envs: int = 8
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None

    # Network architecture
    policy_type: str = "MlpPolicy"
    net_arch: List[int] = field(default_factory=lambda: [64, 64])
    activation_fn: str = "tanh"

    # Environment settings
    max_episode_steps: int = 3000
    reward_progress_weight: float = 1.0
    reward_speed_weight: float = 0.1
    reward_collision_penalty: float = -5.0
    reward_lap_complete: float = 100.0
    reward_time_penalty: float = -0.01

    # Saving and logging
    save_freq: int = 10_000
    eval_freq: int = 10_000
    n_eval_episodes: int = 5
    log_interval: int = 1
    tensorboard_log: str = "./logs/tensorboard"
    model_save_path: str = "./models"

    # Experiment settings
    seed: Optional[int] = None
    device: str = "auto"
    verbose: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "total_timesteps": self.total_timesteps,
            "n_envs": self.n_envs,
            "learning_rate": self.learning_rate,
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_range": self.clip_range,
            "clip_range_vf": self.clip_range_vf,
            "ent_coef": self.ent_coef,
            "vf_coef": self.vf_coef,
            "max_grad_norm": self.max_grad_norm,
            "target_kl": self.target_kl,
            "policy_type": self.policy_type,
            "net_arch": self.net_arch,
            "activation_fn": self.activation_fn,
            "max_episode_steps": self.max_episode_steps,
            "reward_progress_weight": self.reward_progress_weight,
            "reward_speed_weight": self.reward_speed_weight,
            "reward_collision_penalty": self.reward_collision_penalty,
            "reward_lap_complete": self.reward_lap_complete,
            "reward_time_penalty": self.reward_time_penalty,
            "save_freq": self.save_freq,
            "eval_freq": self.eval_freq,
            "n_eval_episodes": self.n_eval_episodes,
            "log_interval": self.log_interval,
            "tensorboard_log": self.tensorboard_log,
            "model_save_path": self.model_save_path,
            "seed": self.seed,
            "device": self.device,
            "verbose": self.verbose,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrainingConfig":
        """Create config from dictionary"""
        return cls(**{k: v for k, v in config_dict.items() if hasattr(cls, k)})

    def save(self, filepath: str) -> None:
        """Save config to JSON file"""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "TrainingConfig":
        """Load config from JSON file"""
        with open(filepath, "r") as f:
            return cls.from_dict(json.load(f))

    def get_ppo_params(self) -> Dict[str, Any]:
        """Get parameters for PPO model initialization"""
        return {
            "learning_rate": self.learning_rate,
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_range": self.clip_range,
            "clip_range_vf": self.clip_range_vf,
            "ent_coef": self.ent_coef,
            "vf_coef": self.vf_coef,
            "max_grad_norm": self.max_grad_norm,
            "target_kl": self.target_kl,
            "verbose": self.verbose,
            "device": self.device,
            "seed": self.seed,
            "tensorboard_log": self.tensorboard_log,
        }

    def get_env_params(self) -> Dict[str, Any]:
        """Get parameters for environment initialization"""
        return {
            "max_steps": self.max_episode_steps,
            "reward_progress_weight": self.reward_progress_weight,
            "reward_speed_weight": self.reward_speed_weight,
            "reward_collision_penalty": self.reward_collision_penalty,
            "reward_lap_complete": self.reward_lap_complete,
            "reward_time_penalty": self.reward_time_penalty,
        }


# Predefined configurations for different training scenarios

def get_quick_test_config() -> TrainingConfig:
    """Quick test configuration (few timesteps, fast iteration)"""
    return TrainingConfig(
        total_timesteps=10_000,
        n_envs=4,
        save_freq=2_000,
        eval_freq=2_000,
        verbose=2,
    )


def get_development_config() -> TrainingConfig:
    """Development configuration (moderate training)"""
    return TrainingConfig(
        total_timesteps=100_000,
        n_envs=4,
        save_freq=10_000,
        eval_freq=10_000,
    )


def get_production_config() -> TrainingConfig:
    """Production configuration (full training)"""
    return TrainingConfig(
        total_timesteps=1_000_000,
        n_envs=8,
        save_freq=50_000,
        eval_freq=25_000,
        n_eval_episodes=10,
    )


def get_hyperparameter_search_configs() -> List[TrainingConfig]:
    """Generate configs for hyperparameter search"""
    configs = []

    # Learning rate search
    for lr in [1e-4, 3e-4, 1e-3]:
        for ent_coef in [0.0, 0.01, 0.1]:
            config = TrainingConfig(
                total_timesteps=50_000,
                learning_rate=lr,
                ent_coef=ent_coef,
                n_envs=4,
            )
            configs.append(config)

    return configs
