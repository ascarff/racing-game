"""
Training Module for Racing Game ML

This module provides tools for training PPO models on the racing environment.

Components:
- config: Training configuration and hyperparameters
- train: Main training script and functions
- callbacks: Custom callbacks for logging and checkpoints
- evaluate: Model evaluation utilities
- export: Export models for frontend inference

Usage:
    # From command line:
    python -m src.training.train --preset dev

    # Or programmatically:
    from src.training import TrainingConfig, train

    config = TrainingConfig(total_timesteps=100_000)
    model = train(config)

Requirements:
    pip install stable-baselines3 torch tensorboard
"""

# Config module has no external dependencies beyond stdlib
from .config import (
    TrainingConfig,
    get_quick_test_config,
    get_development_config,
    get_production_config,
    get_hyperparameter_search_configs,
)

__all__ = [
    # Config (always available)
    "TrainingConfig",
    "get_quick_test_config",
    "get_development_config",
    "get_production_config",
    "get_hyperparameter_search_configs",
]

# These modules require stable-baselines3
_SB3_AVAILABLE = False
_SB3_INSTALL_MSG = (
    "Training features require stable-baselines3. "
    "Install with: pip install stable-baselines3 torch tensorboard"
)
try:
    import stable_baselines3
    _SB3_AVAILABLE = True
except ImportError:
    import warnings
    warnings.warn(_SB3_INSTALL_MSG, ImportWarning)

if _SB3_AVAILABLE:
    from .callbacks import (
        RacingMetricsCallback,
        BestModelCallback,
        ProgressBarCallback,
        create_training_callbacks,
    )

    from .evaluate import (
        EpisodeResult,
        EvaluationResults,
        load_model,
        create_eval_env,
        evaluate_episode,
        evaluate_model,
        compare_models,
        save_evaluation_results,
        watch_model,
    )

    from .export import (
        export_to_onnx,
        export_to_json,
        create_inference_code,
        export_model,
    )

    from .train import (
        make_env,
        create_vec_env,
        get_policy_kwargs,
        train,
    )

    __all__.extend([
        # Callbacks
        "RacingMetricsCallback",
        "BestModelCallback",
        "ProgressBarCallback",
        "create_training_callbacks",
        # Evaluate
        "EpisodeResult",
        "EvaluationResults",
        "load_model",
        "create_eval_env",
        "evaluate_episode",
        "evaluate_model",
        "compare_models",
        "save_evaluation_results",
        "watch_model",
        # Export
        "export_to_onnx",
        "export_to_json",
        "create_inference_code",
        "export_model",
        # Train
        "make_env",
        "create_vec_env",
        "get_policy_kwargs",
        "train",
    ])
