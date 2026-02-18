"""
Racing Game ML Training Environment
Python implementation of racing game physics for ML training

Modules:
    environment: Gymnasium-compatible racing environment
    training: PPO training pipeline and utilities
"""

__version__ = "0.1.0"

# Make submodules available
from . import environment

# Training module requires stable-baselines3
try:
    from . import training
except ImportError:
    training = None  # Will be None if stable-baselines3 not installed
