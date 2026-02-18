# Racing Game ML Training Environment

A Python-based ML training environment for the 2D racing game. This environment replicates the frontend game physics and provides a Gymnasium-compatible interface for training reinforcement learning agents.

## Features

- **Headless/Accelerated Mode**: Run simulations faster than real-time for efficient training
- **State Extraction**: Full access to car position, velocity, angle, track boundaries, and checkpoints
- **Programmatic Actions**: Discrete or MultiDiscrete action spaces for accelerate, brake, and steering
- **Reward Function**: Configurable rewards for progress, speed, collisions, and lap completion
- **Episode Management**: Automatic reset on crash, lap completion, or timeout
- **Parallel Training**: Support for vectorized environments (stable-baselines3 compatible)

## Installation

```bash
cd racing-game-ml
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from src.environment import RacingEnv

# Create environment
env = RacingEnv()

# Reset and get initial observation
obs, info = env.reset()

# Take a step with action 1 (accelerate)
obs, reward, terminated, truncated, info = env.step(1)

# Print state
print(f"Position: {info['position']}")
print(f"Speed: {info['speed']}")
print(f"Checkpoints: {info['checkpoints_passed']}/{info['total_checkpoints']}")
```

### Random Agent Demo

```python
from src.environment import RacingEnv

env = RacingEnv(render_mode="ansi")
obs, info = env.reset()

total_reward = 0
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    if terminated or truncated:
        print(f"Episode ended. Total reward: {total_reward:.2f}")
        print(f"Lap complete: {info['lap_complete']}")
        break

env.close()
```

### Training with Stable-Baselines3

```python
from stable_baselines3 import PPO
from src.environment import RacingEnv

# Create environment
env = RacingEnv()

# Create PPO agent
model = PPO("MlpPolicy", env, verbose=1)

# Train
model.learn(total_timesteps=100000)

# Save
model.save("racing_agent")

# Test
obs, info = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
```

### Vectorized Training (Parallel)

```python
from stable_baselines3 import PPO
from src.environment import make_vec_env

# Create 8 parallel environments
vec_env = make_vec_env(num_envs=8)

# Train with parallel environments
model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=500000)
```

## Environment Details

### Observation Space

The observation is a 15-dimensional vector (Box space, values normalized to [-1, 1]):

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | x | Normalized X position (0-1) |
| 1 | y | Normalized Y position (0-1) |
| 2 | cos(angle) | Heading X component |
| 3 | sin(angle) | Heading Y component |
| 4 | speed | Normalized speed (-1 to 1) |
| 5 | velocity_x | Normalized X velocity |
| 6 | velocity_y | Normalized Y velocity |
| 7 | is_colliding | Collision flag (0 or 1) |
| 8-12 | distances | Distance to track boundaries (5 rays) |
| 13 | track_progress | Progress around track (0 to 1) |
| 14 | checkpoint_progress | Checkpoints passed ratio |

### Action Space

**Discrete (default)**: 9 actions

| Action | Description |
|--------|-------------|
| 0 | No input (coast) |
| 1 | Accelerate |
| 2 | Brake |
| 3 | Steer left |
| 4 | Steer right |
| 5 | Accelerate + steer left |
| 6 | Accelerate + steer right |
| 7 | Brake + steer left |
| 8 | Brake + steer right |

**MultiDiscrete (optional)**: `[3, 3]`
- First value: 0=none, 1=accelerate, 2=brake
- Second value: 0=none, 1=steer left, 2=steer right

To use MultiDiscrete:
```python
env = RacingEnv(use_multi_discrete=True)
```

### Reward Function

| Component | Default Weight | Description |
|-----------|----------------|-------------|
| Progress | 1.0 | Reward for forward movement around track |
| Speed | 0.1 | Reward for maintaining high speed |
| Collision | -5.0 | Penalty per collision step |
| Lap Complete | 100.0 | Bonus for finishing a lap |
| Time | -0.01 | Small penalty per timestep |

Customize rewards:
```python
env = RacingEnv(
    reward_progress_weight=2.0,
    reward_speed_weight=0.05,
    reward_collision_penalty=-10.0,
    reward_lap_complete=200.0,
    reward_time_penalty=-0.02,
)
```

### Episode Termination

Episodes end when:
- **Lap Complete**: Car crosses finish line after all checkpoints
- **Stuck**: Total collision time exceeds 5 seconds
- **Timeout**: Step count exceeds `max_steps` (default: 3000)

## Physics Constants

The physics match the frontend TypeScript implementation exactly:

| Parameter | Value | Description |
|-----------|-------|-------------|
| max_speed | 350 | Maximum forward speed (px/s) |
| max_reverse_speed | 150 | Maximum reverse speed (px/s) |
| acceleration | 250 | Acceleration rate (px/s^2) |
| braking | 400 | Braking deceleration (px/s^2) |
| friction | 100 | Natural deceleration (px/s^2) |
| base_turn_rate | 3.5 | Base steering rate (rad/s) |
| collision_radius | 20 | Hitbox radius (px) |

### Track Dimensions

| Parameter | Value |
|-----------|-------|
| Canvas Width | 1200 |
| Canvas Height | 800 |
| Track Width | 100 |
| Outer Radius X | 450 |
| Outer Radius Y | 300 |

## Testing

Run tests:
```bash
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/ --cov=src --cov-report=html
```

## Project Structure

```
racing-game-ml/
├── src/
│   ├── __init__.py
│   └── environment/
│       ├── __init__.py
│       ├── track.py          # Track data and collision detection
│       ├── car_physics.py    # Car physics simulation
│       └── racing_env.py     # Gymnasium environment
├── tests/
│   └── test_environment.py
├── requirements.txt
└── README.md
```

## Integration with Frontend

This environment is designed to produce ML models that can control cars in the frontend game. The physics and track layout are identical, so trained agents should transfer directly.

To export actions for the frontend:
```python
# In Python (training)
action = model.predict(obs)[0]

# Convert to frontend input format
input_state = action_to_input(action)
# input_state has: accelerate, brake, steer_left, steer_right
```

## License

Part of the Racing Game project.
