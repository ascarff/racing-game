"""
Gym-compatible Racing Environment for ML training

This environment provides a simulation of the 2D racing game
that can be used with reinforcement learning frameworks like
stable-baselines3.

Features:
- Headless/accelerated mode (faster than real-time)
- State extraction (car position, velocity, track boundaries, checkpoints)
- Programmatic actions (accelerate, brake, steer)
- Reward function (progress, speed, penalize collisions)
- Episode reset on crash/lap completion
- Support for parallel training (vectorized environments)
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .track import (
    track_data,
    is_point_on_track,
    get_distance_to_track_boundaries,
    get_track_progress_angle,
    check_checkpoint_crossing,
    check_start_finish_crossing,
    CANVAS_WIDTH,
    CANVAS_HEIGHT,
)
from .car_physics import (
    CarState,
    CarConfig,
    InputState,
    create_car_state,
    update_car,
    reset_car,
    action_to_input,
    multi_discrete_action_to_input,
    default_car_config,
)


class RacingEnv(gym.Env):
    """
    Racing Game Environment

    A Gym-compatible environment for training RL agents to race around
    an oval track. The environment replicates the physics and track
    from the frontend TypeScript implementation.

    Observation Space:
        Box space with 15 features:
        - [0]: x position (normalized to 0-1)
        - [1]: y position (normalized to 0-1)
        - [2]: cos(angle) - heading x component
        - [3]: sin(angle) - heading y component
        - [4]: speed (normalized to -1 to 1)
        - [5]: velocity_x (normalized)
        - [6]: velocity_y (normalized)
        - [7]: is_colliding (0 or 1)
        - [8-12]: distance to track boundaries (5 rays, normalized)
        - [13]: progress around track (0 to 1)
        - [14]: checkpoint progress (0 to 1)

    Action Space:
        Discrete(9) by default:
        0: No input (coast)
        1: Accelerate
        2: Brake
        3: Steer left
        4: Steer right
        5: Accelerate + steer left
        6: Accelerate + steer right
        7: Brake + steer left
        8: Brake + steer right

        Or MultiDiscrete([3, 3]) if use_multi_discrete=True:
        [accel_brake, steer]
        - accel_brake: 0=none, 1=accelerate, 2=brake
        - steer: 0=none, 1=left, 2=right

    Reward Function:
        - Positive reward for forward progress along track
        - Positive reward for maintaining speed
        - Negative reward for wall collisions
        - Large positive reward for completing a lap
        - Small negative reward per timestep (encourages efficiency)
    """

    metadata = {"render_modes": ["human", "rgb_array", "ansi"], "render_fps": 60}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        use_multi_discrete: bool = False,
        max_steps: int = 3000,
        dt: float = 1 / 60,
        num_rays: int = 5,
        reward_progress_weight: float = 1.0,
        reward_speed_weight: float = 0.1,
        reward_collision_penalty: float = -5.0,
        reward_lap_complete: float = 100.0,
        reward_time_penalty: float = -0.01,
        car_config: Optional[CarConfig] = None,
    ):
        """
        Initialize the racing environment

        Args:
            render_mode: Rendering mode ('human', 'rgb_array', 'ansi', or None)
            use_multi_discrete: If True, use MultiDiscrete action space
            max_steps: Maximum steps per episode
            dt: Simulation timestep in seconds (1/60 = 60 FPS)
            num_rays: Number of boundary distance rays
            reward_progress_weight: Weight for progress reward
            reward_speed_weight: Weight for speed reward
            reward_collision_penalty: Penalty for collisions
            reward_lap_complete: Reward for completing a lap
            reward_time_penalty: Small penalty per timestep
            car_config: Optional custom car configuration
        """
        super().__init__()

        self.render_mode = render_mode
        self.use_multi_discrete = use_multi_discrete
        self.max_steps = max_steps
        self.dt = dt
        self.num_rays = num_rays

        # Reward weights
        self.reward_progress_weight = reward_progress_weight
        self.reward_speed_weight = reward_speed_weight
        self.reward_collision_penalty = reward_collision_penalty
        self.reward_lap_complete = reward_lap_complete
        self.reward_time_penalty = reward_time_penalty

        # Car configuration
        self.car_config = car_config or default_car_config

        # Track reference
        self.track = track_data

        # Define action space
        if use_multi_discrete:
            # [accel/brake, steer]: each with 3 options
            self.action_space = spaces.MultiDiscrete([3, 3])
        else:
            # 9 discrete actions
            self.action_space = spaces.Discrete(9)

        # Define observation space
        # 8 base features + num_rays boundary distances + 2 progress features
        obs_dim = 8 + num_rays + 2
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # Initialize state
        self.car: Optional[CarState] = None
        self.step_count = 0
        self.previous_progress = 0.0
        self.checkpoints_passed: List[int] = []
        self.lap_complete = False
        self.total_collision_time = 0.0

        # Metrics for logging
        self.episode_reward = 0.0
        self.episode_progress = 0.0
        self.episode_collisions = 0
        self.episode_lap_time: Optional[float] = None

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state

        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)

        Returns:
            observation: Initial observation
            info: Additional info dictionary
        """
        super().reset(seed=seed)

        # Create car at starting position
        self.car = create_car_state(
            x=self.track.start_position.x,
            y=self.track.start_position.y,
            angle=self.track.start_angle,
            config=self.car_config
        )

        # Reset tracking variables
        self.step_count = 0
        self.previous_progress = get_track_progress_angle(self.car.x, self.car.y)
        self.checkpoints_passed = []
        self.lap_complete = False
        self.total_collision_time = 0.0

        # Reset metrics
        self.episode_reward = 0.0
        self.episode_progress = 0.0
        self.episode_collisions = 0
        self.episode_lap_time = None

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(
        self,
        action: Union[int, np.ndarray]
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment

        Args:
            action: Action to take (discrete or multi-discrete)

        Returns:
            observation: New observation
            reward: Reward for this step
            terminated: Whether episode ended (lap complete or crash)
            truncated: Whether episode was truncated (max steps)
            info: Additional info dictionary
        """
        assert self.car is not None, "Environment must be reset before stepping"

        # Convert action to input state
        if self.use_multi_discrete:
            input_state = multi_discrete_action_to_input(action)
        else:
            input_state = action_to_input(action)

        # Store old position for checkpoint detection
        old_x, old_y = self.car.x, self.car.y
        old_progress = get_track_progress_angle(old_x, old_y)

        # Update physics
        update_car(self.car, input_state, self.dt)
        self.step_count += 1

        # Get new progress
        new_progress = get_track_progress_angle(self.car.x, self.car.y)

        # Calculate reward
        reward = self._calculate_reward(old_progress, new_progress)
        self.episode_reward += reward

        # Check for checkpoint crossings
        self._check_checkpoints(old_x, old_y)

        # Check for lap completion
        if self._check_lap_complete(old_x, old_y):
            self.lap_complete = True
            reward += self.reward_lap_complete
            self.episode_reward += self.reward_lap_complete
            self.episode_lap_time = self.step_count * self.dt

        # Track collision time
        if self.car.is_colliding:
            self.total_collision_time += self.dt
            self.episode_collisions += 1

        # Check termination conditions
        terminated = self.lap_complete
        truncated = self.step_count >= self.max_steps

        # Check if car is stuck (too much collision time)
        if self.total_collision_time > 5.0:  # 5 seconds of collision = stuck
            terminated = True

        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """
        Get current observation from environment state

        Returns:
            Observation array
        """
        assert self.car is not None

        # Normalize values
        norm_x = self.car.x / CANVAS_WIDTH
        norm_y = self.car.y / CANVAS_HEIGHT
        cos_angle = math.cos(self.car.angle)
        sin_angle = math.sin(self.car.angle)
        norm_speed = self.car.speed / self.car.config.max_speed
        norm_vx = self.car.velocity_x / self.car.config.max_speed
        norm_vy = self.car.velocity_y / self.car.config.max_speed
        is_colliding = 1.0 if self.car.is_colliding else 0.0

        # Get boundary distances (normalized)
        distances = get_distance_to_track_boundaries(
            self.car.x, self.car.y, self.car.angle, self.num_rays
        )
        max_distance = 300.0
        norm_distances = [d / max_distance for d in distances]

        # Progress features
        track_progress = get_track_progress_angle(self.car.x, self.car.y) / (2 * math.pi)
        checkpoint_progress = len(self.checkpoints_passed) / (len(self.track.checkpoints) + 1)

        # Build observation
        observation = np.array([
            norm_x,
            norm_y,
            cos_angle,
            sin_angle,
            norm_speed,
            norm_vx,
            norm_vy,
            is_colliding,
            *norm_distances,
            track_progress,
            checkpoint_progress,
        ], dtype=np.float32)

        # Clip to observation space bounds
        observation = np.clip(observation, -1.0, 1.0)

        return observation

    def _calculate_reward(self, old_progress: float, new_progress: float) -> float:
        """
        Calculate reward for current step

        Args:
            old_progress: Progress angle before step
            new_progress: Progress angle after step

        Returns:
            Reward value
        """
        assert self.car is not None

        reward = 0.0

        # Progress reward (forward movement around track)
        progress_delta = new_progress - old_progress

        # Handle wraparound (crossing from ~2*PI to ~0)
        if progress_delta < -math.pi:
            progress_delta += 2 * math.pi
        elif progress_delta > math.pi:
            progress_delta -= 2 * math.pi

        # Positive reward for forward progress
        if progress_delta > 0:
            reward += progress_delta * self.reward_progress_weight
            self.episode_progress += progress_delta

        # Speed reward (encourage maintaining speed)
        normalized_speed = max(0, self.car.speed) / self.car.config.max_speed
        reward += normalized_speed * self.reward_speed_weight

        # Collision penalty
        if self.car.is_colliding:
            reward += self.reward_collision_penalty

        # Time penalty (encourage efficiency)
        reward += self.reward_time_penalty

        return reward

    def _check_checkpoints(self, old_x: float, old_y: float) -> None:
        """
        Check if car crossed any checkpoints

        Args:
            old_x: Previous X position
            old_y: Previous Y position
        """
        assert self.car is not None

        for checkpoint in self.track.checkpoints:
            if checkpoint.id not in self.checkpoints_passed:
                if check_checkpoint_crossing(
                    old_x, old_y, self.car.x, self.car.y, checkpoint
                ):
                    self.checkpoints_passed.append(checkpoint.id)

    def _check_lap_complete(self, old_x: float, old_y: float) -> bool:
        """
        Check if car completed a lap

        A lap is complete when:
        - All checkpoints have been passed
        - Start/finish line is crossed

        Args:
            old_x: Previous X position
            old_y: Previous Y position

        Returns:
            True if lap is complete
        """
        assert self.car is not None

        # Need all checkpoints first
        if len(self.checkpoints_passed) < len(self.track.checkpoints):
            return False

        # Check for start/finish line crossing
        return check_start_finish_crossing(old_x, old_y, self.car.x, self.car.y)

    def _get_info(self) -> Dict[str, Any]:
        """
        Get info dictionary for current state

        Returns:
            Info dictionary with debugging/logging data
        """
        assert self.car is not None

        return {
            "step": self.step_count,
            "position": (self.car.x, self.car.y),
            "angle": self.car.angle,
            "speed": self.car.speed,
            "is_colliding": self.car.is_colliding,
            "checkpoints_passed": len(self.checkpoints_passed),
            "total_checkpoints": len(self.track.checkpoints),
            "lap_complete": self.lap_complete,
            "total_collision_time": self.total_collision_time,
            "episode_reward": self.episode_reward,
            "episode_progress": self.episode_progress,
            "episode_collisions": self.episode_collisions,
            "episode_lap_time": self.episode_lap_time,
        }

    def render(self) -> Optional[Union[np.ndarray, str]]:
        """
        Render the environment

        Returns:
            Rendered frame (numpy array for rgb_array, string for ansi)
        """
        if self.render_mode is None:
            return None

        if self.render_mode == "ansi":
            return self._render_ansi()
        elif self.render_mode == "human":
            self._render_human()
            return None
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()

        return None

    def _render_ansi(self) -> str:
        """Render environment state as text"""
        assert self.car is not None

        lines = [
            f"Step: {self.step_count}",
            f"Position: ({self.car.x:.1f}, {self.car.y:.1f})",
            f"Angle: {math.degrees(self.car.angle):.1f} deg",
            f"Speed: {self.car.speed:.1f}",
            f"Colliding: {self.car.is_colliding}",
            f"Checkpoints: {len(self.checkpoints_passed)}/{len(self.track.checkpoints)}",
            f"Progress: {get_track_progress_angle(self.car.x, self.car.y) / (2 * math.pi) * 100:.1f}%",
        ]
        return "\n".join(lines)

    def _render_human(self) -> None:
        """Render to console (human readable)"""
        print(self._render_ansi())

    def _render_rgb_array(self) -> np.ndarray:
        """
        Render environment as RGB image

        Returns:
            RGB image array (height, width, 3)
        """
        # Simple visualization using numpy
        # Scale down for performance
        scale = 4
        width = CANVAS_WIDTH // scale
        height = CANVAS_HEIGHT // scale

        # Create blank image (green background)
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:, :] = [30, 100, 50]  # Dark green grass

        # Draw track (simplified)
        for y in range(height):
            for x in range(width):
                real_x = x * scale
                real_y = y * scale
                if is_point_on_track(real_x, real_y):
                    img[y, x] = [80, 80, 80]  # Gray track

        # Draw car (red dot)
        if self.car is not None:
            car_x = int(self.car.x / scale)
            car_y = int(self.car.y / scale)
            if 0 <= car_x < width and 0 <= car_y < height:
                # Draw car as a small rectangle
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        px, py = car_x + dx, car_y + dy
                        if 0 <= px < width and 0 <= py < height:
                            img[py, px] = [255, 0, 0]  # Red car

        return img

    def close(self) -> None:
        """Clean up environment resources"""
        pass

    def get_state(self) -> Dict[str, Any]:
        """
        Get full internal state for serialization/debugging

        Returns:
            Dictionary with all state variables
        """
        if self.car is None:
            return {}

        return {
            "car": {
                "x": self.car.x,
                "y": self.car.y,
                "angle": self.car.angle,
                "velocity_x": self.car.velocity_x,
                "velocity_y": self.car.velocity_y,
                "speed": self.car.speed,
                "is_colliding": self.car.is_colliding,
            },
            "step_count": self.step_count,
            "checkpoints_passed": self.checkpoints_passed.copy(),
            "lap_complete": self.lap_complete,
            "total_collision_time": self.total_collision_time,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Set internal state from serialized state

        Args:
            state: State dictionary from get_state()
        """
        if "car" in state:
            car_state = state["car"]
            self.car = create_car_state(
                x=car_state["x"],
                y=car_state["y"],
                angle=car_state["angle"],
                config=self.car_config
            )
            self.car.velocity_x = car_state["velocity_x"]
            self.car.velocity_y = car_state["velocity_y"]
            self.car.speed = car_state["speed"]
            self.car.is_colliding = car_state["is_colliding"]

        self.step_count = state.get("step_count", 0)
        self.checkpoints_passed = state.get("checkpoints_passed", []).copy()
        self.lap_complete = state.get("lap_complete", False)
        self.total_collision_time = state.get("total_collision_time", 0.0)


def make_vec_env(
    num_envs: int = 4,
    **env_kwargs
) -> gym.vector.VectorEnv:
    """
    Create a vectorized environment for parallel training

    Args:
        num_envs: Number of parallel environments
        **env_kwargs: Additional arguments passed to RacingEnv

    Returns:
        Vectorized environment
    """
    def make_env():
        return RacingEnv(**env_kwargs)

    return gym.vector.SyncVectorEnv([make_env for _ in range(num_envs)])


def make_async_vec_env(
    num_envs: int = 4,
    **env_kwargs
) -> gym.vector.VectorEnv:
    """
    Create an asynchronous vectorized environment for parallel training

    Note: Requires environments to be picklable

    Args:
        num_envs: Number of parallel environments
        **env_kwargs: Additional arguments passed to RacingEnv

    Returns:
        Asynchronous vectorized environment
    """
    def make_env():
        return RacingEnv(**env_kwargs)

    return gym.vector.AsyncVectorEnv([make_env for _ in range(num_envs)])


# Register environment with gymnasium
gym.register(
    id="RacingGame-v0",
    entry_point="src.environment.racing_env:RacingEnv",
)
