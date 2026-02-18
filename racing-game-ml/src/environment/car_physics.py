"""
Car physics system for the 2D racing game
Handles acceleration, steering, momentum, and collision detection

This module replicates the TypeScript car.ts implementation in Python
for ML training purposes. All physics constants match frontend exactly.
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from .track import is_point_on_track, get_track_normal


@dataclass
class CarConfig:
    """
    Car physics configuration
    Tunable parameters for car behavior
    ALL VALUES MUST MATCH FRONTEND EXACTLY
    """
    # Maximum forward speed in pixels per second
    max_speed: float = 350.0
    # Maximum reverse speed in pixels per second
    max_reverse_speed: float = 150.0
    # Acceleration rate in pixels per second squared
    acceleration: float = 250.0
    # Braking deceleration rate in pixels per second squared
    braking: float = 400.0
    # Natural friction/drag deceleration when not accelerating
    friction: float = 100.0
    # Base turn rate in radians per second at low speed
    base_turn_rate: float = 3.5
    # Turn rate multiplier based on speed (higher = less turning at high speed)
    turn_speed_factor: float = 0.8
    # Minimum speed required to turn (prevents spinning in place)
    min_speed_to_turn: float = 5.0
    # Speed reduction on collision (0-1, where 0.5 = lose half speed)
    collision_speed_loss: float = 0.3
    # How strongly the car is pushed back onto track on collision
    collision_push_strength: float = 50.0
    # Car hitbox radius for collision detection
    collision_radius: float = 20.0


# Default configuration matching frontend
default_car_config = CarConfig()


@dataclass
class CarState:
    """Complete car state including physics"""
    # Position
    x: float
    y: float
    # Rotation angle in radians
    angle: float
    # Velocity components
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    # Speed (magnitude of velocity)
    speed: float = 0.0
    # Whether the car is currently colliding with a wall
    is_colliding: bool = False
    # Physics configuration
    config: CarConfig = field(default_factory=CarConfig)

    @property
    def position(self) -> Tuple[float, float]:
        """Get position as tuple"""
        return (self.x, self.y)

    @property
    def velocity(self) -> Tuple[float, float]:
        """Get velocity as tuple"""
        return (self.velocity_x, self.velocity_y)


@dataclass
class InputState:
    """Input state representing current control inputs"""
    accelerate: bool = False
    brake: bool = False
    steer_left: bool = False
    steer_right: bool = False


def create_car_state(
    x: float,
    y: float,
    angle: float,
    config: Optional[CarConfig] = None
) -> CarState:
    """
    Create initial car state at a given position and angle

    Args:
        x: X position
        y: Y position
        angle: Initial angle in radians
        config: Optional physics configuration

    Returns:
        New CarState
    """
    return CarState(
        x=x,
        y=y,
        angle=angle,
        velocity_x=0.0,
        velocity_y=0.0,
        speed=0.0,
        is_colliding=False,
        config=config or default_car_config
    )


def _update_steering(car: CarState, input_state: InputState, dt: float) -> None:
    """
    Update car steering based on input

    Args:
        car: Car state to update (modified in place)
        input_state: Current input
        dt: Delta time in seconds
    """
    # Only allow turning if car is moving
    if abs(car.speed) < car.config.min_speed_to_turn:
        return

    # Calculate turn rate based on speed
    # Turn rate decreases as speed increases for more realistic feel
    speed_ratio = abs(car.speed) / car.config.max_speed
    turn_rate = car.config.base_turn_rate * (1 - speed_ratio * car.config.turn_speed_factor)

    # Apply steering
    steer_direction = 0.0
    if input_state.steer_left:
        steer_direction -= 1.0
    if input_state.steer_right:
        steer_direction += 1.0

    # Reverse steering direction when going backwards
    if car.speed < 0:
        steer_direction *= -1

    car.angle += steer_direction * turn_rate * dt

    # Normalize angle to -PI to PI range
    while car.angle > math.pi:
        car.angle -= math.pi * 2
    while car.angle < -math.pi:
        car.angle += math.pi * 2


def _update_speed(car: CarState, input_state: InputState, dt: float) -> None:
    """
    Update car speed based on input (acceleration, braking, friction)

    Args:
        car: Car state to update (modified in place)
        input_state: Current input
        dt: Delta time in seconds
    """
    config = car.config
    speed_change = 0.0

    if input_state.accelerate:
        # Accelerate forward
        speed_change += config.acceleration * dt

    if input_state.brake:
        if car.speed > 0:
            # Braking while moving forward
            speed_change -= config.braking * dt
        else:
            # Reversing when stopped or already moving backward
            speed_change -= config.acceleration * 0.6 * dt  # Reverse is slower

    # Apply natural friction/drag when not accelerating
    if not input_state.accelerate and not input_state.brake:
        if car.speed > 0:
            speed_change -= config.friction * dt
            # Don't let friction make us go backwards
            if car.speed + speed_change < 0:
                speed_change = -car.speed
        elif car.speed < 0:
            speed_change += config.friction * dt
            # Don't let friction make us go forwards
            if car.speed + speed_change > 0:
                speed_change = -car.speed

    # Update speed
    car.speed += speed_change

    # Clamp speed to limits
    car.speed = max(-config.max_reverse_speed, min(config.max_speed, car.speed))

    # Update velocity vector based on speed and angle
    car.velocity_x = math.cos(car.angle) * car.speed
    car.velocity_y = math.sin(car.angle) * car.speed


def _update_position(car: CarState, dt: float) -> Tuple[float, float]:
    """
    Update car position based on velocity

    Args:
        car: Car state to update (modified in place)
        dt: Delta time in seconds

    Returns:
        Old position (x, y) before update
    """
    old_x, old_y = car.x, car.y
    car.x += car.velocity_x * dt
    car.y += car.velocity_y * dt
    return (old_x, old_y)


def _handle_collisions(car: CarState) -> None:
    """
    Handle collision detection and response

    This is a robust collision handler for ML training that ensures
    the car always stays on track. When a collision is detected:
    1. Push the car back onto the track (strong push)
    2. Reduce speed significantly
    3. Remove velocity component pointing off-track

    Args:
        car: Car state to update (modified in place)
    """
    config = car.config

    # Check if car center is on track
    center_on_track = is_point_on_track(car.x, car.y)

    # Also check points around the car for more accurate collision
    check_points = [
        (car.x + math.cos(car.angle) * config.collision_radius,
         car.y + math.sin(car.angle) * config.collision_radius),
        (car.x - math.cos(car.angle) * config.collision_radius,
         car.y - math.sin(car.angle) * config.collision_radius),
        (car.x + math.cos(car.angle + math.pi / 2) * config.collision_radius * 0.6,
         car.y + math.sin(car.angle + math.pi / 2) * config.collision_radius * 0.6),
        (car.x + math.cos(car.angle - math.pi / 2) * config.collision_radius * 0.6,
         car.y + math.sin(car.angle - math.pi / 2) * config.collision_radius * 0.6),
    ]

    all_points_on_track = all(is_point_on_track(px, py) for px, py in check_points)

    if not center_on_track or not all_points_on_track:
        car.is_colliding = True

        # Get normal pointing toward the track
        normal = get_track_normal(car.x, car.y)

        # Strong push back onto track - keep pushing until we're on track
        push_strength = 5.0  # Start with small push
        iterations = 0
        max_iterations = 50

        while not is_point_on_track(car.x, car.y) and iterations < max_iterations:
            car.x += normal[0] * push_strength
            car.y += normal[1] * push_strength
            iterations += 1
            # Increase push strength if we're not making progress
            if iterations % 10 == 0:
                push_strength *= 1.5

        # Reduce speed significantly on collision
        car.speed *= (1 - config.collision_speed_loss)

        # Remove velocity component perpendicular to track (pointing off-track)
        # Calculate velocity dot product with normal
        vel_dot_normal = car.velocity_x * normal[0] + car.velocity_y * normal[1]

        # If velocity is pointing away from track (against the normal), remove that component
        if vel_dot_normal < 0:
            car.velocity_x -= vel_dot_normal * normal[0]
            car.velocity_y -= vel_dot_normal * normal[1]

        # Recalculate speed from velocity
        car.speed = math.sqrt(car.velocity_x ** 2 + car.velocity_y ** 2)

        # Update angle to match velocity direction (if moving)
        if car.speed > 1:
            car.angle = math.atan2(car.velocity_y, car.velocity_x)
    else:
        car.is_colliding = False


def update_car(
    car: CarState,
    input_state: InputState,
    delta_time: float
) -> Tuple[float, float]:
    """
    Update car physics based on input and deltaTime
    This is the main physics update function called each frame

    Args:
        car: Current car state (will be mutated)
        input_state: Current input state
        delta_time: Time since last frame in seconds

    Returns:
        Old position (x, y) before update (useful for checkpoint detection)
    """
    # Clamp deltaTime to prevent physics explosion on lag spikes
    dt = min(delta_time, 0.05)

    # === STEERING ===
    _update_steering(car, input_state, dt)

    # === ACCELERATION / BRAKING ===
    _update_speed(car, input_state, dt)

    # === APPLY VELOCITY ===
    old_pos = _update_position(car, dt)

    # === COLLISION DETECTION AND RESPONSE ===
    _handle_collisions(car)

    return old_pos


def reset_car(car: CarState, x: float, y: float, angle: float) -> CarState:
    """
    Reset car to a specific position and angle
    Useful for respawning or starting a new race

    Args:
        car: Car state to reset (will be mutated)
        x: New X position
        y: New Y position
        angle: New angle

    Returns:
        Updated car state (same reference, mutated)
    """
    car.x = x
    car.y = y
    car.angle = angle
    car.velocity_x = 0.0
    car.velocity_y = 0.0
    car.speed = 0.0
    car.is_colliding = False
    return car


def action_to_input(action: int) -> InputState:
    """
    Convert discrete action to input state

    Action space (9 discrete actions):
        0: No input (coast)
        1: Accelerate
        2: Brake
        3: Steer left
        4: Steer right
        5: Accelerate + steer left
        6: Accelerate + steer right
        7: Brake + steer left
        8: Brake + steer right

    Args:
        action: Discrete action index (0-8)

    Returns:
        InputState corresponding to action
    """
    actions = {
        0: InputState(),  # No input
        1: InputState(accelerate=True),
        2: InputState(brake=True),
        3: InputState(steer_left=True),
        4: InputState(steer_right=True),
        5: InputState(accelerate=True, steer_left=True),
        6: InputState(accelerate=True, steer_right=True),
        7: InputState(brake=True, steer_left=True),
        8: InputState(brake=True, steer_right=True),
    }
    return actions.get(action, InputState())


def multi_discrete_action_to_input(action: np.ndarray) -> InputState:
    """
    Convert MultiDiscrete action to input state

    Action space: [accelerate/brake, steer]
        accelerate/brake: 0=none, 1=accelerate, 2=brake
        steer: 0=none, 1=left, 2=right

    Args:
        action: numpy array [accel_brake, steer]

    Returns:
        InputState corresponding to action
    """
    accel_brake = action[0]
    steer = action[1]

    return InputState(
        accelerate=(accel_brake == 1),
        brake=(accel_brake == 2),
        steer_left=(steer == 1),
        steer_right=(steer == 2)
    )
