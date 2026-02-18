"""
Racing Game Environment Package
Contains track, physics, and Gym-compatible environment
"""

from .track import (
    TrackData,
    is_point_on_track,
    get_track_normal,
    get_track_center_point,
    track_data,
    CANVAS_WIDTH,
    CANVAS_HEIGHT,
    TRACK_WIDTH,
)
from .car_physics import (
    CarConfig,
    CarState,
    default_car_config,
    create_car_state,
    update_car,
    reset_car,
)
from .racing_env import RacingEnv

__all__ = [
    # Track
    "TrackData",
    "is_point_on_track",
    "get_track_normal",
    "get_track_center_point",
    "track_data",
    "CANVAS_WIDTH",
    "CANVAS_HEIGHT",
    "TRACK_WIDTH",
    # Physics
    "CarConfig",
    "CarState",
    "default_car_config",
    "create_car_state",
    "update_car",
    "reset_car",
    # Environment
    "RacingEnv",
]
