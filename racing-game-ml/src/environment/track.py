"""
Track definition and data for the 2D racing game
Creates an oval circuit with turns - matches frontend implementation exactly

This module replicates the TypeScript track.ts implementation in Python
for ML training purposes.
"""

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


# Track dimensions - MUST match frontend values exactly
CANVAS_WIDTH = 1200
CANVAS_HEIGHT = 800
TRACK_WIDTH = 100

# Track center and radii for the oval
CENTER_X = CANVAS_WIDTH / 2  # 600
CENTER_Y = CANVAS_HEIGHT / 2  # 400
OUTER_RADIUS_X = 450
OUTER_RADIUS_Y = 300
INNER_RADIUS_X = OUTER_RADIUS_X - TRACK_WIDTH  # 350
INNER_RADIUS_Y = OUTER_RADIUS_Y - TRACK_WIDTH  # 200


@dataclass
class Point:
    """Represents a 2D point in the game world"""
    x: float
    y: float


@dataclass
class LineSegment:
    """Represents a line segment between two points"""
    start: Point
    end: Point


@dataclass
class Checkpoint:
    """Checkpoint for lap tracking and position validation"""
    id: int
    line: LineSegment


@dataclass
class TrackBoundary:
    """Track boundary definition with inner and outer edges"""
    inner: List[Point]
    outer: List[Point]


@dataclass
class StartFinishLine:
    """Start/finish line definition"""
    start: Point
    end: Point
    start_angle: float  # Direction angle in radians for cars starting here


@dataclass
class TrackData:
    """Complete track data exported for use by other game components"""
    boundaries: TrackBoundary
    start_finish_line: StartFinishLine
    checkpoints: List[Checkpoint]
    track_width: float
    dimensions: Tuple[int, int]  # (width, height)
    start_position: Point
    start_angle: float


def generate_ellipse_points(
    center_x: float,
    center_y: float,
    radius_x: float,
    radius_y: float,
    num_points: int
) -> List[Point]:
    """
    Generate points along an ellipse

    Args:
        center_x: X coordinate of ellipse center
        center_y: Y coordinate of ellipse center
        radius_x: Horizontal radius
        radius_y: Vertical radius
        num_points: Number of points to generate

    Returns:
        Array of points along the ellipse
    """
    points = []
    for i in range(num_points):
        angle = (i / num_points) * math.pi * 2
        points.append(Point(
            x=center_x + math.cos(angle) * radius_x,
            y=center_y + math.sin(angle) * radius_y
        ))
    return points


def create_track_boundaries() -> TrackBoundary:
    """Create track boundaries (inner and outer edges)"""
    num_points = 100  # Smooth curves

    return TrackBoundary(
        outer=generate_ellipse_points(CENTER_X, CENTER_Y, OUTER_RADIUS_X, OUTER_RADIUS_Y, num_points),
        inner=generate_ellipse_points(CENTER_X, CENTER_Y, INNER_RADIUS_X, INNER_RADIUS_Y, num_points)
    )


def create_start_finish_line() -> StartFinishLine:
    """
    Create the start/finish line
    Positioned at the bottom of the oval (south position)
    """
    start_x = CENTER_X
    inner_y = CENTER_Y + INNER_RADIUS_Y
    outer_y = CENTER_Y + OUTER_RADIUS_Y

    return StartFinishLine(
        start=Point(x=start_x, y=inner_y),
        end=Point(x=start_x, y=outer_y),
        start_angle=-math.pi / 2  # Facing up (counter-clockwise racing direction)
    )


def create_checkpoints() -> List[Checkpoint]:
    """
    Create checkpoints around the track for lap validation
    Positioned at key points: top, left, right
    """
    return [
        # Top checkpoint
        Checkpoint(
            id=1,
            line=LineSegment(
                start=Point(x=CENTER_X, y=CENTER_Y - INNER_RADIUS_Y),
                end=Point(x=CENTER_X, y=CENTER_Y - OUTER_RADIUS_Y)
            )
        ),
        # Left checkpoint (west)
        Checkpoint(
            id=2,
            line=LineSegment(
                start=Point(x=CENTER_X - INNER_RADIUS_X, y=CENTER_Y),
                end=Point(x=CENTER_X - OUTER_RADIUS_X, y=CENTER_Y)
            )
        ),
        # Right checkpoint (east)
        Checkpoint(
            id=3,
            line=LineSegment(
                start=Point(x=CENTER_X + INNER_RADIUS_X, y=CENTER_Y),
                end=Point(x=CENTER_X + OUTER_RADIUS_X, y=CENTER_Y)
            )
        ),
    ]


def create_track_data() -> TrackData:
    """Create complete track data for the racing game"""
    return TrackData(
        boundaries=create_track_boundaries(),
        start_finish_line=create_start_finish_line(),
        checkpoints=create_checkpoints(),
        track_width=TRACK_WIDTH,
        dimensions=(CANVAS_WIDTH, CANVAS_HEIGHT),
        # Starting position on the track (middle of start/finish line)
        start_position=Point(
            x=CENTER_X,
            y=CENTER_Y + (INNER_RADIUS_Y + OUTER_RADIUS_Y) / 2
        ),
        start_angle=-math.pi / 2  # Facing up
    )


# Singleton track data instance
track_data = create_track_data()


def is_point_on_track(x: float, y: float) -> bool:
    """
    Check if a point is on the track (between inner and outer boundaries)
    Uses simplified ellipse math for efficient checking

    Args:
        x: X coordinate of point
        y: Y coordinate of point

    Returns:
        True if point is on the track
    """
    # Normalize point relative to center
    dx = x - CENTER_X
    dy = y - CENTER_Y

    # Check if point is inside outer ellipse
    outer_check = (dx * dx) / (OUTER_RADIUS_X * OUTER_RADIUS_X) + \
                  (dy * dy) / (OUTER_RADIUS_Y * OUTER_RADIUS_Y)

    # Check if point is outside inner ellipse
    inner_check = (dx * dx) / (INNER_RADIUS_X * INNER_RADIUS_X) + \
                  (dy * dy) / (INNER_RADIUS_Y * INNER_RADIUS_Y)

    return outer_check <= 1 and inner_check >= 1


def is_point_on_track_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Vectorized version of is_point_on_track for batch processing

    Args:
        x: X coordinates (numpy array)
        y: Y coordinates (numpy array)

    Returns:
        Boolean numpy array indicating if each point is on track
    """
    dx = x - CENTER_X
    dy = y - CENTER_Y

    outer_check = (dx * dx) / (OUTER_RADIUS_X * OUTER_RADIUS_X) + \
                  (dy * dy) / (OUTER_RADIUS_Y * OUTER_RADIUS_Y)

    inner_check = (dx * dx) / (INNER_RADIUS_X * INNER_RADIUS_X) + \
                  (dy * dy) / (INNER_RADIUS_Y * INNER_RADIUS_Y)

    return (outer_check <= 1) & (inner_check >= 1)


def get_track_normal(x: float, y: float) -> Tuple[float, float]:
    """
    Get the track normal (direction to push car back onto track)
    Useful for collision response

    For an elliptical track:
    - If inside inner ellipse: push outward (away from center)
    - If outside outer ellipse: push inward (toward center)

    Args:
        x: X coordinate
        y: Y coordinate

    Returns:
        Normalized direction vector (nx, ny) pointing toward track
    """
    dx = x - CENTER_X
    dy = y - CENTER_Y
    length = math.sqrt(dx * dx + dy * dy)

    if length == 0:
        return (0.0, 1.0)  # Push outward if at exact center

    # Check which boundary we're violating
    outer_check = (dx * dx) / (OUTER_RADIUS_X * OUTER_RADIUS_X) + \
                  (dy * dy) / (OUTER_RADIUS_Y * OUTER_RADIUS_Y)
    inner_check = (dx * dx) / (INNER_RADIUS_X * INNER_RADIUS_X) + \
                  (dy * dy) / (INNER_RADIUS_Y * INNER_RADIUS_Y)

    if inner_check < 1:
        # Inside inner ellipse - push outward (away from center)
        return (dx / length, dy / length)
    elif outer_check > 1:
        # Outside outer ellipse - push inward (toward center)
        return (-dx / length, -dy / length)
    else:
        # On track - no push needed, but return outward direction as default
        return (dx / length, dy / length)


def get_track_center_point(angle: float) -> Tuple[float, float]:
    """
    Get a point on the track center line at a given angle
    Useful for AI or positioning

    Args:
        angle: Angle in radians (0 = right, PI/2 = bottom)

    Returns:
        (x, y) point on the track center line
    """
    center_radius_x = (INNER_RADIUS_X + OUTER_RADIUS_X) / 2
    center_radius_y = (INNER_RADIUS_Y + OUTER_RADIUS_Y) / 2

    return (
        CENTER_X + math.cos(angle) * center_radius_x,
        CENTER_Y + math.sin(angle) * center_radius_y
    )


def get_distance_to_track_boundaries(x: float, y: float, angle: float, num_rays: int = 5) -> List[float]:
    """
    Cast rays from a point to find distances to track boundaries
    Used for observation space in ML training

    Args:
        x: X coordinate
        y: Y coordinate
        angle: Current heading angle
        num_rays: Number of rays to cast (spread evenly from -90 to +90 degrees)

    Returns:
        List of distances to boundaries for each ray
    """
    distances = []
    max_distance = 300.0  # Maximum ray distance

    # Spread rays from -90 to +90 degrees relative to heading
    ray_angles = np.linspace(-math.pi / 2, math.pi / 2, num_rays)

    for ray_offset in ray_angles:
        ray_angle = angle + ray_offset
        dx = math.cos(ray_angle)
        dy = math.sin(ray_angle)

        # March along ray to find boundary
        for distance in np.linspace(0, max_distance, 100):
            test_x = x + dx * distance
            test_y = y + dy * distance

            if not is_point_on_track(test_x, test_y):
                distances.append(distance)
                break
        else:
            distances.append(max_distance)

    return distances


def get_track_progress_angle(x: float, y: float) -> float:
    """
    Get the progress around the track as an angle
    Used to calculate forward progress for rewards

    The track is counter-clockwise, starting at the bottom (south)

    Args:
        x: X coordinate
        y: Y coordinate

    Returns:
        Angle in radians representing progress (0 to 2*PI, starting from bottom going counter-clockwise)
    """
    # Calculate angle from center
    dx = x - CENTER_X
    dy = y - CENTER_Y

    # atan2 returns angle where 0 is right, PI/2 is down, PI/-PI is left, -PI/2 is up
    # We want 0 to be at the bottom (start/finish), increasing counter-clockwise
    angle = math.atan2(dy, dx)

    # Shift so bottom (PI/2) becomes 0, and goes counter-clockwise (decreasing angle becomes increasing progress)
    # Original: bottom=PI/2, right=0, top=-PI/2, left=PI/-PI
    # We need: bottom=0, left=PI/2, top=PI, right=3*PI/2
    progress_angle = math.pi / 2 - angle

    # Normalize to 0 to 2*PI
    if progress_angle < 0:
        progress_angle += 2 * math.pi

    return progress_angle


def line_segment_intersection(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    p4: Tuple[float, float]
) -> bool:
    """
    Check if two line segments intersect

    Args:
        p1, p2: First line segment endpoints
        p3, p4: Second line segment endpoints

    Returns:
        True if segments intersect
    """
    def ccw(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> bool:
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))


def check_checkpoint_crossing(
    old_x: float, old_y: float,
    new_x: float, new_y: float,
    checkpoint: Checkpoint
) -> bool:
    """
    Check if the car crossed a checkpoint between two positions

    Args:
        old_x, old_y: Previous position
        new_x, new_y: New position
        checkpoint: Checkpoint to check

    Returns:
        True if checkpoint was crossed
    """
    return line_segment_intersection(
        (old_x, old_y),
        (new_x, new_y),
        (checkpoint.line.start.x, checkpoint.line.start.y),
        (checkpoint.line.end.x, checkpoint.line.end.y)
    )


def check_start_finish_crossing(
    old_x: float, old_y: float,
    new_x: float, new_y: float
) -> bool:
    """
    Check if the car crossed the start/finish line between two positions

    Args:
        old_x, old_y: Previous position
        new_x, new_y: New position

    Returns:
        True if start/finish line was crossed
    """
    sf = track_data.start_finish_line
    return line_segment_intersection(
        (old_x, old_y),
        (new_x, new_y),
        (sf.start.x, sf.start.y),
        (sf.end.x, sf.end.y)
    )
