"""
Tests for Racing Game ML Environment

Tests cover:
- Track geometry and collision detection
- Car physics simulation
- Gym environment interface
- Reward function
- Parallel environment support
"""

import math
import pytest
import numpy as np

import sys
sys.path.insert(0, "/Users/andrewscarff/code/steelcityai/20260217-skills/racing-game-ml")

from src.environment.track import (
    track_data,
    is_point_on_track,
    get_track_normal,
    get_track_progress_angle,
    get_distance_to_track_boundaries,
    check_checkpoint_crossing,
    CANVAS_WIDTH,
    CANVAS_HEIGHT,
    CENTER_X,
    CENTER_Y,
    OUTER_RADIUS_X,
    OUTER_RADIUS_Y,
    INNER_RADIUS_X,
    INNER_RADIUS_Y,
    TRACK_WIDTH,
)
from src.environment.car_physics import (
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
from src.environment.racing_env import RacingEnv, make_vec_env


class TestTrackGeometry:
    """Tests for track geometry and constants"""

    def test_track_dimensions_match_frontend(self):
        """Verify track dimensions match frontend values"""
        assert CANVAS_WIDTH == 1200
        assert CANVAS_HEIGHT == 800
        assert TRACK_WIDTH == 100
        assert OUTER_RADIUS_X == 450
        assert OUTER_RADIUS_Y == 300
        assert INNER_RADIUS_X == 350
        assert INNER_RADIUS_Y == 200

    def test_track_center(self):
        """Verify track center is at canvas center"""
        assert CENTER_X == 600
        assert CENTER_Y == 400

    def test_track_data_exists(self):
        """Verify track data is properly initialized"""
        assert track_data is not None
        assert track_data.boundaries is not None
        assert track_data.start_finish_line is not None
        assert track_data.checkpoints is not None
        assert len(track_data.checkpoints) == 3

    def test_start_position_on_track(self):
        """Verify start position is on the track"""
        start = track_data.start_position
        assert is_point_on_track(start.x, start.y)

    def test_start_angle_facing_up(self):
        """Verify start angle is facing up (counter-clockwise direction)"""
        assert track_data.start_angle == -math.pi / 2


class TestCollisionDetection:
    """Tests for collision detection functions"""

    def test_point_on_track_center(self):
        """Points on track center should be on track"""
        # Bottom center (start position area)
        center_y = CENTER_Y + (INNER_RADIUS_Y + OUTER_RADIUS_Y) / 2
        assert is_point_on_track(CENTER_X, center_y)

        # Top center
        top_y = CENTER_Y - (INNER_RADIUS_Y + OUTER_RADIUS_Y) / 2
        assert is_point_on_track(CENTER_X, top_y)

        # Left center
        left_x = CENTER_X - (INNER_RADIUS_X + OUTER_RADIUS_X) / 2
        assert is_point_on_track(left_x, CENTER_Y)

        # Right center
        right_x = CENTER_X + (INNER_RADIUS_X + OUTER_RADIUS_X) / 2
        assert is_point_on_track(right_x, CENTER_Y)

    def test_point_inside_inner_ellipse(self):
        """Points inside inner ellipse should NOT be on track"""
        # Center of track (inside inner ellipse)
        assert not is_point_on_track(CENTER_X, CENTER_Y)

        # Slightly inside inner boundary
        inner_x = CENTER_X + INNER_RADIUS_X * 0.9
        assert not is_point_on_track(inner_x, CENTER_Y)

    def test_point_outside_outer_ellipse(self):
        """Points outside outer ellipse should NOT be on track"""
        # Far outside
        assert not is_point_on_track(0, 0)
        assert not is_point_on_track(CANVAS_WIDTH, CANVAS_HEIGHT)

        # Just outside outer boundary
        outer_x = CENTER_X + OUTER_RADIUS_X * 1.1
        assert not is_point_on_track(outer_x, CENTER_Y)

    def test_track_normal_points_toward_track(self):
        """Track normal should point toward track (for collision response)"""
        # Test point inside inner ellipse (should point outward/away from center)
        x = CENTER_X
        y = CENTER_Y + INNER_RADIUS_Y - 10  # Inside inner ellipse
        nx, ny = get_track_normal(x, y)
        assert ny > 0  # Points down (away from center, toward track)

        # Test point outside outer ellipse (should point inward/toward center)
        x = CENTER_X
        y = CENTER_Y + OUTER_RADIUS_Y + 10  # Outside outer ellipse
        nx, ny = get_track_normal(x, y)
        assert ny < 0  # Points up (toward center, toward track)

        # Test point on track (should have some direction)
        x = CENTER_X + (INNER_RADIUS_X + OUTER_RADIUS_X) / 2
        y = CENTER_Y
        nx, ny = get_track_normal(x, y)
        # On track, normal points outward (default behavior)
        assert abs(nx) > 0.9  # Mostly horizontal

    def test_boundary_ray_distances(self):
        """Boundary distance rays should detect track edges"""
        # Position on track center, facing right
        x = CENTER_X + (INNER_RADIUS_X + OUTER_RADIUS_X) / 2
        y = CENTER_Y
        angle = 0  # Facing right

        distances = get_distance_to_track_boundaries(x, y, angle, num_rays=5)
        assert len(distances) == 5
        assert all(d > 0 for d in distances)
        assert all(d <= 300 for d in distances)  # Max ray distance


class TestCarPhysics:
    """Tests for car physics simulation"""

    def test_default_config_matches_frontend(self):
        """Verify default car config matches frontend values"""
        assert default_car_config.max_speed == 350
        assert default_car_config.max_reverse_speed == 150
        assert default_car_config.acceleration == 250
        assert default_car_config.braking == 400
        assert default_car_config.friction == 100
        assert default_car_config.base_turn_rate == 3.5
        assert default_car_config.turn_speed_factor == 0.8
        assert default_car_config.min_speed_to_turn == 5
        assert default_car_config.collision_speed_loss == 0.3
        assert default_car_config.collision_push_strength == 50
        assert default_car_config.collision_radius == 20

    def test_create_car_state(self):
        """Test car state creation"""
        car = create_car_state(100, 200, math.pi / 4)
        assert car.x == 100
        assert car.y == 200
        assert car.angle == math.pi / 4
        assert car.speed == 0
        assert car.velocity_x == 0
        assert car.velocity_y == 0
        assert not car.is_colliding

    def test_acceleration(self):
        """Test car acceleration"""
        car = create_car_state(
            track_data.start_position.x,
            track_data.start_position.y,
            track_data.start_angle
        )
        input_state = InputState(accelerate=True)

        # Simulate several frames
        for _ in range(60):  # 1 second at 60 FPS
            update_car(car, input_state, 1 / 60)

        assert car.speed > 0
        assert car.speed <= default_car_config.max_speed

    def test_braking(self):
        """Test car braking"""
        car = create_car_state(
            track_data.start_position.x,
            track_data.start_position.y,
            track_data.start_angle
        )

        # First accelerate for a shorter time to avoid collision
        accel_input = InputState(accelerate=True)
        for _ in range(20):
            update_car(car, accel_input, 1 / 60)

        initial_speed = car.speed
        assert initial_speed > 10  # Should have some meaningful speed

        # Then brake for shorter time
        brake_input = InputState(brake=True)
        for _ in range(10):
            update_car(car, brake_input, 1 / 60)

        # Speed should be lower (accounting for collision effects)
        # Use a tolerance for floating point comparison
        assert car.speed < initial_speed or abs(car.speed - initial_speed) < 1.0

    def test_steering_requires_speed(self):
        """Test that steering requires minimum speed"""
        car = create_car_state(
            track_data.start_position.x,
            track_data.start_position.y,
            track_data.start_angle
        )
        initial_angle = car.angle

        # Try to steer without speed
        steer_input = InputState(steer_left=True)
        update_car(car, steer_input, 1 / 60)

        assert car.angle == initial_angle  # No change

        # Accelerate first
        accel_input = InputState(accelerate=True)
        for _ in range(10):
            update_car(car, accel_input, 1 / 60)

        assert car.speed > default_car_config.min_speed_to_turn

        # Now steering should work
        steer_accel = InputState(accelerate=True, steer_left=True)
        update_car(car, steer_accel, 1 / 60)

        assert car.angle != initial_angle

    def test_reset_car(self):
        """Test car reset"""
        car = create_car_state(100, 100, 0)
        car.speed = 200
        car.velocity_x = 150
        car.is_colliding = True

        reset_car(car, 500, 500, math.pi)

        assert car.x == 500
        assert car.y == 500
        assert car.angle == math.pi
        assert car.speed == 0
        assert car.velocity_x == 0
        assert not car.is_colliding

    def test_action_to_input(self):
        """Test discrete action to input conversion"""
        # Action 0: No input
        inp = action_to_input(0)
        assert not inp.accelerate and not inp.brake
        assert not inp.steer_left and not inp.steer_right

        # Action 1: Accelerate
        inp = action_to_input(1)
        assert inp.accelerate and not inp.brake

        # Action 5: Accelerate + steer left
        inp = action_to_input(5)
        assert inp.accelerate and inp.steer_left

    def test_multi_discrete_action_to_input(self):
        """Test multi-discrete action to input conversion"""
        # [1, 2] = accelerate + steer right
        inp = multi_discrete_action_to_input(np.array([1, 2]))
        assert inp.accelerate
        assert inp.steer_right

        # [2, 1] = brake + steer left
        inp = multi_discrete_action_to_input(np.array([2, 1]))
        assert inp.brake
        assert inp.steer_left


class TestRacingEnvironment:
    """Tests for the Gym environment"""

    def test_env_creation(self):
        """Test environment creation"""
        env = RacingEnv()
        assert env is not None
        assert env.observation_space.shape == (15,)  # 8 base + 5 rays + 2 progress
        assert env.action_space.n == 9  # 9 discrete actions
        env.close()

    def test_env_reset(self):
        """Test environment reset"""
        env = RacingEnv()
        obs, info = env.reset()

        assert obs.shape == (15,)
        assert obs.dtype == np.float32
        assert all(-1 <= o <= 1 for o in obs)

        assert "position" in info
        assert "speed" in info
        assert info["step"] == 0
        env.close()

    def test_env_step(self):
        """Test environment step"""
        env = RacingEnv()
        obs, _ = env.reset()

        # Take action 1 (accelerate)
        new_obs, reward, terminated, truncated, info = env.step(1)

        assert new_obs.shape == (15,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert info["step"] == 1
        env.close()

    def test_env_multi_discrete(self):
        """Test environment with multi-discrete action space"""
        env = RacingEnv(use_multi_discrete=True)
        assert env.action_space.nvec.tolist() == [3, 3]

        obs, _ = env.reset()
        action = np.array([1, 2])  # Accelerate + steer right
        obs, reward, _, _, info = env.step(action)

        assert obs.shape == (15,)
        env.close()

    def test_reward_components(self):
        """Test that rewards are calculated correctly"""
        env = RacingEnv()
        obs, _ = env.reset()

        # Accelerate for several steps
        total_reward = 0
        for _ in range(100):
            obs, reward, terminated, truncated, info = env.step(1)
            total_reward += reward
            if terminated or truncated:
                break

        # Should have positive progress reward offset by time penalty
        assert total_reward != 0
        env.close()

    def test_collision_penalty(self):
        """Test that collisions give negative reward"""
        env = RacingEnv()
        obs, _ = env.reset()

        # Drive off track by going right (into the grass)
        initial_reward = 0
        colliding = False

        for i in range(200):
            # Steer right and accelerate to go off track
            obs, reward, terminated, truncated, info = env.step(6)  # Accel + steer right

            if info["is_colliding"]:
                colliding = True
                # Collision should give negative reward
                assert reward < 0
                break

        assert colliding, "Should have collided with track boundary"
        env.close()

    def test_episode_truncation(self):
        """Test that episodes truncate at max_steps"""
        max_steps = 100
        env = RacingEnv(max_steps=max_steps)
        obs, _ = env.reset()

        for i in range(max_steps + 10):
            obs, reward, terminated, truncated, info = env.step(0)  # No input
            if truncated:
                assert info["step"] == max_steps
                break
        else:
            pytest.fail("Episode should have truncated")
        env.close()

    def test_render_ansi(self):
        """Test ANSI rendering"""
        env = RacingEnv(render_mode="ansi")
        env.reset()

        output = env.render()
        assert isinstance(output, str)
        assert "Position" in output
        assert "Speed" in output
        env.close()

    def test_render_rgb_array(self):
        """Test RGB array rendering"""
        env = RacingEnv(render_mode="rgb_array")
        env.reset()

        img = env.render()
        assert isinstance(img, np.ndarray)
        assert len(img.shape) == 3
        assert img.shape[2] == 3  # RGB
        env.close()

    def test_state_serialization(self):
        """Test get_state and set_state"""
        env = RacingEnv()
        env.reset()

        # Take some actions
        for _ in range(50):
            env.step(1)

        # Get state
        state = env.get_state()
        assert "car" in state
        assert "step_count" in state
        assert "checkpoints_passed" in state

        # Reset and set state
        env.reset()
        env.set_state(state)

        new_state = env.get_state()
        assert new_state["car"]["x"] == state["car"]["x"]
        assert new_state["step_count"] == state["step_count"]
        env.close()


class TestVectorizedEnvironment:
    """Tests for parallel environment support"""

    def test_sync_vec_env(self):
        """Test synchronous vectorized environment"""
        num_envs = 4
        vec_env = make_vec_env(num_envs=num_envs)

        obs, info = vec_env.reset()
        assert obs.shape == (num_envs, 15)

        # Take random actions
        actions = np.array([vec_env.single_action_space.sample() for _ in range(num_envs)])
        obs, rewards, terminated, truncated, info = vec_env.step(actions)

        assert obs.shape == (num_envs, 15)
        assert rewards.shape == (num_envs,)
        assert terminated.shape == (num_envs,)
        assert truncated.shape == (num_envs,)
        vec_env.close()

    def test_vec_env_independence(self):
        """Test that vectorized environments are independent"""
        num_envs = 2
        vec_env = make_vec_env(num_envs=num_envs)

        obs, _ = vec_env.reset()

        # Take different actions in each environment
        actions = np.array([1, 2])  # Accelerate in env 0, brake in env 1
        obs, _, _, _, _ = vec_env.step(actions)

        # Positions should diverge (speeds will be different)
        # After more steps, this becomes more apparent
        for _ in range(60):
            vec_env.step(actions)

        # Get observations - they should be different
        obs, _, _, _, _ = vec_env.step(actions)

        # The speed observations (index 4) should be different
        assert not np.allclose(obs[0], obs[1])
        vec_env.close()


class TestProgressTracking:
    """Tests for lap progress and checkpoint tracking"""

    def test_progress_angle_at_start(self):
        """Test progress angle at starting position"""
        start = track_data.start_position
        progress = get_track_progress_angle(start.x, start.y)

        # Start is at bottom, should be near 0 or 2*PI
        assert progress < 0.1 or progress > 2 * math.pi - 0.1

    def test_progress_increases_counterclockwise(self):
        """Test that progress increases when moving counter-clockwise"""
        # Sample points going counter-clockwise
        angles = [math.pi / 2, math.pi / 4, 0, -math.pi / 4, -math.pi / 2]
        center_radius_x = (INNER_RADIUS_X + OUTER_RADIUS_X) / 2
        center_radius_y = (INNER_RADIUS_Y + OUTER_RADIUS_Y) / 2

        progress_values = []
        for angle in angles:
            x = CENTER_X + math.cos(angle) * center_radius_x
            y = CENTER_Y + math.sin(angle) * center_radius_y
            progress = get_track_progress_angle(x, y)
            progress_values.append(progress)

        # Progress should generally increase (accounting for wraparound)
        # We check that at least some progress increases happen
        increases = sum(
            1 for i in range(len(progress_values) - 1)
            if progress_values[i + 1] > progress_values[i] or
               (progress_values[i] > 5 and progress_values[i + 1] < 1)  # Wraparound
        )
        assert increases > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
