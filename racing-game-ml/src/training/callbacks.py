"""
Training Callbacks for Racing Game ML

Custom callbacks for monitoring training progress, logging metrics,
and saving model checkpoints.
"""

import os
import time
from typing import Any, Dict, List, Optional
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.evaluation import evaluate_policy


class RacingMetricsCallback(BaseCallback):
    """
    Custom callback for tracking racing-specific metrics.

    Tracks:
    - Lap completion rate
    - Average lap time (for completed laps)
    - Crash/collision statistics
    - Best lap time achieved
    - Progress statistics
    """

    def __init__(
        self,
        log_freq: int = 1000,
        verbose: int = 1,
    ):
        """
        Initialize the racing metrics callback.

        Args:
            log_freq: Log metrics every N timesteps
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.log_freq = log_freq

        # Tracking variables
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.lap_times: List[float] = []
        self.laps_completed: int = 0
        self.total_episodes: int = 0
        self.total_collisions: int = 0
        self.best_lap_time: Optional[float] = None
        self.episode_progress: List[float] = []

        # Per-episode tracking
        self.current_episode_collisions: List[int] = []
        self.start_time: float = 0.0

    def _on_training_start(self) -> None:
        """Called at the start of training"""
        self.start_time = time.time()
        self._reset_episode_tracking()

    def _reset_episode_tracking(self) -> None:
        """Reset per-episode tracking variables"""
        n_envs = self.training_env.num_envs if self.training_env else 1
        self.current_episode_collisions = [0] * n_envs

    def _on_step(self) -> bool:
        """
        Called after each environment step.

        Returns:
            True to continue training
        """
        # Get info from environments
        infos = self.locals.get("infos", [])

        for i, info in enumerate(infos):
            # Track collisions
            if info.get("is_colliding", False):
                if i < len(self.current_episode_collisions):
                    self.current_episode_collisions[i] += 1

            # Check for episode end (done flag)
            if "episode" in info:
                # Episode finished - extract metrics
                self.total_episodes += 1
                ep_reward = info["episode"].get("r", 0)
                ep_length = info["episode"].get("l", 0)

                self.episode_rewards.append(ep_reward)
                self.episode_lengths.append(ep_length)

                # Check for lap completion
                if info.get("lap_complete", False):
                    self.laps_completed += 1
                    lap_time = info.get("episode_lap_time")
                    if lap_time is not None:
                        self.lap_times.append(lap_time)
                        if self.best_lap_time is None or lap_time < self.best_lap_time:
                            self.best_lap_time = lap_time
                            if self.verbose >= 1:
                                print(f"New best lap time: {lap_time:.2f}s")

                # Track progress
                progress = info.get("episode_progress", 0)
                self.episode_progress.append(progress)

                # Track collisions
                if i < len(self.current_episode_collisions):
                    self.total_collisions += self.current_episode_collisions[i]
                    self.current_episode_collisions[i] = 0

        # Log metrics periodically
        if self.n_calls % self.log_freq == 0:
            self._log_metrics()

        return True

    def _log_metrics(self) -> None:
        """Log current metrics to TensorBoard and console"""
        if self.total_episodes == 0:
            return

        # Calculate statistics
        recent_episodes = min(100, len(self.episode_rewards))
        recent_rewards = self.episode_rewards[-recent_episodes:]
        recent_lengths = self.episode_lengths[-recent_episodes:]
        recent_progress = self.episode_progress[-recent_episodes:] if self.episode_progress else [0]

        avg_reward = np.mean(recent_rewards)
        avg_length = np.mean(recent_lengths)
        avg_progress = np.mean(recent_progress)

        lap_completion_rate = self.laps_completed / self.total_episodes * 100
        avg_lap_time = np.mean(self.lap_times) if self.lap_times else 0

        # Log to TensorBoard
        if self.logger:
            self.logger.record("racing/lap_completion_rate", lap_completion_rate)
            self.logger.record("racing/laps_completed", self.laps_completed)
            self.logger.record("racing/total_episodes", self.total_episodes)
            self.logger.record("racing/avg_progress", avg_progress)
            self.logger.record("racing/total_collisions", self.total_collisions)

            if self.lap_times:
                self.logger.record("racing/avg_lap_time", avg_lap_time)
                self.logger.record("racing/best_lap_time", self.best_lap_time)

            self.logger.record("racing/avg_reward_100ep", avg_reward)
            self.logger.record("racing/avg_length_100ep", avg_length)

        # Console logging
        if self.verbose >= 1:
            elapsed = time.time() - self.start_time
            print(f"\n--- Racing Metrics @ {self.num_timesteps} steps ({elapsed:.0f}s) ---")
            print(f"Episodes: {self.total_episodes}, Laps: {self.laps_completed} ({lap_completion_rate:.1f}%)")
            print(f"Avg Reward (100ep): {avg_reward:.2f}, Avg Length: {avg_length:.0f}")
            print(f"Avg Progress: {avg_progress:.2f} rad")
            if self.best_lap_time:
                print(f"Best Lap: {self.best_lap_time:.2f}s, Avg Lap: {avg_lap_time:.2f}s")
            print("-" * 50)

    def _on_training_end(self) -> None:
        """Called at the end of training"""
        self._log_metrics()
        self._print_summary()

    def _print_summary(self) -> None:
        """Print final training summary"""
        elapsed = time.time() - self.start_time

        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        print(f"Total timesteps: {self.num_timesteps}")
        print(f"Total episodes: {self.total_episodes}")
        print(f"Training time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
        print(f"Timesteps/sec: {self.num_timesteps / elapsed:.0f}")
        print()
        print(f"Laps completed: {self.laps_completed}")
        if self.total_episodes > 0:
            print(f"Lap completion rate: {self.laps_completed / self.total_episodes * 100:.1f}%")
        if self.best_lap_time:
            print(f"Best lap time: {self.best_lap_time:.2f}s")
        if self.lap_times:
            print(f"Average lap time: {np.mean(self.lap_times):.2f}s")
        if self.episode_rewards:
            print(f"Final avg reward (100ep): {np.mean(self.episode_rewards[-100:]):.2f}")
        print("=" * 60)

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics as dictionary"""
        return {
            "total_timesteps": self.num_timesteps,
            "total_episodes": self.total_episodes,
            "laps_completed": self.laps_completed,
            "lap_completion_rate": self.laps_completed / max(1, self.total_episodes),
            "best_lap_time": self.best_lap_time,
            "avg_lap_time": np.mean(self.lap_times) if self.lap_times else None,
            "total_collisions": self.total_collisions,
            "avg_reward_100ep": np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
            "avg_progress": np.mean(self.episode_progress[-100:]) if self.episode_progress else 0,
        }


class BestModelCallback(BaseCallback):
    """
    Callback to save the best model based on lap completion rate.

    Saves the model whenever the lap completion rate improves.
    """

    def __init__(
        self,
        save_path: str,
        check_freq: int = 1000,
        min_episodes: int = 10,
        verbose: int = 1,
    ):
        """
        Initialize the best model callback.

        Args:
            save_path: Path to save the best model
            check_freq: Check for improvement every N timesteps
            min_episodes: Minimum episodes before starting to save
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.save_path = save_path
        self.check_freq = check_freq
        self.min_episodes = min_episodes

        self.best_lap_rate: float = 0.0
        self.best_avg_reward: float = float("-inf")
        self.episode_count: int = 0
        self.lap_count: int = 0
        self.recent_rewards: List[float] = []

    def _on_step(self) -> bool:
        """Called after each step"""
        # Track episodes and laps
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_count += 1
                self.recent_rewards.append(info["episode"].get("r", 0))
                if len(self.recent_rewards) > 100:
                    self.recent_rewards.pop(0)

                if info.get("lap_complete", False):
                    self.lap_count += 1

        # Check for improvement periodically
        if self.n_calls % self.check_freq == 0 and self.episode_count >= self.min_episodes:
            self._check_and_save()

        return True

    def _check_and_save(self) -> None:
        """Check if current performance is best and save if so"""
        lap_rate = self.lap_count / max(1, self.episode_count)
        avg_reward = np.mean(self.recent_rewards) if self.recent_rewards else 0

        # Save if lap rate improved or (same lap rate but better reward)
        should_save = False
        reason = ""

        if lap_rate > self.best_lap_rate:
            should_save = True
            reason = f"lap rate improved {self.best_lap_rate:.2%} -> {lap_rate:.2%}"
            self.best_lap_rate = lap_rate
            self.best_avg_reward = avg_reward
        elif lap_rate == self.best_lap_rate and avg_reward > self.best_avg_reward:
            should_save = True
            reason = f"avg reward improved {self.best_avg_reward:.2f} -> {avg_reward:.2f}"
            self.best_avg_reward = avg_reward

        if should_save:
            path = Path(self.save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(str(path))

            if self.verbose >= 1:
                print(f"\nSaved best model ({reason})")
                print(f"  Path: {path}")
                print(f"  Lap rate: {lap_rate:.2%}, Avg reward: {avg_reward:.2f}")


class ProgressBarCallback(BaseCallback):
    """
    Simple progress bar callback for training.
    """

    def __init__(self, total_timesteps: int):
        """
        Initialize progress bar.

        Args:
            total_timesteps: Total timesteps for training
        """
        super().__init__(verbose=0)
        self.total_timesteps = total_timesteps
        self.last_percent = 0

    def _on_step(self) -> bool:
        """Update progress bar"""
        percent = int(self.num_timesteps / self.total_timesteps * 100)
        if percent > self.last_percent:
            self.last_percent = percent
            bar_length = 40
            filled = int(bar_length * percent / 100)
            bar = "=" * filled + "-" * (bar_length - filled)
            print(f"\rProgress: [{bar}] {percent}% ({self.num_timesteps}/{self.total_timesteps})", end="")
            if percent == 100:
                print()
        return True


def create_training_callbacks(
    model_save_path: str,
    eval_env: Optional[VecEnv] = None,
    save_freq: int = 10_000,
    eval_freq: int = 10_000,
    n_eval_episodes: int = 5,
    log_freq: int = 1000,
    total_timesteps: int = 500_000,
    verbose: int = 1,
) -> CallbackList:
    """
    Create a list of training callbacks.

    Args:
        model_save_path: Directory for saving models
        eval_env: Environment for evaluation (optional)
        save_freq: Save checkpoint every N timesteps
        eval_freq: Evaluate every N timesteps
        n_eval_episodes: Number of evaluation episodes
        log_freq: Log metrics every N timesteps
        total_timesteps: Total training timesteps (for progress bar)
        verbose: Verbosity level

    Returns:
        CallbackList with all callbacks
    """
    callbacks = []

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=model_save_path,
        name_prefix="racing_model",
        save_replay_buffer=False,
        verbose=verbose,
    )
    callbacks.append(checkpoint_callback)

    # Racing metrics callback
    racing_metrics = RacingMetricsCallback(
        log_freq=log_freq,
        verbose=verbose,
    )
    callbacks.append(racing_metrics)

    # Best model callback
    best_model_path = os.path.join(model_save_path, "best_model")
    best_model_callback = BestModelCallback(
        save_path=best_model_path,
        check_freq=log_freq,
        verbose=verbose,
    )
    callbacks.append(best_model_callback)

    # Evaluation callback (if eval env provided)
    if eval_env is not None:
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(model_save_path, "eval_best"),
            log_path=os.path.join(model_save_path, "eval_logs"),
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            verbose=verbose,
        )
        callbacks.append(eval_callback)

    # Progress bar (only if not verbose)
    if verbose == 0:
        progress_callback = ProgressBarCallback(total_timesteps)
        callbacks.append(progress_callback)

    return CallbackList(callbacks)
