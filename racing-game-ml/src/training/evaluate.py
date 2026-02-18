"""
Model Evaluation Utilities for Racing Game ML

Functions for loading trained models, running evaluation episodes,
and reporting statistics.
"""

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

from ..environment.racing_env import RacingEnv


@dataclass
class EpisodeResult:
    """Results from a single evaluation episode"""
    reward: float
    length: int
    lap_completed: bool
    lap_time: Optional[float]
    progress: float
    collisions: int
    final_position: Tuple[float, float]
    final_speed: float


@dataclass
class EvaluationResults:
    """Aggregated results from evaluation episodes"""
    n_episodes: int
    mean_reward: float
    std_reward: float
    mean_length: float
    std_length: float
    lap_completion_rate: float
    n_laps_completed: int
    mean_lap_time: Optional[float]
    best_lap_time: Optional[float]
    worst_lap_time: Optional[float]
    mean_progress: float
    mean_collisions: float
    total_collisions: int
    evaluation_time: float
    episodes: List[EpisodeResult]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding detailed episode data)"""
        result = asdict(self)
        # Convert episodes to simplified format
        result["episodes"] = [asdict(ep) for ep in self.episodes]
        return result

    def __str__(self) -> str:
        """Human-readable summary"""
        lines = [
            "=" * 60,
            "EVALUATION RESULTS",
            "=" * 60,
            f"Episodes: {self.n_episodes}",
            f"Evaluation time: {self.evaluation_time:.1f}s",
            "",
            "Performance:",
            f"  Mean reward: {self.mean_reward:.2f} (+/- {self.std_reward:.2f})",
            f"  Mean length: {self.mean_length:.0f} (+/- {self.std_length:.0f})",
            f"  Mean progress: {self.mean_progress:.2f} rad",
            "",
            "Lap Statistics:",
            f"  Laps completed: {self.n_laps_completed}/{self.n_episodes} ({self.lap_completion_rate:.1%})",
        ]

        if self.mean_lap_time is not None:
            lines.extend([
                f"  Mean lap time: {self.mean_lap_time:.2f}s",
                f"  Best lap time: {self.best_lap_time:.2f}s",
                f"  Worst lap time: {self.worst_lap_time:.2f}s",
            ])

        lines.extend([
            "",
            "Collisions:",
            f"  Total collisions: {self.total_collisions}",
            f"  Mean per episode: {self.mean_collisions:.1f}",
            "=" * 60,
        ])

        return "\n".join(lines)


def load_model(model_path: str, device: str = "auto") -> PPO:
    """
    Load a trained PPO model.

    Args:
        model_path: Path to the saved model (without .zip extension)
        device: Device to load model on ('auto', 'cpu', 'cuda')

    Returns:
        Loaded PPO model
    """
    # Handle path with or without .zip extension
    path = Path(model_path)
    if path.suffix != ".zip":
        path = path.with_suffix(".zip")

    if not path.exists():
        # Try without .zip
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

    model = PPO.load(str(path), device=device)
    return model


def create_eval_env(
    render_mode: Optional[str] = None,
    **env_kwargs
) -> RacingEnv:
    """
    Create an environment for evaluation.

    Args:
        render_mode: Rendering mode ('human', 'rgb_array', 'ansi', None)
        **env_kwargs: Additional environment arguments

    Returns:
        Racing environment instance
    """
    return RacingEnv(render_mode=render_mode, **env_kwargs)


def evaluate_episode(
    model: PPO,
    env: RacingEnv,
    deterministic: bool = True,
    render: bool = False,
) -> EpisodeResult:
    """
    Run a single evaluation episode.

    Args:
        model: Trained PPO model
        env: Racing environment
        deterministic: Use deterministic actions (no exploration)
        render: Whether to render the environment

    Returns:
        Episode result
    """
    obs, info = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    collisions = 0

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1

        if info.get("is_colliding", False):
            collisions += 1

        if render:
            env.render()

    return EpisodeResult(
        reward=total_reward,
        length=steps,
        lap_completed=info.get("lap_complete", False),
        lap_time=info.get("episode_lap_time"),
        progress=info.get("episode_progress", 0.0),
        collisions=collisions,
        final_position=info.get("position", (0, 0)),
        final_speed=info.get("speed", 0.0),
    )


def evaluate_model(
    model: PPO,
    n_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    verbose: int = 1,
    env_kwargs: Optional[Dict[str, Any]] = None,
) -> EvaluationResults:
    """
    Evaluate a trained model over multiple episodes.

    Args:
        model: Trained PPO model
        n_episodes: Number of evaluation episodes
        deterministic: Use deterministic actions
        render: Whether to render
        verbose: Verbosity level
        env_kwargs: Additional environment arguments

    Returns:
        Aggregated evaluation results
    """
    env_kwargs = env_kwargs or {}
    render_mode = "human" if render else None
    env = create_eval_env(render_mode=render_mode, **env_kwargs)

    episodes: List[EpisodeResult] = []
    start_time = time.time()

    for i in range(n_episodes):
        if verbose >= 1:
            print(f"Evaluating episode {i + 1}/{n_episodes}...", end="\r")

        result = evaluate_episode(model, env, deterministic, render)
        episodes.append(result)

        if verbose >= 2:
            status = "LAP!" if result.lap_completed else "---"
            print(f"Episode {i + 1}: reward={result.reward:.2f}, "
                  f"length={result.length}, progress={result.progress:.2f} {status}")

    eval_time = time.time() - start_time
    env.close()

    if verbose >= 1:
        print()  # Clear progress line

    # Aggregate results
    rewards = [ep.reward for ep in episodes]
    lengths = [ep.length for ep in episodes]
    lap_times = [ep.lap_time for ep in episodes if ep.lap_time is not None]
    n_laps = sum(1 for ep in episodes if ep.lap_completed)

    results = EvaluationResults(
        n_episodes=n_episodes,
        mean_reward=np.mean(rewards),
        std_reward=np.std(rewards),
        mean_length=np.mean(lengths),
        std_length=np.std(lengths),
        lap_completion_rate=n_laps / n_episodes,
        n_laps_completed=n_laps,
        mean_lap_time=np.mean(lap_times) if lap_times else None,
        best_lap_time=min(lap_times) if lap_times else None,
        worst_lap_time=max(lap_times) if lap_times else None,
        mean_progress=np.mean([ep.progress for ep in episodes]),
        mean_collisions=np.mean([ep.collisions for ep in episodes]),
        total_collisions=sum(ep.collisions for ep in episodes),
        evaluation_time=eval_time,
        episodes=episodes,
    )

    if verbose >= 1:
        print(results)

    return results


def compare_models(
    model_paths: List[str],
    n_episodes: int = 10,
    deterministic: bool = True,
    verbose: int = 1,
    env_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, EvaluationResults]:
    """
    Compare multiple trained models.

    Args:
        model_paths: List of paths to model files
        n_episodes: Number of episodes per model
        deterministic: Use deterministic actions
        verbose: Verbosity level
        env_kwargs: Additional environment arguments

    Returns:
        Dictionary mapping model path to evaluation results
    """
    results = {}

    for path in model_paths:
        if verbose >= 1:
            print(f"\nEvaluating model: {path}")
            print("-" * 40)

        try:
            model = load_model(path)
            eval_results = evaluate_model(
                model,
                n_episodes=n_episodes,
                deterministic=deterministic,
                verbose=verbose,
                env_kwargs=env_kwargs,
            )
            results[path] = eval_results
        except Exception as e:
            print(f"Error evaluating {path}: {e}")

    # Print comparison summary
    if verbose >= 1 and len(results) > 1:
        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)
        print(f"{'Model':<30} {'Lap Rate':>10} {'Avg Reward':>12} {'Best Lap':>10}")
        print("-" * 62)

        for path, result in sorted(results.items(), key=lambda x: -x[1].lap_completion_rate):
            name = Path(path).stem[:28]
            lap_rate = f"{result.lap_completion_rate:.1%}"
            avg_reward = f"{result.mean_reward:.2f}"
            best_lap = f"{result.best_lap_time:.2f}s" if result.best_lap_time else "N/A"
            print(f"{name:<30} {lap_rate:>10} {avg_reward:>12} {best_lap:>10}")

        print("=" * 60)

    return results


def save_evaluation_results(
    results: EvaluationResults,
    filepath: str,
) -> None:
    """
    Save evaluation results to JSON file.

    Args:
        results: Evaluation results to save
        filepath: Output file path
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(results.to_dict(), f, indent=2)

    print(f"Results saved to: {path}")


def watch_model(
    model_path: str,
    n_episodes: int = 5,
    deterministic: bool = True,
    delay: float = 0.0,
) -> None:
    """
    Watch a trained model play the game.

    Args:
        model_path: Path to the model file
        n_episodes: Number of episodes to watch
        deterministic: Use deterministic actions
        delay: Delay between steps (seconds)
    """
    model = load_model(model_path)
    env = create_eval_env(render_mode="ansi")

    for episode in range(n_episodes):
        print(f"\n{'=' * 40}")
        print(f"Episode {episode + 1}/{n_episodes}")
        print("=" * 40)

        obs, info = env.reset()
        done = False
        total_reward = 0.0
        step = 0

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step += 1

            # Print state every N steps
            if step % 60 == 0:  # Every ~1 second at 60 FPS
                print(env.render())
                print(f"Step: {step}, Reward: {total_reward:.2f}")

            if delay > 0:
                time.sleep(delay)

        print(f"\nEpisode finished!")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Steps: {step}")
        print(f"Lap completed: {info.get('lap_complete', False)}")
        if info.get('episode_lap_time'):
            print(f"Lap time: {info['episode_lap_time']:.2f}s")


# CLI entry point for evaluation
def main():
    """CLI entry point for model evaluation"""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate trained racing models")
    parser.add_argument("model_path", type=str, help="Path to trained model")
    parser.add_argument("-n", "--n-episodes", type=int, default=10,
                        help="Number of evaluation episodes")
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic actions (with exploration)")
    parser.add_argument("--render", action="store_true",
                        help="Render episodes (slow)")
    parser.add_argument("--watch", action="store_true",
                        help="Watch mode with text rendering")
    parser.add_argument("-o", "--output", type=str,
                        help="Save results to JSON file")
    parser.add_argument("-v", "--verbose", type=int, default=1,
                        help="Verbosity level (0-2)")

    args = parser.parse_args()

    if args.watch:
        watch_model(
            args.model_path,
            n_episodes=args.n_episodes,
            deterministic=not args.stochastic,
        )
    else:
        model = load_model(args.model_path)
        results = evaluate_model(
            model,
            n_episodes=args.n_episodes,
            deterministic=not args.stochastic,
            render=args.render,
            verbose=args.verbose,
        )

        if args.output:
            save_evaluation_results(results, args.output)


if __name__ == "__main__":
    main()
