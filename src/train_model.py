import os
import random

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

from src.agent import AgentFactory
from src.hyperparameters import Hyperparameters
from src.results import EvalHistory, Results, calculate_eval_reward_stats

N_EVAL_EPISODES = 10


def train_adaptive(
    level: str,
    params: Hyperparameters,
    seed: int = 0,
    max_episode_steps: int = 750,
    n_eval_episodes: int = 25,
    eval_interval_episodes: int = 100,
    plateau_patience: int = 10,
    plateau_threshold: float = 5.0,
    success_threshold: float = 250.0,
    success_window: int = 5,
    max_iterations: int = 10000,
    verbose: bool = False,
) -> tuple[Results, DQN]:
    """
    Train until plateau or consistent success, with a max iteration cap.
    Returns the trained agent and eval history as (timestep, eval_mean) pairs.
    """
    seed_everything(seed)

    avg_episode_steps = convert_max_steps_to_avg_steps(max_episode_steps)
    total_timesteps = max_iterations * avg_episode_steps

    environment = make_environment(level, max_episode_steps, seed=seed)
    eval_env = make_environment(level, max_episode_steps, seed=seed + 1)
    agent = AgentFactory.create_dqn_agent(
        environment, params, seed=seed, total_timesteps=total_timesteps
    )

    eval_interval = eval_interval_episodes * avg_episode_steps
    results = Results()

    callback = AdaptiveStopCallback(
        eval_env=eval_env,
        eval_interval=eval_interval,
        n_eval_episodes=n_eval_episodes,
        plateau_patience=plateau_patience,
        plateau_threshold=plateau_threshold,
        success_threshold=success_threshold,
        success_window=success_window,
        verbose=verbose,
        results=results,
    )

    agent.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        reset_num_timesteps=True,
    )

    seed_everything(seed + 1)
    _, _ = eval_and_watch(
        agent,
        level,
        goal_reward=success_threshold,
        seed=seed,
        max_episode_steps=max_episode_steps,
        n_eval_episodes=n_eval_episodes,
        watch=verbose,
    )

    environment.close()
    eval_env.close()

    return results, agent


class AdaptiveStopCallback(BaseCallback):
    """
    Stops training when either:
    - eval_mean hasn't improved by plateau_threshold in plateau_patience consecutive evals
    - eval_mean has stayed above success_threshold for success_window consecutive evals
    Evaluates every eval_interval timesteps.
    """

    def __init__(
        self,
        eval_env: gym.Env,  # type: ignore[type-arg]
        eval_interval: int,
        n_eval_episodes: int,
        plateau_patience: int,
        plateau_threshold: float,
        success_threshold: float,
        success_window: int,
        verbose: bool = False,
        results: Results = Results(),
    ) -> None:
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.eval_interval = eval_interval
        self.n_eval_episodes = n_eval_episodes
        self.plateau_patience = plateau_patience
        self.plateau_threshold = plateau_threshold
        self.success_threshold = success_threshold
        self.success_window = success_window

        self._best_eval: float = float("-inf")
        self._no_improvement_count: int = 0
        self._success_count: int = 0
        self._steps_since_eval: int = 0

        self.results = results
        self._episode_reward: float = 0.0

    def _on_step(self) -> bool:
        reward: float = float(self.locals["rewards"][0])
        self._episode_reward += reward

        self._steps_since_eval += 1

        if self._steps_since_eval < self.eval_interval:
            return True

        self._steps_since_eval = 0

        rewards = evaluate_model(self.model, self.eval_env, self.n_eval_episodes)  # type: ignore[arg-type]
        eval_history = calculate_eval_reward_stats(rewards, self.success_threshold)
        self.results.eval_history.append(eval_history)

        if self.verbose:
            print(
                f"  [eval @ {self.num_timesteps}] {eval_history.mean:.2f} +/- {eval_history.std:.2f} | Goal hit: {eval_history.hit_pct:.0f}%"
            )

        # Check success
        if eval_history.mean >= self.success_threshold:
            self._success_count += 1
            if self._success_count >= self.success_window:
                print(
                    f"  [STOP] Consistently above {self.success_threshold} for {self.success_window} evals"
                )
                return False
        else:
            self._success_count = 0

        # Check plateau
        if eval_history.mean > self._best_eval + self.plateau_threshold:
            self._best_eval = eval_history.mean
            self._no_improvement_count = 0
        else:
            self._no_improvement_count += 1
            if self._no_improvement_count >= self.plateau_patience:
                print(
                    f"  [STOP] No improvement in {self.plateau_patience} evals (best={self._best_eval:.1f})"
                )
                return False

        return True


def evaluate_model(
    agent: DQN,
    environment: gym.Env,  # type: ignore[type-arg]
    n_eval_episodes: int = N_EVAL_EPISODES,
) -> list[float]:
    """Run deterministic episodes to get a clean measure of policy performance."""
    rewards: list[float] = []
    for _ in range(n_eval_episodes):
        obs, _ = environment.reset()
        done = False
        episode_reward = 0.0
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = environment.step(action)
            episode_reward += float(reward)
            done = terminated or truncated
        rewards.append(episode_reward)
    return rewards


def eval_and_watch(
    agent: DQN,
    level: str,
    goal_reward: float = 200,
    seed: int = 0,
    max_episode_steps: int = 750,
    n_eval_episodes: int = N_EVAL_EPISODES,
    watch: bool = True,
) -> tuple[list[float], EvalHistory]:
    """Evaluate a trained agent deterministically and optionally render episodes."""
    seed_everything(seed + 1)
    eval_env = make_environment(level, max_episode_steps, seed=seed)
    eval_rewards = evaluate_model(agent, eval_env, n_eval_episodes=n_eval_episodes)
    eval_env.close()

    eval_history = calculate_eval_reward_stats(eval_rewards, goal_reward)
    print(
        f"Eval mean: {eval_history.mean:.2f} +/- {eval_history.std:.2f} | Goal hit: {eval_history.hit_pct:.0f}%"
    )

    if watch:
        input("Press enter to watch 5 rendered episodes...")
        render_env = make_environment(level, max_episode_steps, seed=seed, render=True)
        for ep in range(5):
            obs, _ = render_env.reset()
            episode_reward = 0.0
            done = False
            while not done:
                action, _ = agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = render_env.step(action)
                episode_reward += float(reward)
                done = terminated or truncated
            print(f"Episode {ep + 1}: {episode_reward:.1f}")
        render_env.close()

    return eval_rewards, eval_history


def seed_everything(seed: int) -> None:
    """Seed all sources of randomness for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def make_environment(
    level: str, max_episode_steps: int, seed: int = 0, render: bool = False
) -> gym.Env:  # type: ignore[type-arg]
    render_mode = "human" if render else None
    env = gym.make(level, max_episode_steps=max_episode_steps, render_mode=render_mode)
    env.reset(seed=seed)
    return env


def convert_max_steps_to_avg_steps(max_steps: int) -> int:
    # Rough approximation for the average case
    return max_steps // 3
