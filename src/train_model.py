import os
import random
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

from src.agent import AgentFactory, HitGoalCallback
from src.hyperparameters import Hyperparameters
from src.results import Results

N_EVAL_EPISODES = 10


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


def run_training(
    agent: DQN,
    level: str,
    iterations: int,
    goal_reward: float = 200,
    seed: int = 0,
    max_episode_steps: int = 750,
    verbose: bool = False,
    reset: bool = True,
) -> Results:
    """Core training loop — works for fresh agents and continuing from a checkpoint."""
    environment = make_environment(level, max_episode_steps, seed=seed)
    result = Results()
    callback = HitGoalCallback(result, goal_reward, verbose=verbose)
    avg_episode_steps = max_episode_steps // 3
    agent.set_env(environment)
    agent.learn(
        total_timesteps=iterations * avg_episode_steps,
        callback=callback,
        reset_num_timesteps=reset,
        progress_bar=verbose,
    )
    environment.close()
    return result


def eval_and_watch(
    agent: DQN,
    level: str,
    goal_reward: float = 200,
    seed: int = 0,
    max_episode_steps: int = 750,
    n_eval_episodes: int = N_EVAL_EPISODES,
    watch: bool = True,
) -> tuple[float, float, float, list[float]]:
    """Evaluate a trained agent deterministically and optionally render episodes."""
    seed_everything(seed + 1)
    eval_env = make_environment(level, max_episode_steps, seed=seed)
    eval_rewards = evaluate_model(agent, eval_env, n_eval_episodes=n_eval_episodes)
    eval_env.close()

    eval_mean = sum(eval_rewards) / len(eval_rewards)
    eval_std = (
        sum((r - eval_mean) ** 2 for r in eval_rewards) / len(eval_rewards)
    ) ** 0.5
    hit_pct = sum(1 for r in eval_rewards if r >= goal_reward) / len(eval_rewards) * 100
    print(f"Eval mean: {eval_mean:.2f} +/- {eval_std:.2f} | Goal hit: {hit_pct:.0f}%")

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

    return eval_mean, eval_std, hit_pct, eval_rewards


def train_model(
    level: str,
    params: Hyperparameters,
    iterations: int = 1500,
    goal_reward: float = 200,
    verbose: bool = False,
    seed: int = 0,
    max_episode_steps: int = 750,
    n_eval_episodes: int = N_EVAL_EPISODES,
) -> tuple[Results, DQN]:
    seed_everything(seed)

    avg_episode_steps = max_episode_steps // 3
    total_timesteps = iterations * avg_episode_steps

    environment = make_environment(level, max_episode_steps, seed=seed)
    agent = AgentFactory.create_dqn_agent(
        environment, params, seed=seed, total_timesteps=total_timesteps
    )
    environment.close()

    result = run_training(
        agent,
        level,
        iterations,
        goal_reward=goal_reward,
        seed=seed,
        max_episode_steps=max_episode_steps,
        verbose=verbose,
        reset=True,
    )

    seed_everything(seed + 1)
    _, _, _, eval_rewards = eval_and_watch(
        agent,
        level,
        goal_reward=goal_reward,
        seed=seed,
        max_episode_steps=max_episode_steps,
        n_eval_episodes=n_eval_episodes,
        watch=verbose,
    )
    result.add_eval(eval_rewards, goal_reward)

    return result, agent


def continue_training(
    agent: DQN,
    level: str,
    iterations: int,
    goal_reward: float = 200,
    seed: int = 0,
    max_episode_steps: int = 750,
) -> Results:
    """Continue training an already-loaded agent for additional iterations."""
    return run_training(
        agent,
        level,
        iterations,
        goal_reward=goal_reward,
        seed=seed,
        max_episode_steps=max_episode_steps,
        reset=False,
    )


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
        goal_reward: float,
        eval_interval: int,
        n_eval_episodes: int,
        plateau_patience: int,
        plateau_threshold: float,
        success_threshold: float,
        success_window: int,
        verbose: bool = False,
    ) -> None:
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.goal_reward = goal_reward
        self.eval_interval = eval_interval
        self.n_eval_episodes = n_eval_episodes
        self.plateau_patience = plateau_patience
        self.plateau_threshold = plateau_threshold
        self.success_threshold = success_threshold
        self.success_window = success_window

        self.eval_history: list[tuple[int, float]] = []
        self._best_eval: float = float("-inf")
        self._no_improvement_count: int = 0
        self._success_count: int = 0
        self._last_eval_step: int = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_step < self.eval_interval:
            return True
        self._last_eval_step = self.num_timesteps

        rewards = evaluate_model(self.model, self.eval_env, self.n_eval_episodes)  # type: ignore[arg-type]
        eval_mean = sum(rewards) / len(rewards)
        self.eval_history.append((self.num_timesteps, eval_mean))

        if self.verbose:
            print(f"  [eval @ {self.num_timesteps}] mean={eval_mean:.1f}")

        # Check success
        if eval_mean >= self.success_threshold:
            self._success_count += 1
            if self._success_count >= self.success_window:
                print(
                    f"  [STOP] Consistently above {self.success_threshold} for {self.success_window} evals"
                )
                return False
        else:
            self._success_count = 0

        # Check plateau
        if eval_mean > self._best_eval + self.plateau_threshold:
            self._best_eval = eval_mean
            self._no_improvement_count = 0
        else:
            self._no_improvement_count += 1
            if self._no_improvement_count >= self.plateau_patience:
                print(
                    f"  [STOP] No improvement in {self.plateau_patience} evals (best={self._best_eval:.1f})"
                )
                return False

        return True


def train_adaptive(
    level: str,
    params: Hyperparameters,
    seed: int = 0,
    max_episode_steps: int = 750,
    goal_reward: float = 200,
    n_eval_episodes: int = 25,
    eval_interval_episodes: int = 100,
    plateau_patience: int = 10,
    plateau_threshold: float = 5.0,
    success_threshold: float = 250.0,
    success_window: int = 5,
    max_iterations: int = 10000,
) -> tuple[DQN, list[tuple[int, float]]]:
    """
    Train until plateau or consistent success, with a max iteration cap.
    Returns the trained agent and eval history as (timestep, eval_mean) pairs.
    """
    seed_everything(seed)

    avg_episode_steps = max_episode_steps // 3
    total_timesteps = max_iterations * avg_episode_steps

    environment = make_environment(level, max_episode_steps, seed=seed)
    eval_env = make_environment(level, max_episode_steps, seed=seed + 1)
    agent = AgentFactory.create_dqn_agent(
        environment, params, seed=seed, total_timesteps=total_timesteps
    )

    eval_interval = eval_interval_episodes * avg_episode_steps

    callback = AdaptiveStopCallback(
        eval_env=eval_env,
        goal_reward=goal_reward,
        eval_interval=eval_interval,
        n_eval_episodes=n_eval_episodes,
        plateau_patience=plateau_patience,
        plateau_threshold=plateau_threshold,
        success_threshold=success_threshold,
        success_window=success_window,
        verbose=True,
    )

    agent.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        reset_num_timesteps=True,
    )

    environment.close()
    eval_env.close()

    return agent, callback.eval_history
