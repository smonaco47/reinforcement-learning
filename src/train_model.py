import os
import random
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import DQN

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
) -> gym.Env[Any, Any]:
    render_mode = "human" if render else None
    env = gym.make(level, max_episode_steps=max_episode_steps, render_mode=render_mode)
    env.reset(seed=seed)
    return env


def evaluate_model(
    agent: DQN,
    environment: gym.Env[Any, Any],
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
    goal_reward: float = 200.0,
    seed: int = 0,
    max_episode_steps: int = 750,
    verbose: bool = False,
    reset: bool = True,
) -> Results:
    """Core training loop — works for both fresh agents and continuing from a checkpoint."""
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
    goal_reward: float = 200.0,
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
    goal_reward: float = 200.0,
    verbose: bool = False,
    seed: int = 0,
    max_episode_steps: int = 750,
    n_eval_episodes: int = N_EVAL_EPISODES,
) -> tuple[Results, DQN]:
    seed_everything(seed)

    environment = make_environment(level, max_episode_steps, seed=seed)
    agent = AgentFactory.create_dqn_agent(environment, params, seed=seed)
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
    goal_reward: float = 200.0,
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


def train_until_solved(
    agent: DQN,
    level: str,
    params: Hyperparameters,
    seed: int = 0,
    max_episode_steps: int = 750,
    goal_reward: float = 200.0,
    n_eval_episodes: int = 25,
    eval_interval: int = 100,  # evaluate every N iterations
    plateau_window: int = 5,  # evals with no improvement before stopping
    plateau_threshold: float = 2.0,  # min improvement in eval_mean to not be stalled
    solved_threshold: float = 250.0,  # eval_mean considered "solved"
    solved_window: int = 3,  # consecutive evals above threshold to stop
    max_iterations: int = 10000,  # hard cap
) -> tuple[list[float], list[int], DQN]:
    """
    Train until one of three conditions:
    - Stalled: eval_mean hasn't improved by plateau_threshold in plateau_window evals
    - Solved: eval_mean >= solved_threshold for solved_window consecutive evals
    - Max: max_iterations reached

    Returns (eval_means, eval_steps, agent).
    """
    eval_means: list[float] = []
    eval_steps: list[int] = []
    best_eval: float = float("-inf")
    stall_count: int = 0
    solved_count: int = 0
    total_iterations: int = 0

    print(
        f"Training with early stopping — plateau_window={plateau_window}, "
        f"solved_threshold={solved_threshold}, max_iterations={max_iterations}"
    )

    while total_iterations < max_iterations:
        chunk = min(eval_interval, max_iterations - total_iterations)
        run_training(
            agent,
            level,
            chunk,
            goal_reward=goal_reward,
            seed=seed,
            max_episode_steps=max_episode_steps,
            verbose=False,
            reset=(total_iterations == 0),
        )
        total_iterations += chunk

        seed_everything(seed + 1)
        eval_env = make_environment(level, max_episode_steps, seed=seed)
        eval_rewards = evaluate_model(agent, eval_env, n_eval_episodes=n_eval_episodes)
        eval_env.close()

        eval_mean = sum(eval_rewards) / len(eval_rewards)
        eval_means.append(eval_mean)
        eval_steps.append(total_iterations)

        print(
            f"  [{total_iterations}/{max_iterations}] eval_mean={eval_mean:.1f} "
            f"best={best_eval:.1f} stall={stall_count}/{plateau_window} "
            f"solved={solved_count}/{solved_window}"
        )

        # Check solved
        if eval_mean >= solved_threshold:
            solved_count += 1
            if solved_count >= solved_window:
                print(
                    f"Solved! eval_mean={eval_mean:.1f} >= {solved_threshold} "
                    f"for {solved_window} consecutive evals"
                )
                break
        else:
            solved_count = 0

        # Check plateau
        if eval_mean > best_eval + plateau_threshold:
            best_eval = eval_mean
            stall_count = 0
        else:
            stall_count += 1
            if stall_count >= plateau_window:
                print(
                    f"Stalled — no improvement > {plateau_threshold} "
                    f"over {plateau_window} evals (best={best_eval:.1f})"
                )
                break

    else:
        print(f"Reached max_iterations={max_iterations}")

    return eval_means, eval_steps, agent
