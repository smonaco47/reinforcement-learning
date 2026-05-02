import os
import random

import gymnasium as gym
import numpy as np
import torch

from src.agent import AgentFactory, HitGoalCallback
from src.hyperparameters import Hyperparameters
from src.results import Results

N_EVAL_EPISODES = 10


def seed_everything(seed):
    """Seed all sources of randomness for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make pytorch deterministic — slight performance cost but required for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def make_environment(level, max_episode_steps, seed=0, render=False):
    render_mode = "human" if render else None
    env = gym.make(level, max_episode_steps=max_episode_steps, render_mode=render_mode)
    env.reset(seed=seed)
    return env


def evaluate_model(agent, environment, n_eval_episodes=N_EVAL_EPISODES):
    """Run deterministic episodes to get a clean measure of policy performance."""
    rewards = []
    for _ in range(n_eval_episodes):
        obs, _ = environment.reset()
        done = False
        episode_reward = 0.0
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = environment.step(action)
            episode_reward += reward
            done = terminated or truncated
        rewards.append(episode_reward)
    return rewards


def run_training(agent, level, iterations, goal_reward=200, seed=0,
                 max_episode_steps=750, verbose=False, reset=True):
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




def eval_and_watch(agent, level, goal_reward=200, seed=0, max_episode_steps=750,
                   n_eval_episodes=N_EVAL_EPISODES, watch=True):
    """Evaluate a trained agent deterministically and optionally render episodes."""
    seed_everything(seed + 1)
    eval_env = make_environment(level, max_episode_steps, seed=seed)
    eval_rewards = evaluate_model(agent, eval_env, n_eval_episodes=n_eval_episodes)
    eval_env.close()

    eval_mean = sum(eval_rewards) / len(eval_rewards)
    eval_std = (sum((r - eval_mean) ** 2 for r in eval_rewards) / len(eval_rewards)) ** 0.5
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
                episode_reward += reward
                done = terminated or truncated
            print(f"Episode {ep + 1}: {episode_reward:.1f}")
        render_env.close()

    return eval_mean, eval_std, hit_pct


def train_model(level, params, iterations=1500, goal_reward=200, verbose=False, seed=0,
                max_episode_steps=750, n_eval_episodes=N_EVAL_EPISODES):
    seed_everything(seed)

    environment = make_environment(level, max_episode_steps, seed=seed)
    agent = AgentFactory.create_dqn_agent(environment, params, seed=seed)
    environment.close()

    result = run_training(agent, level, iterations, goal_reward=goal_reward,
                          seed=seed, max_episode_steps=max_episode_steps,
                          verbose=verbose, reset=True)

    eval_env = make_environment(level, max_episode_steps, seed=seed)
    seed_everything(seed + 1)
    eval_rewards = evaluate_model(agent, eval_env, n_eval_episodes=n_eval_episodes)
    eval_env.close()
    result.add_eval(eval_rewards, goal_reward)

    eval_and_watch(agent, level, goal_reward=goal_reward, seed=seed,
                   max_episode_steps=max_episode_steps, n_eval_episodes=0,
                   watch=verbose)

    return result, agent

    
def continue_training(agent, level, iterations, goal_reward=200, seed=0,
                      max_episode_steps=750):
    """Continue training an already-loaded agent for additional iterations."""
    return run_training(agent, level, iterations, goal_reward=goal_reward,
                        seed=seed, max_episode_steps=max_episode_steps, reset=False)