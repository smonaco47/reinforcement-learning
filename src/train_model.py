import random

import gymnasium as gym
import numpy as np

from src.agent import AgentFactory, HitGoalCallback
from src.hyperparameters import Hyperparameters
from src.results import Results

N_EVAL_EPISODES = 10


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


def train_model(level, params, iterations=1500, goal_reward=200, verbose=False, seed=0,
                max_episode_steps=750, n_eval_episodes=N_EVAL_EPISODES):
    np.random.seed(seed)
    random.seed(seed)

    environment = make_environment(level, max_episode_steps, seed=seed)

    result = Results()
    callback = HitGoalCallback(result, goal_reward, verbose=verbose)

    agent = AgentFactory.create_dqn_agent(environment, params, seed=seed)

    # Use average episode length of ~200 steps for LunarLander as a reasonable estimate.
    # This makes `iterations` mean approximately that many episodes of training.
    avg_episode_steps = max_episode_steps // 3
    agent.learn(
        total_timesteps=iterations * avg_episode_steps,
        callback=callback,
        progress_bar=verbose,
    )

    # Always evaluate deterministically after training so CSV results
    # reflect true policy performance rather than noisy training rewards.
    eval_env = make_environment(level, max_episode_steps, seed=seed)
    eval_rewards = evaluate_model(agent, eval_env, n_episodes=n_episodes)
    eval_env.close()
    result.add_eval(eval_rewards)

    if verbose:
        print(f"Eval mean: {result.eval_mean:.2f} +/- {result.eval_std:.2f}")
        input("Finished training, press enter to watch 5 episodes...")
        render_env = make_environment(level, max_episode_steps, seed=seed, render=True)
        for _ in range(5):
            obs, _ = render_env.reset()
            episode_reward = 0.0
            done = False
            while not done:
                action, _ = agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = render_env.step(action)
                episode_reward += reward
                done = terminated or truncated
            result.add_result(episode_reward)
            print(f"Render episode reward: {episode_reward}")
        render_env.close()
        result.print_summary()

    environment.close()
    return result, agent
