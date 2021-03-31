import random

import numpy as np
from tensorforce import Environment

from src.agent import AgentFactory
from src.hyperparameters import Hyperparameters
from src.results import Results


def run_episode(agent, environment, render=False):
    states = environment.reset()
    terminal = False

    sum_rewards = 0.0
    steps = 0

    while not terminal:
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)
        sum_rewards += reward
        steps += 1

        if (render):
            environment.environment.render()

    return sum_rewards


def train_model(environment, agent, iterations=10000, goal_reward=200, verbose=False, seed=0):
    environment.environment.seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    result = Results()

    for i in range(iterations):

        if verbose and (i % 200) == 0 and i > 1:
            result.print_summary()

        episode_reward = run_episode(agent, environment)

        hit_goal = episode_reward >= goal_reward
        result.add_result(episode_reward, hit_goal)

        if hit_goal:
            print(f"Hit goal in {i} iterations with a reward of {episode_reward}!")

    if verbose:
        input("Finished training, press here to continue...")
        for _ in range(5):
            episode_reward = run_episode(agent, environment, render=True)
            result.add_result(episode_reward)
        result.print_summary()

    agent.close()
    return result


if __name__ == "__main__":
    RANDOM_SEED = 0

    # environment = Environment.create(
    #     environment='gym', level='LunarLander-v2', max_episode_timesteps=750
    # )
    environment = Environment.create(
        environment='gym', level='CartPole', max_episode_timesteps=400
    )

    params = Hyperparameters()
    params.randomize()

    agent = AgentFactory.create_tensorforce_agent(environment, params, RANDOM_SEED)

    train_model(environment, agent, verbose=True, iterations=500, seed=RANDOM_SEED, goal_reward=400)

    environment.close()
