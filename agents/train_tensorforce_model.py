import random

import numpy as np
from tensorforce import Environment

from agents.agent import AgentFactory
from agents.hyperparameters import Hyperparameters
from agents.results import Results


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


def train_model(environment, params: Hyperparameters, max_iterations=10000, goal_reward=200, verbose=False, seed=0):
    environment.environment.seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    result = Results()

    agent = AgentFactory.create_tensorforce_agent(environment, params, seed)

    for i in range(max_iterations):

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

    lander_env = Environment.create(
        environment='gym', level='LunarLander-v2', max_episode_timesteps=750
    )

    params = Hyperparameters()
    # params = Hyperparameters.from_csv("0.00010895333588082155,4.0102600954974266e-07,0.998792570730397,4639,exponential,0.07806314924326674,0.007102957506168539,0.9972166634078218,860,exponential,95,0.0134")

    train_model(lander_env, params, verbose=True, max_iterations=5, seed=RANDOM_SEED)

    lander_env.close()
