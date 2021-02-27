import random

import numpy as np
from tensorforce import Agent, Environment

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

    agent = create_agent(environment, params, seed)

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


def create_agent(environment, params, seed):
    lr_dict = dict(
        type=params.lr_type, unit='episodes', num_steps=params.lr_steps,
        initial_value=params.lr_initial
    )
    if params.lr_type == 'linear':
        lr_dict["final_value"] = params.lr_final
    else:
        lr_dict["decay_rate"] = params.lr_decay

    explore_dict = dict(
        type=params.explore_type, unit='episodes', num_steps=params.explore_steps,
        initial_value=params.explore_initial, final_value=params.explore_final
    )
    if params.explore_type == 'linear':
        explore_dict["final_value"] = params.explore_final
    else:
        explore_dict["decay_rate"] = params.explore_decay

    agent = Agent.create(
        agent='tensorforce',
        environment=environment,
        memory=params.memory,
        update=dict(unit='timesteps', batch_size=params.batch_size),
        optimizer=dict(type='adam', learning_rate=lr_dict),
        exploration=explore_dict,
        policy=dict(network='auto'),
        objective='policy_gradient',
        reward_estimation=dict(horizon=params.horizon, discount=params.discount),
        config={'seed': seed}
    )
    return agent


if __name__ == "__main__":
    RANDOM_SEED = 0

    lander_env = Environment.create(
        environment='gym', level='LunarLander-v2', max_episode_timesteps=750
    )

    params = Hyperparameters()
    # params = Hyperparameters.from_csv("0.00010895333588082155,4.0102600954974266e-07,0.998792570730397,4639,exponential,0.07806314924326674,0.007102957506168539,0.9972166634078218,860,exponential,95,0.0134")

    train_model(lander_env, params, verbose=True, max_iterations=2205, seed=RANDOM_SEED)

    lander_env.close()
