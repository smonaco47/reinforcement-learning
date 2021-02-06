from tensorforce import Agent, Environment

from agents.hyperparameters import Hyperparameters
from agents.results import Results


def run_episode(agent, environment, render=False):
    states = environment.reset()
    terminal = False

    sum_rewards = 0.0
    while not terminal:
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)
        sum_rewards += reward

        if (render):
            environment.environment.render()

    return sum_rewards


def train_model(environment, params: Hyperparameters, max_iterations=10000, goal_reward=200, show_final=False):
    result = Results()

    agent = create_agent(environment, params)

    for i in range(max_iterations):

        if (i % 500) == 0 and i > 1:
            result.print_summary()

        episode_reward = run_episode(agent, environment)
        result.add_result(episode_reward)

        if episode_reward >= goal_reward:
            print(f"Solved in {i} iterations!")

    if show_final:
        input("Finished training, press here to continue...")
        for _ in range(5):
            episode_reward = run_episode(agent, environment, render=True)
            result.add_result(episode_reward)
        result.print_summary()

    agent.close()
    return result


def create_agent(environment, params):
    lr_dict = dict(
        type=params.lr_type, unit='timesteps', num_steps=params.lr_steps,
        initial_value=params.lr_initial
    )
    if params.lr_type == 'linear':
        lr_dict["final_value"] = params.lr_final
    else:
        lr_dict["decay_rate"] = params.lr_decay
    explore_dict = dict(
        type=params.explore_type, unit='timesteps', num_steps=params.explore_steps,
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
        reward_estimation=dict(horizon=params.horizon)
    )
    return agent


if __name__ == "__main__":
    lander_env = Environment.create(
        environment='gym', level='LunarLander-v2', max_episode_timesteps=100
    )
    params = Hyperparameters()
    train_model(lander_env, params, show_final=True)
    lander_env.close()
