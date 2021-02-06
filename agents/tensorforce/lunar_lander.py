from tensorforce import Agent, Environment
from datetime import datetime

environment = Environment.create(
    environment='gym', level='LunarLander-v2', max_episode_timesteps=100
)


# Instantiate a Tensorforce agent
agent = Agent.create(
    agent='tensorforce',
    environment=environment,
    memory=10000,
    update=dict(unit='timesteps', batch_size=128),
    optimizer=dict(type='adam', learning_rate=dict(
        type='linear', unit='timesteps', num_steps=50000,
        initial_value=1e-3, final_value=1e-5
    )),
    exploration=dict(
        type='linear', unit='timesteps', num_steps=10000,
        initial_value=0.1, final_value=0.01
    ),
    policy=dict(network='auto'),
    objective='policy_gradient',
    reward_estimation=dict(horizon=30)
)

rewards = []
num_iterations = 10000
for i in range(num_iterations):
    states = environment.reset()
    terminal = False

    sum_rewards = 0.0
    while not terminal:
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)
        sum_rewards += reward

    rewards.append(sum_rewards)

    if (i % 500) == 0 and i > 1:
        print(f"{datetime.now().strftime('%H:%M:%S')}\t{i}\t{sum(rewards[-11:-1])/10}")

    if sum_rewards >= 200:
        print(f"Solved in {i} iterations!")
        break

input("Finished training, press here to continue...")

for _ in range(10):

    states = environment.reset()
    terminal = False

    sum_rewards = 0.0
    while not terminal:
        environment.environment.render()
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)
        sum_rewards += reward

    rewards.append(sum_rewards)

print(f"Final\t{sum(rewards[-11:-1])/10}")

agent.close()
environment.close()
