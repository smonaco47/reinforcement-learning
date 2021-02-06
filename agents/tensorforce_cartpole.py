from tensorforce import Agent, Environment

# Pre-defined or custom environment
environment = Environment.create(
    environment='gym', level='CartPole', max_episode_timesteps=400
)

# Instantiate a Tensorforce agent
agent = Agent.create(
    agent='tensorforce',
    environment=environment,  # alternatively: states, actions, (max_episode_timesteps)
    memory=10000,
    update=dict(unit='timesteps', batch_size=64),
    optimizer=dict(type='adam', learning_rate=1e-3),
    policy=dict(network='auto'),
    objective='policy_gradient',
    reward_estimation=dict(horizon=20)
)

rewards = []
num_iterations = 3000
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

    if (i % 200) == 0 and i > 1:
        print(f"{i}\t{sum(rewards[-11:-1])/10}")

    if sum_rewards == 400:
        print(f"Solved in {i} iterations!")
        break

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
