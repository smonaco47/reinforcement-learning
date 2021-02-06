import gym
env = gym.make('CartPole-v0')
env.reset()
print("Action space", env.action_space)
print("Observation space", env.observation_space)
print(f"Min: {env.observation_space.low}\t Max: {env.observation_space.high}")
for _ in range(1000):
    env.render()
    obs, reward, done, info = env.step(env.action_space.sample())
    print(obs)
    print(reward)
    print(done)
    if done:
        print("Episode complete")
        break
env.close()