from datetime import datetime

from tensorforce import Environment

from agents.agent import AgentFactory
from agents.hyperparameters import Hyperparameters
from agents.train_tensorforce_model import train_model

RANDOM_SEED = 0
i = 0
for line in open("output/ddqn v2/results/best_simplified.csv",'r'):
    params, iterations = line.split('!')
    iterations = int(iterations)
    print(params, iterations)

    try:
        i += 1
        environment = Environment.create(
            environment='gym', level='LunarLander-v2', max_episode_timesteps=750
        )

        print(f"{datetime.now().strftime('%H:%M:%S')}\t{i}")

        params = Hyperparameters.from_csv(params)

        agent = AgentFactory.create_ddqn_agent(environment, params, RANDOM_SEED)
        result = train_model(environment, agent, iterations=iterations, verbose=True, seed=RANDOM_SEED,
                             goal_reward=200)

        environment.close()

    except Exception as e:
        print("Failed for some reason", e)
