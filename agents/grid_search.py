from datetime import datetime

from tensorforce import Environment

from agents.agent import AgentFactory
from agents.hyperparameters import Hyperparameters
from agents.train_tensorforce_model import train_model
from agents.results import Results

params = Hyperparameters()

iterations = 1500
RANDOM_SEED = 0

with open(f"output/{datetime.now().strftime('%m-%d-%H-%M-%S')}.csv", 'w') as out_file:
    out_file.write(f"{Hyperparameters.csv_header()},{Results.csv_header(iterations)}\n")
    out_file.flush()

    i = 0
    while True:

        try:
            i += 1
            environment = Environment.create(
                environment='gym', level='LunarLander-v2', max_episode_timesteps=750
            )

            print(f"{datetime.now().strftime('%H:%M:%S')}\t{i}")

            params.randomize(iterations)

            agent = AgentFactory.create_ddqn_agent(environment, params, RANDOM_SEED)
            result = train_model(environment, agent, iterations=iterations, verbose=False, seed=RANDOM_SEED,
                                 goal_reward=200)
            out_file.write(f"{params.csv_params()},{result.csv_result()}\n")
            out_file.flush()

            environment.close()

        except Exception as e:
            print("Failed for some reason", e)
