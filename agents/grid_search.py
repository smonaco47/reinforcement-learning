from datetime import datetime

from tensorforce import Environment

from agents.hyperparameters import Hyperparameters
from agents.tensorforce_agent import train_model
from agents.results import Results

params = Hyperparameters()

iterations = 5000
RANDOM_SEED = 0

with open(f"output/{datetime.now().strftime('%m-%d-%H-%M-%S')}.csv", 'w') as out_file:
    out_file.write(f"{Hyperparameters.csv_header()},{Results.csv_header(iterations)}\n")
    out_file.flush()

    i = 0
    while True:

        try:
            i += 1
            lander_env = Environment.create(
                environment='gym', level='LunarLander-v2', max_episode_timesteps=750
            )

            print(f"{datetime.now().strftime('%H:%M:%S')}\t{i}")

            params.randomize(iterations)
            result = train_model(lander_env, params, max_iterations=iterations, verbose=False, seed=RANDOM_SEED)
            out_file.write(f"{params.csv_params()},{result.csv_result()}\n")
            out_file.flush()

            lander_env.close()

        except Exception as e:
            print("Failed for some reason", e)