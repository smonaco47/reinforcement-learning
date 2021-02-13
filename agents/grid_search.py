from datetime import datetime

from tensorforce import Environment

from agents.hyperparameters import Hyperparameters
from agents.tensorforce_agent import train_model
from agents.results import Results

params = Hyperparameters()

iterations = 2500
RANDOM_SEED = 0

with open(f"output-{datetime.now().strftime('%H-%M-%S')}.csv", 'w') as out_file:
    out_file.write(f"{Hyperparameters.csv_header()},{Results.csv_header(iterations)}\n")
    out_file.flush()

    for i in range(100):

        lander_env = Environment.create(
            environment='gym', level='LunarLander-v2', max_episode_timesteps=750
        )

        print(f"{datetime.now().strftime('%H:%M:%S')}\t{i}")

        params.randomize()
        result = train_model(lander_env, params, max_iterations=iterations, show_final=False, seed=RANDOM_SEED)
        out_file.write(f"{params.csv_params()},{result.csv_result()}\n")
        out_file.flush()

        lander_env.close()