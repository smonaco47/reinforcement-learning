from datetime import datetime

from tensorforce import Environment

from agents.hyperparameters import Hyperparameters
from agents.tensorforce.lunar_lander import train_model
from agents.results import Results

params = Hyperparameters()

lander_env = Environment.create(
    environment='gym', level='LunarLander-v2', max_episode_timesteps=300
)

iterations = 10000

with open(f"output-{datetime.now().strftime('%H-%M-%S')}.txt", 'w') as out_file:
    out_file.write(f"{Hyperparameters.csv_header()},{Results.csv_header(iterations)}\n")
    for i in range(100):
        print(f"{datetime.now().strftime('%H:%M:%S')}\t{i}")
        params.randomize()
        result = train_model(lander_env, params, max_iterations=iterations, show_final=False)
        out_file.write(f"{params.csv_params()},{result.csv_result()}\n")
    lander_env.close()
