# Reinforcement Learning — LunarLander Hyperparameter Search

Trains a DQN agent to solve the [LunarLander-v3](https://gymnasium.farama.org/environments/box2d/lunar_lander/) environment using [Stable Baselines3](https://stable-baselines3.readthedocs.io/), with a random grid search over hyperparameters to find the best configuration.

## Requirements

- Python 3.13
- [Poetry](https://python-poetry.org/)
- `swig` (required to build box2d for LunarLander)

## Setup

### 1. Install swig

```powershell
winget install swigwin
```

Restart your terminal after so `swig.exe` is on your PATH.

### 2. Install Poetry

```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -3.13 -
```

Restart your terminal, then verify:

```powershell
poetry --version
```

### 3. Install dependencies

```powershell
poetry install
```

### 4. Install box2d

We install `gymnasium` without the `box2d` extra and instead install `box2d-py` directly:

```powershell
pip install box2d-py swig
```

This works around a bug in pygame 2.6.1 on Windows + Python 3.13 where `gymnasium[box2d]` triggers an import of `CarRacing` at startup, which pulls in pygame and crashes with `AttributeError: module 'pygame.rect' has no attribute 'FRect'`. Installing box2d separately gives LunarLander access to the physics engine it needs without triggering the broken CarRacing import chain.

### 5. Activate the environment

```powershell
poetry env activate
```

If PowerShell blocks the activation script, run this first:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Alternatively, skip activation entirely and prefix all commands with `poetry run`.

## Project structure

```
├── src/
│   ├── agent.py            # DQN agent factory and training callback
│   ├── hyperparameters.py  # Hyperparameter class with randomization
│   ├── results.py          # Tracks episode rewards and eval scores
│   ├── train_model.py      # Core training and evaluation logic
│   └── cli.py              # All CLI commands
├── output/                 # Created automatically — CSVs and saved models
├── pyproject.toml
└── README.md
```

## Usage

All commands are available through a single CLI entry point:

```powershell
python -m src.cli --help
```

---

### `train` — train a single model

Trains one model with randomized hyperparameters, evaluates it, then prompts to watch 5 rendered episodes:

```powershell
python -m src.cli train
```

| Option | Default | Description |
|--------|---------|-------------|
| `--level` | `LunarLander-v3` | Gymnasium environment ID |
| `--iterations` | `1500` | Training iterations |
| `--goal-reward` | `200` | Reward threshold to count as goal hit |
| `--seed` | `0` | Random seed |
| `--max-episode-steps` | `750` | Max steps per episode |
| `--n-eval-episodes` | `10` | Episodes for deterministic evaluation |

```powershell
# CartPole
python -m src.cli train --level CartPole-v1 --iterations 500 --goal-reward 400 --max-episode-steps 400
```

---

### `grid-search` — run hyperparameter search

Trains many agents with randomized hyperparameters and writes results to timestamped CSVs in `output/`:

```powershell
python -m src.cli grid-search
```

| Option | Default | Description |
|--------|---------|-------------|
| `--level` | `LunarLander-v3` | Gymnasium environment ID |
| `--iterations` | `1500` | Training iterations per run |
| `--limit` | `300` | Number of runs (0 = unlimited) |
| `--max-episode-steps` | `750` | Max steps per episode |
| `--goal-reward` | `200` | Reward threshold to count as goal hit |
| `--n-eval-episodes` | `25` | Episodes for deterministic evaluation |
| `--output-folder` | `output` | Folder to write CSV results |
| `--seed` | random | Base random seed |
| `--narrow/--no-narrow` | `--no-narrow` | Use narrowed hyperparameter ranges |
| `--lr-initial-min` | `5e-4` | `lr_initial` lower bound (narrow mode) |
| `--lr-initial-max` | `1e-2` | `lr_initial` upper bound (narrow mode) |
| `--discount-min` | `0.95` | `discount` lower bound (narrow mode) |
| `--discount-max` | `1.0` | `discount` upper bound (narrow mode) |

```powershell
# unlimited runs
python -m src.cli grid-search --limit 0

# longer training per run
python -m src.cli grid-search --iterations 3000 --limit 100

# second-stage narrow search
python -m src.cli grid-search --narrow --lr-initial-min 5e-4 --lr-initial-max 1e-2 --discount-min 0.95
```

---

### `analyze` — consolidate results and plot

Merges all CSVs in the output folder into `consolidated_output.csv` and generates a scatter plot of each hyperparameter vs eval mean reward:

```powershell
python -m src.cli analyze
```

| Option | Default | Description |
|--------|---------|-------------|
| `--output-folder` | `output` | Folder containing CSV results |
| `--no-plot` | off | Skip plotting, just consolidate |
| `--no-save` | off | Skip saving consolidated CSV, just plot |

```powershell
# just consolidate, no plot
python -m src.cli analyze --no-plot

# just plot from existing consolidated CSV
python -m src.cli analyze --no-save
```

Sort `consolidated_output.csv` by `eval_mean` to find the best hyperparameter configurations.

---

### `run-best` — re-run best hyperparameters

Re-trains the best configurations with longer training and saves the models. Create `output/best_simplified.csv` by copying your best rows from `consolidated_output.csv` and appending `!<iterations>` to each line:

```
0.001,0.0001,...!3000
```

Then run:

```powershell
python -m src.cli run-best
```

| Option | Default | Description |
|--------|---------|-------------|
| `--input-file` | `output/best_simplified.csv` | Input file |
| `--output-folder` | `output` | Folder to save models |
| `--level` | `LunarLander-v3` | Gymnasium environment ID |
| `--max-episode-steps` | `750` | Max steps per episode |
| `--goal-reward` | `200` | Reward threshold to count as goal hit |
| `--seed` | `0` | Random seed |
| `--n-eval-episodes` | `25` | Episodes for deterministic evaluation |

Models are saved to `output/model_<n>_eval<score>.zip`.

---

### Load a saved model

```python
from stable_baselines3 import DQN
agent = DQN.load("output/model_1_eval247")
```

## Second-stage focused search

After running 50-100 initial grid search iterations, run `analyze` to plot hyperparameter correlations and sort `consolidated_output.csv` by `eval_mean` to identify promising ranges. Then switch to narrow mode:

```powershell
python -m src.cli grid-search --narrow --lr-initial-min 5e-4 --lr-initial-max 1e-2 --discount-min 0.95
```

From early results, the clearest signals are:
- `lr_initial` below `1e-4` consistently produces poor results — keep it above `5e-4`
- High `discount` (0.99–1.0) appears in the best runs

The goal of the second stage is to concentrate compute on the promising region of hyperparameter space rather than continuing to sample obviously bad combinations.

## How it works

### Hyperparameter search

`Hyperparameters.randomize()` samples random values for learning rate, exploration schedule, batch size, replay buffer size, discount factor, and horizon using log-uniform sampling for exponential parameters (learning rate, exploration) so each order of magnitude gets equal representation. The grid search trains a fresh agent for each combination and records both the noisy training rewards and a clean deterministic evaluation score (`eval_mean`, `eval_std`) after training. This means the CSV reflects true policy performance rather than training noise.

### Algorithm

The agent uses [DQN](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html) from Stable Baselines3, which enables double-Q learning by default. The learning rate follows either a linear or exponential decay schedule over training. Exploration uses epsilon-greedy decay from `explore_initial` down to `explore_final`.

### Training loop

SB3 manages the training loop internally via `agent.learn(total_timesteps=...)`. Episode rewards and goal hits are tracked via a custom `HitGoalCallback`. Training runs for approximately `iterations * 250` timesteps (using 250 as the average episode length for LunarLander).

### Goal threshold

LunarLander-v3 is considered solved at a mean reward of **200** over 100 consecutive episodes. Individual goal hits during training are logged but the `eval_mean` from deterministic evaluation is the primary metric for comparing hyperparameter sets.

## Notes

- The original implementation used [Tensorforce](https://github.com/tensorforce/tensorforce), which is no longer maintained and incompatible with Python 3.11+. It was replaced with Stable Baselines3.
- The original used `gym` (OpenAI). This project uses `gymnasium` (the maintained Farama fork).