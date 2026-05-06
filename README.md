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

### 6. Install dev dependencies (type checking)

```powershell
poetry install --with dev
```

Run mypy:

```powershell
poetry run mypy src/
```

## Project structure

```
├── src/
│   ├── agent.py            # DQN agent factory and training callback
│   ├── hyperparameters.py  # Hyperparameter class with randomization
│   ├── results.py          # Tracks episode rewards and eval scores
│   ├── train_model.py      # Core training and evaluation logic
│   └── cli.py              # All CLI commands
├── output/                 # Created automatically — CSVs, saved models, run dirs
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

Trains many agents with randomized hyperparameters and writes results to timestamped CSVs in `output/`. Any run where `eval_mean >= goal_reward` has its model saved automatically to `output/model_gs_<seed>_eval<score>.zip`.

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

```

---

### `train-from-csv` — adaptive training from CSV rows

Re-trains rows from `consolidated_output.csv` with automatic early stopping — no need to guess how many iterations to run. Training stops when:

- **Plateau** — `eval_mean` hasn't improved by `--plateau-threshold` in `--plateau-patience` consecutive evals
- **Solved** — `eval_mean` has stayed above `--success-threshold` for `--success-window` consecutive evals
- **Hard cap** — `--max-iterations` is reached regardless

Copy any rows you want to re-train from `consolidated_output.csv` into a new file (keep the header row), then run:

```powershell
python -m src.cli train-from-csv output/my_best_rows.csv
```

Each run gets its own output directory `output/train_from_csv_<seed>_<timestamp>/` containing:
- `model_eval<score>.zip` — the trained model
- `results.txt` — eval summary and hyperparameters
- `eval_progress.png` — plot of eval_mean over training timesteps (saved to file, not opened)

| Option | Default | Description |
|--------|---------|-------------|
| `INPUT_FILE` | — | CSV file with rows from `consolidated_output.csv` (header required) |
| `--level` | `LunarLander-v3` | Gymnasium environment ID |
| `--output-folder` | `output` | Folder for models and plots |
| `--max-episode-steps` | `750` | Max steps per episode |
| `--goal-reward` | `200` | Reward threshold |
| `--n-eval-episodes` | `25` | Episodes per eval checkpoint |
| `--eval-interval` | `100` | Episodes between eval checkpoints |
| `--plateau-patience` | `10` | Evals with no improvement before stopping |
| `--plateau-threshold` | `5.0` | Min improvement to not count as plateau |
| `--success-threshold` | `250.0` | eval_mean above this counts as solved |
| `--success-window` | `5` | Consecutive solved evals before stopping |
| `--max-iterations` | `10000` | Hard cap on training iterations |
| `--seed` | from CSV | Override the seed from the CSV row |

```powershell
# train with defaults
python -m src.cli train-from-csv output/my_best_rows.csv

# longer patience, lower success bar
python -m src.cli train-from-csv output/my_best_rows.csv --plateau-patience 20 --success-threshold 220
```

---

### `run-model` — evaluate or continue training a saved model

Loads a saved model and evaluates it deterministically, optionally continuing training first. Models are saved automatically by `grid-search` when `eval_mean >= goal_reward`, and can be found in `output/model_gs_<seed>_eval<score>.zip`.

```powershell
python -m src.cli run-model MODEL_PATH
```

| Option | Default | Description |
|--------|---------|-------------|
| `MODEL_PATH` | — | Path to saved model zip, e.g. `output/model_gs_12345_eval210` |
| `--level` | `LunarLander-v3` | Gymnasium environment ID |
| `--iterations` | `0` | Additional training iterations (0 = eval only) |
| `--max-episode-steps` | `750` | Max steps per episode |
| `--goal-reward` | `200` | Reward threshold to count as goal hit |
| `--seed` | `0` | Fallback seed — auto-detected from filename if model is named `model_gs_<seed>_eval<score>` |
| `--n-eval-episodes` | `25` | Episodes for deterministic evaluation |
| `--watch/--no-watch` | `--watch` | Watch rendered episodes after eval |
| `--csv-row` | none | Full CSV row from `consolidated_output.csv` to reconstruct original hyperparameter schedules |
| `--output-folder` | `output` | Used to locate `consolidated_output.csv` header when `--csv-row` is set |

```powershell
# eval and watch (seed auto-detected from filename)
python -m src.cli run-model output/model_gs_12345_eval210

# continue training with baked-in hyperparameters then eval
python -m src.cli run-model output/model_gs_12345_eval210 --iterations 1500

# retrain from scratch with full schedule reconstruction
python -m src.cli run-model output/model_gs_12345_eval210 --iterations 3000 --csv-row "12345,0.00731,..."

# eval only, no render, more episodes for a reliable score
python -m src.cli run-model output/model_gs_12345_eval210 --no-watch --n-eval-episodes 100
```

**Two modes of continued training:**

Without `--csv-row`, training continues from the saved checkpoint using the hyperparameters baked into the model (discount, batch size, current LR). This is best when the model is already performing well and you just want more training time.

With `--csv-row`, the original LR and exploration schedules are reconstructed from the CSV row and training starts fresh with those hyperparameters. This is better when you want to retrain from scratch with a known-good configuration for more iterations than the grid search used. Copy the full row directly from `consolidated_output.csv`.

---

## Recommended workflow

1. **Wide search** — run `grid-search` for 200–300 iterations with default settings. Models that solve the environment are saved automatically.

2. **Analyze** — run `analyze` to plot correlations. Look at the scatter plots to identify which hyperparameters correlate with high `eval_mean`.

3. **Narrow search** — run `grid-search --narrow` with the promising ranges from step 2. Another 100–200 iterations is usually enough.

4. **Evaluate saved models** — use `run-model` on any `.zip` model saved by `grid-search` to watch it land and get a reliable eval score over more episodes.

5. **Continue training** — if a model is close but not consistent, use `run-model --iterations 1500` to continue from the checkpoint with baked-in hyperparameters, or pass `--csv-row` to retrain from scratch with the full original schedule.

## Second-stage focused search

After running 50-100 initial grid search iterations, run `analyze` to plot hyperparameter correlations and identify promising ranges. From early results, the clearest signals are:

- `lr_initial` below `1e-4` consistently produces poor results — keep it above `5e-4`
- High `discount` (0.99–1.0) appears in the best runs

```powershell
python -m src.cli grid-search --narrow --lr-initial-min 5e-4 --lr-initial-max 1e-2 --discount-min 0.95
```

## How it works

### Hyperparameter search

`Hyperparameters.randomize()` samples random values for learning rate, exploration schedule, batch size, replay buffer size, discount factor, and horizon using log-uniform sampling for exponential parameters (learning rate, exploration) so each order of magnitude gets equal representation. The grid search trains a fresh agent for each combination and records both the noisy training rewards and a clean deterministic evaluation score (`eval_mean`, `eval_std`, `eval_hit_pct`) after training. This means the CSV reflects true policy performance rather than training noise.

### Algorithm

The agent uses [DQN](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html) from Stable Baselines3, which enables double-Q learning by default. The learning rate follows either a linear or exponential decay schedule over training. Exploration uses epsilon-greedy decay from `explore_initial` down to `explore_final`.

### Training architecture

`train_model.py` has a single core training loop (`run_training`) shared by both fresh training and continued training from a checkpoint:

- **`run_training`** — the core loop. Calls `agent.learn` and returns a `Results` object. The `reset` flag distinguishes fresh training from continuation.
- **`continue_training`** — thin wrapper around `run_training(reset=False)`, used by `run-model`.
- **`train_model`** — creates the agent, calls `run_training(reset=True)`, then evaluates.
- **`eval_and_watch`** — deterministic evaluation and optional rendering, used by both `train_model` and `run-model`.

### Reproducibility

All sources of randomness are seeded via `seed_everything`: Python's `random`, NumPy, PyTorch (including CUDA), and `PYTHONHASHSEED`. `torch.backends.cudnn.deterministic = True` forces deterministic CUDA operations. The eval phase uses `seed + 1` so eval results are reproducible independently of training length. Note that perfect reproducibility across different hardware or library versions is not guaranteed by PyTorch.

### Goal threshold

LunarLander-v3 is considered solved at a mean reward of **200** over 100 consecutive episodes. The `eval_hit_pct` column records the percentage of deterministic eval episodes that hit this threshold — a more reliable signal than `eval_mean` alone since a high mean can hide inconsistency.

## Type checking

All source files are typed and checked with mypy strict mode. To run:

```powershell
poetry run mypy src/
```

Third-party libraries (SB3, gymnasium, matplotlib, numpy) don't ship complete stubs so `ignore_missing_imports = true` is set in `pyproject.toml`. `warn_return_any = false` is also set to allow `Any` in places where SB3/gymnasium interfaces require it.

## Notes

- The original implementation used [Tensorforce](https://github.com/tensorforce/tensorforce), which is no longer maintained and incompatible with Python 3.11+. It was replaced with Stable Baselines3.
- The original used `gym` (OpenAI). This project uses `gymnasium` (the maintained Farama fork).
