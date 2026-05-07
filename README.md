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

# longer training per run — recommended for better convergence
python -m src.cli grid-search --iterations 3000 --limit 100

# narrow search with high discount — the most effective configuration found
python -m src.cli grid-search --iterations 3000 --narrow --discount-min 0.98 --discount-max 1.0
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
| `--n-eval-episodes` | `50` | Episodes per eval checkpoint — higher values give more stable signal |
| `--eval-interval` | `100` | Episodes between eval checkpoints |
| `--plateau-patience` | `10` | Evals with no improvement before stopping |
| `--plateau-threshold` | `20.0` | Min improvement to not count as plateau — set higher than eval noise (~30-50 pts) |
| `--success-threshold` | `250.0` | eval_mean above this counts as solved |
| `--success-window` | `5` | Consecutive solved evals before stopping |
| `--max-iterations` | `10000` | Hard cap on training iterations |
| `--seed` | from CSV | Override the seed from the CSV row |

```powershell
# train with defaults
python -m src.cli train-from-csv output/my_best_rows.csv

# more conservative stopping — good for runs that are close but not consistent
python -m src.cli train-from-csv output/my_best_rows.csv --plateau-patience 15 --plateau-threshold 20 --n-eval-episodes 50
```

**Note on plateau threshold:** With 25 eval episodes, `eval_mean` has a natural variance of 30-50 points between checkpoints even when the policy hasn't changed. Set `--plateau-threshold` above this noise floor — `20.0` is a reasonable minimum, and increasing `--n-eval-episodes` to 50-100 gives a more stable signal that makes lower thresholds meaningful.

---

### `run-model` — evaluate a saved model

Loads a saved model and evaluates it deterministically. Models saved by `grid-search` are named `output/model_gs_<seed>_eval<score>.zip`. Models saved by `train-from-csv` are in their own run directory.

The seed is auto-detected from the model filename so you don't need to pass `--seed` for models saved by `grid-search`.

```powershell
python -m src.cli run-model MODEL_PATH
```

| Option | Default | Description |
|--------|---------|-------------|
| `MODEL_PATH` | — | Path to saved model zip, e.g. `output/model_gs_12345_eval210` |
| `--level` | `LunarLander-v3` | Gymnasium environment ID |
| `--max-episode-steps` | `750` | Max steps per episode |
| `--goal-reward` | `200` | Reward threshold to count as goal hit |
| `--seed` | `0` | Fallback seed — auto-detected from filename if present |
| `--n-eval-episodes` | `25` | Episodes for deterministic evaluation |
| `--watch/--no-watch` | `--watch` | Watch rendered episodes after eval |

```powershell
# eval and watch
python -m src.cli run-model output/model_gs_12345_eval210

# eval only, no render, more episodes for a reliable score
python -m src.cli run-model output/model_gs_12345_eval210 --no-watch --n-eval-episodes 100
```

---

## Recommended workflow

1. **Wide search** — run `grid-search --iterations 3000` for 100-200 iterations. Models that hit `eval_mean >= 200` are saved automatically.

2. **Analyze** — run `analyze` to generate `consolidated_output.csv` and scatter plots. Sort by `eval_mean` to find the best configs. Look for hyperparameters that correlate with high scores.

3. **Narrow search** — run `grid-search --narrow` with the ranges from step 2. `--discount-min 0.98` is strongly recommended based on empirical results — high discount is the clearest signal for LunarLander.

4. **Adaptive re-training** — use `train-from-csv` on your best rows to train longer with automatic stopping. Each run saves a model, summary, and progress plot.

5. **Evaluate** — use `run-model` on saved models to watch the agent land and get a reliable eval score over more episodes.

---

## How it works

### Hyperparameter search

`Hyperparameters.randomize()` samples random values for learning rate, exploration schedule, batch size, replay buffer size, and discount factor. Exponential parameters (learning rate, exploration) use log-uniform sampling so each order of magnitude gets equal representation. The grid search trains a fresh agent for each combination and records a clean deterministic evaluation score (`eval_mean`, `eval_std`, `eval_hit_pct`) after training, so the CSV reflects true policy performance rather than training noise.

### Key findings

Through empirical grid search, the most important hyperparameters for LunarLander-v3 turned out to be:

- **`discount >= 0.98`** — the single strongest signal. LunarLander requires long-horizon planning to connect early stabilization to the final landing reward. Low discount prevents the agent from learning this.
- **`lr_initial` in `[5e-4, 1e-2]`** — values below `1e-4` consistently fail to learn.
- **`memory >= 50000`** — smaller buffers get overwritten too quickly for stable learning.
- **`iterations >= 3000`** — 1500 iterations is often not enough to converge.

### Algorithm

The agent uses [DQN](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html) from Stable Baselines3, which enables double-Q learning by default. The learning rate follows either a linear or exponential decay schedule. Exploration uses epsilon-greedy decay from `explore_initial` down to `explore_final`, scaled correctly to the actual training budget so exploration doesn't collapse early.

### Training architecture

`train_model.py` has a single core training loop (`run_training`) shared by all training paths:

- **`run_training`** — the core loop. Calls `agent.learn` and returns a `Results` object. The `reset` flag distinguishes fresh training from continuation.
- **`train_model`** — creates the agent, calls `run_training(reset=True)`, then evaluates. Used by `grid-search` and `train`.
- **`train_adaptive`** — creates an agent and runs `agent.learn` with `AdaptiveStopCallback`, which stops on plateau or consistent success. Used by `train-from-csv`.
- **`eval_and_watch`** — deterministic evaluation and optional rendering, used by `train_model` and `run-model`.
- **`continue_training`** — thin wrapper around `run_training(reset=False)` for continuing from a checkpoint.

### Reproducibility

All sources of randomness are seeded via `seed_everything`: Python's `random`, NumPy, PyTorch (including CUDA), and `PYTHONHASHSEED`. `torch.backends.cudnn.deterministic = True` forces deterministic CUDA operations. The eval phase uses `seed + 1` so eval results are reproducible independently of training length. The seed used for each grid search run is recorded in the CSV and auto-detected from saved model filenames. Note that perfect reproducibility across different hardware or library versions is not guaranteed by PyTorch.

### Goal threshold

LunarLander-v3 is considered solved at a mean reward of **200** over 100 consecutive episodes. The `eval_hit_pct` column records the percentage of deterministic eval episodes that hit this threshold — a more reliable signal than `eval_mean` alone since a high mean can hide inconsistency.

## Type checking

All source files are typed and checked with mypy strict mode:

```powershell
poetry run mypy src/
```

Third-party libraries (SB3, gymnasium, matplotlib, numpy) don't ship complete stubs so `ignore_missing_imports = true` is set in `pyproject.toml`. `warn_return_any = false` is also set to allow `Any` in places where SB3/gymnasium interfaces require it.

## Notes

- The original implementation used [Tensorforce](https://github.com/tensorforce/tensorforce), which is no longer maintained and incompatible with Python 3.11+. It was replaced with Stable Baselines3.
- The original used `gym` (OpenAI). This project uses `gymnasium` (the maintained Farama fork).
- Several bugs in the original port caused poor grid search performance: exploration fraction was hardcoded to 1500 iterations regardless of actual training length, `train_freq` was tied to `batch_size` causing infrequent updates, the replay buffer minimum was too small at 10k, and a `horizon` parameter was being searched over but had no effect on SB3 DQN. Fixing these dramatically improved results.