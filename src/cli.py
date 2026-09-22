import csv
import os
import random
import re
from datetime import datetime
from typing import Any

import click
import matplotlib

matplotlib.use("Agg")  # non-interactive backend — no windows opened
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN

from src.hyperparameters import Hyperparameters
from src.results import Results
from src.train_model import (
    eval_and_watch,
    make_environment,
    seed_everything,
    train_adaptive,
)

# --- Constants ---

SKIP_FILES = {"consolidated_output.csv"}

KEEP_COLUMNS = [
    "seed",
    "lr_initial",
    "lr_final",
    "lr_decay",
    "lr_steps",
    "lr_type",
    "explore_initial",
    "explore_final",
    "explore_decay",
    "explore_steps",
    "explore_type",
    "discount",
    "batch_size",
    "memory",
    "max",
    "max_group",
    "hit_goal",
    "eval_mean",
    "eval_std",
    "eval_hit_pct",
]

PLOT_KEYS = [
    "discount",
    "lr_initial",
    "lr_final",
    "lr_decay",
    "lr_steps",
    "explore_initial",
    "explore_final",
    "explore_decay",
    "explore_steps",
    "batch_size",
    "memory",
    "eval_hit_pct",
]

PARAM_COLS = [
    "lr_initial",
    "lr_final",
    "lr_decay",
    "lr_steps",
    "lr_type",
    "explore_initial",
    "explore_final",
    "explore_decay",
    "explore_steps",
    "explore_type",
    "discount",
    "batch_size",
    "memory",
]


# --- CLI group ---


@click.group()
def cli() -> None:
    """Reinforcement Learning hyperparameter search for LunarLander / CartPole."""
    pass


# --- train ---


@cli.command()
@click.option(
    "--level",
    default="LunarLander-v3",
    show_default=True,
    help="Gymnasium environment ID",
)
@click.option(
    "--training_iterations", default=1500, show_default=True, help="Training iterations"
)
@click.option(
    "--goal-reward",
    default=250.0,
    show_default=True,
    help="Reward threshold to count as goal hit",
)
@click.option("--seed", default=0, show_default=True, help="Random seed")
@click.option(
    "--max-episode-steps", default=750, show_default=True, help="Max steps per episode"
)
@click.option(
    "--n-eval-episodes",
    default=10,
    show_default=True,
    help="Episodes for deterministic evaluation",
)
@click.option(
    "--verbose",
    is_flag=True,
    type=bool,
    help="show detailed debugging information",
)
def train(
    level: str,
    training_iterations: int,
    goal_reward: float,
    seed: int,
    max_episode_steps: int,
    n_eval_episodes: int,
    verbose: bool,
) -> None:
    """Train a single DQN agent with randomized hyperparameters."""
    params = Hyperparameters()
    params.randomize()
    train_adaptive(
        level=level,
        params=params,
        max_iterations=training_iterations,
        success_threshold=goal_reward,
        verbose=verbose,
        seed=seed,
        max_episode_steps=max_episode_steps,
        n_eval_episodes=n_eval_episodes,
    )


# --- grid-search ---


@cli.command()
@click.option(
    "--level",
    default="LunarLander-v3",
    show_default=True,
    help="Gymnasium environment ID",
)
@click.option(
    "--search-iterations",
    default=300,
    show_default=True,
    help="Number of combinations to try (0 = unlimited)",
)
@click.option(
    "--training-iterations",
    default=1500,
    show_default=True,
    help="Training iterations per run",
)
@click.option(
    "--plateau-patience",
    default=10,
    show_default=True,
    help="Evals with no improvement before stopping",
)
@click.option(
    "--plateau-threshold",
    default=10,
    show_default=True,
    help="Min eval_mean improvement to not count as plateau",
)
@click.option(
    "--success-window",
    default=5,
    show_default=True,
    help="Consecutive evals above success threshold before stopping",
)
@click.option(
    "--max-episode-steps", default=750, show_default=True, help="Max steps per episode"
)
@click.option(
    "--goal-reward",
    default=250.0,
    show_default=True,
    help="Reward threshold to count as goal hit",
)
@click.option(
    "--eval-interval",
    default=100,
    show_default=True,
    help="Episodes between eval periods",
)
@click.option(
    "--n-eval-episodes",
    default=25,
    show_default=True,
    help="Episodes for deterministic evaluation",
)
@click.option(
    "--output-folder",
    default="output",
    show_default=True,
    help="Folder to write CSV results",
)
@click.option(
    "--seed", default=None, type=int, help="Base random seed (default: random)"
)
@click.option(
    "--lr-initial-min",
    default=5e-4,
    show_default=True,
    help="lr_initial lower bound",
)
@click.option(
    "--lr-initial-max",
    default=1e-2,
    show_default=True,
    help="lr_initial upper bound",
)
@click.option(
    "--discount-min",
    default=0.95,
    show_default=True,
    help="discount lower bound",
)
@click.option(
    "--discount-max",
    default=1.0,
    show_default=True,
    help="discount upper bound",
)
@click.option(
    "--verbose",
    is_flag=True,
    type=bool,
    help="show detailed debugging information",
)
def grid_search(
    level: str,
    search_iterations: int,
    training_iterations: int,
    plateau_patience: int,
    plateau_threshold: int,
    success_window: int,
    max_episode_steps: int,
    goal_reward: float,
    eval_interval: int,
    n_eval_episodes: int,
    output_folder: str,
    seed: int | None,
    lr_initial_min: float,
    lr_initial_max: float,
    discount_min: float,
    discount_max: float,
    verbose: bool,
) -> None:
    """Run a random hyperparameter grid search."""
    base_seed = seed if seed is not None else random.randint(0, 1_000_000)
    os.makedirs(output_folder, exist_ok=True)

    click.echo(
        f"Starting grid search — level={level}, training_iterations={training_iterations}, "
        f"search_iterations={'unlimited' if search_iterations == 0 else search_iterations}, base_seed={base_seed}"
    )

    params = Hyperparameters()
    i = 0

    with open(
        f"{output_folder}/{datetime.now().strftime('%m-%d-%H-%M-%S')}.csv", "w"
    ) as out_file:
        out_file.write(
            f"seed,{Hyperparameters.csv_header()},{Results.csv_header(training_iterations // eval_interval)}\n"
        )
        out_file.flush()

        while search_iterations == 0 or i < search_iterations:
            try:
                i += 1
                run_seed = base_seed + i
                click.echo(
                    f"{datetime.now().strftime('%H:%M:%S')}\t{i}\tSEED: {run_seed}"
                )

                params.randomize(
                    steps=training_iterations,
                    lr_initial_range=(lr_initial_min, lr_initial_max),
                    discount_range=(discount_min, discount_max),
                )

                result, agent = train_adaptive(
                    level=level,
                    params=params,
                    seed=run_seed,
                    max_episode_steps=max_episode_steps,
                    n_eval_episodes=n_eval_episodes,
                    eval_interval_episodes=eval_interval,
                    max_iterations=training_iterations,
                    verbose=verbose,
                    plateau_patience=plateau_patience,
                    plateau_threshold=plateau_threshold,
                    success_window=success_window,
                    success_threshold=goal_reward,
                )

                out_file.write(
                    f"{run_seed},{params.csv_params()},{result.csv_result()}\n"
                )
                out_file.flush()

                last_eval = result.eval_history[-1]
                if last_eval.mean >= goal_reward:
                    model_name = (
                        f"{output_folder}/model_gs_{run_seed}_eval{last_eval.mean:.0f}"
                    )
                    agent.save(model_name)
                    click.echo(
                        f"  [SAVED] {model_name}.zip (eval_mean={last_eval.mean:.1f})"
                    )

                    if result.eval_history:
                        steps = [h.num_timestamps for h in result.eval_history]
                        means = [h.mean for h in result.eval_history]
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.plot(steps, means, marker="o", markersize=3)
                        ax.axhline(
                            y=goal_reward,
                            color="green",
                            linestyle="--",
                            label=f"Goal ({goal_reward})",
                        )
                        ax.axhline(
                            y=goal_reward,
                            color="gold",
                            linestyle="--",
                            label=f"Success ({goal_reward})",
                        )
                        ax.set_xlabel("Timesteps")
                        ax.set_ylabel("Eval Mean Reward")
                        ax.set_title(
                            f"Training Progress — seed={run_seed} final={last_eval.mean:.0f}"
                        )
                        ax.legend()
                        ax.grid(True)
                        plt.tight_layout()
                        plot_path = f"{output_folder}/eval_model_gs_{run_seed}_eval{last_eval.mean:.0f}_progress.png"
                        plt.savefig(plot_path)
                        plt.close()
                        click.echo(f"  Saved plot: {plot_path}")
                del agent

            except Exception as e:
                click.echo(f"Failed on run {i}: {e}", err=True)


# --- analyze ---


@cli.command()
@click.option(
    "--output-folder",
    default="output",
    show_default=True,
    help="Folder containing CSV results",
)
@click.option(
    "--no-plot", is_flag=True, default=False, help="Skip plotting, just consolidate"
)
@click.option(
    "--no-save",
    is_flag=True,
    default=False,
    help="Skip saving consolidated CSV, just plot",
)
def analyze(output_folder: str, no_plot: bool, no_save: bool) -> None:
    """Consolidate grid search CSVs and plot hyperparameter correlations."""
    csv_files = [
        entry.path
        for entry in os.scandir(output_folder)
        if entry.path.endswith(".csv")
        and os.path.basename(entry.path) not in SKIP_FILES
    ]

    click.echo(f"Found {len(csv_files)} CSV file(s):")
    results: list[list[str]] = []
    for path in csv_files:
        skipped = added = 0
        with open(path, "r") as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            header = next(reader)
            missing = [col for col in KEEP_COLUMNS if col not in header]
            if missing:
                click.echo(f"  [WARN] {path} missing columns {missing} — skipping")
                continue
            indices = [header.index(col) for col in KEEP_COLUMNS]
            for i, row in enumerate(reader, start=2):
                if len(row) < max(indices) + 1:
                    skipped += 1
                    continue
                results.append([row[idx] for idx in indices])
                added += 1
        click.echo(f"  [OK] {path}: {added} added, {skipped} skipped")

    click.echo(f"\nTotal rows: {len(results)}")

    if not results:
        click.echo("[ERROR] No results collected", err=True)
        raise SystemExit(1)

    data = np.array(results)

    if not no_save:
        out_path = f"{output_folder}/consolidated_output.csv"
        np.savetxt(out_path, np.vstack([KEEP_COLUMNS, data]), "%s", delimiter=",")
        click.echo(f"Saved {out_path} ({len(results)} rows)")

    if not no_plot:

        def col_data(col: str) -> Any:
            return data[:, KEEP_COLUMNS.index(col)].astype(np.float64)

        eval_mean = col_data("eval_mean")
        fig, axs = plt.subplots(
            len(PLOT_KEYS), figsize=(8, 2 * len(PLOT_KEYS)), sharex="all"
        )
        fig.suptitle("Hyperparameter Results")
        for idx, key in enumerate(PLOT_KEYS):
            d = col_data(key)
            axs[idx].scatter(eval_mean, d, alpha=0.6)
            axs[idx].set(ylabel=key)
            if max(d) < 1:
                axs[idx].set(yscale="log")
            axs[idx].grid(True)
        axs[-1].set(xlabel="Eval Mean Reward")
        plt.tight_layout()
        fig_path = f"{output_folder}/fig.png"
        plt.savefig(fig_path)
        plt.close()
        click.echo(f"Saved {fig_path}")


# --- run-model ---


@cli.command()
@click.argument("model-path")
@click.option(
    "--level",
    default="LunarLander-v3",
    show_default=True,
    help="Gymnasium environment ID",
)
@click.option(
    "--max-episode-steps", default=750, show_default=True, help="Max steps per episode"
)
@click.option(
    "--goal-reward",
    default=200.0,
    show_default=True,
    help="Reward threshold to count as goal hit",
)
@click.option(
    "--n-eval-episodes",
    default=25,
    show_default=True,
    help="Episodes for deterministic evaluation",
)
@click.option(
    "--seed",
    default=0,
    show_default=True,
    help="Random seed (auto-detected from filename if present)",
)
@click.option(
    "--watch/--no-watch",
    default=True,
    show_default=True,
    help="Watch rendered episodes after eval",
)
def run_model(
    model_path: str,
    level: str,
    max_episode_steps: int,
    goal_reward: float,
    n_eval_episodes: int,
    seed: int,
    watch: bool,
) -> None:
    """Load a saved model and evaluate it.

    MODEL_PATH is the path to the saved model zip, e.g. output/model_gs_12345_eval210

    Examples:

        python -m src.cli run-model output/model_gs_12345_eval210
        python -m src.cli run-model output/model_gs_12345_eval210 --no-watch --n-eval-episodes 100
    """
    model_seed = seed
    match = re.search(r"model_gs_(\d+)_eval", model_path)
    if match:
        model_seed = int(match.group(1))
        click.echo(f"Using seed {model_seed} from model filename")

    seed_everything(model_seed)

    click.echo(f"Loading model from {model_path}...")
    env = make_environment(level, max_episode_steps, seed=model_seed)
    agent = DQN.load(model_path, env=env)
    env.close()

    click.echo(f"Evaluating over {n_eval_episodes} deterministic episodes...")
    eval_and_watch(
        agent,
        level,
        goal_reward=goal_reward,
        seed=model_seed,
        max_episode_steps=max_episode_steps,
        n_eval_episodes=n_eval_episodes,
        watch=watch,
    )


if __name__ == "__main__":
    cli()
