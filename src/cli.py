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
    evaluate_model,
    make_environment,
    seed_everything,
    train_adaptive,
    train_model,
)

# --- Constants ---

SKIP_FILES = {"consolidated_output.csv", "output_for_dt.csv", "best_simplified.csv"}

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
    "--iterations", default=1500, show_default=True, help="Training iterations"
)
@click.option(
    "--goal-reward",
    default=200.0,
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
def train(
    level: str,
    iterations: int,
    goal_reward: float,
    seed: int,
    max_episode_steps: int,
    n_eval_episodes: int,
) -> None:
    """Train a single DQN agent with randomized hyperparameters."""
    params = Hyperparameters()
    params.randomize()
    train_model(
        level=level,
        params=params,
        iterations=iterations,
        goal_reward=goal_reward,
        verbose=True,
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
    "--iterations", default=1500, show_default=True, help="Training iterations per run"
)
@click.option(
    "--limit", default=300, show_default=True, help="Number of runs (0 = unlimited)"
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
    "--output-folder",
    default="output",
    show_default=True,
    help="Folder to write CSV results",
)
@click.option(
    "--seed", default=None, type=int, help="Base random seed (default: random)"
)
@click.option(
    "--narrow/--no-narrow",
    default=False,
    show_default=True,
    help="Use narrowed hyperparameter ranges",
)
@click.option(
    "--lr-initial-min",
    default=5e-4,
    show_default=True,
    help="lr_initial lower bound (narrow mode)",
)
@click.option(
    "--lr-initial-max",
    default=1e-2,
    show_default=True,
    help="lr_initial upper bound (narrow mode)",
)
@click.option(
    "--discount-min",
    default=0.95,
    show_default=True,
    help="discount lower bound (narrow mode)",
)
@click.option(
    "--discount-max",
    default=1.0,
    show_default=True,
    help="discount upper bound (narrow mode)",
)
def grid_search(
    level: str,
    iterations: int,
    limit: int,
    max_episode_steps: int,
    goal_reward: float,
    n_eval_episodes: int,
    output_folder: str,
    seed: int | None,
    narrow: bool,
    lr_initial_min: float,
    lr_initial_max: float,
    discount_min: float,
    discount_max: float,
) -> None:
    """Run a random hyperparameter grid search."""
    base_seed = seed if seed is not None else random.randint(0, 1_000_000)
    os.makedirs(output_folder, exist_ok=True)

    click.echo(
        f"Starting grid search — level={level}, iterations={iterations}, "
        f"limit={'unlimited' if limit == 0 else limit}, base_seed={base_seed}, "
        f"mode={'narrow' if narrow else 'wide'}"
    )

    params = Hyperparameters()
    i = 0

    with open(
        f"{output_folder}/{datetime.now().strftime('%m-%d-%H-%M-%S')}.csv", "w"
    ) as out_file:
        out_file.write(
            f"seed,{Hyperparameters.csv_header()},{Results.csv_header(iterations)}\n"
        )
        out_file.flush()

        while limit == 0 or i < limit:
            try:
                i += 1
                run_seed = base_seed + i
                click.echo(
                    f"{datetime.now().strftime('%H:%M:%S')}\t{i}\tSEED: {run_seed}"
                )

                if narrow:
                    params.randomize_narrow(
                        steps=iterations,
                        lr_initial_range=(lr_initial_min, lr_initial_max),
                        discount_range=(discount_min, discount_max),
                    )
                else:
                    params.randomize(iterations)

                result, agent = train_model(
                    level=level,
                    params=params,
                    iterations=iterations,
                    goal_reward=goal_reward,
                    verbose=False,
                    seed=run_seed,
                    max_episode_steps=max_episode_steps,
                    n_eval_episodes=n_eval_episodes,
                )

                out_file.write(
                    f"{run_seed},{params.csv_params()},{result.csv_result()}\n"
                )
                out_file.flush()

                if result.eval_mean is not None and result.eval_mean >= goal_reward:
                    model_name = f"{output_folder}/model_gs_{run_seed}_eval{result.eval_mean:.0f}"
                    agent.save(model_name)
                    click.echo(
                        f"  [SAVED] {model_name}.zip (eval_mean={result.eval_mean:.1f})"
                    )
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


# --- train-from-csv ---


@cli.command()
@click.argument("input-file")
@click.option(
    "--level",
    default="LunarLander-v3",
    show_default=True,
    help="Gymnasium environment ID",
)
@click.option(
    "--output-folder",
    default="output",
    show_default=True,
    help="Folder to save models and plots",
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
    help="Episodes per evaluation checkpoint",
)
@click.option(
    "--eval-interval",
    default=100,
    show_default=True,
    help="Episodes between eval checkpoints",
)
@click.option(
    "--plateau-patience",
    default=10,
    show_default=True,
    help="Evals with no improvement before stopping",
)
@click.option(
    "--plateau-threshold",
    default=5.0,
    show_default=True,
    help="Min eval_mean improvement to not count as plateau",
)
@click.option(
    "--success-threshold",
    default=250.0,
    show_default=True,
    help="eval_mean above this is considered solved",
)
@click.option(
    "--success-window",
    default=5,
    show_default=True,
    help="Consecutive evals above success threshold before stopping",
)
@click.option(
    "--max-iterations",
    default=10000,
    show_default=True,
    help="Hard cap on training iterations",
)
@click.option(
    "--seed",
    default=None,
    type=int,
    help="Override seed (default: use seed column from CSV)",
)
def train_from_csv(
    input_file: str,
    level: str,
    output_folder: str,
    max_episode_steps: int,
    goal_reward: float,
    n_eval_episodes: int,
    eval_interval: int,
    plateau_patience: int,
    plateau_threshold: float,
    success_threshold: float,
    success_window: int,
    max_iterations: int,
    seed: int | None,
) -> None:
    """Train adaptively from rows in a consolidated_output.csv-format file.

    INPUT_FILE is a CSV with rows in consolidated_output.csv format (header required).
    Copy any rows you want to re-train directly from consolidated_output.csv.

    Stops automatically when stalled or consistently solved. Saves model,
    results summary, and eval progress plot per run.

    Example:

        python -m src.cli train-from-csv output/my_best_rows.csv
    """
    os.makedirs(output_folder, exist_ok=True)

    with open(input_file, "r") as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows = list(reader)

    if not rows:
        click.echo("[ERROR] No rows found in input file", err=True)
        raise SystemExit(1)

    missing = [col for col in PARAM_COLS + ["seed"] if col not in headers]
    if missing:
        click.echo(f"[ERROR] Input file missing columns: {missing}", err=True)
        raise SystemExit(1)

    seed_idx = headers.index("seed")
    param_indices = [headers.index(col) for col in PARAM_COLS]

    click.echo(f"Found {len(rows)} row(s) to train")

    for row_num, row in enumerate(rows, start=1):
        run_seed = seed if seed is not None else int(row[seed_idx])
        params_str = ",".join(row[idx] for idx in param_indices)
        params = Hyperparameters.from_csv(params_str)

        click.echo(
            f"\n[{row_num}/{len(rows)}] seed={run_seed} "
            f"lr={params.lr_initial:.2e} discount={params.discount:.3f} "
            f"batch={params.batch_size} memory={params.memory}"
        )

        try:
            agent, eval_history = train_adaptive(
                level=level,
                params=params,
                seed=run_seed,
                max_episode_steps=max_episode_steps,
                goal_reward=goal_reward,
                n_eval_episodes=n_eval_episodes,
                eval_interval_episodes=eval_interval,
                plateau_patience=plateau_patience,
                plateau_threshold=plateau_threshold,
                success_threshold=success_threshold,
                success_window=success_window,
                max_iterations=max_iterations,
            )

            # Final deterministic eval
            seed_everything(run_seed + 1)
            eval_env = make_environment(level, max_episode_steps, seed=run_seed)
            final_rewards = evaluate_model(
                agent, eval_env, n_eval_episodes=n_eval_episodes
            )
            eval_env.close()

            final_mean = sum(final_rewards) / len(final_rewards)
            final_std = (
                sum((r - final_mean) ** 2 for r in final_rewards) / len(final_rewards)
            ) ** 0.5
            final_hit_pct = (
                sum(1 for r in final_rewards if r >= goal_reward)
                / len(final_rewards)
                * 100
            )

            timestamp = datetime.now().strftime("%m-%d-%H-%M-%S")
            run_dir = f"{output_folder}/train_from_csv_{run_seed}_{timestamp}"
            os.makedirs(run_dir, exist_ok=True)

            model_path = f"{run_dir}/model_eval{final_mean:.0f}"
            agent.save(model_path)
            click.echo(f"  Saved model: {model_path}.zip")

            with open(f"{run_dir}/results.txt", "w") as sf:
                sf.write(f"Seed:          {run_seed}\n")
                sf.write(f"Eval mean:     {final_mean:.2f}\n")
                sf.write(f"Eval std:      {final_std:.2f}\n")
                sf.write(f"Goal hit %:    {final_hit_pct:.0f}%\n")
                sf.write(f"Total evals:   {len(eval_history)}\n")
                sf.write(
                    f"Final step:    {eval_history[-1][0] if eval_history else 0}\n"
                )
                sf.write("\nHyperparameters:\n")
                for col, idx in zip(PARAM_COLS, param_indices):
                    sf.write(f"  {col}: {row[idx]}\n")

            click.echo(
                f"  Final eval: {final_mean:.2f} +/- {final_std:.2f} | Goal hit: {final_hit_pct:.0f}%"
            )

            if eval_history:
                steps = [h[0] for h in eval_history]
                means = [h[1] for h in eval_history]
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(steps, means, marker="o", markersize=3)
                ax.axhline(
                    y=goal_reward,
                    color="green",
                    linestyle="--",
                    label=f"Goal ({goal_reward})",
                )
                ax.axhline(
                    y=success_threshold,
                    color="gold",
                    linestyle="--",
                    label=f"Success ({success_threshold})",
                )
                ax.set_xlabel("Timesteps")
                ax.set_ylabel("Eval Mean Reward")
                ax.set_title(
                    f"Training Progress — seed={run_seed} final={final_mean:.0f}"
                )
                ax.legend()
                ax.grid(True)
                plt.tight_layout()
                plot_path = f"{run_dir}/eval_progress.png"
                plt.savefig(plot_path)
                plt.close()
                click.echo(f"  Saved plot: {plot_path}")

        except Exception as e:
            click.echo(f"  [ERROR] Failed on row {row_num}: {e}", err=True)


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
