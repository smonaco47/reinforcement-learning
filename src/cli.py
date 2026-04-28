import os
import csv
import random
from datetime import datetime

import click
import numpy as np
import matplotlib.pyplot as plt

from src.hyperparameters import Hyperparameters
from src.train_model import train_model
from src.results import Results

# --- Constants ---

SKIP_FILES = {'consolidated_output.csv', 'output_for_dt.csv'}

KEEP_COLUMNS = [
    "lr_initial", "lr_final", "lr_decay", "lr_steps", "lr_type",
    "explore_initial", "explore_final", "explore_decay", "explore_steps", "explore_type",
    "horizon", "discount", "batch_size", "memory",
    "max", "max_group", "hit_goal", "eval_mean", "eval_std"
]

PLOT_KEYS = [
    "discount", "horizon", "lr_initial", "lr_final", "lr_decay", "lr_steps",
    "explore_initial", "explore_final", "explore_decay", "explore_steps",
    "batch_size", "memory"
]


# --- CLI group ---

@click.group()
def cli():
    """Reinforcement Learning hyperparameter search for LunarLander / CartPole."""
    pass


# --- train ---

@cli.command()
@click.option("--level", default="LunarLander-v3", show_default=True, help="Gymnasium environment ID")
@click.option("--iterations", default=1500, show_default=True, help="Training iterations")
@click.option("--goal-reward", default=200, show_default=True, help="Reward threshold to count as goal hit")
@click.option("--seed", default=0, show_default=True, help="Random seed")
@click.option("--max-episode-steps", default=750, show_default=True, help="Max steps per episode")
@click.option("--n-eval-episodes", default=10, show_default=True, help="Episodes for deterministic evaluation")
def train(level, iterations, goal_reward, seed, max_episode_steps, n_eval_episodes):
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
@click.option("--level", default="LunarLander-v3", show_default=True, help="Gymnasium environment ID")
@click.option("--iterations", default=1500, show_default=True, help="Training iterations per run")
@click.option("--limit", default=300, show_default=True, help="Number of runs (0 = unlimited)")
@click.option("--max-episode-steps", default=750, show_default=True, help="Max steps per episode")
@click.option("--goal-reward", default=200, show_default=True, help="Reward threshold to count as goal hit")
@click.option("--n-eval-episodes", default=25, show_default=True, help="Episodes for deterministic evaluation")
@click.option("--output-folder", default="output", show_default=True, help="Folder to write CSV results")
@click.option("--seed", default=None, type=int, help="Base random seed (default: random)")
@click.option("--narrow/--no-narrow", default=False, show_default=True, help="Use narrowed hyperparameter ranges")
@click.option("--lr-initial-min", default=5e-4, show_default=True, help="lr_initial lower bound (narrow mode)")
@click.option("--lr-initial-max", default=1e-2, show_default=True, help="lr_initial upper bound (narrow mode)")
@click.option("--discount-min", default=0.95, show_default=True, help="discount lower bound (narrow mode)")
@click.option("--discount-max", default=1.0, show_default=True, help="discount upper bound (narrow mode)")
def grid_search(level, iterations, limit, max_episode_steps, goal_reward, n_eval_episodes,
                output_folder, seed, narrow, lr_initial_min, lr_initial_max, discount_min, discount_max):
    """Run a random hyperparameter grid search."""
    base_seed = seed if seed is not None else random.randint(0, 1_000_000)
    os.makedirs(output_folder, exist_ok=True)

    click.echo(f"Starting grid search — level={level}, iterations={iterations}, "
               f"limit={'unlimited' if limit == 0 else limit}, base_seed={base_seed}, "
               f"mode={'narrow' if narrow else 'wide'}")

    params = Hyperparameters()
    i = 0

    with open(f"{output_folder}/{datetime.now().strftime('%m-%d-%H-%M-%S')}.csv", 'w') as out_file:
        out_file.write(f"{Hyperparameters.csv_header()},{Results.csv_header(iterations)}\n")
        out_file.flush()

        while limit == 0 or i < limit:
            try:
                i += 1
                run_seed = base_seed + i
                click.echo(f"{datetime.now().strftime('%H:%M:%S')}\t{i}\tSEED: {run_seed}")

                if narrow:
                    params.randomize_narrow(
                        steps=iterations,
                        lr_initial_range=(lr_initial_min, lr_initial_max),
                        discount_range=(discount_min, discount_max),
                    )
                else:
                    params.randomize(iterations)

                result, _ = train_model(
                    level=level,
                    params=params,
                    iterations=iterations,
                    goal_reward=goal_reward,
                    verbose=False,
                    seed=run_seed,
                    max_episode_steps=max_episode_steps,
                    n_eval_episodes=n_eval_episodes,
                )

                out_file.write(f"{params.csv_params()},{result.csv_result()}\n")
                out_file.flush()

            except Exception as e:
                click.echo(f"Failed on run {i}: {e}", err=True)


# --- analyze ---

@cli.command()
@click.option("--output-folder", default="output", show_default=True, help="Folder containing CSV results")
@click.option("--no-plot", is_flag=True, default=False, help="Skip plotting, just consolidate")
@click.option("--no-save", is_flag=True, default=False, help="Skip saving consolidated CSV, just plot")
def analyze(output_folder, no_plot, no_save):
    """Consolidate grid search CSVs and plot hyperparameter correlations."""
    csv_files = [
        entry.path for entry in os.scandir(output_folder)
        if entry.path.endswith('.csv') and os.path.basename(entry.path) not in SKIP_FILES
    ]

    click.echo(f"Found {len(csv_files)} CSV file(s):")
    results = []
    for path in csv_files:
        skipped = added = 0
        with open(path, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
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
        np.savetxt(out_path, np.vstack([KEEP_COLUMNS, data]), '%s', delimiter=",")
        click.echo(f"Saved {out_path} ({len(results)} rows)")

    if not no_plot:
        def col_data(col):
            return data[:, KEEP_COLUMNS.index(col)].astype(np.float64)

        eval_mean = col_data("eval_mean")
        fig, axs = plt.subplots(len(PLOT_KEYS), figsize=(8, 2 * len(PLOT_KEYS)), sharex='all')
        fig.suptitle('Hyperparameter Results')
        for idx, key in enumerate(PLOT_KEYS):
            d = col_data(key)
            axs[idx].scatter(eval_mean, d, alpha=0.6)
            axs[idx].set(ylabel=key)
            if max(d) < 1:
                axs[idx].set(yscale='log')
            axs[idx].grid(True)
        axs[-1].set(xlabel='Eval Mean Reward')
        plt.tight_layout()
        fig_path = f"{output_folder}/fig.png"
        plt.savefig(fig_path)
        click.echo(f"Saved {fig_path}")
        plt.show()


# --- run-best ---

@cli.command()
@click.option("--input-file", default="output/best_simplified.csv", show_default=True,
              help="CSV with best hyperparameters. Format: <csv_params>!<iterations> per line.")
@click.option("--output-folder", default="output", show_default=True, help="Folder to save models")
@click.option("--level", default="LunarLander-v3", show_default=True, help="Gymnasium environment ID")
@click.option("--max-episode-steps", default=750, show_default=True, help="Max steps per episode")
@click.option("--goal-reward", default=200, show_default=True, help="Reward threshold to count as goal hit")
@click.option("--seed", default=0, show_default=True, help="Random seed")
@click.option("--n-eval-episodes", default=25, show_default=True, help="Episodes for deterministic evaluation")
def run_best(input_file, output_folder, level, max_episode_steps, goal_reward, seed, n_eval_episodes):
    """Re-run the best hyperparameters from a file with longer training.

    Create the input file by copying your best rows from consolidated_output.csv
    and appending !<iterations> to each line, e.g.:

        0.001,0.0001,...!3000
    """
    i = 0
    for line in open(input_file, 'r'):
        line = line.strip()
        if not line:
            continue
        try:
            i += 1
            params_str, iterations = line.split('!')
            iterations = int(iterations)
            click.echo(f"{datetime.now().strftime('%H:%M:%S')}\t{i}\titerations={iterations}")

            params = Hyperparameters.from_csv(params_str)
            result, agent = train_model(
                level=level,
                params=params,
                iterations=iterations,
                goal_reward=goal_reward,
                verbose=True,
                seed=seed,
                max_episode_steps=max_episode_steps,
                n_eval_episodes=n_eval_episodes,
            )

            model_name = f"{output_folder}/model_{i}_eval{result.eval_mean:.0f}"
            agent.save(model_name)
            click.echo(f"Saved model to {model_name}.zip")

        except Exception as e:
            click.echo(f"Failed on line {i}: {e}", err=True)


if __name__ == "__main__":
    cli()