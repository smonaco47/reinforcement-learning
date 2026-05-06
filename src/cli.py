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
    "horizon",
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
    "horizon",
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


# --- CLI group ---


@click.group()
def cli():
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
    default=200,
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
    default=200,
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
    level,
    iterations,
    limit,
    max_episode_steps,
    goal_reward,
    n_eval_episodes,
    output_folder,
    seed,
    narrow,
    lr_initial_min,
    lr_initial_max,
    discount_min,
    discount_max,
):
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

                if result.eval_mean >= goal_reward:
                    model_name = f"{output_folder}/model_gs_{run_seed}_eval{result.eval_mean:.0f}"
                    agent.save(model_name)
                    click.echo(
                        f"  [SAVED] Model saved to {model_name}.zip (eval_mean={result.eval_mean:.1f})"
                    )
                del agent

            except Exception as e:
                click.echo(f"Failed on run {i}: {e}", err=True)
                raise e


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
def analyze(output_folder, no_plot, no_save):
    """Consolidate grid search CSVs and plot hyperparameter correlations."""
    csv_files = [
        entry.path
        for entry in os.scandir(output_folder)
        if entry.path.endswith(".csv")
        and os.path.basename(entry.path) not in SKIP_FILES
    ]

    click.echo(f"Found {len(csv_files)} CSV file(s):")
    results = []
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

        def col_data(col):
            return data[:, KEEP_COLUMNS.index(col)].astype(np.float64)

        x = col_data("eval_hit_pct")
        fig, axs = plt.subplots(
            len(PLOT_KEYS), figsize=(8, 2 * len(PLOT_KEYS)), sharex="all"
        )
        fig.suptitle("Hyperparameter Results")
        for idx, key in enumerate(PLOT_KEYS):
            d = col_data(key)
            axs[idx].scatter(x, d, alpha=0.6)
            axs[idx].set(ylabel=key)
            if max(d) < 1:
                axs[idx].set(yscale="log")
            axs[idx].grid(True)
        axs[-1].set(xlabel="Eval Mean Reward")
        plt.tight_layout()
        fig_path = f"{output_folder}/fig.png"
        plt.savefig(fig_path)
        click.echo(f"Saved {fig_path}")
        plt.show()


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
    default=200,
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
    model_path, level, max_episode_steps, goal_reward, n_eval_episodes, seed, watch
):
    """Load a saved model and evaluate it deterministically.

    MODEL_PATH is the path to the saved model zip, e.g. output/model_gs_12345_eval210

    Examples:

        python -m src.cli run-model output/model_gs_12345_eval210
        python -m src.cli run-model output/model_gs_12345_eval210 --no-watch --n-eval-episodes 100
    """
    import re
    from stable_baselines3 import DQN
    from src.train_model import make_environment, seed_everything, eval_and_watch

    model_seed = seed
    match = re.search(r"model_gs_(\d+)_eval", model_path)
    if match:
        model_seed = int(match.group(1))
        click.echo(f"Using seed {model_seed} from model filename")

    seed_everything(model_seed)
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


# --- train-from-csv ---


@cli.command()
@click.argument("csv-file")
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
    help="Episodes per eval checkpoint",
)
@click.option(
    "--eval-interval",
    default=100,
    show_default=True,
    help="Training iterations between evals",
)
@click.option(
    "--plateau-window",
    default=5,
    show_default=True,
    help="Evals with no improvement before stopping",
)
@click.option(
    "--plateau-threshold",
    default=2.0,
    show_default=True,
    help="Min eval_mean improvement to not be stalled",
)
@click.option(
    "--solved-threshold",
    default=250.0,
    show_default=True,
    help="eval_mean considered solved",
)
@click.option(
    "--solved-window",
    default=3,
    show_default=True,
    help="Consecutive evals above threshold to stop",
)
@click.option(
    "--max-iterations",
    default=10000,
    show_default=True,
    help="Hard cap on training iterations",
)
@click.option(
    "--output-folder",
    default="output",
    show_default=True,
    help="Folder to save models and plots",
)
@click.option(
    "--consolidated-csv",
    default="output/consolidated_output.csv",
    show_default=True,
    help="consolidated_output.csv to read headers from",
)
def train_from_csv(
    csv_file,
    level,
    max_episode_steps,
    goal_reward,
    n_eval_episodes,
    eval_interval,
    plateau_window,
    plateau_threshold,
    solved_threshold,
    solved_window,
    max_iterations,
    output_folder,
    consolidated_csv,
):
    """Train models from hyperparameter rows in a CSV file until solved or stalled.

    CSV_FILE should contain rows in the same format as consolidated_output.csv —
    just copy the rows you want to re-train directly from that file.

    Stopping conditions (whichever comes first):
      - Stalled: eval_mean hasn't improved by --plateau-threshold in --plateau-window evals
      - Solved: eval_mean >= --solved-threshold for --solved-window consecutive evals
      - Max: --max-iterations reached

    For each row, saves the model and a plot of eval_mean over training to output/.
    """
    import re
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend — don't open windows
    import matplotlib.pyplot as plt
    from src.train_model import (
        make_environment,
        seed_everything,
        train_until_solved,
        eval_and_watch,
    )
    from src.agent import AgentFactory

    os.makedirs(output_folder, exist_ok=True)

    # Read header from consolidated_output.csv
    with open(consolidated_csv, "r") as f:
        headers = next(csv.reader(f))

    seed_idx = headers.index("seed")
    param_cols = [
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
        "horizon",
        "discount",
        "batch_size",
        "memory",
    ]
    param_indices = [headers.index(col) for col in param_cols]

    # Read rows to train from
    with open(csv_file, "r") as f:
        rows = [line.strip() for line in f if line.strip()]

    click.echo(f"Found {len(rows)} row(s) to train")

    for row_num, line in enumerate(rows, start=1):
        parts = line.split(",")
        seed = int(parts[seed_idx])
        params_str = ",".join(parts[idx] for idx in param_indices)
        params = Hyperparameters.from_csv(params_str)

        click.echo(
            f"[{row_num}/{len(rows)}] seed={seed} lr={params.lr_initial:.2e} "
            f"discount={params.discount:.3f} batch={params.batch_size}"
        )

        seed_everything(seed)
        env = make_environment(level, max_episode_steps, seed=seed)
        agent = AgentFactory.create_dqn_agent(env, params, seed=seed)
        env.close()

        eval_means, eval_steps, agent = train_until_solved(
            agent=agent,
            level=level,
            params=params,
            seed=seed,
            max_episode_steps=max_episode_steps,
            goal_reward=goal_reward,
            n_eval_episodes=n_eval_episodes,
            eval_interval=eval_interval,
            plateau_window=plateau_window,
            plateau_threshold=plateau_threshold,
            solved_threshold=solved_threshold,
            solved_window=solved_window,
            max_iterations=max_iterations,
        )

        final_eval = eval_means[-1] if eval_means else 0.0
        stop_reason = (
            "solved"
            if len(eval_means) >= solved_window
            and all(e >= solved_threshold for e in eval_means[-solved_window:])
            else "stalled"
            if len(eval_means) >= plateau_window
            else "max_iterations"
        )

        # Save model
        model_name = (
            f"{output_folder}/model_csv_{row_num}_seed{seed}_eval{final_eval:.0f}"
        )
        agent.save(model_name)
        click.echo(
            f"Saved model to {model_name}.zip (stop={stop_reason}, final_eval={final_eval:.1f})"
        )

        # Save eval_mean plot
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(eval_steps, eval_means, marker="o", markersize=3, linewidth=1.5)
        ax.axhline(
            y=solved_threshold,
            color="green",
            linestyle="--",
            alpha=0.7,
            label=f"solved ({solved_threshold})",
        )
        ax.axhline(
            y=goal_reward,
            color="orange",
            linestyle="--",
            alpha=0.7,
            label=f"goal ({goal_reward})",
        )
        ax.set_xlabel("Training iterations")
        ax.set_ylabel("Eval mean reward")
        ax.set_title(
            f"Row {row_num} | seed={seed} | lr={params.lr_initial:.2e} | "
            f"discount={params.discount:.3f} | stop={stop_reason}"
        )
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plot_path = f"{output_folder}/plot_csv_{row_num}_seed{seed}.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        click.echo(f"Saved plot to {plot_path}")

        # Print hyperparameter summary
        click.echo(
            f"  lr_initial={params.lr_initial:.2e}  lr_final={params.lr_final:.2e}  "
            f"lr_steps={params.lr_steps}  lr_type={params.lr_type}"
        )
        click.echo(
            f"  explore_initial={params.explore_initial:.3f}  explore_final={params.explore_final:.4f}  "
            f"explore_steps={params.explore_steps}"
        )
        click.echo(
            f"  discount={params.discount:.3f}  horizon={params.horizon}  "
            f"batch_size={params.batch_size}  memory={params.memory}"
        )
        click.echo(
            f"  final eval_mean={final_eval:.2f}  iterations={eval_steps[-1] if eval_steps else 0}"
        )


if __name__ == "__main__":
    cli()
