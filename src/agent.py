from typing import Any, Callable

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

from src.hyperparameters import Hyperparameters
from src.results import Results

# Fixed train frequency — decoupled from batch_size.
# SB3 default is 4; tying it to batch_size means large batches update very infrequently.
TRAIN_FREQ = 4


class HitGoalCallback(BaseCallback):
    """Callback that tracks goal hits and episode rewards into a Results object."""

    def __init__(
        self, results: Results, goal_reward: float, verbose: bool = False
    ) -> None:
        super().__init__(verbose)
        self.results = results
        self.goal_reward = goal_reward
        self._episode_reward: float = 0.0

    def _on_step(self) -> bool:
        reward: float = float(self.locals["rewards"][0])
        done: bool = bool(self.locals["dones"][0])
        self._episode_reward += reward

        if done:
            hit_goal = self._episode_reward >= self.goal_reward
            self.results.add_result(self._episode_reward, hit_goal)
            self._episode_reward = 0.0

        return True


class AgentFactory:
    @classmethod
    def create_dqn_agent(
        cls,
        environment: Any,
        params: Hyperparameters,
        seed: int = 0,
        total_timesteps: int = 375_000,
    ) -> DQN:
        """
        DQN with double-Q (SB3 default).
        https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html

        total_timesteps must match the actual training budget so that
        exploration_fraction correctly scales explore_steps across the full run.
        Without this, exploration collapses early when training longer than 1500 episodes.
        """
        lr_schedule = cls._build_lr_schedule(params)
        exploration_fraction = cls._exploration_fraction(params, total_timesteps)

        return DQN(
            policy="MlpPolicy",
            env=environment,
            learning_rate=lr_schedule,
            buffer_size=params.memory,
            batch_size=params.batch_size,
            gamma=params.discount,
            train_freq=TRAIN_FREQ,
            learning_starts=params.batch_size,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=params.explore_initial,
            exploration_final_eps=params.explore_final,
            seed=seed,
            verbose=0,
            device="auto",
        )

    @classmethod
    def _exploration_fraction(
        cls, params: Hyperparameters, total_timesteps: int
    ) -> float:
        """
        Convert explore_steps (episode count) to a fraction of total_timesteps.
        Assumes average episode length of 250 steps for LunarLander.
        Clamped to [0, 1].
        """
        avg_episode_steps = 250
        return min((params.explore_steps * avg_episode_steps) / total_timesteps, 1.0)

    @classmethod
    def _build_lr_schedule(cls, params: Hyperparameters) -> Callable[[float], float]:
        """
        Returns a LR schedule function.
        SB3 passes progress_remaining (1.0 → 0.0) over the course of training.
        """
        if params.lr_type == "linear":

            def lr_schedule(progress_remaining: float) -> float:
                return (
                    params.lr_final
                    + (params.lr_initial - params.lr_final) * progress_remaining
                )
        else:

            def lr_schedule(progress_remaining: float) -> float:
                return max(params.lr_final, params.lr_initial * (progress_remaining**2))

        return lr_schedule
