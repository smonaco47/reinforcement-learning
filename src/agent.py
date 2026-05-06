from typing import Any, Callable

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

from src.hyperparameters import Hyperparameters
from src.results import Results

# Decoupled from batch_size — update the network every 4 steps regardless of batch size
DEFAULT_TRAIN_FREQ = 4

# Minimum memory size for LunarLander — smaller buffers get overwritten before
# the agent can learn stable patterns
MIN_MEMORY = 50_000


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
        DQN is the closest SB3 equivalent to tensorforce's DoubleDQN.
        SB3's DQN uses double-Q by default.
        https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html

        total_timesteps is required to correctly scale exploration_fraction —
        it should match the total_timesteps passed to agent.learn().
        """
        lr_schedule = cls._build_lr_schedule(params)

        # Scale exploration fraction against actual training length, not a hardcoded value.
        # explore_steps is in episodes; convert to timesteps using avg episode length.
        avg_episode_steps = 250
        explore_timesteps = params.explore_steps * avg_episode_steps
        exploration_fraction = min(explore_timesteps / total_timesteps, 1.0)

        # Enforce minimum memory — buffers smaller than 50k get overwritten too
        # fast for the agent to learn stable patterns on LunarLander.
        buffer_size = max(params.memory, MIN_MEMORY)

        agent = DQN(
            policy="MlpPolicy",
            env=environment,
            learning_rate=lr_schedule,
            buffer_size=buffer_size,
            batch_size=params.batch_size,
            gamma=params.discount,
            train_freq=DEFAULT_TRAIN_FREQ,  # decouple from batch_size
            learning_starts=params.batch_size,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=params.explore_initial,
            exploration_final_eps=params.explore_final,
            seed=seed,
            verbose=0,
            device="auto",
        )
        return agent

    @classmethod
    def _build_lr_schedule(cls, params: Hyperparameters) -> Callable[[float], float]:
        """
        Returns a schedule function for the learning rate.
        SB3 schedules receive `progress_remaining` (1.0 -> 0.0 over training).
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
