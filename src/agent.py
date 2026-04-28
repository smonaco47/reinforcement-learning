import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback


class HitGoalCallback(BaseCallback):
    """Callback that tracks goal hits and episode rewards into a Results object."""

    def __init__(self, results, goal_reward, verbose=False):
        super().__init__(verbose)
        self.results = results
        self.goal_reward = goal_reward
        self._episode_reward = 0.0

    def _on_step(self) -> bool:
        reward = self.locals["rewards"][0]
        done = self.locals["dones"][0]
        self._episode_reward += reward

        if done:
            hit_goal = self._episode_reward >= self.goal_reward
            self.results.add_result(self._episode_reward, hit_goal)
            if hit_goal:
                print(f"Hit goal in episode {len(self.results.results)} "
                      f"with a reward of {self._episode_reward}!")
            self._episode_reward = 0.0

        return True


class AgentFactory:
    @classmethod
    def create_dqn_agent(cls, environment, params, seed=0):
        """
        DQN is the closest SB3 equivalent to tensorforce's DoubleDQN.
        SB3's DQN uses double-Q by default (use_double_dqn=True).
        https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html
        """
        lr_schedule = cls._build_lr_schedule(params)

        agent = DQN(
            policy="MlpPolicy",
            env=environment,
            learning_rate=lr_schedule,
            buffer_size=params.memory,
            batch_size=params.batch_size,
            gamma=params.discount,
            train_freq=params.batch_size,
            learning_starts=params.batch_size,
            exploration_fraction=params.explore_steps / 1500,  # fraction of training for exploration decay
            exploration_initial_eps=params.explore_initial,
            exploration_final_eps=params.explore_final,
            seed=seed,
            verbose=0,
            device="cpu",  # CPU is typically faster for low-dim envs like LunarLander
        )
        return agent

    @classmethod
    def _build_lr_schedule(cls, params):
        """
        Returns a schedule function for the learning rate.
        SB3 schedules receive `progress_remaining` (1.0 -> 0.0 over training).
        """
        if params.lr_type == 'linear':
            def lr_schedule(progress_remaining):
                # progress_remaining goes 1.0 -> 0.0
                return params.lr_final + (params.lr_initial - params.lr_final) * progress_remaining
        else:
            # exponential: use decay rate per episode approximated over progress
            def lr_schedule(progress_remaining):
                return max(params.lr_final, params.lr_initial * (progress_remaining ** 2))

        return lr_schedule
