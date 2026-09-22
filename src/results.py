from dataclasses import dataclass


@dataclass
class EvalHistory:
    num_timestamps: int
    mean: float
    std: float
    hit_pct: float


class Results:
    def __init__(self) -> None:
        self.hit_goal: int = 0
        self.eval_history: list[EvalHistory] = []

    @classmethod
    def csv_header(cls, n_expected_evals: int) -> str:
        headers = [
            "final_eval_mean",
            "final_eval_std",
            "final_eval_hit_pct",
        ]
        headers.extend(str(i) for i in range(n_expected_evals))
        return ",".join(headers)

    def csv_result(self) -> str:
        final_eval_history = self.eval_history[-1]
        values: list[float | int] = [
            final_eval_history.mean,
            final_eval_history.std,
            final_eval_history.hit_pct,
        ]
        values.extend(e.mean for e in self.eval_history)
        return ",".join(str(v) for v in values)


def calculate_eval_reward_stats(
    eval_rewards: list[float], goal_reward: float
) -> EvalHistory:
    eval_mean = sum(eval_rewards) / len(eval_rewards)
    eval_std = (
        sum((r - eval_mean) ** 2 for r in eval_rewards) / len(eval_rewards)
    ) ** 0.5
    hit_pct = sum(1 for r in eval_rewards if r >= goal_reward) / len(eval_rewards) * 100
    return EvalHistory(len(eval_rewards), eval_mean, eval_std, hit_pct)
