from src.results import EvalHistory, Results, calculate_eval_reward_stats


def test_calculating_eval_reward_stats() -> None:
    eval_rewards: list[float] = [5, 10, 15, 20, 25, 30]
    eval_history = calculate_eval_reward_stats(eval_rewards, 20)
    assert eval_history == EvalHistory(
        num_timestamps=6, mean=17.5, std=8.539125638299666, hit_pct=50
    )


def test_calculating_eval_reward_stats_single_result() -> None:
    eval_rewards = [5.0]
    eval_history = calculate_eval_reward_stats(eval_rewards, 20)
    assert eval_history == EvalHistory(num_timestamps=1, mean=5, std=0, hit_pct=0)


def test_csv_serialization() -> None:
    result = Results()
    result.eval_history = [
        EvalHistory(num_timestamps=10, mean=5, std=2, hit_pct=0.3),
        EvalHistory(num_timestamps=10, mean=10, std=5, hit_pct=0.5),
        EvalHistory(num_timestamps=10, mean=20, std=7, hit_pct=0.7),
    ]
    assert (
        Results.csv_header(3)
        == "final_eval_mean,final_eval_std,final_eval_hit_pct,0,1,2"
    )
    assert result.csv_result() == "20,7,0.7,5,10,20"
