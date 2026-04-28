from datetime import datetime


class Results:
    def __init__(self):
        self.results = []
        self.hit_goal = 0
        self.eval_mean = None
        self.eval_std = None

    def add_result(self, result, hit_goal=False):
        self.results.append(result)

        if hit_goal:
            self.hit_goal += 1

    def add_eval(self, eval_rewards):
        self.eval_mean = sum(eval_rewards) / len(eval_rewards)
        self.eval_std = (sum((r - self.eval_mean) ** 2 for r in eval_rewards) / len(eval_rewards)) ** 0.5

    def print_summary(self):
        len_results = len(self.results)
        rewards_to_print = min(len_results, 100)
        avg_rewards = sum(self.results[len_results - rewards_to_print:-1]) / rewards_to_print
        print(f"{datetime.now().strftime('%H:%M:%S')}\t{len_results}\t{avg_rewards}")
        if self.eval_mean is not None:
            print(f"Eval mean: {self.eval_mean:.2f} +/- {self.eval_std:.2f}")

    @classmethod
    def csv_header(cls, iterations) -> str:
        step_size = iterations // 25
        headers = ["max", "max_group", "hit_goal", "eval_mean", "eval_std"]
        headers.extend(str(i) for i in range(step_size, iterations + 1, step_size))
        return ",".join(headers)

    def csv_result(self):
        len_results = len(self.results)
        step_size = len_results // 25
        buckets = [sum(self.results[i:i + step_size]) / step_size for i in range(0, len_results, step_size)]
        values = [max(self.results), max(buckets), self.hit_goal,
                  self.eval_mean if self.eval_mean is not None else 0,
                  self.eval_std if self.eval_std is not None else 0]
        values.extend(buckets)
        str_values = [str(value) for value in values]
        return ",".join(str_values)