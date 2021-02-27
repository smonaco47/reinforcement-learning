from datetime import datetime


class Results:
    def __init__(self):
        self.results = []
        self.hit_goal=0

    def add_result(self, result, hit_goal=False):
        self.results.append(result)

        if hit_goal:
            self.hit_goal+=1

    def print_summary(self):
        len_results = len(self.results)
        rewards_to_print = min(len_results, 100)
        avg_rewards = sum(self.results[len_results - rewards_to_print:-1]) / rewards_to_print
        print(f"{datetime.now().strftime('%H:%M:%S')}\t{len_results}\t{avg_rewards}")

    @classmethod
    def csv_header(cls, iterations) -> str:
        step_size = iterations // 25
        headers = ["max", "max_group", "hit_goal"]
        headers.extend(str(i) for i in range(step_size, iterations + 1, step_size))
        return ",".join(headers)

    def csv_result(self):
        len_results = len(self.results)
        step_size = len_results // 25
        buckets = [sum(self.results[i:i+step_size]) / step_size for i in range(0, len_results, step_size)]
        values = [max(self.results), max(buckets), self.hit_goal ]
        values.extend(buckets)
        str_values = [str(value) for value in values]
        return ",".join(str_values)