class Results:
    results = []

    def add_result(self, result):
        self.results.append(result)

    def print_summary(self):
        len_results = len(self.results)
        rewards_to_print = min(len_results, 100)
        avg_rewards = sum(self.results[len_results - rewards_to_print:-1]) / rewards_to_print
        print(f"{datetime.now().strftime('%H:%M:%S')}\t{len_results}\t{avg_rewards}")

    @classmethod
    def csv_header(cls, iterations) -> str:
        step_size = iterations // 25
        indices = [str(i) for i in range(step_size, iterations + 1, step_size)]
        return ",".join(indices)

    def csv_result(self):
        len_results = len(self.results)
        step_size = len_results // 25
        indices = [str( sum(self.results[i:i+step_size]) / step_size) for i in range(0, len_results, step_size)]
        return ",".join(indices)