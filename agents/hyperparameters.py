import random
from datetime import datetime


class Hyperparameters:
    lr_initial = 1e-3
    lr_final = 1e-4
    lr_steps = 1000
    lr_decay = 0.99
    lr_type = 'linear'

    explore_initial = 0.1
    explore_final = 0.0001
    explore_steps = 1000
    explore_decay = 0.99
    explore_type = 'linear'

    batch_size = 64
    memory = 10000
    horizon = 200

    def randomize(self):
        random.seed(datetime.now())
        lr_range = self.random_range_exponential(-6, -2)
        self.lr_initial = lr_range[1]
        self.lr_final = lr_range[0]
        self.lr_decay = random.uniform(0.99, 0.99999)
        self.lr_steps = random.randrange(5000)
        self.lr_type = random.choice(['linear', 'exponential'])

        explore_range = self.random_range(0, 0.9)
        self.explore_initial = explore_range[1]
        self.explore_final = explore_range[0]
        self.explore_decay = random.uniform(0.99, 0.99999)
        self.explore_steps = random.randrange(2500)
        self.explore_type = random.choice(['linear', 'exponential'])

    @classmethod
    def random_range(cls, lower, upper):
        values = (random.uniform(lower, upper), random.uniform(lower, upper))
        return sorted(values)

    @classmethod
    def random_range_exponential(cls, lower, upper):
        exponent_1 = random.uniform(lower, upper)
        exponent_2 = random.uniform(lower, upper)
        base_1 = random.uniform(1, 9)
        base_2 = random.uniform(1, 9)
        values = (base_1 * (10 ** exponent_1), base_2 * (10 ** exponent_2))
        return sorted(values)

    @classmethod
    def csv_header(cls) -> str:
        return "lr_initial,lr_final,lr_decay,lr_steps,lr_type,explore_initial,explore_final,explore_decay,explore_steps,explore_type"

    def csv_params(self) -> str:
        return f"{self.lr_initial},{self.lr_final},{self.lr_decay},{self.lr_steps},{self.lr_type},{self.explore_initial},{self.explore_final},{self.explore_decay},{self.explore_steps},{self.explore_type}"

    @classmethod
    def from_csv(cls, val: str):
        val = val.split(',')
        params = Hyperparameters()
        params.lr_initial = float(val[0])
        params.lr_final = float(val[1])
        params.lr_decay = float(val[2])
        params.lr_steps = int(val[3])
        params.lr_type = val[4]
        params.explore_initial = float(val[5])
        params.explore_final = float(val[6])
        params.explore_decay = float(val[7])
        params.explore_steps = int(val[8])
        params.explore_type = val[9]
        return params
