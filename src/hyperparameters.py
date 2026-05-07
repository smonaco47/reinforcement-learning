import math
import random


def log_uniform(low: float, high: float) -> float:
    """Sample uniformly in log space — gives equal probability to each order of magnitude."""
    return math.exp(random.uniform(math.log(low), math.log(high)))


class Hyperparameters:
    lr_initial: float = 1e-3
    lr_final: float = 1e-4
    lr_steps: int = 1000
    lr_decay: float = 0.99
    lr_type: str = "linear"

    explore_initial: float = 0.1
    explore_final: float = 0.0001
    explore_steps: int = 1000
    explore_decay: float = 0.99
    explore_type: str = "exponential"

    batch_size: int = 64
    memory: int = 50_000
    discount: float = 1.0

    def randomize(self, steps: int = 1000) -> None:
        lr_range = self.random_range_exponential(-6, -3)
        self.lr_initial = lr_range[1]
        self.lr_final = lr_range[0]
        self.lr_steps = random.randrange(max(1, steps))
        self.lr_decay = self.calc_decay(self.lr_initial, self.lr_final, self.lr_steps)

        explore_range = self.random_range(0.01, 0.3)
        self.explore_initial = explore_range[1]
        self.explore_final = explore_range[0] / 10
        self.explore_steps = random.randrange(max(1, steps))
        self.explore_decay = self.calc_decay(
            self.explore_initial, self.explore_final, self.explore_steps
        )

        self.discount = random.choice([x / 100 for x in range(90, 101)])

        self.batch_size = 2 ** random.randint(4, 8)
        self.memory = random.choice([50_000, 100_000, 200_000])

    def randomize_narrow(
        self,
        steps: int = 1000,
        lr_initial_range: tuple[float, float] = (1e-5, 1e-3),
        lr_final_range: tuple[float, float] = (1e-6, 1e-4),
        explore_initial_range: tuple[float, float] = (0.05, 0.3),
        explore_final_range: tuple[float, float] = (0.001, 0.05),
        discount_range: tuple[float, float] = (0.95, 1.0),
        batch_sizes: tuple[int, ...] = (32, 64, 128),
        memories: tuple[int, ...] = (50000, 100000, 200000),
    ) -> None:
        """Second-stage randomization with narrowed bounds based on first-stage results."""
        self.lr_initial = log_uniform(*lr_initial_range)
        self.lr_final = log_uniform(*lr_final_range)
        if self.lr_final >= self.lr_initial:
            self.lr_final = self.lr_initial / 10

        self.lr_steps = random.randrange(max(1, steps))
        self.lr_decay = self.calc_decay(self.lr_initial, self.lr_final, self.lr_steps)

        self.explore_initial = log_uniform(*explore_initial_range)
        self.explore_final = log_uniform(*explore_final_range)
        if self.explore_final >= self.explore_initial:
            self.explore_final = self.explore_initial / 10

        self.explore_steps = random.randrange(max(1, steps))
        self.explore_decay = self.calc_decay(
            self.explore_initial, self.explore_final, self.explore_steps
        )

        self.discount = random.uniform(*discount_range)
        self.batch_size = random.choice(batch_sizes)
        self.memory = random.choice(memories)

    @classmethod
    def calc_decay(cls, initial: float, final: float, steps: int) -> float:
        return (final / initial) ** (1 / steps)

    @classmethod
    def random_range(cls, lower: float, upper: float) -> tuple[float, float]:
        values = (random.uniform(lower, upper), random.uniform(lower, upper))
        return (min(values), max(values))

    @classmethod
    def random_range_exponential(cls, lower: int, upper: int) -> tuple[float, float]:
        low = 10.0**lower
        high = 10.0**upper
        values = (log_uniform(low, high), log_uniform(low, high))
        return (min(values), max(values))

    @classmethod
    def csv_header(cls) -> str:
        return "lr_initial,lr_final,lr_decay,lr_steps,lr_type,explore_initial,explore_final,explore_decay,explore_steps,explore_type,discount,batch_size,memory"

    def csv_params(self) -> str:
        return (
            f"{self.lr_initial},{self.lr_final},{self.lr_decay},{self.lr_steps},{self.lr_type},"
            f"{self.explore_initial},{self.explore_final},{self.explore_decay},{self.explore_steps},{self.explore_type},"
            f"{self.discount},{self.batch_size},{self.memory}"
        )

    @classmethod
    def from_csv(cls, str_val: str) -> "Hyperparameters":
        val = str_val.split(",")
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
        params.discount = float(val[10])
        params.batch_size = int(val[11])
        params.memory = int(val[12])
        return params

    @classmethod
    def random_decay(cls) -> float:
        num = f"0.{random.randrange(8, 10)}{random.randrange(10)}{random.randrange(10)}{random.randrange(10)}"
        return float(num)
