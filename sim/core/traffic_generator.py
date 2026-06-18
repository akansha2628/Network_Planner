from typing import List, Set, Optional
import random
import math

class NetworkTrafficGenerator:
    def __init__(self,
                 number_of_nodes: int,
                 datarates: List[float],
                 lambda_0: float,
                 mean_holding_time: float,
                 Current_global_time: float,
                 rng: Optional[random.Random] = None):
        self.nodes_set: Set[int] = set(range(number_of_nodes))
        self.datarates_set: Set[float] = set(datarates)
        self.lambda_0: float = lambda_0
        self.mean_holding_time: float = mean_holding_time
        self.Current_global_time: float = Current_global_time

        # # ✅ use local RNG (falls back to global random if not provided)
        # self.rng = rng if rng is not None else random
        if rng is None:
            raise ValueError("RNG must be provided")
        self.rng = rng

    def generate_connection_data(self):
        src, dest = self.rng.sample(list(self.nodes_set), 2)
        rate = self.rng.choice(list(self.datarates_set))
        return src, dest, rate

    def generate_arrival_event(self, lambda_sd):
        time = self.random_exponential(lambda_sd)
        self.Current_global_time += time

        mu = 1.0 / self.mean_holding_time
        return self.Current_global_time, self.random_exponential(mu)

    def random_exponential(self, lambda_sd: float):
        pV = 1.0
        while pV == 1.0:
            pV = self.rng.uniform(0.0, 1.0)

        return (-1.0 / lambda_sd) * math.log(1.0 - pV)

    def get_connection(self, lambda_sd):
        return self.generate_arrival_event(lambda_sd)


