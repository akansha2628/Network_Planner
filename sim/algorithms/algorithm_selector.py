"""Algorithm selection utilities for the optical network simulator."""

from typing import Dict, List, Type

from sim.algorithms.base_algorithm import UpgradeAlgorithm
from sim.algorithms.no_upgrade import NoUpgradeAlgorithm
from sim.algorithms.greedy_no_sd_threshold import GreedyAlgorithm
from sim.algorithms.Incremental_upgrade_no_sd_threshold import IncrementalAlgorithm
from sim.algorithms.jump_to_max_no_sd_threshold import JumpToMaxAlgorithm
from sim.algorithms.random_no_sd_threshold import RandomAlgorithm
from sim.algorithms.max_initial import MaxInitialAlgorithm


class AlgorithmSelector:
    """Map user-facing algorithm names to implementation classes."""

    _algorithms: Dict[str, Type[UpgradeAlgorithm]] = {
        "no_upgrade": NoUpgradeAlgorithm,
        "greedy": GreedyAlgorithm,
        "increment": IncrementalAlgorithm,
        "max_initial": MaxInitialAlgorithm,
        "jump_to_max": JumpToMaxAlgorithm,
        "random": RandomAlgorithm,
    }

    @classmethod
    def get_algorithm(cls, algorithm_name: str) -> UpgradeAlgorithm:
        """Return an algorithm instance for a configured algorithm name."""
        normalized_name = algorithm_name.lower()
        if normalized_name not in cls._algorithms:
            available = ", ".join(cls._algorithms.keys())
            raise ValueError(
                f"Unknown algorithm '{algorithm_name}'. Available algorithms: {available}"
            )
        return cls._algorithms[normalized_name]()

    @classmethod
    def list_algorithms(cls) -> List[str]:
        """Return the supported algorithm names."""
        return list(cls._algorithms.keys())

    @classmethod
    def register_algorithm(cls, name: str, algorithm_class: Type[UpgradeAlgorithm]) -> None:
        """Register an additional algorithm implementation."""
        if not issubclass(algorithm_class, UpgradeAlgorithm):
            raise ValueError("Algorithm class must inherit from UpgradeAlgorithm")
        cls._algorithms[name.lower()] = algorithm_class
