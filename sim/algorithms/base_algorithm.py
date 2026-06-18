"""
Base class for upgrade algorithms.

All upgrade algorithms should inherit from this class and implement
the required methods.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple


class UpgradeAlgorithm(ABC):
    """Base class for network upgrade algorithms."""

    def __init__(self, name: str):
        """
        Initialize upgrade algorithm.

        Args:
            name: Name of the algorithm
        """
        self.name = name

    @abstractmethod
    def check_need_for_upgrade(
            self,
            current_traffic,
            current_time: float,
            PATHS: dict,
            current_link_status_forward: dict,
            current_link_status_backward: dict,
            seed: int,
            **kwargs
    ) -> bool:
        """
        Check if network upgrade is needed.

        Args:
            current_traffic: Current traffic matrix
            current_time: Current simulation time
            PATHS: Precomputed paths
            current_link_status_forward: Forward link status
            current_link_status_backward: Backward link status
            seed: Random seed
            **kwargs: Additional algorithm-specific parameters

        Returns:
            True if upgrade is needed, False otherwise
        """
        pass

    @abstractmethod
    def select_links_for_upgrade(
            self,
            traffic_rates,
            PATHS: dict,
            sd_block_prob: dict,
            block_threshold: float,
            working_topology,
            LINK_INDEX,
            link_status_forward: dict,
            link_status_backward: dict,
            current_time: float,
            **kwargs
    ) -> Tuple[List[int], Dict, Any]:
        """
        Select links to upgrade.

        Args:
            traffic_rates: Traffic rate matrix
            PATHS: Precomputed paths
            sd_block_prob: Per source-destination blocking probability
            block_threshold: Blocking probability threshold
            working_topology: Current network topology
            LINK_INDEX: Link index matrix
            link_status_forward: Forward link status
            link_status_backward: Backward link status
            current_time: Current simulation time
            **kwargs: Additional algorithm-specific parameters

        Returns:
            Tuple of (selected_links, link_scores, additional_info)
        """
        pass

    def __str__(self):
        return f"UpgradeAlgorithm({self.name})"