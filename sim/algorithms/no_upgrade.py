"""
No Upgrade Algorithm

Baseline algorithm that never upgrades the network.
Useful for comparison and testing.
"""

from typing import List, Dict, Any, Tuple
from sim.algorithms.base_algorithm import UpgradeAlgorithm

class NoUpgradeAlgorithm(UpgradeAlgorithm):
    """Algorithm that never performs upgrades."""
    
    def __init__(self):
        super().__init__("no_upgrade")
    
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
        Always returns False (no upgrade needed).
        
        Returns:
            False - never upgrade
        """
        print(f"\n[{self.name}] No upgrade check - always returns False")
        return False
    
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
        Returns empty list (no links to upgrade).
        
        Returns:
            Empty list and empty dictionaries
        """
        print(f"\n[{self.name}] No link selection - returning empty list")
        return [], {}, None
