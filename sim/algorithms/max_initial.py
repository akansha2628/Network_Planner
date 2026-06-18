"""
Variation 1: Maximum Initial State Algorithm
All links start at maximum capacity: 3 fibers lit, 3 cores each, C+L bands

This is a BASELINE algorithm - no upgrade logic runs.
Network starts at max and just routes traffic (similar to no-upgrade baseline).
"""

from typing import Tuple, List, Dict, Any
from sim.algorithms.base_algorithm import UpgradeAlgorithm
from sim.core import Num_cores
from sim.core.constants import C_BAND_SLOTS, TOTAL_SLOTS, FIBERS, Num_cores
import copy


class MaxInitialAlgorithm(UpgradeAlgorithm):
    """
    Baseline algorithm: network starts at maximum capacity, no upgrades ever performed.

    Initial state:
    - 3 fibers lit (fibers 0, 1, 2)
    - 3 cores per fiber (cores 0, 1, 2)
    - C+L bands on all cores

    This serves as an upper-bound performance baseline.
    No upgrade checking or planning occurs.
    """

    def __init__(self):
        super().__init__("max_initial")

    def initialize_link_status(self, num_links=None):
        """
        Initialize all links at MAXIMUM state:
        - 3 fibers lit (0, 1, 2)
        - Each fiber has 3 cores
        - Each core has C+L bands
        - Remaining fibers are dark
        """
        if num_links is None:
            from sim.core import topology as Topology
            num_links = Topology.LINKS

        link_status_forward = {}
        link_status_backward = {}

        for linkid in range(num_links):
            fibers = {}
            for fiber_id in range(FIBERS):
                if fiber_id < 3:  # First 3 fibers are lit with max config
                    # Multi-core structure: fiber[core_id] = {...}
                    fibers[fiber_id] = {}
                    for core_id in range(Num_cores):
                        fibers[fiber_id][core_id] = {
                            "lit": True,
                            "cores": 3,
                            "bands": ["C", "L"],
                            "slots": [0 for _ in range(TOTAL_SLOTS)]
                        }
                else:  # Remaining fibers are dark
                    fibers[fiber_id] = {}
                    for core_id in range(Num_cores):
                        fibers[fiber_id][core_id] = {
                            "lit": False,
                            "cores": 3,
                            "bands": [],
                            "slots": []
                        }

            link_status_forward[linkid] = fibers
            link_status_backward[linkid] = copy.deepcopy(fibers)

        print(f"[{self.name}] ✓ Initialized at MAXIMUM capacity: 3 fibers × 3 cores × C+L bands")
        print(f"[{self.name}] ℹ  Running as BASELINE (no upgrade logic will execute)")
        return link_status_forward, link_status_backward

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
        Always return False - no upgrades needed or possible.
        Network is already at maximum capacity.
        """
        return False

    def select_links_for_upgrade(
            self,
            traffic_rates,
            PATHS: dict,
            sd_block_prob: dict,
            sd_blocked_counts: dict = None,
            block_threshold: float = 0.0,
            working_topology=None,
            LINK_INDEX=None,
            link_status_forward: dict = None,
            link_status_backward: dict = None,
            current_time: float = 0.0,
            **kwargs
    ) -> Tuple[List[int], Dict[int, float], List]:
        """
        Always return empty - no upgrades possible (already at max).
        This should never be called since check_need_for_upgrade returns False.
        """
        return [], {}, []