"""
Variation 2: Jump-to-Max Upgrade Algorithm
Starts minimal (single-core C-band), but first upgrade jumps to maximum state
"""

from typing import Tuple, List, Dict, Any
from sim.algorithms.base_algorithm import UpgradeAlgorithm
from sim.upgrade.need_for_upgrade import check_need_for_upgrade
from sim.upgrade.link_selection import select_links_from_congested_paths_per_sd

def compute_link_utilization(link_id, link_status_forward):
    fibers = link_status_forward.get(link_id, {})
    used_slots = 0
    total_slots = 0

    for fiber_id, fiber in fibers.items():
        if isinstance(fiber, dict) and "slots" in fiber:
            if fiber.get("lit", False):
                slots = fiber.get("slots", [])
                total_slots += len(slots)
                used_slots += sum(1 for x in slots if x != 0)
        else:
            for core_id, core in fiber.items():
                if isinstance(core, dict) and core.get("lit", False):
                    slots = core.get("slots", [])
                    total_slots += len(slots)
                    used_slots += sum(1 for x in slots if x != 0)

    return 0.0 if total_slots == 0 else used_slots / total_slots

class JumpToMaxAlgorithm(UpgradeAlgorithm):
    """
    Algorithm variant where upgrades jump directly to maximum state.

    Initial state: Same as standard (fiber 0, single core, C band)

    Upgrade behavior:
    - When upgrade is needed on a link → jump to maximum state:
      * Convert ALL fibers to 3-core structure
      * Light first 3 fibers (0, 1, 2)
      * Enable C+L bands on all lit cores
    - Once upgraded → link at max, no further upgrades on that link
    """

    def __init__(self):
        super().__init__("jump_to_max")
        # ❌ REMOVED: self.upgraded_links = set()

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
        Standard upgrade check using projection logic.
        """
        print(f"\n[{self.name}] Checking need for upgrade...")
        return check_need_for_upgrade(
            current_time,
            PATHS,
            current_traffic,
            current_link_status_forward,
            current_link_status_backward,
            seed
        )

    def select_links_for_upgrade(
            self,
            traffic_rates,
            PATHS,
            sd_block_prob,
            sd_blocked_counts=None,
            block_threshold=0.0,
            working_topology=None,
            LINK_INDEX=None,
            link_status_forward=None,
            link_status_backward=None,
            current_time=0.0,
            **kwargs
    ):

        print(f"\n[{self.name}] Selecting links using path congestion")

        # =========================================================
        # NO SD-THRESHOLD SELECTION
        # Rank all SD pairs with positive blocking probability.
        # PlanChecker will later handle performance, budget, and time feasibility.
        # =========================================================
        ranked_sd_pairs = sorted(
            [(sd, bp) for sd, bp in sd_block_prob.items() if bp > 0.0],
            key=lambda x: x[1],
            reverse=True
        ) if sd_block_prob else []

        # Fallback: if all SD-pair BP values are zero but the network still
        # triggered an upgrade, rank all available SD pairs by BP.
        if not ranked_sd_pairs and sd_block_prob:
            ranked_sd_pairs = sorted(
                sd_block_prob.items(),
                key=lambda x: x[1],
                reverse=True
            )

        critical_sd_pairs = [sd for sd, bp in ranked_sd_pairs]

        if not critical_sd_pairs:
            return [], {}, []

        print(f"[{self.name}] NO SD-THRESHOLD SELECTION ENABLED")
        print(f"[{self.name}] Ignoring block_threshold={block_threshold}")
        print(f"[{self.name}] Ranked SD pairs considered = {len(critical_sd_pairs)}")
        print(f"[{self.name}] Top ranked SD pairs by SD blocking:")
        for sd, bp in ranked_sd_pairs[:10]:
            print(f"  SD {sd}: SD_BP={bp:.6f}")

        link_selection_mode = kwargs.get("link_selection_mode", "path_congestion")

        if link_selection_mode == "path_congestion":
            print(f"\n[{self.name}] Selecting all links on most congested paths")
        elif link_selection_mode == "most_congested":
            print(f"\n[{self.name}] Selecting highest-utilization link from each most congested path")

        path_stats = kwargs.get("path_stats")
        path_rank = int(kwargs.get("path_rank", 0))

        if path_stats is None:
            print("path_stats missing")
            return [], {}, []

        selected_links = select_links_from_congested_paths_per_sd(
            critical_sd_pairs,
            PATHS,
            path_stats,
            LINK_INDEX
        )

        link_scores = {}

        for sd in critical_sd_pairs:
            s, d = sd

            candidate_paths = []
            try:
                candidate_paths = PATHS[s][d]
            except Exception:
                candidate_paths = []

            if candidate_paths is None:
                candidate_paths = []

            ranked_paths = []

            for p_idx, node_path in enumerate(candidate_paths):

                key = (min(s, d), max(s, d), p_idx)
                row = path_stats.get(key)

                if not row:
                    continue

                arrivals = row.get("path_arrivals", 0)
                blocked = row.get("path_blocked", 0)

                if arrivals <= 0:
                    continue

                bp = blocked / arrivals

                ranked_paths.append(
                    (bp, node_path, p_idx)
                )

            ranked_paths.sort(
                key=lambda x: x[0],
                reverse=True
            )

            if path_rank >= len(ranked_paths):
                continue

            best_bp, best_path, chosen_p_idx = ranked_paths[path_rank]

            path_link_ids = []
            for i in range(len(best_path) - 1):
                u = best_path[i]
                v = best_path[i + 1]
                lid = LINK_INDEX[u][v]
                if lid != -1:
                    path_link_ids.append(lid)

            link_selection_mode = kwargs.get("link_selection_mode", "path_congestion")

            if link_selection_mode == "path_congestion":
                for lid in path_link_ids:
                    link_scores[lid] = link_scores.get(lid, 0.0) + max(best_bp, 0.0)

            elif link_selection_mode == "most_congested":
                if path_link_ids:
                    best_link = max(
                        path_link_ids,
                        key=lambda lid: compute_link_utilization(lid, link_status_forward)
                    )

                    util_score = compute_link_utilization(best_link, link_status_forward)
                    link_scores[best_link] = max(link_scores.get(best_link, 0.0), util_score)

            else:
                raise ValueError(f"Unknown link_selection_mode: {link_selection_mode}")

        link_selection_mode = kwargs.get("link_selection_mode", "path_congestion")

        if link_selection_mode == "path_congestion":
            selected_links = [
                lid for lid in selected_links
                if not self._is_link_at_max(lid, link_status_forward)
            ]

            link_scores = {
                lid: score for lid, score in link_scores.items()
                if lid in selected_links
            }

        elif link_selection_mode == "most_congested":
            selected_links = [
                lid for lid in link_scores.keys()
                if not self._is_link_at_max(lid, link_status_forward)
            ]

            link_scores = {
                lid: score for lid, score in link_scores.items()
                if lid in selected_links
            }

        else:
            raise ValueError(f"Unknown link_selection_mode: {link_selection_mode}")

        top_contrib = sorted(
            link_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        print("Top contributing links:")
        for lid, score in top_contrib[:5]:
            print(f"  Link {lid}: score={score:.6f}")


        print(f"[{self.name}] Link selection mode = {link_selection_mode}")
        print(f"[{self.name}] Final selected links = {selected_links}")

        return selected_links, link_scores, top_contrib

    def _is_link_at_max(self, link_id: int, link_status_forward: dict) -> bool:
        """
        Jump-to-max definition:
        - Fibers 0,1,2 are 3-core (multi-core structure)
        - And on each of fibers 0,1,2, lit cores have both C and L enabled
        """
        fibers = link_status_forward.get(link_id)
        if not isinstance(fibers, dict):
            return False

        for fid in (0, 1, 2):
            fiber = fibers.get(fid)
            if not isinstance(fiber, dict) or 0 not in fiber:
                return False

            core_keys = [k for k in fiber.keys() if isinstance(k, int)]
            if len(core_keys) != 3:
                return False

            # since jump_to_max sets ALL 3 cores lit on lit fibers, we can safely check core 0
            core0 = fiber.get(0, {})
            if not core0.get("lit", False):
                return False
            if set(core0.get("bands", [])) != {"C", "L"}:
                return False

        return True

    def get_upgrade_decision(
            self,
            link_id: int,
            link_status_forward: dict,
            link_status_backward: dict
    ) -> List[Dict[str, Any]]:
        """
        Jump-to-max decision:
          - core_upgrade (to 3-core, fiber 0 C-only)
          - band_upgrade on fiber 0
          - new_fiber_CL on fibers 1 and 2
        """

        if self._is_link_at_max(link_id, link_status_forward):
            return []

        return [
            {
                "upgrade_type": "core_upgrade",
                "fiber_id": None,
                "core_type": 1,
            },
            {
                "upgrade_type": "band_upgrade",
                "fiber_id": 0,
                "core_type": 3,
            },
            {
                "upgrade_type": "new_fiber_CL",
                "fiber_id": 1,
                "core_type": 3,
            },
            {
                "upgrade_type": "new_fiber_CL",
                "fiber_id": 2,
                "core_type": 3,
            }
        ]

