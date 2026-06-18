"""
Incremental Upgrade Algorithm (formerly voting_based)

NO-SD-THRESHOLD VARIANT\n\nPath-based upgrade selection:
For each critical SD pair, inspect candidate paths, choose the requested
congestion-ranked path, and upgrade all links on those chosen paths.

path_rank:
    0 -> most congested path
    1 -> second most congested path
    2 -> third most congested path

Upgrades remain incremental:
C -> C+L -> new fiber -> 3-core (applied step-by-step).
"""

from typing import List, Dict, Tuple
from sim.algorithms.base_algorithm import UpgradeAlgorithm
from sim.upgrade.need_for_upgrade import check_need_for_upgrade


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


class IncrementalAlgorithm(UpgradeAlgorithm):
    """
    Incremental upgrade algorithm:
    1. Check if upgrade is needed
    2. Identify critical SD pairs
    3. For each critical SD pair, inspect candidate paths
    4. Choose the congestion-ranked path requested by path_rank
    5. Upgrade links on the union of those selected paths
    """

    def __init__(self):
        super().__init__("increment")

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
        print(f"\n[{self.name}] Checking need for upgrade (incremental)")

        return check_need_for_upgrade(
            current_time=current_time,
            PATHS=PATHS,
            current_traffic=current_traffic,
            current_link_status_forward=current_link_status_forward,
            current_link_status_backward=current_link_status_backward,
            seed=seed
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

        path_stats = kwargs.get("path_stats")
        path_rank = int(kwargs.get("path_rank", 0))
        link_selection_mode = kwargs.get("link_selection_mode", "path_congestion")

        if path_stats is None:
            print("path_stats missing")
            return [], {}, []

        # ============================================================
        # NO SD-THRESHOLD SELECTION
        # Instead of filtering SD pairs using block_threshold,
        # rank all SD pairs with positive blocking probability.
        # This keeps the congestion calculation the same, but removes
        # the SD-pair threshold as requested.
        # ============================================================

        ranked_sd_pairs = sorted(
            [
                (sd, bp)
                for sd, bp in sd_block_prob.items()
                if bp > 0.0
            ],
            key=lambda x: x[1],
            reverse=True
        )

        # Fallback: if all SD-pair blocking values are zero but this
        # function is called, rank all SD pairs anyway.
        if not ranked_sd_pairs and sd_block_prob:
            ranked_sd_pairs = sorted(
                sd_block_prob.items(),
                key=lambda x: x[1],
                reverse=True
            )

        critical_sd_pairs = [
            sd for sd, bp in ranked_sd_pairs
        ]

        if not critical_sd_pairs:
            return [], {}, []

        print("\n" + "=" * 72)
        print(f"[{self.name}] NO SD-THRESHOLD SELECTION ENABLED")
        print(f"[{self.name}] Ignoring SD block_threshold={block_threshold}")
        print(f"[{self.name}] Ranked SD pairs considered = {len(critical_sd_pairs)}")
        print("=" * 72)

        print(f"[{self.name}] Top ranked SD pairs by SD blocking:")
        for sd, bp in ranked_sd_pairs[:10]:
            print(f"  SD {sd}: SD_BP={bp:.6f}")

        if link_selection_mode == "path_congestion":
            print(
                f"\n[{self.name}] Selecting all links on congestion-ranked paths "
                f"(path_rank={path_rank})"
            )
        elif link_selection_mode == "most_congested":
            print(
                f"\n[{self.name}] Selecting highest-utilization link from each "
                f"congestion-ranked path (path_rank={path_rank})"
            )
        else:
            raise ValueError(f"Unknown link_selection_mode: {link_selection_mode}")

        selected_links = []
        link_scores = {}

        for sd in critical_sd_pairs:
            s, d = sd

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
                print(
                    f"[{self.name}] SD pair {sd}: path_rank={path_rank} not available; skipping."
                )
                continue

            best_bp, selected_path, chosen_p_idx = ranked_paths[path_rank]

            print(
                f"[{self.name}] SD pair {sd}: selected path index {chosen_p_idx} "
                f"with path BP={best_bp:.6f}"
            )

            path_link_ids = []

            for i in range(len(selected_path) - 1):
                u = selected_path[i]
                v = selected_path[i + 1]

                lid = LINK_INDEX[u][v]

                if lid != -1:
                    path_link_ids.append(lid)

            if link_selection_mode == "path_congestion":
                for lid in path_link_ids:
                    if self._is_link_at_max(lid, link_status_forward):
                        continue

                    selected_links.append(lid)
                    link_scores[lid] = link_scores.get(lid, 0.0) + max(best_bp, 0.0)

            elif link_selection_mode == "most_congested":
                available_path_links = [
                    lid for lid in path_link_ids
                    if not self._is_link_at_max(lid, link_status_forward)
                ]

                if available_path_links:
                    best_link = max(
                        available_path_links,
                        key=lambda lid: compute_link_utilization(lid, link_status_forward)
                    )

                    util_score = compute_link_utilization(best_link, link_status_forward)
                    selected_links.append(best_link)
                    link_scores[best_link] = max(
                        link_scores.get(best_link, 0.0),
                        util_score
                    )

        selected_links = list(dict.fromkeys(selected_links))

        link_scores = {
            lid: score for lid, score in link_scores.items()
            if lid in selected_links
        }

        top_contrib = sorted(
            link_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        print("Top contributing links:")
        for lid, score in top_contrib[:5]:
            print(f"  Link {lid}: score={score:.6f}")

        print(f"[{self.name}] Link selection mode = {link_selection_mode}")
        print(f"[{self.name}] Path rank = {path_rank}")
        print(f"[{self.name}] Final selected links = {selected_links}")

        return selected_links, link_scores, top_contrib

    def _is_link_at_max(self, link_id: int, link_status_forward: dict) -> bool:
        """
        Max-state definition:
        - Fibers 0,1,2 are 3-core
        - On each of fibers 0,1,2, lit cores have both C and L enabled
        """
        fibers = link_status_forward.get(link_id)

        if not isinstance(fibers, dict):
            return False

        for fid in (0, 1, 2):
            fiber = fibers.get(fid)

            if not isinstance(fiber, dict) or 0 not in fiber:
                return False

            core_keys = [
                k for k in fiber.keys()
                if isinstance(k, int)
            ]

            if len(core_keys) != 3:
                return False

            core0 = fiber.get(0, {})

            if not core0.get("lit", False):
                return False

            if set(core0.get("bands", [])) != {"C", "L"}:
                return False

        return True

