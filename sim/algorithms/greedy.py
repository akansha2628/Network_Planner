"""
Greedy Upgrade Algorithm

A greedy algorithm that selects links with highest utilization
or blocking and upgrades them directly to maximum state.

path_rank:
    0 -> most congested path
    1 -> second most congested path
    2 -> third most congested path
"""

from typing import List, Dict, Any, Tuple
from sim.algorithms.base_algorithm import UpgradeAlgorithm
from sim.upgrade.need_for_upgrade import check_need_for_upgrade
from sim.core.constants import C_BAND_SLOTS, TOTAL_SLOTS


def fiber_mode(fiber_obj: dict) -> str:
    return "single" if isinstance(fiber_obj, dict) and "lit" in fiber_obj else "multi"


def is_fiber_lit(fiber_obj: dict) -> bool:
    if fiber_mode(fiber_obj) == "single":
        return bool(fiber_obj.get("lit", False))

    return any(
        isinstance(core, dict) and core.get("lit", False)
        for core in fiber_obj.values()
    )


def get_current_cores(fiber_obj: dict) -> int:
    if fiber_mode(fiber_obj) == "single":
        return int(fiber_obj.get("cores", 1))

    for core in fiber_obj.values():
        if isinstance(core, dict) and "cores" in core:
            return int(core["cores"])

    return 3


def band_set_of_fiber(fiber_obj: dict) -> set:
    if fiber_mode(fiber_obj) == "single":
        return set(fiber_obj.get("bands", []))

    for core in fiber_obj.values():
        if isinstance(core, dict) and core.get("lit", False):
            return set(core.get("bands", []))

    return set()


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

    if total_slots == 0:
        return 0.0

    return used_slots / total_slots


def choose_upgrade_type_greedy_max(link_id: int, link_status_forward: dict, link_status_backward: dict):
    """
    Greedy behavior:
      - first time on a single-core link:
            [core_upgrade, band_upgrade on first lit fiber]
      - later on 3-core link:
            if any lit fiber is C-only -> band_upgrade that fiber
            else if dark fiber exists -> new_fiber_CL on next dark fiber
            else -> None
    """

    if link_id not in link_status_forward:
        raise ValueError(f"ERROR: link_id {link_id} missing in link_status_forward")

    forward_fibers = link_status_forward[link_id]

    lit_fibers = sorted([
        fid for fid, f in forward_fibers.items()
        if is_fiber_lit(f)
    ])

    dark_fibers = sorted([
        fid for fid, f in forward_fibers.items()
        if not is_fiber_lit(f)
    ])

    if not lit_fibers:
        raise ValueError(f"ERROR: No lit fibers found for link {link_id}")

    first_lit_fid = min(lit_fibers)
    current_cores = get_current_cores(forward_fibers[first_lit_fid])

    if current_cores == 1:
        return [
            {
                "upgrade_type": "core_upgrade",
                "fiber_id": None,
                "core_type": 1
            },
            {
                "upgrade_type": "band_upgrade",
                "fiber_id": first_lit_fid,
                "core_type": 3
            }
        ]

    if current_cores == 3:
        for fid in lit_fibers:
            if band_set_of_fiber(forward_fibers[fid]) == {"C"}:
                return [
                    {
                        "upgrade_type": "band_upgrade",
                        "fiber_id": fid,
                        "core_type": 3
                    }
                ]

        if dark_fibers:
            chosen_fiber = min(dark_fibers)
            return [
                {
                    "upgrade_type": "new_fiber_CL",
                    "fiber_id": chosen_fiber,
                    "core_type": 3
                }
            ]

        return [
            {
                "upgrade_type": None,
                "fiber_id": None,
                "core_type": 3
            }
        ]

    return [
        {
            "upgrade_type": None,
            "fiber_id": None,
            "core_type": current_cores
        }
    ]


class GreedyAlgorithm(UpgradeAlgorithm):
    """
    Greedy algorithm for network upgrades.
    """

    def __init__(self):
        super().__init__("greedy")

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

        print(f"\n[{self.name}] Checking need for upgrade using greedy strategy")

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

        critical_sd_pairs = [
            sd for sd, bp in sd_block_prob.items()
            if bp >= block_threshold
        ]

        if not critical_sd_pairs and sd_block_prob:
            ranked_sd = sorted(
                sd_block_prob.items(),
                key=lambda x: x[1],
                reverse=True
            )

            critical_sd_pairs = [
                sd for sd, bp in ranked_sd[:3]
            ]

        if not critical_sd_pairs:
            return [], {}, []

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


