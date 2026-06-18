import random
from typing import List, Dict, Any
from sim.algorithms.base_algorithm import UpgradeAlgorithm
from sim.upgrade.need_for_upgrade import check_need_for_upgrade
from sim.upgrade.link_selection import select_links_from_congested_paths_per_sd


class RandomAlgorithm(UpgradeAlgorithm):

    def __init__(self):
        super().__init__("random")

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
        print(f"\n[{self.name}] Checking need for upgrade...")
        return check_need_for_upgrade(
            current_time,
            PATHS,
            current_traffic,
            current_link_status_forward,
            current_link_status_backward,
            seed
        )

    def _is_link_at_max(self, link_id: int, link_status_forward: dict) -> bool:
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

            core0 = fiber.get(0, {})
            if not core0.get("lit", False):
                return False
            if set(core0.get("bands", [])) != {"C", "L"}:
                return False

        return True

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

        print(f"\n[{self.name}] Selecting links randomly from congested candidate pool")

        if link_status_forward is None:
            return [], {}, []

        rng = random.Random(int(current_time) + 12345)

        # ------------------------------------------------------------
        # Step 1: build eligible links (not already maxed out)
        # ------------------------------------------------------------
        eligible_links = [
            lid for lid in link_status_forward.keys()
            if not self._is_link_at_max(lid, link_status_forward)
        ]

        if not eligible_links:
            print(f"[{self.name}] No eligible links available")
            return [], {}, []

        # ------------------------------------------------------------
        # Step 2: identify critical SD pairs
        # Same spirit as incremental, but random will only use this
        # to build a candidate pool, not as final deterministic output.
        # ------------------------------------------------------------
        # ============================================================
        # Step 2: identify critical SD pairs WITHOUT SD threshold
        # Rank all SD pairs with positive blocking probability.
        # Random still uses this only to build a candidate pool.
        # ============================================================
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

        print(f"[{self.name}] NO SD-THRESHOLD SELECTION ENABLED for candidate pool")
        print(f"[{self.name}] Ignoring block_threshold={block_threshold}")
        print(f"[{self.name}] Ranked SD pairs considered = {len(critical_sd_pairs)}")
        for sd, bp in ranked_sd_pairs[:10]:
            print(f"  SD {sd}: SD_BP={bp:.6f}")

        path_stats = kwargs.get("path_stats")

        candidate_links = []

        # ------------------------------------------------------------
        # Step 3: use congested-path selector to form candidate pool
        # ------------------------------------------------------------
        if critical_sd_pairs and path_stats is not None and LINK_INDEX is not None:
            try:
                candidate_links = select_links_from_congested_paths_per_sd(
                    critical_sd_pairs,
                    PATHS,
                    path_stats,
                    LINK_INDEX
                )
            except Exception as e:
                print(f"[{self.name}] Candidate pool generation failed: {e}")
                candidate_links = []

        # keep only eligible links
        candidate_links = [
            lid for lid in candidate_links
            if lid in eligible_links
        ]

        # remove duplicates while preserving order
        candidate_links = list(dict.fromkeys(candidate_links))

        # ------------------------------------------------------------
        # Step 4: fallback if candidate pool is empty
        # ------------------------------------------------------------
        if not candidate_links:
            print(f"[{self.name}] No congested candidate pool found, falling back to eligible links")
            candidate_links = eligible_links[:]

        # ------------------------------------------------------------
        # Step 5: limit pool size so random remains random-but-focused
        # ------------------------------------------------------------
        # DT12-friendly setting:
        #   pool = top 8 candidate links at most
        pool_size = min(8, len(candidate_links))
        candidate_pool = candidate_links[:pool_size]

        if not candidate_pool:
            return [], {}, []

        # ------------------------------------------------------------
        # Step 6: randomly sample from the candidate pool
        # DT12-friendly setting:
        #   choose up to 4 links from pool
        # ------------------------------------------------------------
        # # num_links = min(4, len(candidate_pool))
        # num_links = rng.randint(1, min(4, len(candidate_pool)))
        # selected_links = rng.sample(candidate_pool, num_links)

        max_links = min(4, len(candidate_pool))
        num_links = rng.randint(1, max_links)
        selected_links = rng.sample(candidate_pool, num_links)

        # simple scores for compatibility with downstream sorting
        link_scores = {lid: 1.0 for lid in selected_links}
        top_contrib = [(lid, 1.0) for lid in selected_links]

        print(f"[{self.name}] Eligible links: {eligible_links}")
        print(f"[{self.name}] Candidate pool: {candidate_pool}")
        print(f"[{self.name}] Selected {num_links} links: {selected_links}")

        return selected_links, link_scores, top_contrib

    def get_random_upgrade_decision(self, link_id, link_status_forward):
        """
        Random technology selection for Random baseline.

        This does NOT follow the normal progression order.
        It randomly chooses among physically valid upgrade options:
          - band_upgrade on any lit C-only fiber
          - new_fiber_C on any dark fiber
          - new_fiber_CL on any dark fiber
          - core_upgrade only if current link is still single-core

        It prevents repeated core upgrades.
        """

        import random

        fibers = link_status_forward.get(link_id, {})
        if not isinstance(fibers, dict) or not fibers:
            return []

        def is_single_core_fiber(f):
            return isinstance(f, dict) and "lit" in f

        def is_fiber_lit(f):
            if is_single_core_fiber(f):
                return bool(f.get("lit", False))
            return any(
                isinstance(core, dict) and core.get("lit", False)
                for core in f.values()
            )

        def get_core_type(f):
            if is_single_core_fiber(f):
                return int(f.get("cores", 1))
            for core in f.values():
                if isinstance(core, dict) and "cores" in core:
                    return int(core["cores"])
            return 3

        def get_band_set(f):
            if is_single_core_fiber(f):
                return set(f.get("bands", []))
            for core in f.values():
                if isinstance(core, dict) and core.get("lit", False):
                    return set(core.get("bands", []))
            return set()

        lit_fibers = sorted([
            fid for fid, f in fibers.items()
            if is_fiber_lit(f)
        ])

        dark_fibers = sorted([
            fid for fid, f in fibers.items()
            if not is_fiber_lit(f)
        ])

        if not lit_fibers:
            return []

        current_cores = get_core_type(fibers[lit_fibers[0]])

        valid_options = []

        # 1) Band upgrade on any lit C-only fiber
        for fid in lit_fibers:
            if get_band_set(fibers[fid]) == {"C"}:
                valid_options.append({
                    "upgrade_type": "band_upgrade",
                    "fiber_id": fid,
                    "core_type": current_cores
                })

        # 2) Light any dark fiber with C
        for fid in dark_fibers:
            valid_options.append({
                "upgrade_type": "new_fiber_C",
                "fiber_id": fid,
                "core_type": current_cores
            })

        # 3) Light any dark fiber directly with C+L
        for fid in dark_fibers:
            valid_options.append({
                "upgrade_type": "new_fiber_CL",
                "fiber_id": fid,
                "core_type": current_cores
            })

        # 4) Core upgrade only if current link is single-core
        # This prevents repeated core_upgrade.
        if current_cores == 1:
            valid_options.append({
                "upgrade_type": "core_upgrade",
                "fiber_id": None,
                "core_type": 1
            })

        if not valid_options:
            return []

        chosen = random.choice(valid_options)

        print(
            f"[{self.name}] Link {link_id} → Selected upgrade: "
            f"{chosen['upgrade_type']} "
            f"(fiber={chosen['fiber_id']}, core={chosen['core_type']})"
        )

        return [chosen]

