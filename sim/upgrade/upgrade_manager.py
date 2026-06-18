import random
from sim.core.constants import *
from sim.routing.rsa import *
import math
import copy
from collections import defaultdict


def initialize_link_status(num_links=None):
    """
    Initializes forward and backward link status.
    First fiber in each link bundle is lit, single-core, C-band.
    Others are dark.
    
    Args:
        num_links: Number of links in the network. If None, uses LINKS from Topology module.
    """
    if num_links is None:
        from sim.core import topology as Topology
        num_links = Topology.LINKS

    link_status_forward = {}
    link_status_backward = {}

    for linkid in range(num_links):
        fibers = {}
        for fiber_id in range(FIBERS):
            if fiber_id == 0:
                fibers[fiber_id] = {
                    "lit": True,
                    "cores": 1,
                    "bands": ["C"],
                    "slots": [0 for _ in range(C_BAND_SLOTS)]
                }
            else:
                fibers[fiber_id] = {
                    "lit": False,
                    "cores": 1,
                    "bands": [],
                    "slots": []
                }

        link_status_forward[linkid] = fibers
        link_status_backward[linkid] = copy.deepcopy(fibers)

    print("Link status initialization complete.")
    return link_status_forward, link_status_backward


def initialize_link_status_max_state(num_links=None):
    """
    VARIATION 1: Initialize all links at MAXIMUM state.
    - All fibers are lit (default 3 fibers, or specify num_fibers)
    - Each fiber has 3 cores
    - Each core has C+L bands

    Args:
        num_links: Number of links in the network
        num_fibers: Number of fibers to initialize (default 3)
    """
    if num_links is None:
        from sim.core import topology as Topology
        num_links = Topology.LINKS

    link_status_forward = {}
    link_status_backward = {}

    for linkid in range(num_links):
        fibers = {}
        for fiber_id in range(FIBERS):
            # All fibers are lit with 3 cores, C+L bands
            fibers[fiber_id] = {
                0: {
                    "lit": True,
                    "cores": 3,
                    "bands": ["C", "L"],
                    "slots": [0 for _ in range(TOTAL_SLOTS)]
                },
                1: {
                    "lit": True,
                    "cores": 3,
                    "bands": ["C", "L"],
                    "slots": [0 for _ in range(TOTAL_SLOTS)]
                },
                2: {
                    "lit": True,
                    "cores": 3,
                    "bands": ["C", "L"],
                    "slots": [0 for _ in range(TOTAL_SLOTS)]
                }
            }

        link_status_forward[linkid] = fibers
        link_status_backward[linkid] = copy.deepcopy(fibers)

    print(f"Link status initialization complete (MAX STATE - {FIBERS} fibers, 3 cores, C+L).")
    return link_status_forward, link_status_backward

def free_connection_spectrum(conn, link_status_forward: dict, link_status_backward: dict):
    """
    Frees the spectrum occupied by `conn` in both directions, handling BOTH data layouts:

    Single-core fiber layout:
      link_status[link_id][fiber_id] = {"slots": [...], "cores": 1, ...}

    3-core fiber layout:
      link_status[link_id][fiber_id][core_id] = {"slots": [...], "cores": 3, ...}

    Assumes conn has:
      - conn.path
      - conn.link_ids
      - conn.fibers_used
      - conn.cores_used
      - conn.fs_index
      - conn.slice_window_size
    """

    fs0 = conn.fs_index
    fs1 = conn.fs_index + conn.slice_window_size  # end (exclusive)

    for i in range(len(conn.path) - 1):
        src_depart = conn.path[i]
        dest_depart = conn.path[i + 1]

        link_id = conn.link_ids[i]
        fiber_id = conn.fibers_used[i]
        core_id = conn.cores_used[i]

        # Choose direction dict
        status = link_status_forward if src_depart < dest_depart else link_status_backward

        if link_id not in status:
            continue
        if fiber_id not in status[link_id]:
            continue

        fiber_obj = status[link_id][fiber_id]

        # ✅ single-core: has top-level "slots"
        if isinstance(fiber_obj, dict) and "slots" in fiber_obj:
            slots = fiber_obj["slots"]
            for s in range(fs0, fs1):
                slots[s] = 0

        # ✅ 3-core: fiber_obj is dict keyed by core_id
        else:
            if core_id not in fiber_obj:
                continue
            slots = fiber_obj[core_id]["slots"]
            for s in range(fs0, fs1):
                slots[s] = 0


def node_path_to_link_ids(node_path, LINK_INDEX):
    """
    node_path like [0, 2, 5, 1]  ->  [Topology.LINK_INDEX[0][2], Topology.LINK_INDEX[2][5], Topology.LINK_INDEX[5][1]]
    """
    if not node_path or len(node_path) < 2:
        return []
    from sim.core import topology as Topology

    link_ids = []
    for i in range(len(node_path) - 1):
        u, v = node_path[i], node_path[i + 1]
        lid = Topology.LINK_INDEX[u][v]
        if lid == -1:
            # Path invalid for this topology (should not happen if PATHS computed correctly)
            return []
        link_ids.append(lid)
    return link_ids

def network_connectivity(candidate_links: list,
                         working_topology: list,
                         LINK_INDEX: list):
    """
    Multi-stage network connectivity verification for upgrade planning.

    Steps:
      1. Individual-link safety test (Prim/DFS style).
      2. Remove all individually-safe links together.
      3. If disconnected → identify isolated nodes and restore the
         lowest-utilization link touching each isolated node.
      4. If still disconnected → identify remaining critical links.
      5. Return the final list of links that CAN be safely removed (upgraded).
    """

    n = len(working_topology)

    # ------------------------------------------------------
    # STEP 1: INDIVIDUAL LINK SAFETY CHECK
    # ------------------------------------------------------
    safe_links = []

    for link in candidate_links:

        # Find (u, v) of this link
        u = v = None
        for i in range(n):
            for j in range(n):
                if LINK_INDEX[i][j] == link:
                    u, v = i, j
                    break
            if u is not None:
                break


        temp_topo_single = copy.deepcopy(working_topology)

        # Remove this link
        temp_topo_single[u][v] = 0
        temp_topo_single[v][u] = 0

        # Test connectivity
        if is_network_connected(temp_topo_single):
            safe_links.append(link)
        else:
            print(f"[INDIVIDUAL] Link {link} is NOT safe to remove.")

    # If no link is safe individually → no upgrades this period
    # if not safe_links:
    #     print("No individual links are safe to remove -> no upgrades possible this period.")
    #     return []
    return safe_links


def is_network_connected(topology: list) -> bool:
    """
    Checks if the network remains connected using a Prim-style traversal.
    """

    n = len(topology)
    visited = [False] * n

    start = next((i for i in range(n) if sum(topology[i]) > 0), None)
    if start is None:
        print("ERROR: entire topology is disconnected!")
        return False

    visited[start] = True
    num_visited = 1

    while True:
        min_edge = (None, None, math.inf)
        for u in range(n):
            if visited[u]:
                for v in range(n):
                    if not visited[v] and topology[u][v] > 0:
                        if topology[u][v] < min_edge[2]:
                            min_edge = (u, v, topology[u][v])

        u, v, w = min_edge
        if v is not None:
            visited[v] = True
            num_visited += 1
        else:
            break

    return num_visited == n

from collections import Counter, defaultdict

def print_upgrade_decisions_summary(day: float, safe_links: list, upgrade_decisions: dict):
    """
    upgrade_decisions[link_id] = list[dict] where dict has keys:
      - upgrade_type: str
      - fiber_id: int or None
      - core_type: int (1 or 3) representing current core structure before applying this decision
    """
    print("\n" + "-"*70)
    print(f"UPGRADE SUMMARY @ day {day:.1f}")
    print("-"*70)

    type_counter = Counter()
    type_to_links = defaultdict(list)

    for link_id in safe_links:
        decs = upgrade_decisions.get(link_id, [])
        if isinstance(decs, dict):
            decs = [decs]

        for dec in decs:
            u = dec.get("upgrade_type")
            fid = dec.get("fiber_id", None)
            ct = dec.get("core_type", None)

            type_counter[u] += 1
            type_to_links[u].append((link_id, fid, ct))

    # Print per-link details
    for u, items in type_to_links.items():
        print(f"\n• {u}: {len(items)} action(s)")
        for (link_id, fid, ct) in items:
            print(f"   - link {link_id}, fiber {fid}, core_type ={ct}")

    # Aggregate counts you asked for
    print("\nCounts:")
    for k in ["band_upgrade", "new_fiber_C","new_fiber_CL", "core_upgrade"]:
        print(f"  {k}: {type_counter.get(k, 0)}")

    print("-"*70 + "\n")

def fiber_mode(fiber_obj: dict) -> str:
    """
    Returns: "single" or "multi"
    """
    return "single" if isinstance(fiber_obj, dict) and "lit" in fiber_obj else "multi"


def is_fiber_lit(fiber_obj: dict) -> bool:
    """
    Single-core: fiber_obj["lit"]
    Multi-core: any(core["lit"] for core in fiber_obj.values())
    """
    if fiber_mode(fiber_obj) == "single":
        return bool(fiber_obj.get("lit", False))

    # multi-core
    return any(
        isinstance(core, dict) and core.get("lit", False)
        for core in fiber_obj.values()
    )


def get_current_cores(fiber_obj: dict) -> int:
    """
    Single-core: fiber_obj["cores"]
    Multi-core: fiber_obj[0]["cores"] (or first core that has "cores")
    """
    if fiber_mode(fiber_obj) == "single":
        return int(fiber_obj.get("cores", 1))

    for core in fiber_obj.values():
        if isinstance(core, dict) and "cores" in core:
            return int(core["cores"])
    return 3  # fallback


def band_set_of_fiber(fiber_obj: dict) -> set:
    """
    Single-core: set(fiber_obj["bands"])
    Multi-core: set(bands of a lit core) else empty set
    """
    if fiber_mode(fiber_obj) == "single":
        return set(fiber_obj.get("bands", []))

    # multi-core: take bands from a lit core
    for core in fiber_obj.values():
        if isinstance(core, dict) and core.get("lit", False):
            return set(core.get("bands", []))
    return set()


def choose_upgrade_type(link_id: int, link_status_forward: dict, link_status_backward: dict, algorithm:str):
    """
    SIMPLE RULES (as you stated):

    SINGLE-CORE link:
      1) If any lit fiber is C-only -> band_upgrade (pick smallest fiber id)
      2) Else if there is any dark fiber -> new_fiber_C (pick smallest dark fiber id)
      3) Else (all fibers are lit AND all lit are C+L) -> core_upgrade

    THREE-CORE link:
      1) If any lit fiber is C-only -> band_upgrade on that fiber (all 3 cores)
      2) Else if there is any dark fiber -> new_fiber_C on that fiber (all 3 cores)
      3) Else (no dark fibers left; lit ones are already C+L) -> no further upgrade
    """
    if link_id not in link_status_forward:
        raise ValueError(f"choose_upgrade_type(): link_id {link_id} missing in link_status_forward")

    forward_fibers = link_status_forward[link_id]

    lit_fibers = sorted([fid for fid, f in forward_fibers.items() if is_fiber_lit(f)])
    dark_fibers = sorted([fid for fid, f in forward_fibers.items() if not is_fiber_lit(f)])

    if not lit_fibers:
        raise ValueError(f"choose_upgrade_type(): No lit fibers found for link {link_id}")

    # Determine current core mode from a lit fiber
    first_lit_fid = lit_fibers[0]
    current_cores = get_current_cores(forward_fibers[first_lit_fid])

    # --------------------------
    # SINGLE-CORE MODE
    # --------------------------
    if current_cores == 1:
        # 1) band upgrade if any lit fiber is C-only
        c_only_lit = [fid for fid in lit_fibers if band_set_of_fiber(forward_fibers[fid]) == {"C"}]
        if c_only_lit:
            return {"upgrade_type": "band_upgrade", "fiber_id": c_only_lit[0], "core_type": 1}

        # Check if all currently-lit fibers are already C+L
        all_lit_are_cl = all(band_set_of_fiber(forward_fibers[fid]) == {"C", "L"} for fid in lit_fibers)

        # if algorithm.name == "greedy"
        # 2) if any dark fiber exists, light a new one with C (this matches your rule)
        if dark_fibers:
            return {"upgrade_type": "new_fiber_C", "fiber_id": dark_fibers[0], "core_type": 1}

        # 3) if no dark fibers left AND all lit are C+L -> core upgrade
        if all_lit_are_cl:
            return {"upgrade_type": "core_upgrade", "fiber_id": None, "core_type": 1}

        return {"upgrade_type": None, "fiber_id": None, "core_type": 1}

    # --------------------------
    # THREE-CORE MODE
    # --------------------------
    if current_cores == 3:
        # 1) band upgrade if any lit fiber is C-only
        c_only_lit = [fid for fid in lit_fibers if band_set_of_fiber(forward_fibers[fid]) == {"C"}]
        if c_only_lit:
            return {"upgrade_type": "band_upgrade", "fiber_id": c_only_lit[0], "core_type": 3}

        if algorithm.name == "greedy":
         # 2) light a new fiber with C if possible
          if dark_fibers:
            return {"upgrade_type": "new_fiber_CL", "fiber_id": dark_fibers[0], "core_type": 3}

        else:
         if dark_fibers:
            return {"upgrade_type": "new_fiber_C", "fiber_id": dark_fibers[0], "core_type": 3}

        # 3) no more upgrades
        return {"upgrade_type": None, "fiber_id": None, "core_type": 3}

    raise ValueError(f"choose_upgrade_type(): Unexpected core count {current_cores} on link {link_id}")

def choose_upgrade_type_plan_checker(
        link_id: int,
        link_status_forward: dict,
        link_status_backward: dict,
        algorithm,
        round_idx: int
):
    """
    Plan-checker-specific selector.

    Rule:
    - Round 1 after initial failure: use normal staged logic.
    - Round 2 onward:
        * if next action is new_fiber_C -> escalate to new_fiber_CL
        * if next action is band_upgrade -> escalate to new_fiber_CL
    This uses the same fiber_id.

    Why:
    - new_fiber_CL is immediate in your model
    - band_upgrade has downtime
    - after new_fiber_C, the next normal step is often band_upgrade
    """

    normal_upg = choose_upgrade_type(
        link_id=link_id,
        link_status_forward=link_status_forward,
        link_status_backward=link_status_backward,
        algorithm=algorithm
    )

    if normal_upg is None or normal_upg.get("upgrade_type") is None:
        return normal_upg

    # Keep first progressive round exactly as normal
    if round_idx == 1:
        return normal_upg

    upg_type = normal_upg.get("upgrade_type")

    # From round 2 onward, strengthen weak / downtime-causing next steps
    if upg_type in ("new_fiber_C", "band_upgrade"):
        print(f"[PLAN_CHECKER] Escalating link {link_id}: {upg_type} -> new_fiber_CL")
        return {
            "upgrade_type": "new_fiber_CL",
            "fiber_id": normal_upg.get("fiber_id"),
            "core_type": normal_upg.get("core_type")
        }

    return normal_upg

def preprocess_upgrade_decisions(upgrade_decisions: dict) -> dict:
    """
    If a link has multiple upgrade steps and the LAST step is core_upgrade,
    then any earlier band_upgrade/new_fiber_* steps in that same list are wasted
    (core_upgrade rebuilds the structure). So we keep ONLY the last core_upgrade.
    """
    cleaned = {}

    for link_id, decs in upgrade_decisions.items():
        # normalize to list
        dec_list = [decs] if isinstance(decs, dict) else list(decs)

        if dec_list and dec_list[-1].get("upgrade_type") == "core_upgrade":
            cleaned[link_id] = [dec_list[-1]]  # keep only core_upgrade
        else:
            cleaned[link_id] = dec_list

    return cleaned


def perform_upgrade(
        safe_links: list,
        upgrade_decisions: dict,
        link_status_forward: dict,
        link_status_backward: dict,
        C_BAND_SLOTS: int,
        TOTAL_SLOTS: int,
        current_time: float,
        algorithm: str
):
    """
    Applies upgrade decisions for each link in safe_links.

    - upgrade_decisions[link_id] can be a dict (single step) or list of dicts (multi-step).
    - If a multi-step ends with core_upgrade, we automatically drop earlier steps for that link.
    - Supports both single-core and 3-core fiber representations.

    Decision format:
      {"upgrade_type": "...", "fiber_id": int|None, "core_type": 1|3}
    where core_type is the CURRENT core mode before applying that decision.
    """

    # 1) Drop wasted steps (band/new_fiber) if core_upgrade is the final step for that link
    upgrade_decisions = preprocess_upgrade_decisions(upgrade_decisions)

    def ensure_three_core_fiber_shape(fiber_dicts: dict, fid: int):
        """
        If fiber_dicts[fid] is unexpectedly still single-core-shaped, convert that fiber to 3-core-shaped.
        (All cores initially unlit, no bands, empty slots.)
        """
        f = fiber_dicts[fid]
        if isinstance(f, dict) and "slots" in f:
            fiber_dicts[fid] = {}
            for c in range(Num_cores):  # expected 3
                fiber_dicts[fid][c] = {"lit": False, "cores": 3, "bands": [], "slots": []}

    def apply_one_decision(link_id: int, fiber_dicts: dict, dec: dict):
        upgrade_type = dec.get("upgrade_type")
        fiber_id = dec.get("fiber_id")

        if upgrade_type is None:
            raise ValueError(f"perform_upgrade(): upgrade_type=None for link {link_id}: {dec}")

        if "core_type" not in dec:
            raise ValueError(f"perform_upgrade(): missing core_type for link {link_id}: {dec}")
        num_cores = dec["core_type"]

        if upgrade_type in ("band_upgrade", "new_fiber_C", "new_fiber_CL"):
            if fiber_id is None:
                raise ValueError(f"perform_upgrade(): {upgrade_type} missing fiber_id for link {link_id}: {dec}")
            if fiber_id not in fiber_dicts:
                raise ValueError(f"perform_upgrade(): fiber_id {fiber_id} not in link {link_id} fibers")

        if upgrade_type == "core_upgrade" and num_cores != 1:
            raise ValueError(f"core_upgrade expects current core_type=1, got {num_cores} for link {link_id}")

        # -------------------------
        # SINGLE-CORE MODE
        # -------------------------
        if num_cores == 1:
            if upgrade_type == "band_upgrade":
                slots = fiber_dicts[fiber_id]["slots"]
                if len(slots) < TOTAL_SLOTS:
                    slots.extend([0] * (TOTAL_SLOTS - len(slots)))
                fiber_dicts[fiber_id]["bands"] = ["C", "L"]
                return

            if upgrade_type == "new_fiber_C":
                f = fiber_dicts[fiber_id]
                if not f.get("lit", False):
                    f["lit"] = True
                    f["cores"] = 1
                    f["bands"] = ["C"]
                    f["slots"] = [0] * C_BAND_SLOTS
                return

            if upgrade_type == "new_fiber_CL":
                f = fiber_dicts[fiber_id]
                if not f.get("lit", False):
                    f["lit"] = True
                    f["cores"] = 1
                    f["bands"] = ["C", "L"]
                    f["slots"] = [0] * TOTAL_SLOTS
                return

            if upgrade_type == "core_upgrade":
                # Convert ALL fibers to 3-core structure; only fiber 0 lit initially with C-band.
                fiber_ids = list(fiber_dicts.keys())

                if algorithm.name == "greedy":
                    for fid in fiber_ids:
                        is_lit = (fid == 0)
                        fiber_dicts[fid] = {}
                        for c in range(Num_cores):
                            fiber_dicts[fid][c] = {
                                "lit": is_lit,
                                "cores": 3,
                                "bands": ["C"] if is_lit else [],
                                "slots": ([0] * C_BAND_SLOTS) if is_lit else []
                            }
                    return

                # if algorithm.name == "greedy":
                #    # fiber_ids = list(fiber_dicts.keys())
                #    for fid in fiber_ids:
                #         is_lit = (fid == 0)
                #         fiber_dicts[fid] = {}
                #         for c in range(Num_cores):  # expected 3
                #             fiber_dicts[fid][c] = {
                #             "lit": is_lit,
                #             "cores": 3,
                #             "bands": ["C", "L"] if is_lit else [],  # Changed: C+L instead of just C
                #             "slots": ([0] * TOTAL_SLOTS) if is_lit else []  # Changed: TOTAL_SLOTS for C+L
                #             }
                #    return

                elif algorithm.name == "jump_to_max":
                    # Jump-to-max:
                    # first convert to 3-core structure,
                    # keep only fiber 0 lit with C-band,
                    # then later decisions do:
                    #   band_upgrade on fiber 0
                    #   new_fiber_CL on fibers 1 and 2
                    for fid in fiber_ids:
                        is_lit = (fid == 0)
                        fiber_dicts[fid] = {}
                        for c in range(Num_cores):
                            fiber_dicts[fid][c] = {
                                "lit": is_lit,
                                "cores": 3,
                                "bands": ["C"] if is_lit else [],
                                "slots": ([0] * C_BAND_SLOTS) if is_lit else []
                            }
                    return
                # elif algorithm.name == "jump_to_max":
                #     # Jump-to-max: Light FIRST 3 FIBERS (0, 1, 2) with C+L, 3 cores each
                #     for fid in fiber_ids:
                #         is_lit = (fid < 3)  # Fibers 0, 1, 2 are lit
                #         fiber_dicts[fid] = {}
                #         for c in range(Num_cores):  # expected 3
                #             fiber_dicts[fid][c] = {
                #                 "lit": is_lit,
                #                 "cores": 3,
                #                 "bands": ["C", "L"] if is_lit else [],
                #                 "slots": ([0] * TOTAL_SLOTS) if is_lit else []
                #             }
                #     return
                else:
                 fiber_ids = list(fiber_dicts.keys())
                 for fid in fiber_ids:
                    is_lit = (fid == 0)
                    fiber_dicts[fid] = {}
                    for c in range(Num_cores):  # expected 3
                        fiber_dicts[fid][c] = {
                            "lit": is_lit,
                            "cores": 3,
                            "bands": ["C"] if is_lit else [],
                            "slots": ([0] * C_BAND_SLOTS) if is_lit else []
                        }
                 return

            raise ValueError(f"perform_upgrade(): invalid upgrade_type '{upgrade_type}' for core_type=1 (link {link_id})")

        # -------------------------
        # THREE-CORE MODE
        # -------------------------
        if num_cores == 3:
            if upgrade_type == "band_upgrade":
                ensure_three_core_fiber_shape(fiber_dicts, fiber_id)
                for c in range(Num_cores):
                    core_info = fiber_dicts[fiber_id][c]
                    slots = core_info["slots"]
                    if len(slots) < TOTAL_SLOTS:
                        slots.extend([0] * (TOTAL_SLOTS - len(slots)))
                    core_info["bands"] = ["C", "L"]
                return

            if upgrade_type == "new_fiber_C":
                ensure_three_core_fiber_shape(fiber_dicts, fiber_id)
                for c in range(Num_cores):
                    core_info = fiber_dicts[fiber_id][c]
                    core_info["lit"] = True
                    core_info["cores"] = 3
                    core_info["bands"] = ["C"]
                    core_info["slots"] = [0] * C_BAND_SLOTS
                return

            if upgrade_type == "new_fiber_CL":
                ensure_three_core_fiber_shape(fiber_dicts, fiber_id)
                for c in range(Num_cores):
                    core_info = fiber_dicts[fiber_id][c]
                    core_info["lit"] = True
                    core_info["cores"] = 3
                    core_info["bands"] = ["C", "L"]
                    core_info["slots"] = [0] * TOTAL_SLOTS
                return

            raise ValueError(f"perform_upgrade(): invalid upgrade_type '{upgrade_type}' for core_type=3 (link {link_id})")

        raise ValueError(f"perform_upgrade(): unexpected core_type={num_cores} for link {link_id}")

    # ============================================================
    # MAIN LOOP: apply decisions to forward and backward directions
    # ============================================================
    for link_id in safe_links:
        if link_id not in upgrade_decisions:
            raise ValueError(f"perform_upgrade(): missing decision for link {link_id}")

        decs = upgrade_decisions[link_id]
        decisions = [decs] if isinstance(decs, dict) else list(decs)

        for dec in decisions:
            apply_one_decision(link_id, link_status_forward[link_id], dec)
            apply_one_decision(link_id, link_status_backward[link_id], dec)


def reset_all_slots_empty(link_status_forward: dict, link_status_backward: dict):
        """
        Keep CURRENT structure (lit/bands/cores) but clear spectrum occupancy (set all slots to 0).
        Works for both single-core and 3-core representations.
        """

        def reset_direction(ls: dict):
            for link_id, fibers in ls.items():
                for fiber_id, fiber in fibers.items():

                    # Case A: fiber is dict with "slots"
                    if isinstance(fiber, dict) and "slots" in fiber:
                        slots = fiber.get("slots", [])
                        if not slots:
                            continue

                        # slots = [0,0,...]
                        if isinstance(slots[0], (int, float)):
                            for i in range(len(slots)):
                                slots[i] = 0
                        # slots = [[...],[...],[...]]
                        else:
                            for c in range(len(slots)):
                                for i in range(len(slots[c])):
                                    slots[c][i] = 0

                    # Case B: nested dict fiber[core_id]["slots"]
                    else:
                        for core_id, core_struct in fiber.items():
                            slots = core_struct.get("slots", [])
                            for i in range(len(slots)):
                                slots[i] = 0

        reset_direction(link_status_forward)
        reset_direction(link_status_backward)
        return link_status_forward, link_status_backward

def perform_upgrade_second(
        link_id: int,
        upgrade_decisions: dict,
        link_status_forward: dict,
        link_status_backward: dict,
        C_BAND_SLOTS: int,
        TOTAL_SLOTS: int,
        current_time: float,
        algorithm: str
):
    if link_id not in upgrade_decisions:
        raise ValueError(f"perform_upgrade_second(): missing decision for link {link_id}")

    dec = upgrade_decisions[link_id]
    last_dec = dec if isinstance(dec, dict) else dec[-1]

    upgrade_type = last_dec.get("upgrade_type")
    fiber_id = last_dec.get("fiber_id")

    if upgrade_type is None:
        raise ValueError(f"perform_upgrade_second(): upgrade_type=None for link {link_id}")

    if "core_type" not in last_dec:
        raise ValueError(f"perform_upgrade_second(): missing core_type for link {link_id}: {last_dec}")
    num_cores = last_dec["core_type"]

    if upgrade_type in ("band_upgrade", "new_fiber_C", "new_fiber_CL") and fiber_id is None:
        raise ValueError(f"perform_upgrade_second(): {upgrade_type} requires fiber_id for link {link_id}: {last_dec}")

    if upgrade_type == "core_upgrade" and num_cores != 1:
        raise ValueError(f"core_upgrade expects current core_type=1, got {num_cores} for link {link_id}")

    def ensure_three_core_fiber_shape(fiber_dicts: dict, fid: int):
        """
        If fiber_dicts[fid] is single-core-shaped, convert it to 3-core-shaped.

        CONDITION ENFORCED:
          - Only fiber 0 may be lit by default (all 3 cores lit).
          - All other fibers are unlit (all 3 cores unlit).
        """
        f = fiber_dicts[fid]
        if isinstance(f, dict) and "slots" in f:
            make_lit = (fid == 0 and bool(f.get("lit", False)))

            # Bands/slots only meaningful if lit (fiber 0)
            bands = list(f.get("bands", [])) if make_lit else []
            slots = list(f.get("slots", [])) if make_lit else []

            fiber_dicts[fid] = {}
            for c in range(Num_cores):  # expected 3
                fiber_dicts[fid][c] = {
                    "lit": make_lit,
                    "cores": 3,
                    "bands": bands.copy(),
                    "slots": slots.copy()
                }

    def apply_upgrade(fiber_dicts: dict):
        # =========================
        # SINGLE-CORE MODE
        # =========================
        if num_cores == 1:
            if upgrade_type == "band_upgrade":
                old_slots = fiber_dicts[fiber_id]["slots"]
                if len(old_slots) < TOTAL_SLOTS:
                    old_slots.extend([0] * (TOTAL_SLOTS - len(old_slots)))
                fiber_dicts[fiber_id]["bands"] = ["C", "L"]
                return

            if upgrade_type == "new_fiber_C":
                f = fiber_dicts[fiber_id]
                if not f.get("lit", False):
                    f["lit"] = True
                    f["cores"] = 1
                    f["bands"] = ["C"]
                    f["slots"] = [0] * C_BAND_SLOTS
                return

            # if upgrade_type == "new_fiber_CL":
            #     f = fiber_dicts[fiber_id]
            #     if not f.get("lit", False):
            #         f["lit"] = True
            #         f["cores"] = 1
            #         f["bands"] = ["C", "L"]
            #         f["slots"] = [0] * TOTAL_SLOTS
            #     return
            if upgrade_type == "new_fiber_CL":
                f = fiber_dicts[fiber_id]

                # Case A: dark fiber -> light with C+L
                if not f.get("lit", False):
                    f["lit"] = True
                    f["cores"] = 1
                    f["bands"] = ["C", "L"]
                    f["slots"] = [0] * TOTAL_SLOTS
                    return

                # Case B: already lit fiber (likely from new_fiber_C) -> upgrade it to C+L
                bands = set(f.get("bands", []))
                bands.update(["C", "L"])
                f["bands"] = ["C", "L"]  # canonical order

                slots = f.get("slots", [])
                if len(slots) < TOTAL_SLOTS:
                    slots.extend([0] * (TOTAL_SLOTS - len(slots)))  # preserve existing allocations
                f["slots"] = slots

                # lit stays True, cores stays 1
                return

            if upgrade_type == "core_upgrade":
                # Convert ALL fibers to 3-core structure; only fiber 0 lit initially with C-band.
                fiber_ids = list(fiber_dicts.keys())

                if algorithm.name == "greedy":
                    for fid in fiber_ids:
                        is_lit = (fid == 0)
                        fiber_dicts[fid] = {}
                        for c in range(Num_cores):
                            fiber_dicts[fid][c] = {
                                "lit": is_lit,
                                "cores": 3,
                                "bands": ["C"] if is_lit else [],
                                "slots": ([0] * C_BAND_SLOTS) if is_lit else []
                            }
                    return

                # if algorithm.name == "greedy":
                #     fiber_ids = list(fiber_dicts.keys())
                #     for fid in fiber_ids:
                #         is_lit = (fid == 0)
                #         fiber_dicts[fid] = {}
                #         for c in range(Num_cores):  # expected 3
                #             fiber_dicts[fid][c] = {
                #                 "lit": is_lit,
                #                 "cores": 3,
                #                 "bands": ["C", "L"] if is_lit else [],  # Changed: C+L instead of just C
                #                 "slots": ([0] * TOTAL_SLOTS) if is_lit else []  # Changed: TOTAL_SLOTS for C+L
                #             }
                #     return
                elif algorithm.name == "jump_to_max":
                    # Jump-to-max:
                    # first convert to 3-core structure,
                    # keep only fiber 0 lit with C-band,
                    # then later decisions do:
                    #   band_upgrade on fiber 0
                    #   new_fiber_CL on fibers 1 and 2
                    for fid in fiber_ids:
                        is_lit = (fid == 0)
                        fiber_dicts[fid] = {}
                        for c in range(Num_cores):
                            fiber_dicts[fid][c] = {
                                "lit": is_lit,
                                "cores": 3,
                                "bands": ["C"] if is_lit else [],
                                "slots": ([0] * C_BAND_SLOTS) if is_lit else []
                            }
                    return

            raise ValueError(f"perform_upgrade_second(): invalid upgrade_type '{upgrade_type}' for core_type=1 (link {link_id})")

        # =========================
        # THREE-CORE MODE
        # =========================
        if num_cores == 3:
            if fiber_id not in fiber_dicts:
                raise ValueError(f"perform_upgrade_second(): fiber_id {fiber_id} not in link {link_id} fibers")

            if upgrade_type == "band_upgrade":
                ensure_three_core_fiber_shape(fiber_dicts, fiber_id)
                fid_obj = fiber_dicts[fiber_id]
                for c in range(Num_cores):
                    core_info = fid_obj[c]
                    slots = core_info["slots"]
                    if len(slots) < TOTAL_SLOTS:
                        slots.extend([0] * (TOTAL_SLOTS - len(slots)))
                    core_info["bands"] = ["C", "L"]
                return

            if upgrade_type == "new_fiber_C":
                ensure_three_core_fiber_shape(fiber_dicts, fiber_id)
                fid_obj = fiber_dicts[fiber_id]
                for c in range(Num_cores):
                    core_info = fid_obj[c]
                    core_info["lit"] = True
                    core_info["cores"] = 3
                    core_info["bands"] = ["C"]
                    core_info["slots"] = [0] * C_BAND_SLOTS
                return

            # if upgrade_type == "new_fiber_CL":
            #     ensure_three_core_fiber_shape(fiber_dicts, fiber_id)
            #     fid_obj = fiber_dicts[fiber_id]
            #     for c in range(Num_cores):
            #         core_info = fid_obj[c]
            #         core_info["lit"] = True
            #         core_info["cores"] = 3
            #         core_info["bands"] = ["C", "L"]
            #         core_info["slots"] = [0] * TOTAL_SLOTS
            #     return
            if upgrade_type == "new_fiber_CL":
                ensure_three_core_fiber_shape(fiber_dicts, fiber_id)
                f = fiber_dicts[fiber_id]  # {0:{...},1:{...},2:{...}}

                # Case A: dark fiber -> light with C+L
                # (check core 0 since all cores share state)
                if not f[0].get("lit", False):
                    for c in range(Num_cores):
                        core_info = f[c]
                        core_info["lit"] = True
                        core_info["cores"] = 3
                        core_info["bands"] = ["C", "L"]
                        core_info["slots"] = [0] * TOTAL_SLOTS
                    return

                # Case B: already lit fiber (likely from new_fiber_C) -> extend to C+L
                for c in range(Num_cores):
                    core_info = f[c]

                    # bands -> C+L
                    bands = set(core_info.get("bands", []))
                    bands.update(["C", "L"])
                    core_info["bands"] = ["C", "L"]

                    # extend slots to TOTAL_SLOTS
                    slots = core_info.get("slots", [])
                    if len(slots) < TOTAL_SLOTS:
                        slots.extend([0] * (TOTAL_SLOTS - len(slots)))
                    core_info["slots"] = slots

                    # keep lit/core values explicit
                    core_info["lit"] = True
                    core_info["cores"] = 3

                return

            raise ValueError(f"perform_upgrade_second(): invalid upgrade_type '{upgrade_type}' for core_type=3 (link {link_id})")

        raise ValueError(f"perform_upgrade_second(): unexpected core_type={num_cores} for link {link_id}")

    apply_upgrade(link_status_forward[link_id])
    apply_upgrade(link_status_backward[link_id])


# ------------------------------------------------------------
# remove_links_from_topology (UPDATED)
# ------------------------------------------------------------
def remove_links_from_topology(
        topology: list,
        topology_link_length: list,
        link_index: list,
        links_to_remove: dict,          # {link_id: "core_upgrade"}
        link_status_forward: dict,
        link_status_backward: dict,
        removed_links_info: dict,
        current_time: float,
        algorithm_name: str = "",     # ✅ NEW
):
    """
    LINK-level downtime:
      - removes adjacency
      - removes link from link_status dicts
      - stores full link state for restore

    Downtime policy:
      - this function sets a temporary/default restore_time
      - for combined link jobs, sequential_upgrade.py overrides restore_time later
    """

    new_topology = [row[:] for row in topology]

    for link_id, upgrade_type in links_to_remove.items():

        original_length = None
        removed_nodes = None

        # find endpoints
        for i in range(len(link_index)):
            for j in range(len(link_index[i])):
                if link_index[i][j] == link_id:
                    original_length = topology_link_length[i][j]
                    new_topology[i][j] = 0
                    new_topology[j][i] = 0
                    removed_nodes = (i, j)
                    break
            if original_length is not None:
                break

        if original_length is None:
            raise ValueError(f"Link {link_id} not found in LINK_INDEX.")

        stored_forward = link_status_forward.get(link_id)
        stored_backward = link_status_backward.get(link_id)

        # remove from live
        link_status_forward.pop(link_id, None)
        link_status_backward.pop(link_id, None)

        # Temporary/default downtime used only at removal time.
        # For combined link jobs, sequential_upgrade.py overrides restore_time later.
        if upgrade_type == "core_upgrade":
            downtime_days = int(t_3C)
        elif upgrade_type == "band_upgrade":
            downtime_days = int(t_b)
        elif upgrade_type == "new_fiber_CL":
            downtime_days = int(t_CL1)
        else:
            downtime_days = 0

        restore_time = current_time + downtime_days

        removed_links_info[("link", link_id)] = {
            "scope": "link",
            "nodes": removed_nodes,
            "start_time": current_time,
            "restore_time": restore_time,
            "upgrade_type": upgrade_type,
            "stored_forward": stored_forward,
            "stored_backward": stored_backward,
            "downtime_days": downtime_days,  # optional but helpful for logs
            "algorithm_name": algorithm_name, # optional
        }

    return new_topology, link_status_forward, link_status_backward, removed_links_info

def get_upgrade_downtime_days(upg: dict, algorithm_name: str) -> int:
    """
    Return downtime in days for one upgrade decision.
    Must match actual execution semantics.
    """

    if upg is None:
        return 0

    utype = upg.get("upgrade_type")

    if utype == "new_fiber_C":
        return 0

    if utype == "new_fiber_CL":
        return int(t_CL1)

    if utype == "band_upgrade":
        return int(t_b)

    if utype == "core_upgrade":
        return int(t_3C)

    return 0


def filter_upgrades_within_period_budget(
        safe_links: list,
        upgrade_decisions: dict,
        algorithm_name: str,
        max_downtime_days: int = 90
):
    """
    Keep only those links whose cumulative downtime stays within the upgrade-period budget.
    Links are processed in the order given in safe_links.

    Rule per link:
      - one decision  -> use that decision time
      - multiple decisions -> use max(decision times)

    Returns:
        filtered_safe_links
        filtered_upgrade_decisions
        used_downtime
    """

    filtered_safe_links = []
    filtered_upgrade_decisions = {}
    used_downtime = 0.0

    for link_id in safe_links:
        dec = upgrade_decisions.get(link_id)

        if dec is None:
            continue

        if isinstance(dec, list):
            valid_decs = [d for d in dec if d is not None and d.get("upgrade_type") is not None]

            if not valid_decs:
                link_downtime = 0.0
            else:
                times = [get_upgrade_downtime_days(d, algorithm_name) for d in valid_decs]

                if len(times) == 1:
                    link_downtime = float(times[0])
                else:
                    link_downtime = float(max(times))
        else:
            link_downtime = float(get_upgrade_downtime_days(dec, algorithm_name))

        # Keep this link only if total used downtime stays within budget
        if used_downtime + link_downtime <= max_downtime_days:
            filtered_safe_links.append(link_id)
            filtered_upgrade_decisions[link_id] = dec
            used_downtime += link_downtime
        else:
            print(
                f"⚠️ Skipping link {link_id}: "
                f"adding {link_downtime} days would exceed "
                f"{max_downtime_days}-day downtime budget."
            )

    return filtered_safe_links, filtered_upgrade_decisions, used_downtime

# ------------------------------------------------------------
# restore_removed_links_after_downtime (UPDATED)
# ------------------------------------------------------------
def restore_removed_links_after_downtime(
        key,                         # ("link", link_id) OR ("fiber", link_id, fiber_id)
        info: dict,
        current_time: float,
        link_status_forward: dict,
        link_status_backward: dict,
        current_topology: list,
        removed_links_info: dict):
    """
    Restores:
      - ("link", link_id): restore topology adjacency + link_status dict entry
      - ("fiber", link_id, fiber_id): restore only the stored fiber back into that link
    """

    scope = info.get("scope")

    if scope == "link":
        _, link_id = key
        print(f"Restoring LINK {link_id}")

        src, dst = info["nodes"]
        current_topology[src][dst] = 1
        current_topology[dst][src] = 1

        link_status_forward[link_id] = info["stored_forward"]
        link_status_backward[link_id] = info["stored_backward"]

        del removed_links_info[key]
        print(f"Link {link_id} restored at time {current_time}")
        return current_topology, link_status_forward, link_status_backward, removed_links_info

    elif scope == "fiber":
        _, link_id, fiber_id = key
        print(f"Restoring FIBER {fiber_id} on link {link_id}")

        if link_id not in link_status_forward or link_id not in link_status_backward:
            raise ValueError(f"Cannot restore fiber: link {link_id} missing in status dicts")

        link_status_forward[link_id][fiber_id] = info["stored_forward_fiber"]
        link_status_backward[link_id][fiber_id] = info["stored_backward_fiber"]

        del removed_links_info[key]
        print(f"Fiber {fiber_id} on link {link_id} restored at time {current_time}")
        return current_topology, link_status_forward, link_status_backward, removed_links_info

    else:
        raise ValueError(f"Unknown restore scope: {scope}")


