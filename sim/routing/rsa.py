from typing import List, Tuple, Optional, Dict, Any
import math
from collections import defaultdict
from sim.core import topology as Topology
from sim.core.constants import *

def _is_single_core_fiber(fiber: dict) -> bool:
    # single-core fibers have "lit" at the fiber level
    return isinstance(fiber, dict) and "lit" in fiber

def _fiber_is_lit(fiber: dict) -> bool:
    if not isinstance(fiber, dict):
        return False
    if _is_single_core_fiber(fiber):
        return bool(fiber.get("lit", False))
    # multi-core: any lit core
    return any(isinstance(core, dict) and core.get("lit", False) for core in fiber.values())

def _fiber_band_set(fiber: dict) -> set:
    if not isinstance(fiber, dict):
        return set()
    if _is_single_core_fiber(fiber):
        return set(fiber.get("bands", []))
    # multi-core: bands from a lit core
    for core in fiber.values():
        if isinstance(core, dict) and core.get("lit", False):
            return set(core.get("bands", []))
    return set()

def _iter_lit_core_slots(fiber: dict):
    """
    Yields (core_index, slots_list) for each lit core.
    For single-core, yields (0, slots).
    """
    if not isinstance(fiber, dict):
        return
    if _is_single_core_fiber(fiber):
        if fiber.get("lit", False):
            yield 0, fiber.get("slots", [])
        return
    for core_index, core in fiber.items():
        if isinstance(core, dict) and core.get("lit", False):
            yield int(core_index), core.get("slots", [])


def get_modulation_format(
        src: int,
        dest: int,
        path: List[int],
        link_status_forward: dict,
        link_status_backward: dict,
        LINK_INDEX: List[List[int]],
        MOR_BY_BAND: dict
    ):
    """
    Determine the best (highest-rate) modulation format index for a path.

    Adds a 'reason' output to distinguish failure causes:
      - ("reach") : path exceeds modulation reach limits
      - ("fiber") : one or more links have no lit fiber available

    Returns:
      (mf_index, reason)
        - mf_index (int) if feasible
        - float('inf'), 'reach'  if reach limit exceeded
        - float('inf'), 'fiber'  if no lit fiber
    """

    # ----------------------------------------------------------
    # Compute total path length
    # ----------------------------------------------------------
    link_length = 0
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]
        link_length += Topology.TOPOLOGY_LINK_LENGTHS[u][v]

    # missing key # ----------------------------------------------------------
    # # Classify each link in the path
    # # ----------------------------------------------------------
    link_classifications = []  # list of "C-only", "C+L-only", or "mixed"
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]
        linkid = LINK_INDEX[u][v]

        # choose forward/backward dictionary depending on direction
        ls = link_status_forward.get(linkid) if u < v else link_status_backward.get(linkid)

        # Missing link entry → cannot route
        if not ls:
            return float('inf'), None, "fiber"

        # gather band sets for lit fibers only
        # lit_band_sets = []
        # for fid, fiber in ls.items():
        #     if fiber.get("lit", True):
        #         bands = fiber.get("bands", [])
        #         lit_band_sets.append(set(bands))
        lit_band_sets = []
        for fid, fiber in ls.items():
            if not _fiber_is_lit(fiber):
                continue
            lit_band_sets.append(_fiber_band_set(fiber))

        # If there are no lit fibers → fail due to fiber unavailability
        if not lit_band_sets:
            return float('inf'), None, "fiber"

        # classify this link
        all_c_only = all(bands == {"C"} for bands in lit_band_sets)
        all_cplusl = all(("C" in bands and "L" in bands) and len(bands) >= 2 for bands in lit_band_sets)

        if all_c_only:
            link_classifications.append("C-only")
        elif all_cplusl:
            link_classifications.append("C+L-only")
        else:
            link_classifications.append("mixed")

    # ----------------------------------------------------------
    # Decide which MOR column to use
    # ----------------------------------------------------------
    if all(lc == "C-only" for lc in link_classifications):
        mor_key = "C"
    elif all(lc == "C+L-only" for lc in link_classifications):
        mor_key = "C+L_L"
    else:
        mor_key = "C+L_C"

    mor_thresholds = MOR_BY_BAND.get(mor_key)
    if mor_thresholds is None:
        raise ValueError(f"MOR_BY_BAND m'{mor_key}'")

    # ----------------------------------------------------------
    # Find the highest feasible modulation format
    # ----------------------------------------------------------
    for mf_index in range(NUMBER_OF_MFs - 1, -1, -1):
        if link_length <= mor_thresholds[mf_index]:
            return mf_index, mor_key, None  # success, no failure reason

    # if no format can reach → failure due to reach
    return float('inf'), mor_key, "reach"


def get_slice_window_size(datarate_gbps: int, mf_index: int) -> int:
    n_fs = math.ceil(datarate_gbps / (DELTA * 1e-9 * SPECTRAL_EFFICIENCY[mf_index] * N_TRX)) * N_TRX
    return int(n_fs)  # explicitly convert to int

def _normalize_slots_to_cores(slots):
    if slots is None or len(slots) == 0:
        return []
    if isinstance(slots[0], (int, float)):
        return [slots]
    return slots

def _get_ls(linkid, s, d, link_status_forward, link_status_backward):
    return link_status_forward[linkid] if s < d else link_status_backward[linkid]

def _build_link_candidates(ls: Dict[int, Dict[str, Any]]) -> List[Tuple[int, int, int]]:
    """
    Candidates for a single link in stable order:
      (fiber_id, core_index, cap_len)
    Only lit fibers/cores.
    """
    cands: List[Tuple[int, int, int]] = []
    for fiber_id in sorted(ls.keys()):
        fiber = ls[fiber_id]
        if not _fiber_is_lit(fiber):
            continue

        for core_index, slots in _iter_lit_core_slots(fiber):
            cands.append((fiber_id, core_index, len(slots)))

    return cands


def check_if_slice_window_is_free(current_global_time: float,
                                  start_index: int,
                                  slice_window_size: int,
                                  core_slots: List[float]) -> bool:

    end_index = start_index + slice_window_size

    # Window bounds
    if end_index > len(core_slots):
        return False

    # All slots in the window must be free up to current time
    for f_index in range(start_index, end_index):
        if core_slots[f_index] > current_global_time:
            return False

    return True

def check_spectrum_first_fit(
        path: List[int],
        slice_window_size: int,
        LINK_INDEX: List[List[int]],
        current_global_time: float,
        link_status_forward: dict,
        link_status_backward: dict
):
    """
    Implements YOUR policy:

    Outer priority:
      - Choose (fiber/core) on first link (in order)
      - For that choice, scan fs_index from 0..cap-1
      - For each fs_index, try to satisfy link2..end by trying their candidates in order
      - If fail at some later link, go back to first link and try next fs_index
      - If fs exhausted for first-link candidate, move to next first-link candidate
    """

    # Precompute per-link candidates + link IDs
    linkids: List[int] = []
    per_link_cands: List[List[Tuple[int, int, int]]] = []

    for i in range(len(path) - 1):
        s, d = path[i], path[i + 1]
        linkid = LINK_INDEX[s][d]
        linkids.append(linkid)

        ls = _get_ls(linkid, s, d, link_status_forward, link_status_backward)
        cands = _build_link_candidates(ls)
        if not cands:
            return float("inf"), None, None, None
        per_link_cands.append(cands)

    # ---- First link candidates drive the whole search order ----
    first_link_cands = per_link_cands[0]

    for (fid0, core0, cap0) in first_link_cands:

        # For this first-link (fiber/core), we can only search up to cap0
        max_fs = cap0
        if max_fs < slice_window_size:
            continue

        # Scan fs_index on FIRST LINK (this is your “exhaust indices before moving fiber on link1”)
        for fs_index in range(max_fs - slice_window_size + 1):

            # Check first link itself (fast)
            linkid0 = linkids[0]
            s0, d0 = path[0], path[1]
            ls0 = _get_ls(linkid0, s0, d0, link_status_forward, link_status_backward)
            fiber0 = ls0.get(fid0)
            if fiber0 is None or not _fiber_is_lit(fiber0):
                break

            # get slots for the chosen (fid0, core0)
            core_slots0 = None
            for cidx, slots in _iter_lit_core_slots(fiber0):
                if cidx == core0:
                    core_slots0 = slots
                    break
            if core_slots0 is None:
                break

            if fs_index + slice_window_size > len(core_slots0):
                continue

            if not check_if_slice_window_is_free(current_global_time, fs_index, slice_window_size, core_slots0):
                continue

            # If first link works, try to satisfy remaining links using same fs_index
            fibers_used = [fid0]
            cores_used = [core0]

            def satisfy_next_link(link_idx: int) -> bool:
                if link_idx == len(per_link_cands):
                    return True  # all links satisfied

                # Try candidates for this link in order (fiber1 then fiber2 etc.)
                s, d = path[link_idx], path[link_idx + 1]
                linkid = linkids[link_idx]
                ls = _get_ls(linkid, s, d, link_status_forward, link_status_backward)

                for (fid, core, cap) in per_link_cands[link_idx]:
                    # ✅ capacity guard prevents L-band crash on C-only fibers
                    if fs_index + slice_window_size > cap:
                        continue

                    fiber = ls.get(fid)
                    if fiber is None or not _fiber_is_lit(fiber):
                        continue

                    core_slots = None
                    for cidx, slots in _iter_lit_core_slots(fiber):
                        if cidx == core:
                            core_slots = slots
                            break
                    if core_slots is None:
                        continue

                    if fs_index + slice_window_size > len(core_slots):
                        continue

                    if check_if_slice_window_is_free(current_global_time, fs_index, slice_window_size, core_slots):
                        fibers_used.append(fid)
                        cores_used.append(core)

                        if satisfy_next_link(link_idx + 1):
                            return True

                        # backtrack on this link choice
                        fibers_used.pop()
                        cores_used.pop()

                return False

            if satisfy_next_link(1):
                band_region = "L" if fs_index >= C_BAND_SLOTS else "C"
                # print(
                #     f"[RSA] SUCCESS fs_index={fs_index} ({band_region}-band) "
                #     f"slice={slice_window_size} "
                #     f"fibers={fibers_used} cores={cores_used} linkids={linkids}"
                # )
                return fs_index, fibers_used, cores_used, linkids


            # else: link2..end failed -> go back to first link and try next fs_index

        # exhausted all fs_index for this first-link candidate -> move to next first-link fiber/core

    return float("inf"), None, None, None

def assign_spectrum(
    path: List[int],
    start_index: int,
    slice_window_size: int,
    connection_end_time: float,
    mf_index: int,  # kept for interface consistency (not used)
    link_status_forward: dict,
    link_status_backward: dict,
    fibers_used: List[int],
    cores_used: List[int],
):
    """
    Assign spectrum along `path` using the selected (fiber_id, core_id) per hop.

    Works for BOTH structures:

    1) Single-core fiber:
         ls[linkid][fiber_id] = {"lit": bool, "cores": 1, "bands": [...], "slots": [...]}

    2) Multi-core fiber (e.g., 3-core):
         ls[linkid][fiber_id] = {
              0: {"lit": bool, "cores": 3, "bands": [...], "slots": [...]},
              1: {...},
              2: {...}
         }

    Assumes `fibers_used` and `cores_used` have length len(path)-1 and correspond
    hop-by-hop to the path.
    """
    if len(path) < 2:
        raise ValueError("assign_spectrum(): Path must have at least 2 nodes")

    hops = len(path) - 1
    if len(fibers_used) != hops or len(cores_used) != hops:
        raise ValueError(
            f"assign_spectrum(): fibers_used/cores_used length mismatch: "
            f"hops={hops}, fibers_used={len(fibers_used)}, cores_used={len(cores_used)}"
        )

    start = int(start_index)
    width = int(slice_window_size)
    if start < 0 or width <= 0:
        raise ValueError(
            f"assign_spectrum(): Invalid start_index={start_index} or slice_window_size={slice_window_size}"
        )
    end = start + width

    for ln in range(hops):
        s, d = path[ln], path[ln + 1]
        linkid = Topology.LINK_INDEX[s][d]
        if linkid == -1:
            raise ValueError(f"assign_spectrum(): Invalid link in path hop {s}->{d}")

        fiber_id = fibers_used[ln]
        core_id = cores_used[ln]

        ls = link_status_forward[linkid] if s < d else link_status_backward[linkid]
        if linkid not in (link_status_forward if s < d else link_status_backward):
            raise ValueError(f"assign_spectrum(): Missing linkid={linkid} in link status dict")

        if fiber_id not in ls:
            raise ValueError(
                f"assign_spectrum(): Missing fiber_id={fiber_id} for link {linkid} (hop {s}->{d})"
            )

        fiber = ls[fiber_id]

        # -------------------------
        # Single-core fiber case
        # -------------------------
        if _is_single_core_fiber(fiber):
            if not fiber.get("lit", False):
                raise ValueError(
                    f"assign_spectrum(): Attempting to assign on unlit single-core fiber "
                    f"(link {linkid}, fiber {fiber_id})"
                )

            slots = fiber.get("slots", [])
            if end > len(slots):
                raise ValueError(
                    f"Spectrum assignment out of bounds: slot range [{start},{end}) exceeds "
                    f"{len(slots)} slots (link {linkid}, fiber {fiber_id})"
                )

            for i in range(start, end):
                slots[i] = connection_end_time

        # -------------------------
        # Multi-core fiber case
        # -------------------------
        else:
            if core_id not in fiber:
                raise ValueError(
                    f"assign_spectrum(): core_id={core_id} missing in multi-core fiber "
                    f"(link {linkid}, fiber {fiber_id})"
                )

            core = fiber[core_id]
            if not core.get("lit", False):
                raise ValueError(
                    f"assign_spectrum(): Attempting to assign on unlit core "
                    f"(link {linkid}, fiber {fiber_id}, core {core_id})"
                )

            slots = core.get("slots", [])
            if end > len(slots):
                raise ValueError(
                    f"Spectrum assignment out of bounds: slot range [{start},{end}) exceeds "
                    f"{len(slots)} slots (link {linkid}, fiber {fiber_id}, core {core_id})"
                )

            for i in range(start, end):
                slots[i] = connection_end_time

def execute_first_fit(
        src: int,
        dest: int,
        datarate: int,
        arrival_time: float,
        departure_time: float,
        link_status_forward: dict,
        link_status_backward: dict,
        topology: list,
        PATHS
):
    """
    Perform RSA (Routing and Spectrum Assignment) using First-Fit strategy,
    with precomputed K-shortest paths from PATHS[src][dest].

    Returns:
        mf_index,
        fs_index,
        slice_window_size,
        selected_path,
        fibers_used,
        cores_used,
        link_ids,
        attempted_paths_info

    where attempted_paths_info is a list of dicts like:
        {
            "path_index": int,
            "path": [...],
            "status": "accepted" | "blocked" | "skipped_fiber" | "blocked_reach"
        }
    """
    k_shortest = PATHS[src][dest]

    attempted_paths_info = []

    # Handle case where no path exists
    if not k_shortest or len(k_shortest) == 0:
        return (
            float("inf"), float("inf"), float("inf"), None,
            None, None, None, attempted_paths_info
        )

    for p_idx, path in enumerate(k_shortest):

        mf, mor_key, reason = get_modulation_format(
            src, dest, path,
            link_status_forward, link_status_backward,
            Topology.LINK_INDEX, MOR_BY_BAND
        )

        if mf == float("inf"):
            if reason == "reach":
                attempted_paths_info.append({
                    "path_index": p_idx,
                    "path": path,
                    "status": "blocked_reach"
                })
                print(f"Path {path} exceeds reach — stopping search.")
                return (
                    float("inf"), float("inf"), float("inf"), None,
                    None, None, None, attempted_paths_info
                )

            elif reason == "fiber":
                attempted_paths_info.append({
                    "path_index": p_idx,
                    "path": path,
                    "status": "skipped_fiber"
                })
                continue

        # Compute required spectrum slots
        slice_window_size = get_slice_window_size(datarate, mf)

        fs_index, fibers_used, cores_used, link_ids = check_spectrum_first_fit(
            path=path,
            slice_window_size=slice_window_size,
            LINK_INDEX=Topology.LINK_INDEX,
            current_global_time=arrival_time,
            link_status_forward=link_status_forward,
            link_status_backward=link_status_backward,
        )

        # Success
        if fs_index != float("inf"):
            assign_spectrum(
                path=path,
                start_index=fs_index,
                slice_window_size=slice_window_size,
                connection_end_time=departure_time,
                mf_index=mf,
                link_status_forward=link_status_forward,
                link_status_backward=link_status_backward,
                fibers_used=fibers_used,
                cores_used=cores_used
            )

            attempted_paths_info.append({
                "path_index": p_idx,
                "path": path,
                "status": "accepted"
            })

            return (
                mf, fs_index, slice_window_size, path,
                fibers_used, cores_used, link_ids, attempted_paths_info
            )

        # Spectrum blocked on this path
        else:
            attempted_paths_info.append({
                "path_index": p_idx,
                "path": path,
                "status": "blocked"
            })
            continue

    # No available path found
    return (
        float("inf"), float("inf"), float("inf"), None,
        None, None, None, attempted_paths_info
    )


