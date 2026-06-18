from collections import defaultdict, deque
from typing import List, Tuple, Dict, Any, Optional


# ============================================================
# Slot extraction + lit helpers
# ============================================================

def extract_lit_core_slot_lists(fiber_obj: Any) -> List[List[int]]:
    """
    Returns list of slot arrays for LIT resources only:
      - single-core fiber  -> [slots] if lit
      - multi-core fiber   -> [core_slots for lit cores only]
    """
    if not isinstance(fiber_obj, dict):
        return []

    # single-core
    if "slots" in fiber_obj and isinstance(fiber_obj["slots"], list):
        if bool(fiber_obj.get("lit", False)):
            return [fiber_obj["slots"]]
        return []

    # multi-core
    out: List[List[int]] = []
    for _, core_obj in fiber_obj.items():
        if not isinstance(core_obj, dict):
            continue
        if not bool(core_obj.get("lit", False)):
            continue
        slots = core_obj.get("slots", None)
        if isinstance(slots, list):
            out.append(slots)
    return out


def fiber_is_lit(fiber_obj: Any) -> bool:
    """
    Lit rule:
      - single-core: fiber_obj["lit"] must be True
      - multi-core: at least one core has lit=True
    """
    if not isinstance(fiber_obj, dict):
        return False

    # single-core case
    if "slots" in fiber_obj and isinstance(fiber_obj["slots"], list):
        return bool(fiber_obj.get("lit", False))

    # multi-core case
    for _, core_obj in fiber_obj.items():
        if isinstance(core_obj, dict) and bool(core_obj.get("lit", False)):
            return True
    return False


def per_fiber_slot_limit_by_bands(fiber_obj: Any, C_BAND_SLOTS: int, TOTAL_SLOTS: int) -> int:
    """
    Per-resource slot limit:
      - If (C and L) present -> TOTAL_SLOTS
      - else -> C_BAND_SLOTS

    Works for both:
      - single-core: fiber_obj["bands"]
      - multi-core: bands read from a lit core (first lit core found)
    """
    if not isinstance(fiber_obj, dict):
        return C_BAND_SLOTS

    # single-core
    if "slots" in fiber_obj and isinstance(fiber_obj["slots"], list):
        bands = fiber_obj.get("bands", None)
        if not bands:
            return C_BAND_SLOTS
        b = set(bands)
        return TOTAL_SLOTS if ("C" in b and "L" in b) else C_BAND_SLOTS

    # multi-core: read bands from a lit core
    for _, core_obj in fiber_obj.items():
        if isinstance(core_obj, dict) and bool(core_obj.get("lit", False)):
            bands = core_obj.get("bands", None)
            if not bands:
                return C_BAND_SLOTS
            b = set(bands)
            return TOTAL_SLOTS if ("C" in b and "L" in b) else C_BAND_SLOTS

    return C_BAND_SLOTS


# ============================================================
# Contiguous-block capacity helpers
# ============================================================

def count_blocks_of_width(
    slot_list: List[int],
    width: int,
    max_idx: int,
    allow_overlapping: bool = False
) -> int:
    """
    Count how many slices of `width` can be placed in slot_list[0:max_idx].
    Free slot is assumed to be == 0.

    For each contiguous free run of length L:
      - non-overlapping placements: floor(L / width)
      - overlapping placements: (L - width + 1)
    """
    if width <= 0:
        return 0
    if not isinstance(slot_list, list):
        return 0

    limit = min(len(slot_list), max_idx)
    if limit <= 0:
        return 0

    count = 0
    i = 0
    while i < limit:
        if slot_list[i] != 0:
            i += 1
            continue

        j = i
        while j < limit and slot_list[j] == 0:
            j += 1
        run_len = j - i

        if allow_overlapping:
            count += max(0, run_len - width + 1)
        else:
            count += run_len // width

        i = j

    return count


def link_free_capacity(
    ls: Dict[int, Any],
    C_BAND_SLOTS: int,
    TOTAL_SLOTS: int,
    slice_width: int = 3
) -> float:
    """
    Capacity = sum over all lit fibers and lit cores of:
              number of contiguous free blocks of width slice_width,
              counted only within the band-supported slot range of that fiber/core.
    """
    if not ls:
        return 0.0

    free_slices = 0

    for _, fiber in ls.items():
        if not fiber_is_lit(fiber):
            continue

        max_fs = per_fiber_slot_limit_by_bands(fiber, C_BAND_SLOTS, TOTAL_SLOTS)

        for core_slots in extract_lit_core_slot_lists(fiber):
            free_slices += count_blocks_of_width(
                core_slots, slice_width, max_fs, allow_overlapping=False
            )

    return float(free_slices)


# ============================================================
# Capacity graph builder
# ============================================================

def build_capacity_graph(
    TOPOLOGY,
    LINK_INDEX,
    link_status_forward,
    link_status_backward,
    C_BAND_SLOTS,
    TOTAL_SLOTS,
    slice_width: int = 3,
    include_zero_capacity_edges: bool = True,
    eps_for_zero: float = 0.0,
):
    """
    Directed capacity graph cap[u][v] = contiguous-block capacity on link u->v.

    IMPORTANT:
      - If include_zero_capacity_edges=True, we store cap[u][v] even when c == 0,
        so totally-starved links can still appear as cut edges.

      - Dinic will still only traverse edges with residual > 0, so 0 edges won't
        affect max-flow computation (unless eps_for_zero > 0).
    """
    N = len(TOPOLOGY)
    cap: Dict[int, Dict[int, float]] = {i: {} for i in range(N)}

    for u in range(N):
        for v in range(N):
            if u == v or TOPOLOGY[u][v] != 1:
                continue

            lid = LINK_INDEX[u][v]
            if lid == -1:
                continue

            ls = link_status_forward[lid] if u < v else link_status_backward[lid]
            c = link_free_capacity(ls, C_BAND_SLOTS, TOTAL_SLOTS, slice_width=slice_width)

            if c > 0:
                cap[u][v] = float(c)
            else:
                if include_zero_capacity_edges:
                    cap[u][v] = float(eps_for_zero)  # 0.0 by default
                # else: do not store edge at all

    return cap


# ============================================================
# Dinic max-flow for dict-of-dicts
# ============================================================

def dinic_max_flow(cap: Dict[int, Dict[int, float]], s: int, t: int):
    """
    Dinic max-flow for dict-of-dicts capacities.
    Returns (max_flow, flow, residual).

    NOTE:
      - Only edges with capacity > 0 are added to BFS/DFS neighbors.
      - cap may still contain 0 edges for min-cut reporting.
    """
    neighbors: Dict[int, List[int]] = defaultdict(list)
    residual: Dict[int, Dict[int, float]] = {u: dict(vs) for u, vs in cap.items()}
    flow = defaultdict(lambda: defaultdict(float))

    # ensure all nodes exist in residual
    for u in cap:
        residual.setdefault(u, {})

    # Build neighbor lists ONLY for positive capacity edges (efficient)
    for u, vs in cap.items():
        for v, c in vs.items():
            if c <= 0:
                continue

            residual.setdefault(v, {})
            residual[v].setdefault(u, 0.0)
            residual[u].setdefault(v, float(c))

            neighbors[u].append(v)
            neighbors[v].append(u)

    max_flow = 0.0

    def bfs_level() -> Optional[Dict[int, int]]:
        level = {s: 0}
        q = deque([s])
        while q:
            u = q.popleft()
            for v in neighbors[u]:
                if v in level:
                    continue
                if residual.get(u, {}).get(v, 0.0) > 1e-12:
                    level[v] = level[u] + 1
                    q.append(v)
        return level if t in level else None

    def dfs(u: int, pushed: float, level: Dict[int, int], it: Dict[int, int]) -> float:
        if u == t:
            return pushed

        while it[u] < len(neighbors[u]):
            v = neighbors[u][it[u]]
            if residual[u].get(v, 0.0) > 1e-12 and level.get(v, -1) == level[u] + 1:
                tr = dfs(v, min(pushed, residual[u][v]), level, it)
                if tr > 0:
                    residual[u][v] -= tr
                    residual[v][u] = residual[v].get(u, 0.0) + tr
                    flow[u][v] += tr
                    flow[v][u] -= tr
                    return tr
            it[u] += 1
        return 0.0

    while True:
        level = bfs_level()
        if level is None:
            break

        it = defaultdict(int)
        while True:
            pushed = dfs(s, float("inf"), level, it)
            if pushed <= 1e-12:
                break
            max_flow += pushed

    # Make residual keys exist for neighbor pairs (helpful for min-cut BFS)
    for u in list(neighbors.keys()):
        residual.setdefault(u, {})
        for v in neighbors[u]:
            residual[u].setdefault(v, 0.0)

    return max_flow, flow, residual


# ============================================================
# Min-cut extraction
# ============================================================

def min_cut_link_ids(
    cap: Dict[int, Dict[int, float]],
    residual: Dict[int, Dict[int, float]],
    s: int,
    LINK_INDEX
):
    """
    Reachable nodes in residual graph from s define the min-cut partition.
    Cut edges are cap edges from reachable -> non-reachable.
    """
    reachable = {s}
    q = deque([s])

    while q:
        u = q.popleft()
        for v, rc in residual.get(u, {}).items():
            if rc > 1e-12 and v not in reachable:
                reachable.add(v)
                q.append(v)

    cut_links: List[int] = []
    for u in reachable:
        for v in cap.get(u, {}):  # IMPORTANT: iterate cap so 0 edges can appear as cut edges
            if v not in reachable:
                lid = LINK_INDEX[u][v]
                if lid != -1:
                    cut_links.append(lid)

    return cut_links


# ============================================================
# Main selection function
# ============================================================

def select_links_for_upgrade_mincut(
    critical_sd_pairs,
    working_topology,
    LINK_INDEX,
    link_status_forward,
    link_status_backward,
    C_BAND_SLOTS,
    TOTAL_SLOTS,
    slice_width: int = 3,
    sd_blocked_counts: dict = None,
    top_k: int = 50,
    include_zero_capacity_edges: bool = True,
    eps_for_zero: float = 0.0,
):
    """
    Evaluate min-cut for top_k critical SD pairs, on a capacity graph where link
    capacities are contiguous-block placements of width `slice_width`.

    include_zero_capacity_edges=True:
      - store edges even if capacity==0 (so they can be returned as cut edges)

    eps_for_zero:
      - if > 0, gives tiny capacity instead of 0 (heuristic)
    """
    if not critical_sd_pairs:
        return []

    pairs = list(critical_sd_pairs)
    if sd_blocked_counts:
        pairs.sort(
            key=lambda sd: float(sd_blocked_counts.get(sd, sd_blocked_counts.get((sd[1], sd[0]), 0.0))),
            reverse=True
        )
    pairs = pairs[:top_k]

    cap = build_capacity_graph(
        TOPOLOGY=working_topology,
        LINK_INDEX=LINK_INDEX,
        link_status_forward=link_status_forward,
        link_status_backward=link_status_backward,
        C_BAND_SLOTS=C_BAND_SLOTS,
        TOTAL_SLOTS=TOTAL_SLOTS,
        slice_width=slice_width,
        include_zero_capacity_edges=include_zero_capacity_edges,
        eps_for_zero=eps_for_zero,
    )

    link_cut_score = defaultdict(float)

    for (s, d) in pairs:
        if s == d:
            continue

        # DEBUG: notify when s has only zero-capacity outgoing edges (cap entries exist but all are zero)
        if cap.get(s, {}) and all(c <= 0.0 for c in cap[s].values()):
          print(f"[mincut] s={s} has only zero-capacity outgoing edges; cut will include zero-cap edges.")

        # If s has no outgoing edges at all in cap, skip
        if not cap.get(s, {}):
            continue

        max_flow, _, residual = dinic_max_flow(cap, s, d)
        cut_lids = min_cut_link_ids(cap, residual, s, LINK_INDEX)

        # Weighting
        weight = 0.0
        if sd_blocked_counts:
            weight = float(sd_blocked_counts.get((s, d), sd_blocked_counts.get((d, s), 0.0)))
        if weight <= 0.0:
            weight = max_flow if max_flow > 0.0 else 1.0

        for lid in cut_lids:
            link_cut_score[lid] += weight

    if not link_cut_score:
        return []

    return sorted(link_cut_score.keys(), key=lambda lid: link_cut_score[lid], reverse=True)

import math
from collections import defaultdict, deque
from typing import List, Tuple, Dict, Any, Optional


def _path_links_from_nodes(path, LINK_INDEX):
    links = []
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]

        lid = LINK_INDEX[u][v]
        if lid == -1:
            return None

        links.append(lid)

    return links


def select_links_from_congested_paths_per_sd(
    critical_sd_pairs,
    PATHS,
    path_stats,
    LINK_INDEX
):
    """
    For each critical SD pair:
        find most congested path (max blocked/arrivals)
        return all links on that path
    """

    selected_links = set()

    for (s, d) in critical_sd_pairs:

        candidate_paths = PATHS[s][d]
        if not candidate_paths:
            continue

        sd_u = (s, d) if s < d else (d, s)

        best_score = -1
        best_path_index = None

        for p_idx in range(len(candidate_paths)):

            stats = path_stats.get((sd_u[0], sd_u[1], p_idx))
            if not stats:
                continue

            arrivals = stats["path_arrivals"]
            blocked = stats["path_blocked"]

            if arrivals == 0:
                continue

            score = (blocked / arrivals) * math.log1p(arrivals)

            if score > best_score:
                best_score = score
                best_path_index = p_idx

        if best_path_index is None:
            continue

        best_path = candidate_paths[best_path_index]

        link_ids = _path_links_from_nodes(best_path, LINK_INDEX)

        if link_ids:
            selected_links.update(link_ids)

    return sorted(selected_links)
