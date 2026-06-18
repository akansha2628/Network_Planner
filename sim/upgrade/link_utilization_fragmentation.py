from typing import Any, Dict, Tuple
import csv
from pathlib import Path

# ============================================================
# Band definitions
# ============================================================

def band_ranges(C_BAND_SLOTS: int, TOTAL_SLOTS: int):
    # Half-open ranges: C [0, C_BAND_SLOTS), L [C_BAND_SLOTS, TOTAL_SLOTS)
    return {
        "C": (0, C_BAND_SLOTS),
        "L": (C_BAND_SLOTS, TOTAL_SLOTS),
    }

# ============================================================
# Slot extraction helpers
# ============================================================

def extract_resource_slot_lists(fiber_obj: Any):
    """
    Returns list of slot arrays for lit resources only:
      - single-core fiber  -> [slots] if fiber is lit
      - multi-core fiber   -> [core_slots] for lit cores
    """
    if not isinstance(fiber_obj, dict):
        return []

    # single-core layout: top-level dict has slots + lit flag
    if "slots" in fiber_obj and isinstance(fiber_obj["slots"], list):
        if bool(fiber_obj.get("lit", True)):
            return [fiber_obj["slots"]]
        return []

    # multi-core: include only lit cores that have slots
    slot_lists = []
    for _, core_obj in fiber_obj.items():
        if (
            isinstance(core_obj, dict)
            and core_obj.get("lit", False)
            and isinstance(core_obj.get("slots", None), list)
        ):
            slot_lists.append(core_obj["slots"])

    return slot_lists

# ============================================================
# Per-resource metrics
# ============================================================

def resource_util_and_fragmentation(slot_list: list, band_start: int, band_end: int) -> Tuple[float, float, int, int]:
    """
    Utilization = used / total_in_band
    Fragmentation = 1 - (largest_free_block / total_free_in_band)
    Band range uses half-open [band_start, band_end) and is clamped to slot_list length.
    Returns: util, frag, total_in_band, total_free
    """
    if not isinstance(slot_list, list):
        return 0.0, 0.0, 0, 0

    n = len(slot_list)
    if n == 0:
        return 0.0, 0.0, 0, 0

    # Clamp band to [0, n]
    start = max(0, band_start)
    end = min(band_end, n)

    # Band not present for this resource
    if start >= end:
        return 0.0, 0.0, 0, 0

    total = end - start
    used = 0
    total_free = 0
    max_free_run = 0
    cur_free_run = 0

    for s in range(start, end):
        if slot_list[s] != 0:
            used += 1
            max_free_run = max(max_free_run, cur_free_run)
            cur_free_run = 0
        else:
            total_free += 1
            cur_free_run += 1

    max_free_run = max(max_free_run, cur_free_run)

    util = used / total
    frag = 0.0 if total_free == 0 else 1.0 - (max_free_run / total_free)

    return util, frag, total, total_free

# ============================================================
# Per-fiber metrics (aggregate cores)
# ============================================================

def fiber_utilization_in_band(fiber_obj: dict, band_start: int, band_end: int) -> float:
    """
    Utilization for ONE fiber over the band, aggregating across lit cores.
    Util_fiber = total used slots across lit cores in band / total slots across lit cores in band.
    """
    used_sum = 0
    total_sum = 0

    for slot_list in extract_resource_slot_lists(fiber_obj):
        n = len(slot_list)
        if n == 0:
            continue
        start = max(0, band_start)
        end = min(band_end, n)
        if start >= end:
            continue

        total = end - start
        total_sum += total

        used = 0
        for s in range(start, end):
            if slot_list[s] != 0:
                used += 1
        used_sum += used

    return (used_sum / total_sum) if total_sum > 0 else 0.0


def fiber_fragmentation_in_band(fiber_obj: dict, band_start: int, band_end: int) -> float:
    """
    Fragmentation for ONE fiber over the band, computed as the simple average of
    per-core fragmentation across the lit cores of this fiber.
    Returns 0.0 when there are no lit cores or cores have no capacity in the band.
    """
    frags = []

    for slot_list in extract_resource_slot_lists(fiber_obj):
        util_i, frag_i, total_in_band_i, free_in_band_i = resource_util_and_fragmentation(slot_list, band_start, band_end)
        # Include fragmentation for cores that have band capacity (total_in_band_i > 0)
        if total_in_band_i > 0:
            frags.append(frag_i)

    if not frags:
        return 0.0
    return sum(frags) / len(frags)

# ============================================================
# Per-link (directional) aggregation (simple averages across lit fibers)
# ============================================================

def compute_link_dir_band_stats(
    link_status_dir: Dict[int, Dict[int, dict]],
    link_id: int,
    C_BAND_SLOTS: int,
    TOTAL_SLOTS: int,
):
    """
    Returns per-band stats for a directional link:
      {
        "C": (util, frag),
        "L": (util, frag)
      }

    NEW:
      - Utilization is the simple average across lit fibers (each fiber equally weighted).
      - Fragmentation is the simple average across lit fibers (fiber fragmentation is avg of lit cores).
    """
    bands = band_ranges(C_BAND_SLOTS, TOTAL_SLOTS)
    out = {"C": (0.0, 0.0), "L": (0.0, 0.0)}

    fibers = link_status_dir.get(link_id, {})
    if not fibers:
        return out

    for band_name, (band_start, band_end) in bands.items():
        fiber_utils = []
        fiber_frags = []

        for _, fiber_obj in fibers.items():
            # Treat fiber as lit if single-core lit OR any lit core in multi-core
            is_lit = False
            if isinstance(fiber_obj, dict):
                if "slots" in fiber_obj:
                    is_lit = bool(fiber_obj.get("lit", False))
                else:
                    for _, core_obj in fiber_obj.items():
                        if isinstance(core_obj, dict) and core_obj.get("lit", False):
                            is_lit = True
                            break
            if not is_lit:
                continue

            u_fiber = fiber_utilization_in_band(fiber_obj, band_start, band_end)
            f_fiber = fiber_fragmentation_in_band(fiber_obj, band_start, band_end)

            fiber_utils.append(u_fiber)
            fiber_frags.append(f_fiber)

        util_agg = (sum(fiber_utils) / len(fiber_utils)) if fiber_utils else 0.0
        frag_agg = (sum(fiber_frags) / len(fiber_frags)) if fiber_frags else 0.0

        out[band_name] = (util_agg, frag_agg)

    return out

# ============================================================
# CSV helpers (row-wise)
# ============================================================

def init_monthly_rowwise_csvs(results_dir: Path):
    util_csv = results_dir / "monthly_link_utilization_rowwise.csv"
    frag_csv = results_dir / "monthly_link_fragmentation_rowwise.csv"

    header = ["time_day", "forward_C", "forward_L", "backward_C", "backward_L"]

    for p in (util_csv, frag_csv):
        if not p.exists() or p.stat().st_size == 0:
            with open(p, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(header)

    return util_csv, frag_csv

def log_monthly_link_stats_rowwise(
    time_day: float,
    LINKS: int,
    link_status_forward,
    link_status_backward,
    C_BAND_SLOTS: int,
    TOTAL_SLOTS: int,
    util_csv: Path,
    frag_csv: Path,
    print_console: bool = False,
):
    util_rows, frag_rows = [], []

    for lid in range(LINKS):
        f = compute_link_dir_band_stats(link_status_forward, lid, C_BAND_SLOTS, TOTAL_SLOTS)
        b = compute_link_dir_band_stats(link_status_backward, lid, C_BAND_SLOTS, TOTAL_SLOTS)

        util_rows.append([time_day, f["C"][0], f["L"][0], b["C"][0], b["L"][0]])
        frag_rows.append([time_day, f["C"][1], f["L"][1], b["C"][1], b["L"][1]])

    with open(util_csv, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(util_rows)

    with open(frag_csv, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(frag_rows)

    if print_console:
        print("MONTHLY UTILIZATION ROWS (time, Forward C, Forward L, Backward C, Backward L):")
        for lid, row in enumerate(util_rows, start=1):
            print(f"Link {lid}: " + ",".join(map(str, row)))
        print("MONTHLY FRAGMENTATION ROWS (time, Forward C, Forward L, Backward C, Backward L):")
        for lid, row in enumerate(frag_rows, start=1):
            print(f"Link {lid}: " + ",".join(map(str, row)))

