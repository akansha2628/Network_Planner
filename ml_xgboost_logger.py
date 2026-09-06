import csv
from pathlib import Path


TRAIN_SEEDS = set(range(0, 15))   # 0 to 14
TEST_SEEDS = set(range(15, 20))   # 15 to 19


def get_seed_split(seed):
    seed = int(seed)

    if seed in TRAIN_SEEDS:
        return "train"

    if seed in TEST_SEEDS:
        return "test"

    return "unused"


def flatten_traffic_matrix(traffic_rates):
    """
    Save current arrival rate lambda_i_j for every source-destination pair.
    For DT-12, this gives 12 * 11 = 132 lambda columns.
    """
    row = {}
    n = traffic_rates.shape[0]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            row[f"lambda_{i}_{j}"] = float(traffic_rates[i][j])

    return row


def flatten_link_technology_state(link_status_forward, classify_fiber_state):
    """
    Save current technology state for each physical link.

    SC_C  = number of lit single-core C-band fibers
    SC_CL = number of lit single-core C+L fibers
    MC_C  = number of lit multi-core C-band cores
    MC_CL = number of lit multi-core C+L cores
    """
    row = {}

    for link_id, fibers in sorted(link_status_forward.items()):
        sc_c = 0
        sc_cl = 0
        mc_c = 0
        mc_cl = 0

        for _, fiber_obj in fibers.items():
            state = classify_fiber_state(fiber_obj)

            if state == "SC_C":
                sc_c += 1
            elif state == "SC_CL":
                sc_cl += 1
            elif state == "MC_C":
                mc_c += 3
            elif state == "MC_CL":
                mc_cl += 3

        row[f"link_{link_id}_SC_C"] = sc_c
        row[f"link_{link_id}_SC_CL"] = sc_cl
        row[f"link_{link_id}_MC_C"] = mc_c
        row[f"link_{link_id}_MC_CL"] = mc_cl

    return row


def max_contiguous_free_slots(slots):
    """
    Return the largest continuous block of free slots in one slot array.

    Free slot is assumed to be 0.
    Occupied slot is assumed to be non-zero.
    """
    max_run = 0
    current_run = 0

    for value in slots:
        if value == 0:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0

    return max_run


def summarize_slot_array(slots):
    """
    Summarize one slot array.
    """
    if slots is None:
        return {
            "occupied_slots": 0,
            "total_slots": 0,
            "free_slots": 0,
            "max_contiguous_free_slots": 0,
        }

    total_slots = len(slots)
    occupied_slots = sum(1 for value in slots if value != 0)
    free_slots = total_slots - occupied_slots
    max_free_block = max_contiguous_free_slots(slots)

    return {
        "occupied_slots": occupied_slots,
        "total_slots": total_slots,
        "free_slots": free_slots,
        "max_contiguous_free_slots": max_free_block,
    }


def flatten_link_slot_occupancy_state(link_status_forward):
    """
    Save slot occupancy summary for each physical link using ONE direction only.

    This uses link_status_forward only.

    For each physical link, store:
    - occupied_slots
    - total_slots
    - free_slots
    - slot_utilization
    - max_contiguous_free_slots

    Important:
    We do NOT add forward + backward.
    So initially, if only one C-band fiber is lit:
        total_slots = 320
    """

    row = {}

    for link_id, fibers in sorted(link_status_forward.items()):

        occupied_slots = 0
        total_slots = 0
        free_slots = 0
        max_free_block = 0

        for _, fiber_obj in fibers.items():

            if not isinstance(fiber_obj, dict):
                continue

            # --------------------------------------------------
            # Single-core fiber case
            # --------------------------------------------------
            if "lit" in fiber_obj and "slots" in fiber_obj:

                # Do not count dark fibers.
                if not fiber_obj.get("lit", False):
                    continue

                summary = summarize_slot_array(fiber_obj.get("slots", []))

                occupied_slots += summary["occupied_slots"]
                total_slots += summary["total_slots"]
                free_slots += summary["free_slots"]
                max_free_block = max(
                    max_free_block,
                    summary["max_contiguous_free_slots"]
                )

            # --------------------------------------------------
            # Multi-core fiber case
            # --------------------------------------------------
            else:
                core_keys = [k for k in fiber_obj.keys() if isinstance(k, int)]

                for core_id in core_keys:
                    core_obj = fiber_obj.get(core_id, {})

                    if not isinstance(core_obj, dict):
                        continue

                    # Do not count dark cores.
                    if not core_obj.get("lit", False):
                        continue

                    summary = summarize_slot_array(core_obj.get("slots", []))

                    occupied_slots += summary["occupied_slots"]
                    total_slots += summary["total_slots"]
                    free_slots += summary["free_slots"]
                    max_free_block = max(
                        max_free_block,
                        summary["max_contiguous_free_slots"]
                    )

        if total_slots > 0:
            slot_utilization = occupied_slots / total_slots
        else:
            slot_utilization = 0.0

        row[f"link_{link_id}_occupied_slots"] = occupied_slots
        row[f"link_{link_id}_total_slots"] = total_slots
        row[f"link_{link_id}_free_slots"] = free_slots
        row[f"link_{link_id}_slot_utilization"] = slot_utilization
        row[f"link_{link_id}_max_contiguous_free_slots"] = max_free_block

    return row


def append_xgboost_row(
    csv_path,
    seed,
    algorithm_name,
    cycle_number,
    current_day,
    current_blocking_probability,
    traffic_rates,
    link_status_forward,
    upgrade_needed,
    classify_fiber_state,
):
    """
    Append one XGBoost dataset row.

    Call this immediately after check_need_for_upgrade(),
    before applying any upgrade.

    This stores:
    - metadata
    - current network blocking probability
    - traffic matrix arrival rates lambda_i_j
    - current per-link technology state
    - current per-link slot occupancy summary from ONE direction
    - upgrade_needed label
    """

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    row = {}

    # -------------------------
    # Metadata
    # -------------------------
    row["seed"] = int(seed)
    row["split"] = get_seed_split(seed)
    row["algorithm"] = str(algorithm_name)
    row["cycle_number"] = int(cycle_number)
    row["current_day"] = float(current_day)

    # -------------------------
    # Current network-level feature
    # -------------------------
    row["current_BBP"] = float(current_blocking_probability)

    # -------------------------
    # Traffic matrix features
    # -------------------------
    row.update(flatten_traffic_matrix(traffic_rates))

    # -------------------------
    # Link technology features
    # -------------------------
    row.update(
        flatten_link_technology_state(
            link_status_forward=link_status_forward,
            classify_fiber_state=classify_fiber_state,
        )
    )

    # -------------------------
    # Link slot occupancy features
    # ONE direction only: link_status_forward
    # -------------------------
    row.update(
        flatten_link_slot_occupancy_state(
            link_status_forward=link_status_forward
        )
    )

    # -------------------------
    # Label
    # -------------------------
    row["upgrade_needed"] = int(bool(upgrade_needed))

    file_exists = csv_path.exists()

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)