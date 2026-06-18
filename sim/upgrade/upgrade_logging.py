import csv
from pathlib import Path
from typing import Optional

HEADER = [
    "day",
    "period_index",
    "scope",         # immediate | fiber_downtime | link_downtime
    "link_id",
    "upgrade_type",
    "fiber_id",
    "core_type",
]


def init_upgrade_log(results_dir: Path) -> Path:
    """
    Create the upgrade log CSV and write the header once.
    Returns the CSV path.
    """

    out_file = results_dir / "upgrade_log.csv"

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)

    return out_file


def log_upgrade_action(
    csv_file: Path,
    day: float,
    period_index: int,
    scope: str,
    link_id: int,
    upgrade_type: str,
    fiber_id: Optional[int] = None,
    core_type: Optional[int] = None,
):
    """
    Append one upgrade action to the upgrade log CSV.
    """

    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            float(day),
            int(period_index),
            scope,
            int(link_id),
            upgrade_type,
            "" if fiber_id is None else fiber_id,
            "" if core_type is None else core_type,
        ])