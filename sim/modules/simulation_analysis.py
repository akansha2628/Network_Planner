"""
simulation_analysis_basic_threshold_case.py

Clean analysis script for one threshold combination.

Folder naming assumed:
    algorithm_dt12_need_plan_sd

Example:
    increment_dt12_0.02_0.02_0.02
    greedy_dt12_0.02_0.02_0.02
    jump_to_max_dt12_0.02_0.02_0.02
"""

from __future__ import annotations

import ast
import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class SimulationAnalysis:
    @staticmethod
    def save_snapshot(snapshot, filename="simulation_snapshot.pkl"):
        import pickle
        with open(filename, "wb") as f:
            pickle.dump(snapshot, f)
        print(f"✓ Saved simulation snapshot to {filename}")

    @staticmethod
    def load_snapshot(filename="simulation_snapshot.pkl"):
        import pickle
        with open(filename, "rb") as f:
            snapshot = pickle.load(f)
        print(f"✓ Loaded simulation snapshot from {filename}")
        return snapshot

    @staticmethod
    def save_blocking_data(time_points, blocking_prob_points, filename="blocking_results.csv"):
        import csv
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time_days", "blocking_probability"])
            for t, bp in zip(time_points, blocking_prob_points):
                writer.writerow([t, bp])
        print(f"✓ Saved blocking data to {filename}")

    @staticmethod
    def print_simulation_summary(snapshot):
        print("\n" + "=" * 60)
        print("SIMULATION SUMMARY")
        print("=" * 60)
        print(f"Seed: {snapshot.get('seed', 'N/A')}")
        print(f"Final simulation time: {snapshot.get('Current_global_time', 0):.2f} days")
        print(f"Total connections: {snapshot.get('connection_id', 0)}")
        print(f"Counted connections: {snapshot.get('connection_count', 0)}")
        print(f"Blocked connections: {snapshot.get('blocked_connection_count', 0)}")

        if snapshot.get("connection_count", 0) > 0:
            bp = snapshot.get("blocked_connection_count", 0) / max(snapshot.get("connection_count", 1), 1)
            print(f"Final blocking probability: {bp:.4f}")

        print(f"\nNetwork Cost: ${snapshot.get('total_network_cost', 0):,.2f}")
        print("=" * 60 + "\n")

    @staticmethod
    def print_upgrade_timeline(upgrade_cost_timeline):
        if not upgrade_cost_timeline:
            print("No upgrades performed")
            return

        print("\n" + "=" * 60)
        print("UPGRADE TIMELINE")
        print("=" * 60)

        for i, event in enumerate(upgrade_cost_timeline, 1):
            print(f"\nEntry {i} at day {event.get('day', 0):.1f}:")
            print(f"  Equipment: ${event.get('equipment', 0):,.2f}")
            print(f"  Workforce: ${event.get('workforce', 0):,.2f}")
            print(f"  OPEX: ${event.get('opex', 0):,.2f}")
            print(f"  Total: ${event.get('total', 0):,.2f}")

        print("=" * 60 + "\n")

# =========================================================
# USER CONFIG
# =========================================================

# Parent folder containing all your result folders
BASE_RESULTS_DIR = Path(
    r"G:\My Drive\GWU_Phd\Prodigy_Extension\Network_Planner_max_start\results\Results_new"
)

# Threshold case you want to plot
NEED_THRESHOLD = "0.01"
PLAN_THRESHOLD = "0.02"
SD_THRESHOLD = "0.02"

CASE_SUFFIX = f"{NEED_THRESHOLD}_{PLAN_THRESHOLD}_{SD_THRESHOLD}"

OUTDIR = BASE_RESULTS_DIR / f"analysis_dt12_{CASE_SUFFIX}"

SCENARIO_DIRS: Dict[str, Path] = {
     "No Upgrade": BASE_RESULTS_DIR / f"no_upgrade_dt12",
     "Random": BASE_RESULTS_DIR / f"random_dt12",
    "Greedy": BASE_RESULTS_DIR / f"greedy_dt12_{CASE_SUFFIX}",
    "Jump-to-Max": BASE_RESULTS_DIR / f"jump_to_max_dt12_{CASE_SUFFIX}",
    "Adaptive Planning": BASE_RESULTS_DIR / f"increment_dt12_{CASE_SUFFIX}",
}

DISPLAY_ORDER = ["No Upgrade", "Random", "Greedy", "Jump-to-Max", "Adaptive Planning"]

COLORS = {
    "No Upgrade": "red",
    "Random": "#FF8C00",
    "Greedy": "deepskyblue",
    "Jump-to-Max": "green",
    "Adaptive Planning":"#5B1A18",
}

BLOCKING_CSV_NAME = "blocking_results.csv"
FINAL_STATE_NAME = "final_state.pkl"
SIM_LOG_NAME = "simulation_log.txt"

SHOW_THRESHOLD = True
LIFETIME_THRESHOLD = 0.1

# =========================================================
# BASIC HELPERS
# =========================================================

def ensure_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)


def cleanup_old_plots(outdir: Path) -> None:
    if not outdir.exists():
        return
    for p in outdir.glob("*.png"):
        try:
            p.unlink()
        except Exception as e:
            print(f"[WARN] Could not delete {p}: {e}")


def assert_folder(folder: Path, label: str) -> None:
    if folder.exists():
        return
    available = []
    if BASE_RESULTS_DIR.exists():
        available = sorted([p.name for p in BASE_RESULTS_DIR.iterdir() if p.is_dir()])
    raise FileNotFoundError(
        f"Folder not found for {label}: {folder}\n"
        f"BASE_RESULTS_DIR: {BASE_RESULTS_DIR}\n"
        f"Available folders:\n  " + "\n  ".join(available)
    )


def load_snapshot(folder: Path) -> Dict[str, Any]:
    p = folder / FINAL_STATE_NAME
    if not p.exists():
        raise FileNotFoundError(f"Missing final_state.pkl: {p}")
    with open(p, "rb") as f:
        return pickle.load(f)


def load_blocking_csv(folder: Path) -> pd.DataFrame:
    p = folder / BLOCKING_CSV_NAME
    if not p.exists():
        raise FileNotFoundError(f"Missing blocking_results.csv: {p}")
    df = pd.read_csv(p)
    required = {"time_days", "blocking_probability"}
    if not required.issubset(df.columns):
        raise ValueError(f"{p} must contain {required}. Found: {list(df.columns)}")
    df["time_days"] = pd.to_numeric(df["time_days"], errors="coerce")
    df["blocking_probability"] = pd.to_numeric(df["blocking_probability"], errors="coerce")
    return df.dropna(subset=["time_days", "blocking_probability"]).sort_values("time_days").reset_index(drop=True)


def load_cost_timeline(folder: Path) -> pd.DataFrame:
    snap = load_snapshot(folder)
    timeline = snap.get("upgrade_cost_timeline", [])
    df = pd.DataFrame(timeline)
    expected = ["day", "equipment", "workforce", "opex", "total", "cumulative_total"]
    if df.empty:
        return pd.DataFrame(columns=expected)
    for col in expected:
        if col not in df.columns:
            df[col] = 0.0
    for col in expected:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df[expected].sort_values("day").reset_index(drop=True)


def parse_simulation_log_connections(folder: Path) -> Dict[str, int | None]:
    p = folder / SIM_LOG_NAME
    if not p.exists():
        return {"total": None, "blocked": None, "accepted": None}
    text = p.read_text(encoding="utf-8", errors="ignore")

    def last_int(pattern: str) -> int | None:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if not matches:
            return None
        val = matches[-1]
        if isinstance(val, tuple):
            val = val[-1]
        return int(str(val).replace(",", ""))

    total = last_int(r"Total connections:\s*([0-9,]+)")
    blocked = last_int(r"Blocked connections:\s*([0-9,]+)")
    if total is None:
        total = last_int(r"Counted connections:\s*([0-9,]+)")
    if blocked is None:
        blocked = last_int(r"Blocked:\s*([0-9,]+)")
    accepted = total - blocked if total is not None and blocked is not None else None
    return {"total": total, "blocked": blocked, "accepted": accepted}

# =========================================================
# ANNUAL BUDGET CURVE
# =========================================================

def build_annual_budget_curve(
    end_day: float,
    period_opex_budget_0: float,
    period_capex_budget_0: float,
    period_total_budget_0: float,
    inflation_rate: float,
) -> pd.DataFrame:
    """
    Cumulative annual budget curve matching the yearly budget logic:
      - annual budget fixed within each year
      - inflation once per year
      - no carry-forward across years
    """
    max_day = int(np.ceil(end_day))
    days = np.arange(0, max_day + 1, dtype=float)
    cum_opex = np.zeros(max_day + 1)
    cum_capex = np.zeros(max_day + 1)
    cum_total = np.zeros(max_day + 1)
    opex_running = capex_running = total_running = 0.0
    current_year = -1

    for day in range(1, max_day + 1):
        year_idx = (day - 1) // 365
        if year_idx != current_year:
            current_year = year_idx
            growth = (1.0 + inflation_rate) ** year_idx
            opex_running += period_opex_budget_0 * growth
            capex_running += period_capex_budget_0 * growth
            total_running += period_total_budget_0 * growth
        cum_opex[day] = opex_running
        cum_capex[day] = capex_running
        cum_total[day] = total_running

    return pd.DataFrame({
        "day": days,
        "cum_opex_budget": cum_opex,
        "cum_capex_budget": cum_capex,
        "cum_total_budget": cum_total,
    })

# =========================================================
# TECHNOLOGY MIX HELPERS
# =========================================================

def load_upgrade_cycle_summary(folder: Path) -> pd.DataFrame:
    p = folder / "upgrade_cycle_summary.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing upgrade_cycle_summary.csv: {p}")
    df = pd.read_csv(p)
    if "time_day" not in df.columns:
        raise ValueError(f"{p} must contain time_day")
    df["time_day"] = pd.to_numeric(df["time_day"], errors="coerce")
    return df.dropna(subset=["time_day"]).sort_values("time_day").reset_index(drop=True)


def build_technology_mix_dataframe(folder: Path) -> pd.DataFrame:
    cycle_df = load_upgrade_cycle_summary(folder)
    snap = load_snapshot(folder)
    final_link_status = snap.get("link_status_forward", {})
    if not isinstance(final_link_status, dict) or not final_link_status:
        raise ValueError("Could not infer link/fiber structure from final_state.pkl")

    fiber_states: Dict[Any, Dict[Any, str]] = {}
    for link_id, fibers in final_link_status.items():
        if not isinstance(fibers, dict):
            continue
        fiber_states[link_id] = {}
        for fiber_id in sorted(fibers.keys()):
            fiber_states[link_id][fiber_id] = "SC_C" if fiber_id == 0 else "DARK"

    def count_states():
        sc_c = sc_cl = mc_c = mc_cl = 0
        for lid in fiber_states:
            for _, st in fiber_states[lid].items():
                if st == "SC_C": sc_c += 1
                elif st == "SC_CL": sc_cl += 1
                elif st == "MC_C": mc_c += 3
                elif st == "MC_CL": mc_cl += 3
        return sc_c, sc_cl, mc_c, mc_cl

    rows = []
    sc_c, sc_cl, mc_c, mc_cl = count_states()
    rows.append({"upgrade_index": 0, "event_day": 0.0, "SC_C": sc_c, "SC_CL": sc_cl, "MC_C": mc_c, "MC_CL": mc_cl})

    for idx, row in cycle_df.iterrows():
        event_day = float(row["time_day"])
        decisions_raw = row.get("final_decisions", "[]")
        try:
            decisions = ast.literal_eval(decisions_raw) if isinstance(decisions_raw, str) else []
        except Exception:
            decisions = []

        by_link = defaultdict(list)
        for dec in decisions:
            lid = dec.get("link_id")
            if lid is not None:
                by_link[lid].append(dec)

        for lid, decs in by_link.items():
            if lid not in fiber_states:
                continue
            has_core = any(d.get("upgrade_type") == "core_upgrade" for d in decs)
            if has_core:
                for fid, st in list(fiber_states[lid].items()):
                    if st == "SC_C": fiber_states[lid][fid] = "MC_C"
                    elif st == "SC_CL": fiber_states[lid][fid] = "MC_CL"

            for dec in decs:
                upg = dec.get("upgrade_type")
                fid = dec.get("fiber_id")
                core_type = dec.get("core_type", None)
                if upg == "core_upgrade" or fid is None:
                    continue
                if fid not in fiber_states[lid]:
                    fiber_states[lid][fid] = "DARK"
                cur = fiber_states[lid][fid]
                if upg == "band_upgrade":
                    if cur == "SC_C": fiber_states[lid][fid] = "SC_CL"
                    elif cur == "MC_C": fiber_states[lid][fid] = "MC_CL"
                elif upg == "new_fiber_C":
                    fiber_states[lid][fid] = "MC_C" if core_type == 3 else "SC_C"
                elif upg == "new_fiber_CL":
                    fiber_states[lid][fid] = "MC_CL" if core_type == 3 else "SC_CL"

        sc_c, sc_cl, mc_c, mc_cl = count_states()
        rows.append({"upgrade_index": idx + 1, "event_day": event_day, "SC_C": sc_c, "SC_CL": sc_cl, "MC_C": mc_c, "MC_CL": mc_cl})

    return pd.DataFrame(rows)

# =========================================================
# PLOTS
# =========================================================

def plot_blocking(blocking_data: Dict[str, pd.DataFrame]) -> None:
    plt.figure(figsize=(12, 5))
    for label in DISPLAY_ORDER:
        df = blocking_data.get(label)
        if df is None or df.empty:
            continue
        plt.semilogy(df["time_days"], df["blocking_probability"], label=label, color=COLORS.get(label), linewidth=2)
    if SHOW_THRESHOLD:
        plt.axhline(LIFETIME_THRESHOLD, linestyle=":", color="black", linewidth=1.8, label=f"Threshold={LIFETIME_THRESHOLD}")
    plt.title(f"Blocking Probability vs Time (Log Scale) - {CASE_SUFFIX}")
    plt.xlabel("Time (days)")
    plt.ylabel("Blocking Probability")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "blocking_log.png", dpi=220)
    plt.close()

    plt.figure(figsize=(12, 5))
    for label in DISPLAY_ORDER:
        df = blocking_data.get(label)
        if df is None or df.empty:
            continue
        plt.plot(df["time_days"], df["blocking_probability"], label=label, color=COLORS.get(label), linewidth=2)
    if SHOW_THRESHOLD:
        plt.axhline(LIFETIME_THRESHOLD, linestyle=":", color="black", linewidth=1.8, label=f"Threshold={LIFETIME_THRESHOLD}")
    plt.title(f"Blocking Probability vs Time (Linear Scale) - {CASE_SUFFIX}")
    plt.xlabel("Time (days)")
    plt.ylabel("Blocking Probability")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "blocking_linear.png", dpi=220)
    plt.close()


def plot_cumulative_cost(cost_data: Dict[str, pd.DataFrame]) -> None:
    plt.figure(figsize=(12, 5.5))
    max_day = 0.0
    budget_params = None
    for label in DISPLAY_ORDER:
        df = cost_data.get(label)
        if df is None or df.empty:
            continue
        df = df.sort_values("day").reset_index(drop=True)
        max_day = max(max_day, float(df["day"].max()))
        plt.plot(df["day"], df["cumulative_total"], label=label, color=COLORS.get(label), linewidth=2)
        if budget_params is None:
            try:
                budget_params = load_snapshot(SCENARIO_DIRS[label]).get("budget_params", None)
            except Exception:
                budget_params = None

    if budget_params is not None and max_day > 0:
        budget_df = build_annual_budget_curve(
            end_day=max_day,
            period_opex_budget_0=float(budget_params["period_opex_budget_0"]),
            period_capex_budget_0=float(budget_params["period_capex_budget_0"]),
            period_total_budget_0=float(budget_params["period_total_budget_0"]),
            inflation_rate=float(budget_params["inflation_rate"]),
        )
        plt.step(budget_df["day"], budget_df["cum_total_budget"], where="post", linestyle=":", color="black", linewidth=2.5, label="Cumulative Annual Budget")

    plt.title(f"Cumulative Total Cost vs Annual Budget - {CASE_SUFFIX}")
    plt.xlabel("Day")
    plt.ylabel("Cumulative Total Cost ($)")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "cumulative_cost_vs_budget.png", dpi=220)
    plt.close()


def plot_cost_breakdown(cost_data: Dict[str, pd.DataFrame]) -> None:
    labels = [x for x in DISPLAY_ORDER if x in cost_data]
    eq_vals, wf_vals, op_vals, total_vals = [], [], [], []
    for label in labels:
        df = cost_data.get(label)
        if df is None or df.empty:
            eq_vals.append(0.0); wf_vals.append(0.0); op_vals.append(0.0); total_vals.append(0.0); continue
        try:
            life = float(load_snapshot(SCENARIO_DIRS[label]).get("Current_global_time", 1.0))
        except Exception:
            life = 1.0
        life = max(life, 1.0)
        eq_vals.append(float(df["equipment"].sum()) / life)
        wf_vals.append(float(df["workforce"].sum()) / life)
        op_vals.append(float(df["opex"].sum()) / life)
        total_vals.append(float(df["total"].sum()) / life)

    x = np.arange(len(labels)); width = 0.2
    plt.figure(figsize=(11, 5.5))
    plt.bar(x - 1.5 * width, wf_vals, width, label="Workforce")
    plt.bar(x - 0.5 * width, op_vals, width, label="OPEX")
    plt.bar(x + 0.5 * width, eq_vals, width, label="Equipment")
    plt.bar(x + 1.5 * width, total_vals, width, label="Total")
    plt.title(f"Normalized Cost Breakdown - {CASE_SUFFIX}")
    plt.xlabel("Algorithm")
    plt.ylabel("Normalized Cost ($/day)")
    plt.xticks(x, labels, rotation=15, ha="right")
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "cost_breakdown_normalized.png", dpi=220)
    plt.close()


def plot_connection_stats(stats: Dict[str, Dict[str, int | None]]) -> None:
    labels = [x for x in DISPLAY_ORDER if x in stats]
    blocked = np.array([stats[x].get("blocked") or 0 for x in labels], dtype=float)
    accepted = np.array([stats[x].get("accepted") or 0 for x in labels], dtype=float)
    total = np.array([stats[x].get("total") or 0 for x in labels], dtype=float)
    x = np.arange(len(labels)); width = 0.25
    plt.figure(figsize=(10, 5))
    plt.bar(x - width, blocked, width, label="Blocked")
    plt.bar(x, accepted, width, label="Accepted")
    plt.bar(x + width, total, width, label="Total")
    plt.title(f"Connection Statistics - {CASE_SUFFIX}")
    plt.xlabel("Algorithm")
    plt.ylabel("Connection Count")
    plt.xticks(x, labels, rotation=15, ha="right")
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "connection_stats.png", dpi=220)
    plt.close()


def plot_technology_mix_all() -> None:

    technology_order = [
        "Adaptive Planning",
        "Greedy",
        "Jump-to-Max",
        "Random",
    ]

    available = []

    for label in technology_order:
        folder = SCENARIO_DIRS.get(label)

        if folder is None or not folder.exists():
            continue

        try:
            df = build_technology_mix_dataframe(folder)
            available.append((label, df))

        except Exception as e:
            print(f"[WARN] Technology mix unavailable for {label}: {e}")

    if not available:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes = axes.flatten()

    for ax, (label, df) in zip(axes, available):

        x = np.arange(len(df))

        sc_c = df["SC_C"].values
        sc_cl = df["SC_CL"].values
        mc_c = df["MC_C"].values
        mc_cl = df["MC_CL"].values

        ax.bar(
            x,
            sc_c,
            label="SC_C",
            hatch="///",
            edgecolor="black",
        )

        ax.bar(
            x,
            sc_cl,
            bottom=sc_c,
            label="SC_CL",
            hatch="\\\\\\",
            edgecolor="black",
        )

        ax.bar(
            x,
            mc_c,
            bottom=sc_c + sc_cl,
            label="MC_C",
            hatch="xxx",
            edgecolor="black",
        )

        ax.bar(
            x,
            mc_cl,
            bottom=sc_c + sc_cl + mc_c,
            label="MC_CL",
            hatch="...",
            edgecolor="black",
        )

        xtick_labels = [
            f"{int(idx)}\n({int(day)}d)"
            for idx, day in zip(df["upgrade_index"], df["event_day"])
        ]

        ax.set_xticks(x)
        ax.set_xticklabels(xtick_labels)

        ax.set_title(label)

        ax.set_xlabel("Upgrade Index (Day)")
        ax.set_ylabel("Count")

        ax.grid(True, axis="y", linestyle="--", linewidth=0.5)

        ax.legend(fontsize=8)

    plt.tight_layout()

    plt.savefig(
        OUTDIR / "technology_mix_all.png",
        dpi=220,
    )

    plt.close()


# =========================================================
# ANNUAL BUDGET UTILIZATION PLOT
# =========================================================

def build_annual_budget_utilization(folder: Path) -> pd.DataFrame:
    """
    Build year-by-year budget utilization for one algorithm folder.

    This matches your annual budget model:
      - each year receives a fresh annual budget
      - annual budget grows by inflation
      - unused budget does not carry forward
      - spending is grouped by the simulation day's year
    """
    snap = load_snapshot(folder)
    budget_params = snap.get("budget_params", None)

    if not isinstance(budget_params, dict):
        raise ValueError(f"budget_params missing in {folder / FINAL_STATE_NAME}")

    total_budget_0 = float(budget_params["period_total_budget_0"])
    opex_budget_0 = float(budget_params["period_opex_budget_0"])
    capex_budget_0 = float(budget_params["period_capex_budget_0"])
    inflation_rate = float(budget_params["inflation_rate"])

    cost_df = load_cost_timeline(folder)

    # Use lifetime to include years with zero spending as well.
    lifetime = float(snap.get("Current_global_time", 0.0))
    max_year_from_lifetime = int(lifetime // 365) if lifetime > 0 else 0

    if cost_df.empty:
        max_year = max_year_from_lifetime
        yearly_used = pd.DataFrame(columns=["year_idx", "used_total", "used_opex", "used_capex_workforce"])
    else:
        cost_df = cost_df.copy()
        cost_df["year_idx"] = (cost_df["day"] // 365).astype(int)
        cost_df["capex_workforce"] = cost_df["equipment"] + cost_df["workforce"]

        yearly_used = (
            cost_df.groupby("year_idx", as_index=False)
            .agg(
                used_total=("total", "sum"),
                used_opex=("opex", "sum"),
                used_capex_workforce=("capex_workforce", "sum"),
            )
        )

        max_year_from_cost = int(cost_df["year_idx"].max()) if not cost_df.empty else 0
        max_year = max(max_year_from_lifetime, max_year_from_cost)

    rows = []

    for year_idx in range(max_year + 1):
        growth = (1.0 + inflation_rate) ** year_idx

        allocated_total = total_budget_0 * growth
        allocated_opex = opex_budget_0 * growth
        allocated_capex = capex_budget_0 * growth

        used_row = yearly_used[yearly_used["year_idx"] == year_idx]

        if not used_row.empty:
            used_total = float(used_row.iloc[0]["used_total"])
            used_opex = float(used_row.iloc[0]["used_opex"])
            used_capex = float(used_row.iloc[0]["used_capex_workforce"])
        else:
            used_total = 0.0
            used_opex = 0.0
            used_capex = 0.0

        rows.append(
            {
                "year": year_idx + 1,
                "year_idx": year_idx,
                "allocated_total_budget": allocated_total,
                "allocated_opex_budget": allocated_opex,
                "allocated_capex_budget": allocated_capex,
                "used_total": used_total,
                "used_opex": used_opex,
                "used_capex_workforce": used_capex,
                "remaining_total": allocated_total - used_total,
            }
        )

    return pd.DataFrame(rows)


def plot_annual_budget_used_vs_allocated() -> None:
    """
    One graph showing, for every simulation year:
      - annual allocated budget as a black outlined bar
      - actual yearly spending for each algorithm as colored bars

    This is the clearest plot for your no-carry-forward annual budget model.
    """
    yearly_data: Dict[str, pd.DataFrame] = {}
    max_years = 0

    for label in DISPLAY_ORDER:
        folder = SCENARIO_DIRS.get(label)
        if folder is None or not folder.exists():
            continue

        try:
            df = build_annual_budget_utilization(folder)
            yearly_data[label] = df
            max_years = max(max_years, len(df))
        except Exception as e:
            print(f"[WARN] Could not build annual budget utilization for {label}: {e}")

    if not yearly_data or max_years == 0:
        print("[WARN] No annual budget utilization data available.")
        return

    base_x = np.arange(max_years)
    used_width = 0.18
    budget_width = 0.82

    used_width = 0.14

    offsets = {
        "No Upgrade": -2 * used_width,
        "Random": -used_width,
        "Greedy": 0.0,
        "Jump-to-Max": used_width,
        "Adaptive Planning": 2 * used_width,
    }

    plt.figure(figsize=(14, 6.5))

    # Annual budget is the same across algorithms for this threshold case.
    # Use the first available algorithm to draw the budget bars.
    first_label = next(iter(yearly_data))
    budget_df = yearly_data[first_label]

    # Pad budget values if another algorithm lives longer than first_label.
    if len(budget_df) < max_years:
        folder = SCENARIO_DIRS[first_label]
        snap = load_snapshot(folder)
        budget_params = snap.get("budget_params", {})
        total_budget_0 = float(budget_params["period_total_budget_0"])
        inflation_rate = float(budget_params["inflation_rate"])
        budget_values = [total_budget_0 * ((1.0 + inflation_rate) ** i) for i in range(max_years)]
    else:
        budget_values = budget_df["allocated_total_budget"].values[:max_years]

    plt.bar(
        base_x,
        budget_values,
        width=budget_width,
        color="none",
        edgecolor="black",
        linewidth=2.0,
        label="Annual Budget",
        zorder=1,
    )

    for label in DISPLAY_ORDER:
        if label not in yearly_data:
            continue

        df = yearly_data[label]
        x = base_x[: len(df)] + offsets.get(label, 0.0)

        plt.bar(
            x,
            df["used_total"],
            width=used_width,
            color=COLORS.get(label),
            label=f"{label} Used",
            zorder=3,
        )

    plt.xticks(base_x, [f"Year {i + 1}" for i in range(max_years)])
    plt.title(f"Annual Budget Used vs Allocated - {CASE_SUFFIX}")
    plt.xlabel("Simulation Year")
    plt.ylabel("Budget / Spending ($)")
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5, zorder=0)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(OUTDIR / "annual_budget_used_vs_allocated.png", dpi=220)
    plt.close()

    # Also save the numerical table for checking/debugging.
    rows = []
    for label, df in yearly_data.items():
        temp = df.copy()
        temp.insert(0, "algorithm", label)
        rows.append(temp)

    if rows:
        table = pd.concat(rows, ignore_index=True)
        table.to_csv(OUTDIR / "annual_budget_utilization_table.csv", index=False)
        print(f"✅ Saved annual budget utilization table: {OUTDIR / 'annual_budget_utilization_table.csv'}")

def print_summary(blocking_data: Dict[str, pd.DataFrame], cost_data: Dict[str, pd.DataFrame], stats: Dict[str, Dict[str, int | None]]) -> None:
    print("\n" + "=" * 90)
    print(f"SUMMARY FOR THRESHOLD CASE: need={NEED_THRESHOLD}, plan={PLAN_THRESHOLD}, sd={SD_THRESHOLD}")
    print("=" * 90)
    for label in DISPLAY_ORDER:
        folder = SCENARIO_DIRS.get(label)
        if folder is None or not folder.exists():
            continue
        try:
            snap = load_snapshot(folder)
            lifetime = float(snap.get("Current_global_time", 0.0))
        except Exception:
            lifetime = 0.0
        dfb = blocking_data.get(label)
        final_bp = float(dfb["blocking_probability"].iloc[-1]) if dfb is not None and not dfb.empty else np.nan
        dfc = cost_data.get(label)
        final_cost = float(dfc["cumulative_total"].iloc[-1]) if dfc is not None and not dfc.empty else np.nan
        st = stats.get(label, {})
        print(f"\n{label}")
        print(f"  Folder      : {folder.name}")
        print(f"  Lifetime    : {lifetime:.2f} days")
        print(f"  Final BP    : {final_bp:.6f}")
        print(f"  Final Cost  : ${final_cost:,.2f}")
        print(f"  Connections : total={st.get('total')}, accepted={st.get('accepted')}, blocked={st.get('blocked')}")
    print("=" * 90 + "\n")

# =========================================================
# MAIN
# =========================================================

def main() -> None:
    print("BASE_RESULTS_DIR =", BASE_RESULTS_DIR)
    print("OUTDIR =", OUTDIR)
    print("CASE_SUFFIX =", CASE_SUFFIX)
    ensure_outdir(OUTDIR)
    cleanup_old_plots(OUTDIR)

    blocking_data: Dict[str, pd.DataFrame] = {}
    cost_data: Dict[str, pd.DataFrame] = {}
    stats: Dict[str, Dict[str, int | None]] = {}

    for label in DISPLAY_ORDER:
        folder = SCENARIO_DIRS[label]
        try:
            assert_folder(folder, label)
            blocking_data[label] = load_blocking_csv(folder)
            cost_data[label] = load_cost_timeline(folder)
            stats[label] = parse_simulation_log_connections(folder)
            print(f"✓ Loaded {label}: {folder.name}")
        except Exception as e:
            print(f"[WARN] Skipping {label}: {e}")

    if not blocking_data:
        raise RuntimeError("No blocking data loaded. Check SCENARIO_DIRS and folder names.")

    plot_blocking(blocking_data)
    plot_cumulative_cost(cost_data)
    plot_cost_breakdown(cost_data)
    plot_connection_stats(stats)
    plot_technology_mix_all()
    plot_annual_budget_used_vs_allocated()
    print_summary(blocking_data, cost_data, stats)
    print(f"✅ Done. Plots saved to: {OUTDIR}")


if __name__ == "__main__":
    main()
