"""
Optical Network Simulator - Main Entry Point

A modular optical network simulator with network upgrade planning capabilities.
Configure simulation parameters in the CONFIG dictionary below.
"""

import sys
import random
from math import floor

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import math
import copy
import csv
from pathlib import Path
from typing import Dict, Tuple, Any, Optional
import time
from collections import defaultdict


# ============================================================================
# CONFIGURATION - Edit this dictionary to configure your simulation
# ============================================================================

CONFIG = {
    # Topology: 'test5' (5-node test network) or 'dt12' (12-node production network)
    'topology': 'dt12',

    # Algorithm: 'greedy', 'increment', 'max_initial', 'jump_to_max', 'no_upgrade'
    'algorithm': 'greedy',  # or 'max_initial' or 'jump_to_max'

    # Link selection mode:
    # "path_congestion" -> current logic
    # "most_congested" -> for each critical SD pair, select the highest-utilization link
    #                     from its most congested path
     'link_selection_mode': 'path_congestion',

    # Used only when link_selection_mode = "most_congested"
    'link_score_threshold': 0.02,

    # ---------------- Multi-seed running ----------------
    'run_multiple_seeds': True,
    'seed_start': 0,
    'seed_end': 0,
     'max_upgrade_cycles_for_debug': 5,

    # ---------------- Budget envelope settings ----------------
    'budget_enabled': True,

    'budget_selection_mode': 'budget_aware',
    # or
    # 'budget_selection_mode': 'budget_unaware',

    # ---------------- ML/XGBoost data collection ----------------
    'collect_xgboost_data': True,
    'xgboost_csv_name': 'xgboost_need_upgrade_dataset.csv',
    'reset_xgboost_csv_before_run': True,


    # # Allowed daily OPEX budget = factor × initial daily OPEX
    # 'opex_budget_factor': 3,
    #
    # # Daily OPEX budget grows every upgrade period (90 days)
    # 'opex_budget_growth_per_period': 0.2,
    #
    # # If None, auto-compute CAPEX+workforce budget per upgrade period
    # 'period_capex_workforce_budget_0': None,
    #
    # # Starting CAPEX+workforce budget as a fraction of 90-day OPEX budget
    # 'upgrade_budget_factor': 2.0,
    #
    # # CAPEX+workforce budget growth per upgrade period
    # 'capex_workforce_budget_growth_per_period': 0.2,
}

# ============================================================================
# Import simulation modules from sim package
# ============================================================================

from sim.topologies.topology_manager import TopologyManager
from sim.core import topology as Topology
from sim.algorithms.algorithm_selector import AlgorithmSelector
from sim.core.constants import *
from sim.core.traffic_generator import NetworkTrafficGenerator
from sim.routing.shortest_path import build_k_shortest_paths
from ml_xgboost_logger import append_xgboost_row

from sim.upgrade.upgrade_manager import (
    initialize_link_status,
    choose_upgrade_type,
    perform_upgrade,
    restore_removed_links_after_downtime,
    network_connectivity,
    print_upgrade_decisions_summary,
)
# >>> CHANGE 1 END

from sim.algorithms.greedy import choose_upgrade_type_greedy_max
from sim.upgrade.plan_checker import plan_checker
from sim.upgrade.opex_model import (
    compute_running_opex_interval,
    compute_network_opex_per_day,
    count_network_fibers_by_technology,
)

# >>> CHANGE 2 START: import start_link_downtime_job
from sim.upgrade.sequential_upgrade import (
    start_downtime_upgrade_for_link,
    start_link_downtime_job,
    start_combined_fiber_downtime_job,
)
# >>> CHANGE 2 END

from sim.modules.performance_metrics import PerformanceMetrics
from sim.modules.simulation_analysis import SimulationAnalysis
from sim.routing.rsa import execute_first_fit

from sim.upgrade.link_utilization_fragmentation import (
    init_monthly_rowwise_csvs,
    log_monthly_link_stats_rowwise,
)

from sim.upgrade.upgrade_logging import init_upgrade_log, log_upgrade_action
from sim.upgrade.cost_model import compute_upgrade_costs


# ============================================================================
# Helper Functions
# ============================================================================

def create_results_directory(topology: str, algorithm: str) -> Path:
    """Create organized results directory for simulation outputs."""
    results_dir = Path('results') / f"{algorithm}_{topology}"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def sd_key_undirected(s: int, d: int):
    """Create undirected source-destination pair key for tracking."""
    return (s, d) if s < d else (d, s)


def init_connection_request_log(results_dir: Path) -> Path:
    """
    Create CSV to log every arriving connection request.
    """
    out_file = results_dir / "connection_requests.csv"

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "connection_id",
            "source",
            "destination",
            "datarate",
            "arrival_time",
            "holding_time",
            "departure_time",
            "status",              # accepted / blocked
            "mf_index",
            "fs_index",
            "slice_window_size",
            "path",
            "link_ids",
            "fibers_used",
            "cores_used",
        ])

    return out_file

def init_daily_network_technology_log(results_dir: Path) -> Path:
    out_file = results_dir / "daily_network_technology_summary.csv"
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "day",
            "SC_C",
            "SC_CL",
            "MC_C",
            "MC_CL",
            "daily_opex",
            "cumulative_total_cost",
            "blocking_probability",
            "active_connections",
        ])
    return out_file


def init_upgrade_cycle_summary_log(results_dir: Path) -> Path:
    out_file = results_dir / "upgrade_cycle_summary.csv"
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "cycle",
            "time_day",
            "blocking_probability",
            "num_final_links",
            "final_links",
            "final_decisions",
            "cycle_upgrade_level",
        ])
    return out_file

def init_pre_upgrade_per_link_technology_log(results_dir: Path) -> Path:
    out_file = results_dir / "pre_upgrade_per_link_technology.csv"
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "cycle",
            "time_day",
            "link_id",
            "SC_C",
            "SC_CL",
            "MC_C",
            "MC_CL",
        ])
    return out_file

def build_connection_request_row(
    connection_id: int,
    src: int,
    dest: int,
    datarate: float,
    arrival_time: float,
    holding_time: float,
    departure_time: float,
    status: str,
    mf_index,
    fs_index,
    slice_window_size,
    path,
    link_ids,
    fibers_used,
    cores_used,
):
    def safe_str(x):
        if x is None:
            return ""
        if isinstance(x, float) and math.isinf(x):
            return "inf"
        return str(x)

    return [
        connection_id,
        src,
        dest,
        datarate,
        arrival_time,
        holding_time,
        departure_time,
        status,
        safe_str(mf_index),
        safe_str(fs_index),
        safe_str(slice_window_size),
        safe_str(path),
        safe_str(link_ids),
        safe_str(fibers_used),
        safe_str(cores_used),
    ]


class TeeOutput:
    """Redirect output to both console and file simultaneously."""

    def __init__(self, *files):
        self.files = files

    def write(self, message):
        for f in self.files:
            try:
                f.write(message)
            except Exception:
                pass
        self.flush()

    def flush(self):
        for f in self.files:
            try:
                f.flush()
            except Exception:
                pass

UPGRADE_LEVEL_MAP = {
    "band_upgrade": 1,
    "new_fiber_C": 2,
    "new_fiber_CL": 3,
    "core_upgrade": 4,
}


def flatten_upgrade_decisions(decisions):
    """
    Convert upgrade_decisions dict into a flat list of per-link decisions.
    Input format:
        {
            link_id: [decision_dict, decision_dict, ...],
            ...
        }
    """
    rows = []

    if not decisions:
        return rows

    for lid, decs in decisions.items():
        if not isinstance(decs, list):
            continue

        for dec in decs:
            if dec is None:
                continue

            rows.append({
                "link_id": lid,
                "upgrade_type": dec.get("upgrade_type"),
                "fiber_id": dec.get("fiber_id"),
                "core_type": dec.get("core_type"),
            })

    return rows


def get_cycle_upgrade_level(flat_decisions):
    """
    Return one representative level for the whole cycle.
    Highest upgrade type in the cycle is used.
    """
    levels = []

    for dec in flat_decisions:
        upg = dec.get("upgrade_type")
        if upg in UPGRADE_LEVEL_MAP:
            levels.append(UPGRADE_LEVEL_MAP[upg])

    return max(levels) if levels else 0


# ============================================================================
# Simulation Initialization
# ============================================================================

def initialize_simulation(config: Dict[str, Any]):
    """
    Initialize simulation with topology and algorithm.

    Returns:
        tuple: (topology_data, algorithm, results_dir, extended_config)
    """
    # Load topology
    topology_data = TopologyManager.load_topology(config['topology'])

    # Set topology in legacy module for compatibility
    Topology.TOPOLOGY = topology_data['TOPOLOGY']
    Topology.TOPOLOGY_LINK_LENGTHS = topology_data['TOPOLOGY_LINK_LENGTHS']
    Topology.LINK_INDEX = topology_data['LINK_INDEX']
    Topology.N = topology_data['N']
    Topology.LINKS = topology_data['LINKS']

    # Load algorithm
    algorithm = AlgorithmSelector.get_algorithm(config['algorithm'])

    # Create results directory
    results_dir = create_results_directory(config['topology'], config['algorithm'])
    upgrade_log_csv = init_upgrade_log(results_dir)

    # Extend config with simulation parameters and output paths
    extended_config = {
        **config,
        'warmup_connections': WARMUP,
        'blocking_threshold': Total_number_of_allowed_blocked_probability_in_network_lifetime,
        'results_dir': results_dir,
        'blocking_csv': results_dir / 'blocking_results.csv',
        'snapshot_file': results_dir / 'final_state.pkl',
        'blocking_plot_file': results_dir / 'blocking_probability_vs_time.png',
        'cost_plot_file': results_dir / 'cost_vs_time.png',
        'metrics_plot_file': results_dir / 'performance_metrics.png',
        'log_file': results_dir / 'simulation_log.txt',
        'upgrade_log_csv': upgrade_log_csv,
        'connection_request_csv': results_dir / 'connection_requests.csv',
        'budget_plot_file': results_dir / 'budget_vs_cost.png',
    }

    return topology_data, algorithm, results_dir, extended_config



# ============================================================================
# Main Simulation Logic
# ============================================================================

def run_simulation(config: Dict[str, Any]):
    """
    Run the optical network simulation with given configuration.
    """
    topology_data, algorithm, results_dir, config = initialize_simulation(config)
    upgrade_log_csv = config["upgrade_log_csv"]
    connection_request_csv = init_connection_request_log(config["results_dir"])

    daily_network_csv = init_daily_network_technology_log(config["results_dir"])
    upgrade_cycle_csv = init_upgrade_cycle_summary_log(config["results_dir"])
    pre_upgrade_per_link_csv = init_pre_upgrade_per_link_technology_log(config["results_dir"])

    daily_network_file = open(daily_network_csv, "a", newline="", encoding="utf-8")
    daily_network_writer = csv.writer(daily_network_file)
    pre_upgrade_per_link_file = open(pre_upgrade_per_link_csv, "a", newline="", encoding="utf-8")
    pre_upgrade_per_link_writer = csv.writer(pre_upgrade_per_link_file)

    upgrade_cycle_file = open(upgrade_cycle_csv, "a", newline="", encoding="utf-8")
    upgrade_cycle_writer = csv.writer(upgrade_cycle_file)

    config["connection_request_csv"] = connection_request_csv
    connection_log_file = open(connection_request_csv, "a", newline="", encoding="utf-8")
    connection_log_writer = csv.writer(connection_log_file)

    # Wall-clock runtime measurement
    start_wall_time = time.time()

    print("="*70)
    print("OPTICAL NETWORK SIMULATOR")
    print("="*70)
    print(f"Topology: {config['topology']}")
    print(f"Algorithm: {algorithm.name}")
    print(f"Seed: {config['seed']}")
    print("="*70 + "\n")

    # Extract topology information
    TOPOLOGY = topology_data['TOPOLOGY']
    TOPOLOGY_LINK_LENGTHS = topology_data['TOPOLOGY_LINK_LENGTHS']
    LINK_INDEX = topology_data['LINK_INDEX']
    N = topology_data['N']
    LINKS = topology_data['LINKS']

    print(f"✓ Using topology: {topology_data['NAME']} ({N} nodes, {LINKS} links)\n")

    # Monthly per-link logging (row-wise CSV)
    MONTH_DAYS = 30
    current_month = 0
    util_csv, frag_csv = init_monthly_rowwise_csvs(config["results_dir"])

    # Initialize RNG
    seed = config['seed']
    random.seed(seed)
    main_rng = random.Random(seed)

    # Initialize metrics and analysis
    metrics = PerformanceMetrics()
    analysis = SimulationAnalysis()

    # ============================================================
    # PATH STATISTICS (NO lambda)
    # key = (undirected_s, undirected_d, path_index)
    # ============================================================

    path_stats = {}
    # Full-history path statistics across the whole simulation
    path_stats_history = {}

    # Initialize network state
    total_network_cost = 0
    upgrade_cost_timeline = []
    connection_id = 0
    upgrade_decision_history = []
    budget_tracker = {}
    removed_links_info = {}
    Current_global_time = 0
    current_week = 0
    current_upgrade = 0
    last_cost_time = 0.0
    last_logged_day = -1

    # >>> CHANGE 3 START: persistent downtime JOB queue (does not reset each cycle)
    # This holds entries like:
    #   ("fiber", link_id, decision_dict)
    #   ("link",  link_id, [decision_dict, decision_dict, ...])   # core first
    link_execution_queue = []
    per_link_jobs = {}
    # >>> CHANGE 3 END

    # Validate traffic parameters
    if lambda_0 <= 0:
        raise ValueError("lambda_0 must be > 0")
    if MEAN_HOLDING_TIME <= 0:
        raise ValueError("MEAN_HOLDING_TIME must be > 0")
    if Traffic_growth_days <= 0:
        raise ValueError("Traffic_growth_days must be > 0")

    # Validate topology parameters
    if N <= 0:
        raise ValueError("Number of nodes (N) must be > 0")
    if LINKS <= 0:
        raise ValueError("Number of links (LINKS) must be > 0")

    # Validate algorithm
    if algorithm is None:
        raise ValueError("Algorithm cannot be None")

    traffic_rates = np.full((N, N), lambda_0, dtype=float)
    np.fill_diagonal(traffic_rates, 0)

    ALL_DEMANDS = []
    working_topology = copy.deepcopy(TOPOLOGY)

    # Initialize link status
    if hasattr(algorithm, 'initialize_link_status'):
        link_status_forward, link_status_backward = algorithm.initialize_link_status(LINKS)
    else:
        link_status_forward, link_status_backward = initialize_link_status(LINKS)

    if not link_status_forward or not link_status_backward:
        raise ValueError("Link status initialization returned empty dicts")

    # ============================================================
    # MAX-INITIAL: apply one-time day-0 CAPEX + workforce
    # ============================================================
    if algorithm.name == "max_initial":
        # Build synthetic upgrade decisions for ALL links.
        # We use a single core_upgrade per link and let cost_model compute
        # the full jump-to-max cost via algorithm_name="jump_to_max".
        initial_upgrade_decisions = {}

        for link_id in range(LINKS):
            initial_upgrade_decisions[link_id] = [
                {
                    "upgrade_type": "core_upgrade",
                    "fiber_id": None,
                    "core_type": 1
                }
            ]

        initial_links = list(initial_upgrade_decisions.keys())

        initial_cost, cost_summary, cost_details = compute_upgrade_costs(
            safe_links_for_upgrade=initial_links,
            upgrade_decisions=initial_upgrade_decisions,
            algorithm_name="jump_to_max"   # use jump-to-max costing for max-initial baseline
        )

        total_network_cost += float(initial_cost)

        upgrade_cost_timeline.append({
            "day": 0.0,
            "equipment": float(cost_summary.get("equipment_total", 0.0)),
            "workforce": float(cost_summary.get("workforce_total", 0.0)),
            "opex": 0.0,
            "total": float(initial_cost),
            "cumulative_total": float(total_network_cost),
        })

        print(f"[max_initial] Day-0 CAPEX + workforce applied: {initial_cost:.2f}")
        print(f"[max_initial] Equipment: {cost_summary.get('equipment_total', 0.0):.2f}")
        print(f"[max_initial] Workforce: {cost_summary.get('workforce_total', 0.0):.2f}")

    # ------------------------------------------------------------
    # COMMON budget envelope
    # - OPEX budget is derived from actual initial network OPEX
    # - CAPEX/WF budget is a policy value based on your upgrade-budget constants
    # ------------------------------------------------------------
    initial_daily_opex = float(compute_network_opex_per_day(link_status_forward))

    # ------------------------------------------------------------
    # Budget parameters by topology
    #   - test5  -> base budget
    #   - dt12   -> 3x of test5 budget
    # ------------------------------------------------------------
    base_budget_params = {
        "period_days": int(Upgrade_initiation_days),  # 180
        "period_total_budget_0": 750_0000.0,
        "period_opex_budget_0": 100_0000.0,
        "period_capex_budget_0": 650_0000.0,
        "inflation_rate": 0.03,  # annual inflation
    }

    if config["topology"] == "test5":
        budget_params = base_budget_params

    elif config["topology"] == "dt12":
        dt12_multiplier = 3.0  # try 2.0 first; later test 1.75 if needed

        budget_params = {
            "period_days": base_budget_params["period_days"],
            "period_total_budget_0": dt12_multiplier * base_budget_params["period_total_budget_0"],
            "period_opex_budget_0": dt12_multiplier * base_budget_params["period_opex_budget_0"],
            "period_capex_budget_0": dt12_multiplier * base_budget_params["period_capex_budget_0"],
            "inflation_rate": base_budget_params["inflation_rate"],  # still annual
        }

    # elif config["topology"] == "dt12":
    #     budget_params = {
    #         "period_days": base_budget_params["period_days"],
    #         "period_total_budget_0": 12_000_000.0,
    #         "period_opex_budget_0": 2_000_000.0,
    #         "period_capex_budget_0": 10_000_000.0,
    #         "inflation_rate": base_budget_params["inflation_rate"],
    #     }

    else:
        raise ValueError(f"Unsupported topology for budget settings: {config['topology']}")

    print("\nBUDGET ENVELOPE PARAMETERS")
    print("-" * 50)
    print(f"Initial daily OPEX (actual)         : {initial_daily_opex:.2f}")
    print(f"Initial OPEX budget / year          : {budget_params['period_opex_budget_0']:.2f}")
    print(f"Initial CAPEX/WF budget / year      : {budget_params['period_capex_budget_0']:.2f}")
    print(f"Initial TOTAL budget / year         : {budget_params['period_total_budget_0']:.2f}")
    print(f"Upgrade check interval (days)       : {budget_params['period_days']}")
    print("-" * 50)
    # Traffic generator

    traffic_generator = NetworkTrafficGenerator(
        number_of_nodes=N,
        datarates=DATARATE,
        lambda_0=lambda_0,
        mean_holding_time=MEAN_HOLDING_TIME,
        Current_global_time=Current_global_time,
        rng=main_rng
    )

    # Build paths
    PATHS = build_k_shortest_paths(TOPOLOGY)
    if not PATHS:
        raise ValueError("build_k_shortest_paths() returned empty structure")

    print("✓ Simulation initialized\n")
    print("Starting simulation loop...\n")

    def account_running_opex_until(new_time: float):
        """
        Charges OPEX continuously, but logs it cleanly day-by-day:
          - For each day boundary crossed, append one timeline entry
          - Each daily entry contains the OPEX charged for that day segment
          - Upgrade events are separate entries (CAPEX/workforce jumps)
        """
        nonlocal total_network_cost, last_cost_time, upgrade_cost_timeline, last_logged_day

        new_time = float(new_time)
        t = float(last_cost_time)

        if new_time <= t:
            return

        while t < new_time:
            # next integer day boundary
            next_boundary = float(int(t) + 1)
            seg_end = min(new_time, next_boundary)
            dt = seg_end - t
            if dt <= 0:
                break

            # OPEX for this segment (rate based on CURRENT state)
            opex_inc = float(compute_running_opex_interval(link_status_forward, dt))
            total_network_cost += opex_inc

            day_marker = int(seg_end)
            if day_marker > last_logged_day:
                opex_rate = float(compute_network_opex_per_day(link_status_forward))

                upgrade_cost_timeline.append({
                    "day": float(day_marker),
                    "equipment": 0.0,
                    "workforce": 0.0,
                    "opex": float(opex_inc),  # OPEX charged in this segment
                    "opex_rate": float(opex_rate),  # informational
                    "total": float(opex_inc),
                    "cumulative_total": float(total_network_cost),
                })
                last_logged_day = day_marker
                log_daily_network_state(day_marker)
            else:
                # same day marker: merge OPEX into the last row but keep total consistent
                if upgrade_cost_timeline:
                    row = upgrade_cost_timeline[-1]
                    row["opex"] = float(row.get("opex", 0.0)) + float(opex_inc)

                    row["total"] = (
                            float(row.get("equipment", 0.0))
                            + float(row.get("workforce", 0.0))
                            + float(row.get("opex", 0.0))
                    )
                    row["cumulative_total"] = float(total_network_cost)

            t = seg_end

        last_cost_time = float(new_time)

    def log_daily_network_state(day_marker: int):
        tech_counts = count_network_fibers_by_technology(link_status_forward)
        daily_opex = float(compute_network_opex_per_day(link_status_forward))
        blocking_prob = float(metrics.get_blocking_probability())
        active_connections = len(ALL_DEMANDS)

        daily_network_writer.writerow([
            int(day_marker),
            tech_counts["SC_C"],
            tech_counts["SC_CL"],
            tech_counts["MC_C"],
            tech_counts["MC_CL"],
            daily_opex,
            float(total_network_cost),
            blocking_prob,
            active_connections,
        ])

    def classify_fiber_state(fiber_obj):
        if not isinstance(fiber_obj, dict):
            return None

        # -------- Single-core --------
        if "lit" in fiber_obj and "bands" in fiber_obj:
            if not fiber_obj.get("lit", False):
                return None

            bands = set(fiber_obj.get("bands", []))
            if bands == {"C"}:
                return "SC_C"
            elif bands == {"C", "L"}:
                return "SC_CL"
            return None

        # -------- Multi-core --------
        core_keys = [k for k in fiber_obj.keys() if isinstance(k, int)]
        if core_keys:
            lit_cores = []
            all_bands = set()

            for ck in core_keys:
                core = fiber_obj.get(ck, {})
                if isinstance(core, dict) and core.get("lit", False):
                    lit_cores.append(ck)
                    all_bands.update(core.get("bands", []))

            if not lit_cores:
                return None

            if "L" in all_bands:
                return "MC_CL"
            else:
                return "MC_C"

        return None

    def log_pre_upgrade_per_link_state(cycle_idx, time_day):
        for link_id, fibers in link_status_forward.items():

            sc_c = sc_cl = mc_c = mc_cl = 0

            for fiber_id, fiber_obj in fibers.items():
                state = classify_fiber_state(fiber_obj)

                if state == "SC_C":
                    sc_c += 1
                elif state == "SC_CL":
                    sc_cl += 1
                elif state == "MC_C":
                    mc_c += 3  # important rule
                elif state == "MC_CL":
                    mc_cl += 3

            pre_upgrade_per_link_writer.writerow([
                int(cycle_idx),
                float(time_day),
                int(link_id),
                sc_c,
                sc_cl,
                mc_c,
                mc_cl,
            ])

    def start_next_link_job(
            link_execution_queue,
            per_link_jobs,
            current_time,
            period_index,
            working_topology,
            link_status_forward,
            link_status_backward,
            removed_links_info,
            ALL_DEMANDS,
            total_network_cost,
            PATHS,
    ):
        """
        Start the next link's downtime work only if no downtime is active.
        """

        if not link_execution_queue or removed_links_info:
            return (
                link_execution_queue,
                per_link_jobs,
                working_topology,
                link_status_forward,
                link_status_backward,
                removed_links_info,
                ALL_DEMANDS,
                total_network_cost,
                PATHS,
            )

        lid = link_execution_queue.pop(0)
        decs = per_link_jobs.get(lid, [])

        if not decs:
            print(f">>> Link {lid} has no downtime jobs. Moving to next link.")
            return (
                link_execution_queue,
                per_link_jobs,
                working_topology,
                link_status_forward,
                link_status_backward,
                removed_links_info,
                ALL_DEMANDS,
                total_network_cost,
                PATHS,
            )

        has_core = any(d.get("upgrade_type") == "core_upgrade" for d in decs)

        if has_core:
            print(f">>> Starting LINK downtime job on link {lid}: {decs}")

            for dec in decs:
                log_upgrade_action(
                    upgrade_log_csv,
                    day=current_time,
                    period_index=period_index,
                    scope="link_downtime",
                    link_id=lid,
                    upgrade_type=dec.get("upgrade_type", ""),
                    fiber_id=dec.get("fiber_id", None),
                    core_type=dec.get("core_type", None),
                )

            decs = sorted(
                decs,
                key=lambda d: {
                    "core_upgrade": 0,
                    "band_upgrade": 1,
                    "new_fiber_CL": 2,
                }.get(d.get("upgrade_type"), 99)
            )

            (
                working_topology,
                link_status_forward,
                link_status_backward,
                removed_links_info,
                ALL_DEMANDS,
                _,
                _,
                _,
                total_network_cost,
                PATHS,
            ) = start_link_downtime_job(
                link_id=lid,
                decisions=decs,
                pending_downtime_actions=None,
                ALL_DEMANDS=ALL_DEMANDS,
                link_status_forward=link_status_forward,
                link_status_backward=link_status_backward,
                working_topology=working_topology,
                removed_links_info=removed_links_info,
                current_time=current_time,
                blocked_datarate_rerouting=0,
                accepted_datarate_rerouting=0,
                arrived_datarate_rerouting=0,
                total_network_cost=total_network_cost,
                upgrade_cost_timeline=upgrade_cost_timeline,
                algorithm=algorithm,
            )

        else:
            # NEW: treat ALL fiber-level decisions on this link as ONE combined job
            print(f">>> Starting COMBINED FIBER downtime job on link {lid}: {decs}")

            for dec in decs:
                log_upgrade_action(
                    upgrade_log_csv,
                    day=current_time,
                    period_index=period_index,
                    scope="fiber_downtime",
                    link_id=lid,
                    upgrade_type=dec.get("upgrade_type", ""),
                    fiber_id=dec.get("fiber_id", None),
                    core_type=dec.get("core_type", None),
                )

            (
                working_topology,
                link_status_forward,
                link_status_backward,
                removed_links_info,
                ALL_DEMANDS,
                _,
                _,
                _,
                total_network_cost,
                PATHS,
            ) = start_combined_fiber_downtime_job(
                link_id=lid,
                decisions=decs,
                ALL_DEMANDS=ALL_DEMANDS,
                link_status_forward=link_status_forward,
                link_status_backward=link_status_backward,
                working_topology=working_topology,
                removed_links_info=removed_links_info,
                current_time=current_time,
                blocked_datarate_rerouting=0,
                accepted_datarate_rerouting=0,
                arrived_datarate_rerouting=0,
                total_network_cost=total_network_cost,
                upgrade_cost_timeline=upgrade_cost_timeline,
                algorithm=algorithm,
            )

            per_link_jobs[lid] = []
            # first_dec = decs[0]
            # remaining = decs[1:]
            #
            # print(f">>> Starting FIBER downtime job on link {lid}: {first_dec}")
            #
            # log_upgrade_action(
            #     upgrade_log_csv,
            #     day=current_time,
            #     period_index=period_index,
            #     scope="fiber_downtime",
            #     link_id=lid,
            #     upgrade_type=first_dec.get("upgrade_type", ""),
            #     fiber_id=first_dec.get("fiber_id", None),
            #     core_type=first_dec.get("core_type", None),
            # )
            #
            # (
            #     working_topology,
            #     link_status_forward,
            #     link_status_backward,
            #     removed_links_info,
            #     ALL_DEMANDS,
            #     _,
            #     _,
            #     _,
            #     total_network_cost,
            #     PATHS,
            # ) = start_downtime_upgrade_for_link(
            #     scope="fiber",
            #     link_id=lid,
            #     decision=first_dec,
            #     pending_downtime_actions=None,
            #     ALL_DEMANDS=ALL_DEMANDS,
            #     link_status_forward=link_status_forward,
            #     link_status_backward=link_status_backward,
            #     working_topology=working_topology,
            #     removed_links_info=removed_links_info,
            #     current_time=current_time,
            #     blocked_datarate_rerouting=0,
            #     accepted_datarate_rerouting=0,
            #     arrived_datarate_rerouting=0,
            #     total_network_cost=total_network_cost,
            #     upgrade_cost_timeline=upgrade_cost_timeline,
            #     algorithm=algorithm,
            # )
            #
            # if remaining:
            #     per_link_jobs[lid] = remaining
            #     link_execution_queue.insert(0, lid)
            # else:
            #     per_link_jobs[lid] = []

        return (
            link_execution_queue,
            per_link_jobs,
            working_topology,
            link_status_forward,
            link_status_backward,
            removed_links_info,
            ALL_DEMANDS,
            total_network_cost,
            PATHS,
        )

    while True:

        # Weekly traffic growth
        while (Current_global_time // Traffic_growth_days) > current_week:
            print(f">>> Weekly traffic update at day {Current_global_time}")
            current_week += 1

            for i in range(N):
                for j in range(N):
                    if i != j:
                        growth_factor = 1 + main_rng.uniform(0, alpha / 100)
                        traffic_rates[i][j] *= growth_factor

            for i in range(N):
                for j in range(N):
                    if i != j:
                        growth_factor_new = (1 + delta / 100)
                        traffic_rates[i][j] *= growth_factor_new

        # Generate next request
        src, dest, datarate = traffic_generator.generate_connection_data()
        lambda_sd = traffic_rates[src][dest]

        if lambda_sd <= 0:
            raise ValueError("Invalid lambda_sd: must be non-zero and positive.")

        arrival_time, holding_time = traffic_generator.get_connection(lambda_sd)
        Current_global_time = arrival_time
        # ✅ OPEX from last time -> now
        account_running_opex_until(Current_global_time)
        departure_time = arrival_time + holding_time
        period_index = int(Current_global_time // tau)

        # Monthly logging
        month_idx = int(Current_global_time // MONTH_DAYS)
        if month_idx > current_month:
            current_month = month_idx
            log_monthly_link_stats_rowwise(
                time_day=Current_global_time,
                LINKS=LINKS,
                link_status_forward=link_status_forward,
                link_status_backward=link_status_backward,
                C_BAND_SLOTS=C_BAND_SLOTS,
                TOTAL_SLOTS=TOTAL_SLOTS,
                util_csv=util_csv,
                frag_csv=frag_csv,
            )

        connection_id += 1

        # ====================================================================
        # Upgrade Cycle Check (every 3 months)
        # ====================================================================

        if (Current_global_time // Upgrade_initiation_days) > current_upgrade:
            print(f"\n=== Upgrade cycle triggered at day {Current_global_time} ===")
            current_upgrade += 1

            print(">>> Checking whether upgrade is needed...")
            upgrade_needed = algorithm.check_need_for_upgrade(
                current_traffic=traffic_rates,
                current_time=Current_global_time,
                PATHS=PATHS,
                current_link_status_forward=copy.deepcopy(link_status_forward),
                current_link_status_backward=copy.deepcopy(link_status_backward),
                seed=seed
            )

            print(f">>> Upgrade needed = {upgrade_needed}")

            # ============================================================
            # XGBOOST DATA COLLECTION
            # One row per upgrade-check cycle, before applying upgrades
            # ============================================================
            if config.get("collect_xgboost_data", False):
                xgb_csv_path = (
                        config["results_dir"]
                        / config.get("xgboost_csv_name", "xgboost_need_upgrade_dataset.csv")
                )

                append_xgboost_row(
                    csv_path=xgb_csv_path,
                    seed=seed,
                    algorithm_name=algorithm.name,
                    cycle_number=current_upgrade,
                    current_day=Current_global_time,
                    current_blocking_probability=metrics.get_blocking_probability(),
                    traffic_rates=traffic_rates,
                    link_status_forward=link_status_forward,
                    upgrade_needed=upgrade_needed,
                    classify_fiber_state=classify_fiber_state,
                )

                print(f">>> XGBoost data row saved: {xgb_csv_path}")

                # ============================================================
                # DEBUG STOP: stop after a few upgrade-check cycles
                # Use this only to verify the XGBoost CSV.
                # ============================================================
                max_debug_cycles = config.get("max_upgrade_cycles_for_debug", None)

                if max_debug_cycles is not None and current_upgrade >= max_debug_cycles:
                    print(
                        f">>> DEBUG STOP: reached {current_upgrade} upgrade cycles. "
                        f"Stopping early so you can inspect the XGBoost CSV."
                    )
                    break

            if upgrade_needed:
                sd_last_block_prob = copy.deepcopy(metrics.get_sd_blocking_probability())

                # Clear SD stats once for the next measurement window.
                # sd_last_block_prob is already copied and will be reused for all path-rank retries.
                metrics.sd_arrivals.clear()
                metrics.sd_blocked.clear()

                # ============================================================
                # Retry over congestion-ranked paths
                # ============================================================

                plan_ok = False
                final_links = []
                final_upgrade_decisions = {}

                for path_rank in range(3):

                    print("\n" + "=" * 60)
                    print(f">>> TRYING path_rank = {path_rank}")
                    print("=" * 60)

                    # --------------------------------------------------------
                    # Select candidate links
                    # --------------------------------------------------------

                    upgrade_links, link_scores, top_contrib = algorithm.select_links_for_upgrade(
                        traffic_rates=traffic_rates,
                        PATHS=PATHS,
                        sd_block_prob=sd_last_block_prob,
                        block_threshold=PER_SD_BLOCK_THRESHOLD,
                        working_topology=working_topology,
                        LINK_INDEX=LINK_INDEX,
                        link_status_forward=link_status_forward,
                        link_status_backward=link_status_backward,
                        current_time=Current_global_time,
                        path_stats=path_stats,
                        path_rank=path_rank,
                        link_selection_mode=config.get(
                            "link_selection_mode",
                            "path_congestion"
                        ),
                        link_score_threshold=config.get(
                            "link_score_threshold",
                            PER_SD_BLOCK_THRESHOLD
                        ),
                    )

                    print(f">>> Selected links = {upgrade_links}")

                    if not upgrade_links:
                        print(f">>> No upgrade links found for path_rank={path_rank}")
                        continue

                    # --------------------------------------------------------
                    # Connectivity-safe links
                    # --------------------------------------------------------

                    safe_links = network_connectivity(
                        upgrade_links,
                        working_topology,
                        LINK_INDEX
                    )

                    print(f">>> Safe links = {safe_links}")

                    safe_links = sorted(
                        safe_links,
                        key=lambda lid: float(link_scores.get(lid, 0.0)),
                        reverse=True
                    )

                    print(f">>> Safe links after sorting = {safe_links}")

                    # --------------------------------------------------------
                    # Build upgrade decisions
                    # --------------------------------------------------------

                    if config['algorithm'] == 'greedy':

                        upgrade_decisions = {}

                        for link_id in safe_links:

                            decs = choose_upgrade_type_greedy_max(
                                link_id,
                                link_status_forward,
                                link_status_backward
                            )

                            if decs is None:
                                upgrade_decisions[link_id] = []

                            elif isinstance(decs, dict):
                                upgrade_decisions[link_id] = [decs]

                            else:
                                upgrade_decisions[link_id] = list(decs)

                    elif config['algorithm'] == 'jump_to_max':

                        upgrade_decisions = {}

                        for link_id in safe_links:

                            decs = algorithm.get_upgrade_decision(
                                link_id,
                                link_status_forward,
                                link_status_backward
                            )

                            if decs is None:
                                upgrade_decisions[link_id] = []

                            elif isinstance(decs, dict):
                                upgrade_decisions[link_id] = [decs]

                            else:
                                upgrade_decisions[link_id] = list(decs)

                    elif config['algorithm'] == 'random':

                        upgrade_decisions = {}

                        for link_id in safe_links:

                            decs = algorithm.get_random_upgrade_decision(
                                link_id,
                                link_status_forward
                            )

                            if decs is None:
                                upgrade_decisions[link_id] = []

                            elif isinstance(decs, dict):
                                upgrade_decisions[link_id] = [decs]

                            else:
                                upgrade_decisions[link_id] = list(decs)

                    else:

                        upgrade_decisions = {}

                        for link_id in safe_links:

                            decs = choose_upgrade_type(
                                link_id,
                                link_status_forward,
                                link_status_backward,
                                algorithm
                            )

                            if decs is None:
                                upgrade_decisions[link_id] = []

                            elif isinstance(decs, dict):
                                upgrade_decisions[link_id] = [decs]

                            else:
                                upgrade_decisions[link_id] = list(decs)

                    # --------------------------------------------------------
                    # Run PlanChecker
                    # --------------------------------------------------------

                    print(f">>> Running PlanChecker for path_rank={path_rank}")

                    plan_ok, candidate_links, candidate_upgrade_decisions = plan_checker(
                        safe_links=safe_links,
                        upgrade_decisions=upgrade_decisions,
                        current_time=Current_global_time,
                        PATHS=PATHS,
                        current_traffic=traffic_rates,
                        current_link_status_forward=copy.deepcopy(link_status_forward),
                        current_link_status_backward=copy.deepcopy(link_status_backward),
                        seed=seed,
                        algorithm=algorithm,
                        budget_params=budget_params,
                        budget_selection_mode=config.get(
                            "budget_selection_mode",
                            "budget_aware"
                        ),
                        budget_tracker=budget_tracker,
                    )

                    if plan_ok:

                        print(f">>> PlanChecker PASSED for path_rank={path_rank}")

                        final_links = candidate_links
                        final_upgrade_decisions = candidate_upgrade_decisions

                        break

                    else:

                        print(f">>> PlanChecker FAILED for path_rank={path_rank}")

                # ============================================================
                # Final result after retries
                # ============================================================

                if not plan_ok:

                    print(">>> ALL path-rank retries failed.")
                    print(">>> No feasible upgrade plan found for this cycle.")

                else:

                    print(">>> Upgrade plan accepted.")
                    print(">>> PlanChecker PASSED")

                    path_stats.clear()

                    safe_links = final_links
                    upgrade_decisions = final_upgrade_decisions
                    flat_final_decisions = flatten_upgrade_decisions(upgrade_decisions)

                    # ✅ SAVE PRE-UPGRADE STATE (VERY IMPORTANT)
                    log_pre_upgrade_per_link_state(
                        cycle_idx=current_upgrade,
                        time_day=Current_global_time
                    )

                    upgrade_decision_history.append({
                        "cycle": int(current_upgrade),
                        "time_day": float(Current_global_time),
                        "blocking_probability": float(metrics.get_blocking_probability()),
                        "final_decisions": flat_final_decisions,
                        "cycle_upgrade_level": get_cycle_upgrade_level(flat_final_decisions),
                    })

                    upgrade_cycle_writer.writerow([
                        int(current_upgrade),
                        float(Current_global_time),
                        float(metrics.get_blocking_probability()),
                        len(safe_links),
                        str(list(safe_links)),
                        str(flat_final_decisions),
                        get_cycle_upgrade_level(flat_final_decisions),
                    ])
                    print_upgrade_decisions_summary(Current_global_time, safe_links, upgrade_decisions)

                    # ------------------------------------------------------------
                    # Build per-link execution plan
                    #   - Immediate actions are applied globally first
                    #   - Remaining downtime work is executed link-by-link
                    # ------------------------------------------------------------
                    immediate_actions = {}
                    per_link_jobs = {}

                    for lid in safe_links:
                        decs = upgrade_decisions.get(lid, [])

                        if isinstance(decs, dict):
                            decs = [decs]
                        else:
                            decs = list(decs)

                        has_core = any(d.get("upgrade_type") == "core_upgrade" for d in decs)

                        immediate_actions[lid] = []
                        per_link_jobs[lid] = []

                        if has_core:
                            # 🔥 KEEP EVERYTHING TOGETHER
                            per_link_jobs[lid] = list(decs)

                        else:
                            for dec in decs:
                                if dec is None or dec.get("upgrade_type") is None:
                                    continue

                                utype = dec.get("upgrade_type")

                                if utype == "new_fiber_C":
                                    immediate_actions[lid].append(dec)
                                else:
                                    per_link_jobs[lid].append(dec)

                    immediate_actions = {
                        lid: decs for lid, decs in immediate_actions.items() if decs
                    }
                    per_link_jobs = {
                        lid: decs for lid, decs in per_link_jobs.items() if decs
                    }

                    link_execution_queue = [lid for lid in safe_links if lid in per_link_jobs]

                    # ============================================================
                    # ONE-SHOT COST COMPUTATION FOR THE WHOLE UPGRADE CYCLE
                    # ============================================================
                    cycle_actions = {}

                    for lid, decs in immediate_actions.items():
                        cycle_actions.setdefault(lid, []).extend(decs)

                    for lid, decs in per_link_jobs.items():
                        cycle_actions.setdefault(lid, []).extend(decs)

                    cycle_links = sorted(cycle_actions.keys())

                    total_cost_cycle, cost_summary, cost_details = compute_upgrade_costs(
                        safe_links_for_upgrade=cycle_links,
                        upgrade_decisions=cycle_actions,
                        algorithm_name=algorithm.name
                    )

                    equip = float(cost_summary.get("equipment_total", 0.0))
                    work = float(cost_summary.get("workforce_total", 0.0))
                    capex_work_cycle = equip + work

                    total_network_cost += capex_work_cycle

                    day_marker = int(Current_global_time)
                    if not upgrade_cost_timeline or int(upgrade_cost_timeline[-1].get("day", -1)) != day_marker:
                        upgrade_cost_timeline.append({
                            "day": float(day_marker),
                            "equipment": 0.0,
                            "workforce": 0.0,
                            "opex": 0.0,
                            "total": 0.0,
                            "cumulative_total": float(total_network_cost),
                        })

                    row = upgrade_cost_timeline[-1]
                    row["equipment"] = float(row.get("equipment", 0.0)) + equip
                    row["workforce"] = float(row.get("workforce", 0.0)) + work
                    row["total"] = (
                        float(row.get("equipment", 0.0))
                        + float(row.get("workforce", 0.0))
                        + float(row.get("opex", 0.0))
                    )
                    row["cumulative_total"] = float(total_network_cost)

                    print(
                        f">>> ONE-SHOT upgrade cycle cost jump applied: "
                        f"equip={equip:.2f}, workforce={work:.2f}, total={capex_work_cycle:.2f}"
                    )

                    # ------------------------------------------------------------
                    # Apply IMMEDIATE upgrades globally (no downtime)
                    # ------------------------------------------------------------
                    if immediate_actions:
                        immediate_links = sorted(immediate_actions.keys())
                        print(f">>> Immediate (no-downtime) upgrades on links {immediate_links}")

                        for link_id, decs in immediate_actions.items():
                            for dec in decs:
                                log_upgrade_action(
                                    upgrade_log_csv,
                                    day=Current_global_time,
                                    period_index=period_index,
                                    scope="immediate",
                                    link_id=link_id,
                                    upgrade_type=dec.get("upgrade_type", ""),
                                    fiber_id=dec.get("fiber_id", None),
                                    core_type=dec.get("core_type", None),
                                )

                        perform_upgrade(
                            immediate_links,
                            immediate_actions,
                            link_status_forward,
                            link_status_backward,
                            C_BAND_SLOTS,
                            TOTAL_SLOTS,
                            Current_global_time,
                            algorithm
                        )

                    # ------------------------------------------------------------
                    # Start first link's downtime work in PlanChecker order
                    # ------------------------------------------------------------
                    (
                        link_execution_queue,
                        per_link_jobs,
                        working_topology,
                        link_status_forward,
                        link_status_backward,
                        removed_links_info,
                        ALL_DEMANDS,
                        total_network_cost,
                        PATHS,
                    ) = start_next_link_job(
                        link_execution_queue=link_execution_queue,
                        per_link_jobs=per_link_jobs,
                        current_time=Current_global_time,
                        period_index=period_index,
                        working_topology=working_topology,
                        link_status_forward=link_status_forward,
                        link_status_backward=link_status_backward,
                        removed_links_info=removed_links_info,
                        ALL_DEMANDS=ALL_DEMANDS,
                        total_network_cost=total_network_cost,
                        PATHS=PATHS,
                    )

        # ====================================================================
        # Restore Downtime Links (when downtime expires) + start next job
        # ====================================================================

        restore_now = [
            key
            for key, info in removed_links_info.items()
            if info.get("restore_time", math.inf) <= Current_global_time
        ]

        if restore_now:
            print(f">>> Restoring downtime items at day {Current_global_time}")

            for key in restore_now:
                info = removed_links_info[key]
                working_topology, link_status_forward, link_status_backward, removed_links_info = \
                    restore_removed_links_after_downtime(
                        key=key,
                        info=info,
                        current_time=Current_global_time,
                        link_status_forward=link_status_forward,
                        link_status_backward=link_status_backward,
                        current_topology=working_topology,
                        removed_links_info=removed_links_info
                    )

            PATHS = build_k_shortest_paths(working_topology)
            if not PATHS:
                raise ValueError("PATHS empty after restoring downtime items.")

            # Start next link's downtime work after all current downtime items restore
            (
                link_execution_queue,
                per_link_jobs,
                working_topology,
                link_status_forward,
                link_status_backward,
                removed_links_info,
                ALL_DEMANDS,
                total_network_cost,
                PATHS,
            ) = start_next_link_job(
                link_execution_queue=link_execution_queue,
                per_link_jobs=per_link_jobs,
                current_time=Current_global_time,
                period_index=period_index,
                working_topology=working_topology,
                link_status_forward=link_status_forward,
                link_status_backward=link_status_backward,
                removed_links_info=removed_links_info,
                ALL_DEMANDS=ALL_DEMANDS,
                total_network_cost=total_network_cost,
                PATHS=PATHS,
            )

        # ====================================================================
        # Remove Expired Connections
        # ====================================================================

        index_to_remove = []
        for idy, conn in enumerate(ALL_DEMANDS):
            if conn.departure_time <= Current_global_time:
                index_to_remove.append(idy)

        for idy in reversed(index_to_remove):
            conn = ALL_DEMANDS[idy]

            for i in range(len(conn.path) - 1):
                src_depart = conn.path[i]
                dest_depart = conn.path[i + 1]

                link = conn.link_ids[i]
                fiber = conn.fibers_used[i]
                core = conn.cores_used[i]

                if src_depart < dest_depart:
                    if link in link_status_forward and fiber in link_status_forward[link]:
                        fwd_obj = link_status_forward[link][fiber]
                        if isinstance(fwd_obj, dict) and "slots" in fwd_obj:
                            for s in range(conn.fs_index, conn.fs_index + conn.slice_window_size):
                                fwd_obj["slots"][s] = 0
                        else:
                            if core in fwd_obj:
                                for s in range(conn.fs_index, conn.fs_index + conn.slice_window_size):
                                    fwd_obj[core]["slots"][s] = 0
                else:
                    if link in link_status_backward and fiber in link_status_backward[link]:
                        bwd_obj = link_status_backward[link][fiber]
                        if isinstance(bwd_obj, dict) and "slots" in bwd_obj:
                            for s in range(conn.fs_index, conn.fs_index + conn.slice_window_size):
                                bwd_obj["slots"][s] = 0
                        else:
                            if core in bwd_obj:
                                for s in range(conn.fs_index, conn.fs_index + conn.slice_window_size):
                                    bwd_obj[core]["slots"][s] = 0

            del ALL_DEMANDS[idy]

        # ====================================================================
        # Run RSA for New Connection
        # ====================================================================
        mf_index, fs_index, slice_window_size, path, fibers_used, cores_used, link_ids, attempted_paths_info = execute_first_fit(
            src=src,
            dest=dest,
            datarate=datarate,
            arrival_time=arrival_time,
            departure_time=departure_time,
            link_status_forward=link_status_forward,
            link_status_backward=link_status_backward,
            topology=working_topology,
            PATHS=PATHS
        )

        accepted = (mf_index != float('inf') and fs_index != float('inf'))
        status = "accepted" if accepted else "blocked"

        connection_log_writer.writerow(
            build_connection_request_row(
                connection_id=connection_id,
                src=src,
                dest=dest,
                datarate=datarate,
                arrival_time=arrival_time,
                holding_time=holding_time,
                departure_time=departure_time,
                status=status,
                mf_index=mf_index,
                fs_index=fs_index,
                slice_window_size=slice_window_size if accepted else None,
                path=path if accepted else None,
                link_ids=link_ids if accepted else None,
                fibers_used=fibers_used if accepted else None,
                cores_used=cores_used if accepted else None,
            )
        )
        # ============================================================
        # UPDATE PATH STATISTICS
        # ============================================================

        for info in (attempted_paths_info or []):

            p_idx = int(info["path_index"])
            status = info["status"]

            # only count actual spectrum attempts
            if status not in ("accepted", "blocked"):
                continue

            sd_u = sd_key_undirected(src, dest)
            key = (sd_u[0], sd_u[1], p_idx)

            # ------------------------------------------------------------
            # Current-cycle stats (used by link selection)
            # ------------------------------------------------------------
            if key not in path_stats:
                path_stats[key] = {
                    "path_arrivals": 0,
                    "path_accepted": 0,
                    "path_blocked": 0
                }

            path_stats[key]["path_arrivals"] += 1

            if status == "accepted":
                path_stats[key]["path_accepted"] += 1
            else:
                path_stats[key]["path_blocked"] += 1

            # ------------------------------------------------------------
            # Full-history stats (kept until end of simulation)
            # ------------------------------------------------------------
            if key not in path_stats_history:
                path_stats_history[key] = {
                    "path_arrivals": 0,
                    "path_accepted": 0,
                    "path_blocked": 0
                }

            path_stats_history[key]["path_arrivals"] += 1

            if status == "accepted":
                path_stats_history[key]["path_accepted"] += 1
            else:
                path_stats_history[key]["path_blocked"] += 1

        if mf_index != float('inf') and fs_index != float('inf'):
            ALL_DEMANDS.append(ConnectionData(
                path=path,
                link_ids=link_ids,
                fs_index=fs_index,
                slice_window_size=slice_window_size,
                mf_index=mf_index,
                arrival_time=arrival_time,
                holding_time=holding_time,
                departure_time=departure_time,
                datarate=datarate,
                fibers_used=fibers_used,
                cores_used=cores_used
            ))

        # ====================================================================
        # Update Metrics (after warmup)
        # ====================================================================

        if connection_id > config['warmup_connections']:
            sd_u = sd_key_undirected(src, dest)
            metrics.sd_arrivals[sd_u] += 1

            blocked = (mf_index == float('inf') or fs_index == float('inf'))
            if blocked:
                metrics.sd_blocked[sd_u] += 1

            metrics.record_connection(src, dest, datarate, blocked, Current_global_time)

            if metrics.get_blocking_probability() >= config['blocking_threshold']:
                print(
                    f"\nSimulation stopped: blocking threshold "
                    f"({config['blocking_threshold']}) reached at day {Current_global_time:.1f}."
                )
                break

    # ========================================================================
    # Generate Results
    # ========================================================================

    print("\n" + "="*70)
    print("SIMULATION COMPLETE")
    print("="*70)

    analysis.save_blocking_data(
        metrics.time_points,
        metrics.blocking_prob_points,
        config['blocking_csv']
    )

    end_wall_time = time.time()
    total_runtime_sec = end_wall_time - start_wall_time
    account_running_opex_until(Current_global_time)

    snapshot = {
        "seed": seed,
        "Current_global_time": Current_global_time,
        "connection_id": connection_id,
        "connection_count": metrics.connection_count,
        "blocked_connection_count": metrics.blocked_connection_count,
        "arrived_datarate": metrics.arrived_datarate,
        "accepted_datarate": metrics.accepted_datarate,
        "blocked_datarate": metrics.blocked_datarate,
        "time_points": metrics.time_points,
        "blocking_prob_points": metrics.blocking_prob_points,
        "traffic_rates": traffic_rates,
        "total_network_cost": total_network_cost,
        "upgrade_cost_timeline": upgrade_cost_timeline,
        "upgrade_decision_history": upgrade_decision_history,
        "link_status_forward": link_status_forward,
        "link_status_backward": link_status_backward,
        "ALL_DEMANDS": ALL_DEMANDS,
        "working_topology": working_topology,
        "wall_clock_runtime_seconds": total_runtime_sec,
        "budget_params": budget_params,
        "path_stats_history": path_stats_history,
        "daily_network_technology_csv": str(daily_network_csv),
        "upgrade_cycle_summary_csv": str(upgrade_cycle_csv),
    }
    analysis.save_snapshot(snapshot, config['snapshot_file'])
    save_path_stats_history_csv(path_stats_history, config['results_dir'])

    analysis.print_simulation_summary(snapshot)
    metrics.print_summary()
    analysis.print_upgrade_timeline(upgrade_cost_timeline)

    generate_plots(config, metrics, upgrade_cost_timeline, total_network_cost)

    generate_budget_vs_cost_plot(
        config=config,
        upgrade_cost_timeline=upgrade_cost_timeline,
        end_day=Current_global_time,
        budget_params=budget_params
    )

    generate_blocking_vs_upgrade_plot(config, metrics, upgrade_decision_history)

    # ✅ ADD THIS BLOCK HERE
    try:
        from sim.modules.simulation_analysis import plot_technology_mix_by_upgrade_index

        plot_technology_mix_by_upgrade_index(
            folder=config["results_dir"],
            outdir=config["results_dir"],
            algo_name=config["algorithm"]
        )
    except Exception as e:
        print(f"[WARN] Technology mix plot failed: {e}")
    connection_log_file.close()

    # ✅ ADD THESE
    daily_network_file.close()
    upgrade_cycle_file.close()
    pre_upgrade_per_link_file.close()
    print(f"\n✓ All results saved to: {config['results_dir']}")

    end_wall_time = time.time()
    total_runtime_sec = end_wall_time - start_wall_time
    total_runtime_min = total_runtime_sec / 60.0
    total_runtime_hr = total_runtime_min / 60.0

    print("\n" + "-"*60)
    print(f"Total wall-clock runtime:")
    print(f"  {total_runtime_sec:.2f} seconds")
    print(f"  {total_runtime_min:.2f} minutes")
    print(f"  {total_runtime_hr:.2f} hours")
    print("-"*60)

    snapshot["wall_clock_runtime_seconds"] = total_runtime_sec


# ============================================================================
# Plotting Functions
# ============================================================================

def generate_plots(config, metrics, upgrade_cost_timeline, total_network_cost):
    """Generate all simulation plots."""

    print(f"\nGenerating plots and saving to {config['results_dir']}...")

    plt.figure(figsize=(10, 6))
    plt.plot(metrics.time_points, metrics.blocking_prob_points, label="Blocking Probability", linewidth=2)
    plt.axhline(y=config['blocking_threshold'], linestyle="--", color='r', label="Threshold")
    plt.xlabel("Time (days)", fontsize=12)
    plt.ylabel("Blocking Probability", fontsize=12)
    plt.yscale("log")
    plt.title(f"Blocking Probability vs Time - {config['algorithm']} on {config['topology']}", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(config['blocking_plot_file'], dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved blocking probability plot to {config['blocking_plot_file']}")

    if upgrade_cost_timeline:
        plt.figure(figsize=(12, 8))

        # ---- Per-entry series (same length) ----
        days_all = [e["day"] for e in upgrade_cost_timeline]
        equipment_costs = [e.get("equipment", 0.0) for e in upgrade_cost_timeline]
        workforce_costs = [e.get("workforce", 0.0) for e in upgrade_cost_timeline]
        opex_costs = [e.get("opex", 0.0) for e in upgrade_cost_timeline]
        total_costs = [e.get("total", 0.0) for e in upgrade_cost_timeline]

        # ---- Cumulative series (use cumulative_total field) ----
        days_cum = [e["day"] for e in upgrade_cost_timeline if "cumulative_total" in e]
        cumulative_total = [e["cumulative_total"] for e in upgrade_cost_timeline if "cumulative_total" in e]

        plt.subplot(2, 1, 1)
        plt.plot(days_all, equipment_costs, 'o-', label='Equipment Cost', linewidth=2, markersize=4)
        plt.plot(days_all, workforce_costs, 's-', label='Workforce Cost', linewidth=2, markersize=4)
        plt.plot(days_all, opex_costs, '^-', label='OPEX Cost', linewidth=2, markersize=4)
        plt.plot(days_all, total_costs, 'd-', label='Total Cost', linewidth=2, markersize=4, color='black')
        plt.xlabel("Time (days)", fontsize=12)
        plt.ylabel("Cost per Logged Point", fontsize=12)
        plt.title(f"Cost Components vs Time - {config['algorithm']} on {config['topology']}", fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 1, 2)
        plt.plot(days_cum, cumulative_total, 'o-', linewidth=2, markersize=3, color='darkblue')
        plt.xlabel("Time (days)", fontsize=12)
        plt.ylabel("Cumulative Total Cost", fontsize=12)
        plt.title(f"Cumulative Cost vs Time - Total: ${total_network_cost:,.0f}", fontsize=14)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(config['cost_plot_file'], dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved cost vs time plot to {config['cost_plot_file']}")
    else:
        print("✓ No upgrades performed, skipping cost plot")

    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    categories = ['Total', 'Accepted', 'Blocked']
    counts = [metrics.connection_count,
              metrics.connection_count - metrics.blocked_connection_count,
              metrics.blocked_connection_count]
    colors = ['blue', 'green', 'red']
    plt.bar(categories, counts, color=colors, alpha=0.7)
    plt.ylabel('Connection Count', fontsize=12)
    plt.title('Connection Statistics', fontsize=12)
    plt.grid(axis='y', alpha=0.3)

    plt.subplot(2, 2, 2)
    categories = ['Arrived', 'Accepted', 'Blocked']
    datarates = [metrics.arrived_datarate, metrics.accepted_datarate, metrics.blocked_datarate]
    colors = ['blue', 'green', 'red']
    plt.bar(categories, datarates, color=colors, alpha=0.7)
    plt.ylabel('Datarate (Gbps)', fontsize=12)
    plt.title('Datarate Statistics', fontsize=12)
    plt.grid(axis='y', alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.plot(metrics.time_points, metrics.blocking_prob_points, linewidth=2, color='darkred')
    plt.axhline(y=config['blocking_threshold'], linestyle="--", color='black', label="Threshold")
    plt.xlabel("Time (days)", fontsize=12)
    plt.ylabel("Blocking Probability", fontsize=12)
    plt.title('Blocking Probability (Linear Scale)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    if metrics.arrived_datarate > 0:
        throughput_efficiency = (metrics.accepted_datarate / metrics.arrived_datarate) * 100
        blocking_rate = (metrics.blocked_datarate / metrics.arrived_datarate) * 100
        categories = ['Accepted', 'Blocked']
        percentages = [throughput_efficiency, blocking_rate]
        colors = ['green', 'red']
        plt.bar(categories, percentages, color=colors, alpha=0.7)
        plt.ylabel('Percentage (%)', fontsize=12)
        plt.title('Throughput Efficiency', fontsize=12)
        plt.ylim(0, 100)
        plt.grid(axis='y', alpha=0.3)

    plt.suptitle(f"Performance Metrics - {config['algorithm']} on {config['topology']}",
                 fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(config['metrics_plot_file'], dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved performance metrics plot to {config['metrics_plot_file']}")

def build_budget_components_curve(
        end_day: float,
        period_days: int,
        period_opex_budget_0: float,
        period_capex_budget_0: float,
        period_total_budget_0: float,
        inflation_rate: float,
):
    """
    Build cumulative annual budget curves.

    Budget logic:
      - Annual budget is fixed within each year.
      - Inflation is applied once per year.
      - No carry-forward from one year to the next.
      - For plotting cumulative budget, each year contributes one annual budget.
    """
    max_day = int(math.ceil(end_day))
    days = np.arange(0, max_day + 1, dtype=float)

    cum_opex_budget = np.zeros(max_day + 1, dtype=float)
    cum_upgrade_budget = np.zeros(max_day + 1, dtype=float)
    cum_total_budget = np.zeros(max_day + 1, dtype=float)

    opex_cum = 0.0
    upgrade_cum = 0.0
    total_cum = 0.0

    current_year_idx = -1

    for day in range(1, max_day + 1):
        year_idx = (day - 1) // 365

        if year_idx != current_year_idx:
            current_year_idx = year_idx

            growth = (1.0 + inflation_rate) ** year_idx

            annual_opex_budget = period_opex_budget_0 * growth
            annual_upgrade_budget = period_capex_budget_0 * growth
            annual_total_budget = period_total_budget_0 * growth

            opex_cum += annual_opex_budget
            upgrade_cum += annual_upgrade_budget
            total_cum += annual_total_budget

        cum_opex_budget[day] = opex_cum
        cum_upgrade_budget[day] = upgrade_cum
        cum_total_budget[day] = total_cum

    return days, cum_opex_budget, cum_upgrade_budget, cum_total_budget


def build_actual_cumulative_cost_components(upgrade_cost_timeline):
    """
    Build cumulative actual OPEX, cumulative actual CAPEX+workforce,
    and cumulative actual total cost from upgrade_cost_timeline.
    """
    if not upgrade_cost_timeline:
        return [], [], [], []

    timeline_sorted = sorted(upgrade_cost_timeline, key=lambda x: x["day"])

    days = []
    cum_opex = []
    cum_upgrade = []
    cum_total = []

    running_opex = 0.0
    running_upgrade = 0.0

    for row in timeline_sorted:
        running_opex += float(row.get("opex", 0.0))
        running_upgrade += float(row.get("equipment", 0.0)) + float(row.get("workforce", 0.0))

        days.append(float(row["day"]))
        cum_opex.append(running_opex)
        cum_upgrade.append(running_upgrade)
        cum_total.append(running_opex + running_upgrade)

    return days, cum_opex, cum_upgrade, cum_total


def generate_budget_vs_cost_plot(config, upgrade_cost_timeline, end_day: float, budget_params: dict):
    """
    Plot:
      1) cumulative actual total cost vs cumulative total budget
      2) cumulative actual OPEX vs OPEX budget
      3) cumulative actual CAPEX+workforce vs CAPEX+workforce budget
    """
    if not config.get("budget_enabled", False):
        print("✓ Budget plot disabled")
        return

    if not upgrade_cost_timeline:
        print("✓ No cost timeline available, skipping budget plot")
        return

    actual_days, actual_cum_opex, actual_cum_upgrade, actual_cum_total = \
        build_actual_cumulative_cost_components(upgrade_cost_timeline)

    if not actual_days:
        print("✓ No cumulative cost data available, skipping budget plot")
        return

    budget_days, budget_cum_opex, budget_cum_upgrade, budget_cum_total = \
        build_budget_components_curve(
            end_day=end_day,
            period_days=budget_params["period_days"],
            period_opex_budget_0=budget_params["period_opex_budget_0"],
            period_capex_budget_0=budget_params["period_capex_budget_0"],
            period_total_budget_0=budget_params["period_total_budget_0"],
            inflation_rate=budget_params["inflation_rate"],
        )

    interp_actual_total = np.interp(
        budget_days,
        np.array(actual_days, dtype=float),
        np.array(actual_cum_total, dtype=float),
        left=0.0,
        right=float(actual_cum_total[-1]),
    )

    over_budget_mask = interp_actual_total > budget_cum_total

    plt.figure(figsize=(12, 10))

    # 1) Total cost vs total budget
    plt.subplot(3, 1, 1)
    plt.plot(actual_days, actual_cum_total, linewidth=2, label="Actual cumulative total cost")
    plt.plot(budget_days, budget_cum_total, linestyle="--", linewidth=2, label="Cumulative total budget")

    if np.any(over_budget_mask):
        plt.fill_between(
            budget_days,
            budget_cum_total,
            interp_actual_total,
            where=over_budget_mask,
            alpha=0.25,
            label="Over budget"
        )

    for d in range(365, int(math.ceil(end_day)) + 1, 365):
        plt.axvline(d, linestyle=":", alpha=0.3)

    plt.xlabel("Time (days)", fontsize=11)
    plt.ylabel("Cumulative Cost", fontsize=11)
    plt.title(f"Actual Cumulative Cost vs Budget - {config['algorithm']} on {config['topology']}", fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)

    # 2) OPEX vs OPEX budget
    plt.subplot(3, 1, 2)
    plt.plot(actual_days, actual_cum_opex, linewidth=2, label="Actual cumulative OPEX")
    plt.plot(budget_days, budget_cum_opex, linestyle="--", linewidth=2, label="Cumulative OPEX budget")

    for d in range(365, int(math.ceil(end_day)) + 1, 365):
        plt.axvline(d, linestyle=":", alpha=0.3)

    plt.xlabel("Time (days)", fontsize=11)
    plt.ylabel("Cumulative OPEX", fontsize=11)
    plt.title("OPEX Cost vs OPEX Budget", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)

    # 3) CAPEX+workforce vs budget
    plt.subplot(3, 1, 3)
    plt.plot(actual_days, actual_cum_upgrade, linewidth=2, label="Actual cumulative CAPEX + workforce")
    plt.plot(budget_days, budget_cum_upgrade, linestyle="--", linewidth=2, label="Cumulative CAPEX + workforce budget")

    for d in range(365, int(math.ceil(end_day)) + 1, 365):
        plt.axvline(d, linestyle=":", alpha=0.3)

    plt.xlabel("Time (days)", fontsize=11)
    plt.ylabel("Cumulative Upgrade Cost", fontsize=11)
    plt.title("CAPEX + Workforce Cost vs Budget", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(config["budget_plot_file"], dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ Saved cost vs budget plot to {config['budget_plot_file']}")

def generate_blocking_vs_upgrade_plot(config, metrics, upgrade_decision_history):
    """
    Plot blocking probability and cycle-wise upgrade decision level together.
    Left axis  -> blocking probability
    Right axis -> cycle upgrade level
    """

    if not metrics.time_points:
        print("✓ No blocking data available, skipping blocking vs upgrade plot")
        return

    plt.figure(figsize=(12, 6))
    ax1 = plt.gca()

    # Blocking probability line
    ax1.plot(
        metrics.time_points,
        metrics.blocking_prob_points,
        linewidth=2,
        label="Blocking Probability"
    )
    ax1.axhline(
        y=config['blocking_threshold'],
        linestyle="--",
        color="r",
        label="Threshold"
    )
    ax1.set_xlabel("Time (days)", fontsize=12)
    ax1.set_ylabel("Blocking Probability", fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Upgrade level points on second axis
    ax2 = ax1.twinx()

    upgrade_colors = {
        "band_upgrade": "blue",
        "new_fiber_C": "green",
        "new_fiber_CL": "orange",
        "core_upgrade": "red"
    }

    upgrade_levels = {
        "band_upgrade": 1,
        "new_fiber_C": 2,
        "new_fiber_CL": 3,
        "core_upgrade": 4
    }

    if not upgrade_decision_history:
        print("✓ No upgrade decisions recorded, skipping upgrade markers")
    else:
        for entry in upgrade_decision_history:
            day = entry["time_day"]
            decisions = entry.get("final_decisions", [])

            for dec in decisions:
                upg_type = dec.get("upgrade_type")

                if upg_type not in upgrade_levels:
                    continue

                ax2.scatter(
                    day,
                    upgrade_levels[upg_type],
                    color=upgrade_colors[upg_type],
                    s=80
                )
    import matplotlib.patches as mpatches

    legend_patches = [
        mpatches.Patch(color="blue", label="Band Upgrade"),
        mpatches.Patch(color="green", label="New Fiber C"),
        mpatches.Patch(color="orange", label="New Fiber C+L"),
        mpatches.Patch(color="red", label="Core Upgrade"),
    ]

    ax2.legend(handles=legend_patches, loc="upper right")

    ax2.set_ylabel("Upgrade Decision Level", fontsize=12)
    ax2.set_yticks([1, 2, 3, 4])
    ax2.set_yticklabels([
        "Band",
        "New Fiber C",
        "New Fiber C+L",
        "Core"
    ])

    plt.title(
        f"Blocking Probability and Upgrade Decisions - {config['algorithm']} on {config['topology']}",
        fontsize=14
    )
    plt.tight_layout()

    out_file = config['results_dir'] / 'blocking_vs_upgrade_decisions.png'
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ Saved blocking vs upgrade decision plot to {out_file}")

def save_path_stats_history_csv(path_stats_history: dict, results_dir: Path):
    """
    Save full-history path statistics to CSV.

    Columns:
      src, dest, path_index, arrivals, accepted, blocked, blocking_probability
    """
    out_file = results_dir / "path_stats_history.csv"

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "src", "dest", "path_index",
            "path_arrivals", "path_accepted", "path_blocked",
            "path_blocking_probability"
        ])

        for key in sorted(path_stats_history.keys()):
            src, dest, path_idx = key
            row = path_stats_history[key]

            arrivals = row.get("path_arrivals", 0)
            accepted = row.get("path_accepted", 0)
            blocked = row.get("path_blocked", 0)
            bp = (blocked / arrivals) if arrivals > 0 else 0.0

            writer.writerow([
                src, dest, path_idx,
                arrivals, accepted, blocked,
                bp
            ])

    print(f"✓ Saved full path statistics to {out_file}")


# ============================================================================
# Entry Point
# ============================================================================

# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":

    # ============================================================
    # Delete old XGBoost CSV ONCE before running seeds
    # Do NOT put this inside run_simulation().
    # Do NOT put this inside the for seed loop.
    # ============================================================
    if (
        CONFIG.get("collect_xgboost_data", False)
        and CONFIG.get("reset_xgboost_csv_before_run", False)
    ):
        results_dir = create_results_directory(
            CONFIG["topology"],
            CONFIG["algorithm"]
        )

        xgb_csv_path = results_dir / CONFIG.get(
            "xgboost_csv_name",
            "xgboost_need_upgrade_dataset.csv"
        )

        if xgb_csv_path.exists():
            xgb_csv_path.unlink()
            print(f">>> Deleted old XGBoost CSV: {xgb_csv_path}")
        else:
            print(f">>> No old XGBoost CSV found: {xgb_csv_path}")

    # ============================================================
    # Run multiple seeds
    # ============================================================
    if CONFIG.get("run_multiple_seeds", False):

        seed_start = int(CONFIG.get("seed_start", 0))
        seed_end = int(CONFIG.get("seed_end", 19))

        for seed_value in range(seed_start, seed_end + 1):

            run_config = copy.deepcopy(CONFIG)
            run_config["seed"] = seed_value

            results_dir = create_results_directory(
                run_config["topology"],
                run_config["algorithm"]
            )

            log_file_path = results_dir / f"simulation_log_seed_{seed_value}.txt"

            log_file = open(log_file_path, "w", encoding="utf-8", buffering=1)
            sys.stdout = TeeOutput(sys.__stdout__, log_file)

            try:
                print("\n" + "#" * 80)
                print(f"STARTING SEED {seed_value}")
                print("#" * 80 + "\n")

                run_simulation(run_config)

                print("\n" + "#" * 80)
                print(f"FINISHED SEED {seed_value}")
                print("#" * 80 + "\n")

            finally:
                try:
                    sys.stdout = sys.__stdout__
                except Exception:
                    pass

                try:
                    log_file.close()
                except Exception:
                    pass

    # ============================================================
    # Run only one seed
    # ============================================================
    else:
        results_dir = create_results_directory(
            CONFIG["topology"],
            CONFIG["algorithm"]
        )

        log_file_path = results_dir / f"simulation_log_seed_{CONFIG['seed']}.txt"

        log_file = open(log_file_path, "w", encoding="utf-8", buffering=1)
        sys.stdout = TeeOutput(sys.__stdout__, log_file)

        try:
            run_simulation(CONFIG)

        finally:
            try:
                sys.stdout = sys.__stdout__
            except Exception:
                pass

            try:
                log_file.close()
            except Exception:
                pass




# """
# Optical Network Simulator - Main Entry Point
#
# A modular optical network simulator with network upgrade planning capabilities.
# Configure simulation parameters in the CONFIG dictionary below.
# """
#
# import sys
# import random
# from math import floor
#
# import numpy as np
# import matplotlib
# matplotlib.use('Agg')  # Non-interactive backend for headless environments
# import matplotlib.pyplot as plt
# import math
# import copy
# import csv
# from pathlib import Path
# from typing import Dict, Tuple, Any, Optional
# import time
# from collections import defaultdict
#
#
# # ============================================================================
# # CONFIGURATION - Edit this dictionary to configure your simulation
# # ============================================================================
#
# CONFIG = {
#     # Topology: 'test5' (5-node test network) or 'dt12' (12-node production network)
#     'topology': 'test5',
#
#     # Algorithm: 'greedy', 'increment', 'max_initial', 'jump_to_max', 'no_upgrade'
#     'algorithm': 'increment',  # or 'max_initial' or 'jump_to_max'
#
#     # Random seed for reproducibility
#     'seed': 0,
#
#     # ---------------- Budget envelope settings ----------------
#     'budget_enabled': True,
#
#     # # Allowed daily OPEX budget = factor × initial daily OPEX
#     # 'opex_budget_factor': 3,
#     #
#     # # Daily OPEX budget grows every upgrade period (90 days)
#     # 'opex_budget_growth_per_period': 0.2,
#     #
#     # # If None, auto-compute CAPEX+workforce budget per upgrade period
#     # 'period_capex_workforce_budget_0': None,
#     #
#     # # Starting CAPEX+workforce budget as a fraction of 90-day OPEX budget
#     # 'upgrade_budget_factor': 2.0,
#     #
#     # # CAPEX+workforce budget growth per upgrade period
#     # 'capex_workforce_budget_growth_per_period': 0.2,
# }
#
# # ============================================================================
# # Import simulation modules from sim package
# # ============================================================================
#
# from sim.topologies.topology_manager import TopologyManager
# from sim.core import topology as Topology
# from sim.algorithms.algorithm_selector import AlgorithmSelector
# from sim.core.constants import *
# from sim.core.traffic_generator import NetworkTrafficGenerator
# from sim.routing.shortest_path import build_k_shortest_paths
#
# from sim.upgrade.upgrade_manager import (
#     initialize_link_status,
#     choose_upgrade_type,
#     perform_upgrade,
#     restore_removed_links_after_downtime,
#     network_connectivity,
#     print_upgrade_decisions_summary,
#     build_ordered_upgrade_jobs,
# )
# # >>> CHANGE 1 END
#
# from sim.algorithms.greedy import choose_upgrade_type_greedy_max
# from sim.upgrade.plan_checker import plan_checker
# from sim.upgrade.opex_model import compute_running_opex_interval, compute_network_opex_per_day
#
# # >>> CHANGE 2 START: import start_link_downtime_job
# from sim.upgrade.sequential_upgrade import (
#     start_downtime_upgrade_for_link,
#     start_link_downtime_job,     # <-- must exist in sequential_upgrade.py
# )
# # >>> CHANGE 2 END
#
# from sim.modules.performance_metrics import PerformanceMetrics
# from sim.modules.simulation_analysis import SimulationAnalysis
# from sim.routing.rsa import execute_first_fit
#
# from sim.upgrade.link_utilization_fragmentation import (
#     init_monthly_rowwise_csvs,
#     log_monthly_link_stats_rowwise,
# )
#
# from sim.upgrade.upgrade_logging import init_upgrade_log, log_upgrade_action
# from sim.upgrade.cost_model import compute_upgrade_costs
#
#
# # ============================================================================
# # Helper Functions
# # ============================================================================
#
# def create_results_directory(topology: str, algorithm: str) -> Path:
#     """Create organized results directory for simulation outputs."""
#     results_dir = Path('results') / f"{algorithm}_{topology}"
#     results_dir.mkdir(parents=True, exist_ok=True)
#     return results_dir
#
#
# def sd_key_undirected(s: int, d: int):
#     """Create undirected source-destination pair key for tracking."""
#     return (s, d) if s < d else (d, s)
#
# def init_connection_request_log(results_dir: Path) -> Path:
#     """
#     Create CSV to log every arriving connection request.
#     """
#     out_file = results_dir / "connection_requests.csv"
#
#     with open(out_file, "w", newline="", encoding="utf-8") as f:
#         writer = csv.writer(f)
#         writer.writerow([
#             "connection_id",
#             "source",
#             "destination",
#             "datarate",
#             "arrival_time",
#             "holding_time",
#             "departure_time",
#             "status",              # accepted / blocked
#             "mf_index",
#             "fs_index",
#             "slice_window_size",
#             "path",
#             "link_ids",
#             "fibers_used",
#             "cores_used",
#         ])
#
#     return out_file
#
# def build_connection_request_row(
#     connection_id: int,
#     src: int,
#     dest: int,
#     datarate: float,
#     arrival_time: float,
#     holding_time: float,
#     departure_time: float,
#     status: str,
#     mf_index,
#     fs_index,
#     slice_window_size,
#     path,
#     link_ids,
#     fibers_used,
#     cores_used,
# ):
#     def safe_str(x):
#         if x is None:
#             return ""
#         if isinstance(x, float) and math.isinf(x):
#             return "inf"
#         return str(x)
#
#     return [
#         connection_id,
#         src,
#         dest,
#         datarate,
#         arrival_time,
#         holding_time,
#         departure_time,
#         status,
#         safe_str(mf_index),
#         safe_str(fs_index),
#         safe_str(slice_window_size),
#         safe_str(path),
#         safe_str(link_ids),
#         safe_str(fibers_used),
#         safe_str(cores_used),
#     ]
#
#
# class TeeOutput:
#     """Redirect output to both console and file simultaneously."""
#
#     def __init__(self, *files):
#         self.files = files
#
#     def write(self, message):
#         for f in self.files:
#             try:
#                 f.write(message)
#             except Exception:
#                 pass
#         self.flush()
#
#     def flush(self):
#         for f in self.files:
#             try:
#                 f.flush()
#             except Exception:
#                 pass
#
# UPGRADE_LEVEL_MAP = {
#     "band_upgrade": 1,
#     "new_fiber_C": 2,
#     "new_fiber_CL": 3,
#     "core_upgrade": 4,
# }
#
#
# def flatten_upgrade_decisions(decisions):
#     """
#     Convert upgrade_decisions dict into a flat list of per-link decisions.
#     Input format:
#         {
#             link_id: [decision_dict, decision_dict, ...],
#             ...
#         }
#     """
#     rows = []
#
#     if not decisions:
#         return rows
#
#     for lid, decs in decisions.items():
#         if not isinstance(decs, list):
#             continue
#
#         for dec in decs:
#             if dec is None:
#                 continue
#
#             rows.append({
#                 "link_id": lid,
#                 "upgrade_type": dec.get("upgrade_type"),
#                 "fiber_id": dec.get("fiber_id"),
#                 "core_type": dec.get("core_type"),
#             })
#
#     return rows
#
#
# def get_cycle_upgrade_level(flat_decisions):
#     """
#     Return one representative level for the whole cycle.
#     Highest upgrade type in the cycle is used.
#     """
#     levels = []
#
#     for dec in flat_decisions:
#         upg = dec.get("upgrade_type")
#         if upg in UPGRADE_LEVEL_MAP:
#             levels.append(UPGRADE_LEVEL_MAP[upg])
#
#     return max(levels) if levels else 0
#
#
# # ============================================================================
# # Simulation Initialization
# # ============================================================================
#
# def initialize_simulation(config: Dict[str, Any]):
#     """
#     Initialize simulation with topology and algorithm.
#
#     Returns:
#         tuple: (topology_data, algorithm, results_dir, extended_config)
#     """
#     # Load topology
#     topology_data = TopologyManager.load_topology(config['topology'])
#
#     # Set topology in legacy module for compatibility
#     Topology.TOPOLOGY = topology_data['TOPOLOGY']
#     Topology.TOPOLOGY_LINK_LENGTHS = topology_data['TOPOLOGY_LINK_LENGTHS']
#     Topology.LINK_INDEX = topology_data['LINK_INDEX']
#     Topology.N = topology_data['N']
#     Topology.LINKS = topology_data['LINKS']
#
#     # Load algorithm
#     algorithm = AlgorithmSelector.get_algorithm(config['algorithm'])
#
#     # Create results directory
#     results_dir = create_results_directory(config['topology'], config['algorithm'])
#     upgrade_log_csv = init_upgrade_log(results_dir)
#
#     # Extend config with simulation parameters and output paths
#     extended_config = {
#         **config,
#         'warmup_connections': WARMUP,
#         'blocking_threshold': Total_number_of_allowed_blocked_probability_in_network_lifetime,
#         'results_dir': results_dir,
#         'blocking_csv': results_dir / 'blocking_results.csv',
#         'snapshot_file': results_dir / 'final_state.pkl',
#         'blocking_plot_file': results_dir / 'blocking_probability_vs_time.png',
#         'cost_plot_file': results_dir / 'cost_vs_time.png',
#         'metrics_plot_file': results_dir / 'performance_metrics.png',
#         'log_file': results_dir / 'simulation_log.txt',
#         'upgrade_log_csv': upgrade_log_csv,
#         'connection_request_csv': results_dir / 'connection_requests.csv',
#         'budget_plot_file': results_dir / 'budget_vs_cost.png',
#     }
#
#     return topology_data, algorithm, results_dir, extended_config
#
#
#
# # ============================================================================
# # Main Simulation Logic
# # ============================================================================
#
# def run_simulation(config: Dict[str, Any]):
#     """
#     Run the optical network simulation with given configuration.
#     """
#     topology_data, algorithm, results_dir, config = initialize_simulation(config)
#     upgrade_log_csv = config["upgrade_log_csv"]
#     connection_request_csv = init_connection_request_log(config["results_dir"])
#     config["connection_request_csv"] = connection_request_csv
#     connection_log_file = open(connection_request_csv, "a", newline="", encoding="utf-8")
#     connection_log_writer = csv.writer(connection_log_file)
#
#     # Wall-clock runtime measurement
#     start_wall_time = time.time()
#
#     print("="*70)
#     print("OPTICAL NETWORK SIMULATOR")
#     print("="*70)
#     print(f"Topology: {config['topology']}")
#     print(f"Algorithm: {algorithm.name}")
#     print(f"Seed: {config['seed']}")
#     print("="*70 + "\n")
#
#     # Extract topology information
#     TOPOLOGY = topology_data['TOPOLOGY']
#     TOPOLOGY_LINK_LENGTHS = topology_data['TOPOLOGY_LINK_LENGTHS']
#     LINK_INDEX = topology_data['LINK_INDEX']
#     N = topology_data['N']
#     LINKS = topology_data['LINKS']
#
#     print(f"✓ Using topology: {topology_data['NAME']} ({N} nodes, {LINKS} links)\n")
#
#     # Monthly per-link logging (row-wise CSV)
#     MONTH_DAYS = 30
#     current_month = 0
#     util_csv, frag_csv = init_monthly_rowwise_csvs(config["results_dir"])
#
#     # Initialize RNG
#     seed = config['seed']
#     random.seed(seed)
#     main_rng = random.Random(seed)
#
#     # Initialize metrics and analysis
#     metrics = PerformanceMetrics()
#     analysis = SimulationAnalysis()
#
#     # ============================================================
#     # PATH STATISTICS (NO lambda)
#     # key = (undirected_s, undirected_d, path_index)
#     # ============================================================
#
#     path_stats = {}
#     # Full-history path statistics across the whole simulation
#     path_stats_history = {}
#
#     # Initialize network state
#     total_network_cost = 0
#     upgrade_cost_timeline = []
#     connection_id = 0
#     upgrade_decision_history = []
#     removed_links_info = {}
#     Current_global_time = 0
#     current_week = 0
#     current_upgrade = 0
#     last_cost_time = 0.0
#     last_logged_day = -1
#
#     # >>> CHANGE 3 START: persistent downtime JOB queue (does not reset each cycle)
#     # This holds entries like:
#     #   ("fiber", link_id, decision_dict)
#     #   ("link",  link_id, [decision_dict, decision_dict, ...])   # core first
#     pending_downtime_jobs = []
#     # >>> CHANGE 3 END
#
#     # Validate traffic parameters
#     if lambda_0 <= 0:
#         raise ValueError("lambda_0 must be > 0")
#     if MEAN_HOLDING_TIME <= 0:
#         raise ValueError("MEAN_HOLDING_TIME must be > 0")
#     if Traffic_growth_days <= 0:
#         raise ValueError("Traffic_growth_days must be > 0")
#
#     # Validate topology parameters
#     if N <= 0:
#         raise ValueError("Number of nodes (N) must be > 0")
#     if LINKS <= 0:
#         raise ValueError("Number of links (LINKS) must be > 0")
#
#     # Validate algorithm
#     if algorithm is None:
#         raise ValueError("Algorithm cannot be None")
#
#     traffic_rates = np.full((N, N), lambda_0, dtype=float)
#     np.fill_diagonal(traffic_rates, 0)
#
#     ALL_DEMANDS = []
#     working_topology = copy.deepcopy(TOPOLOGY)
#
#     # Initialize link status
#     if hasattr(algorithm, 'initialize_link_status'):
#         link_status_forward, link_status_backward = algorithm.initialize_link_status(LINKS)
#     else:
#         link_status_forward, link_status_backward = initialize_link_status(LINKS)
#
#     if not link_status_forward or not link_status_backward:
#         raise ValueError("Link status initialization returned empty dicts")
#
#     # ------------------------------------------------------------
#     # COMMON budget envelope
#     # - OPEX budget is derived from actual initial network OPEX
#     # - CAPEX/WF budget is a policy value based on your upgrade-budget constants
#     # ------------------------------------------------------------
#     initial_daily_opex = float(compute_network_opex_per_day(link_status_forward))
#
#     budget_params = {
#         "period_days": int(Upgrade_initiation_days),  # 180
#         "period_total_budget_0": 1_000_0000.0,
#         "period_opex_budget_0": 150_0000.0,
#         "period_capex_budget_0": 850_0000.0,
#         "inflation_rate": 0.03,
#     }
#     print("\nBUDGET ENVELOPE PARAMETERS")
#     print("-" * 50)
#     print(f"Initial daily OPEX (actual)         : {initial_daily_opex:.2f}")
#     print(f"Initial OPEX budget / cycle         : {budget_params['period_opex_budget_0']:.2f}")
#     print(f"Initial CAPEX/WF budget / cycle     : {budget_params['period_capex_budget_0']:.2f}")
#     print(f"Initial TOTAL budget / cycle        : {budget_params['period_total_budget_0']:.2f}")
#     print(f"Inflation / period                  : {100 * budget_params['inflation_rate']:.1f}%")
#     print(f"Period length (days)                : {budget_params['period_days']}")
#     print("-" * 50)
#     # Traffic generator
#
#     traffic_generator = NetworkTrafficGenerator(
#         number_of_nodes=N,
#         datarates=DATARATE,
#         lambda_0=lambda_0,
#         mean_holding_time=MEAN_HOLDING_TIME,
#         Current_global_time=Current_global_time,
#         rng=main_rng
#     )
#
#     # Build paths
#     PATHS = build_k_shortest_paths(TOPOLOGY)
#     if not PATHS:
#         raise ValueError("build_k_shortest_paths() returned empty structure")
#
#     print("✓ Simulation initialized\n")
#     print("Starting simulation loop...\n")
#
#     def account_running_opex_until(new_time: float):
#         """
#         Charges OPEX continuously, but logs it cleanly day-by-day:
#           - For each day boundary crossed, append one timeline entry
#           - Each daily entry contains the OPEX charged for that day segment
#           - Upgrade events are separate entries (CAPEX/workforce jumps)
#         """
#         nonlocal total_network_cost, last_cost_time, upgrade_cost_timeline, last_logged_day
#
#         new_time = float(new_time)
#         t = float(last_cost_time)
#
#         if new_time <= t:
#             return
#
#         while t < new_time:
#             # next integer day boundary
#             next_boundary = float(int(t) + 1)
#             seg_end = min(new_time, next_boundary)
#             dt = seg_end - t
#             if dt <= 0:
#                 break
#
#             # OPEX for this segment (rate based on CURRENT state)
#             opex_inc = float(compute_running_opex_interval(link_status_forward, dt))
#             total_network_cost += opex_inc
#
#             day_marker = int(seg_end)
#             if day_marker > last_logged_day:
#                 opex_rate = float(compute_network_opex_per_day(link_status_forward))
#
#                 upgrade_cost_timeline.append({
#                     "day": float(day_marker),
#                     "equipment": 0.0,
#                     "workforce": 0.0,
#                     "opex": float(opex_inc),  # OPEX charged in this segment
#                     "opex_rate": float(opex_rate),  # informational
#                     "total": float(opex_inc),
#                     "cumulative_total": float(total_network_cost),
#                 })
#                 last_logged_day = day_marker
#             else:
#                 # same day marker: merge OPEX into the last row but keep total consistent
#                 if upgrade_cost_timeline:
#                     row = upgrade_cost_timeline[-1]
#                     row["opex"] = float(row.get("opex", 0.0)) + float(opex_inc)
#
#                     row["total"] = (
#                             float(row.get("equipment", 0.0))
#                             + float(row.get("workforce", 0.0))
#                             + float(row.get("opex", 0.0))
#                     )
#                     row["cumulative_total"] = float(total_network_cost)
#
#             t = seg_end
#
#         last_cost_time = float(new_time)
#
#
#     def start_next_link_job(
#         link_execution_queue,
#         per_link_jobs,
#         current_time,
#         period_index,
#         working_topology,
#         link_status_forward,
#         link_status_backward,
#         removed_links_info,
#         ALL_DEMANDS,
#         total_network_cost,
#         PATHS,
#     ):
#         """
#         Starts the next link's downtime work ONLY if no downtime is currently active.
#
#         Returns updated:
#             link_execution_queue,
#             working_topology,
#             link_status_forward,
#             link_status_backward,
#             removed_links_info,
#             ALL_DEMANDS,
#             total_network_cost,
#             PATHS
#         """
#         if not link_execution_queue or removed_links_info:
#             return (
#                 link_execution_queue,
#                 working_topology,
#                 link_status_forward,
#                 link_status_backward,
#                 removed_links_info,
#                 ALL_DEMANDS,
#                 total_network_cost,
#                 PATHS,
#             )
#
#         lid = link_execution_queue.pop(0)
#         decs = per_link_jobs.get(lid, [])
#
#         if not decs:
#             print(f">>> Link {lid} has no downtime jobs. Moving to next link.")
#             return (
#                 link_execution_queue,
#                 working_topology,
#                 link_status_forward,
#                 link_status_backward,
#                 removed_links_info,
#                 ALL_DEMANDS,
#                 total_network_cost,
#                 PATHS,
#             )
#
#         has_core = any(d.get("upgrade_type") == "core_upgrade" for d in decs)
#
#         if has_core:
#             print(f">>> Starting LINK downtime job on link {lid}: {decs}")
#
#             for dec in decs:
#                 log_upgrade_action(
#                     upgrade_log_csv,
#                     day=current_time,
#                     period_index=period_index,
#                     scope="link_downtime",
#                     link_id=lid,
#                     upgrade_type=dec.get("upgrade_type", ""),
#                     fiber_id=dec.get("fiber_id", None),
#                     core_type=dec.get("core_type", None),
#                 )
#
#             (
#                 working_topology,
#                 link_status_forward,
#                 link_status_backward,
#                 removed_links_info,
#                 ALL_DEMANDS,
#                 blocked_datarate_rerouting,
#                 accepted_datarate_rerouting,
#                 arrived_datarate_rerouting,
#                 total_network_cost,
#                 PATHS,
#             ) = start_link_downtime_job(
#                 link_id=lid,
#                 decisions=decs,
#                 pending_downtime_actions=None,
#                 ALL_DEMANDS=ALL_DEMANDS,
#                 link_status_forward=link_status_forward,
#                 link_status_backward=link_status_backward,
#                 working_topology=working_topology,
#                 removed_links_info=removed_links_info,
#                 current_time=current_time,
#                 blocked_datarate_rerouting=0,
#                 accepted_datarate_rerouting=0,
#                 arrived_datarate_rerouting=0,
#                 total_network_cost=total_network_cost,
#                 upgrade_cost_timeline=upgrade_cost_timeline,
#                 algorithm=algorithm
#             )
#
#         else:
#             # Fiber-only link: start ONLY the first fiber downtime decision now.
#             # Remaining fiber decisions for the same link are pushed back to the front
#             # so they run immediately after restore, before the next link starts.
#             first_dec = decs[0]
#             remaining = decs[1:]
#
#             print(f">>> Starting FIBER downtime job on link {lid}: {first_dec}")
#
#             log_upgrade_action(
#                 upgrade_log_csv,
#                 day=current_time,
#                 period_index=period_index,
#                 scope="fiber_downtime",
#                 link_id=lid,
#                 upgrade_type=first_dec.get("upgrade_type", ""),
#                 fiber_id=first_dec.get("fiber_id", None),
#                 core_type=first_dec.get("core_type", None),
#             )
#
#             (
#                 working_topology,
#                 link_status_forward,
#                 link_status_backward,
#                 removed_links_info,
#                 ALL_DEMANDS,
#                 blocked_datarate_rerouting,
#                 accepted_datarate_rerouting,
#                 arrived_datarate_rerouting,
#                 total_network_cost,
#                 PATHS,
#             ) = start_downtime_upgrade_for_link(
#                 scope="fiber",
#                 link_id=lid,
#                 decision=first_dec,
#                 pending_downtime_actions=None,
#                 ALL_DEMANDS=ALL_DEMANDS,
#                 link_status_forward=link_status_forward,
#                 link_status_backward=link_status_backward,
#                 working_topology=working_topology,
#                 removed_links_info=removed_links_info,
#                 current_time=current_time,
#                 blocked_datarate_rerouting=0,
#                 accepted_datarate_rerouting=0,
#                 arrived_datarate_rerouting=0,
#                 total_network_cost=total_network_cost,
#                 upgrade_cost_timeline=upgrade_cost_timeline,
#                 algorithm=algorithm
#             )
#
#             # Put remaining fiber decisions for SAME link back at the front
#             if remaining:
#                 per_link_jobs[lid] = remaining
#                 link_execution_queue.insert(0, lid)
#             else:
#                 per_link_jobs[lid] = []
#
#         return (
#             link_execution_queue,
#             working_topology,
#             link_status_forward,
#             link_status_backward,
#             removed_links_info,
#             ALL_DEMANDS,
#             total_network_cost,
#             PATHS,
#         )
#
#     while True:
#
#         # Weekly traffic growth
#         while (Current_global_time // Traffic_growth_days) > current_week:
#             print(f">>> Weekly traffic update at day {Current_global_time}")
#             current_week += 1
#
#             for i in range(N):
#                 for j in range(N):
#                     if i != j:
#                         growth_factor = 1 + random.uniform(0, alpha / 100)
#                         traffic_rates[i][j] *= growth_factor
#
#             for i in range(N):
#                 for j in range(N):
#                     if i != j:
#                         growth_factor_new = (1 + delta / 100)
#                         traffic_rates[i][j] *= growth_factor_new
#
#         # Generate next request
#         src, dest, datarate = traffic_generator.generate_connection_data()
#         lambda_sd = traffic_rates[src][dest]
#
#         if lambda_sd <= 0:
#             raise ValueError("Invalid lambda_sd: must be non-zero and positive.")
#
#         arrival_time, holding_time = traffic_generator.get_connection(lambda_sd)
#         Current_global_time = arrival_time
#         # ✅ OPEX from last time -> now
#         account_running_opex_until(Current_global_time)
#         departure_time = arrival_time + holding_time
#         period_index = int(Current_global_time // tau)
#
#         # Monthly logging
#         month_idx = int(Current_global_time // MONTH_DAYS)
#         if month_idx > current_month:
#             current_month = month_idx
#             log_monthly_link_stats_rowwise(
#                 time_day=Current_global_time,
#                 LINKS=LINKS,
#                 link_status_forward=link_status_forward,
#                 link_status_backward=link_status_backward,
#                 C_BAND_SLOTS=C_BAND_SLOTS,
#                 TOTAL_SLOTS=TOTAL_SLOTS,
#                 util_csv=util_csv,
#                 frag_csv=frag_csv,
#             )
#
#         connection_id += 1
#
#         # ====================================================================
#         # Upgrade Cycle Check (every 3 months)
#         # ====================================================================
#
#         if (Current_global_time // Upgrade_initiation_days) > current_upgrade:
#             print(f"\n=== Upgrade cycle triggered at day {Current_global_time} ===")
#             current_upgrade += 1
#
#             print(">>> Checking whether upgrade is needed...")
#             upgrade_needed = algorithm.check_need_for_upgrade(
#                 current_traffic=traffic_rates,
#                 current_time=Current_global_time,
#                 PATHS=PATHS,
#                 current_link_status_forward=copy.deepcopy(link_status_forward),
#                 current_link_status_backward=copy.deepcopy(link_status_backward),
#                 seed=seed
#             )
#
#             print(f">>> Upgrade needed = {upgrade_needed}")
#
#             if upgrade_needed:
#                 sd_last_block_prob = metrics.get_sd_blocking_probability()
#
#                 print(">>> Selecting candidate links for upgrade...")
#                 upgrade_links, link_scores, top_contrib = algorithm.select_links_for_upgrade(
#                     traffic_rates=traffic_rates,
#                     PATHS=PATHS,
#                     sd_block_prob=sd_last_block_prob,
#                     block_threshold=PER_SD_BLOCK_THRESHOLD,
#                     working_topology=working_topology,
#                     LINK_INDEX=LINK_INDEX,
#                     link_status_forward=link_status_forward,
#                     link_status_backward=link_status_backward,
#                     current_time=Current_global_time,
#                     path_stats=path_stats
#                 )
#
#                 print(f">>> selected links: {upgrade_links}")
#
#                 safe_links = network_connectivity(
#                     upgrade_links,
#                     working_topology,
#                     LINK_INDEX
#                 )
#                 print(f">>> Safe links = {safe_links}")
#
#                 safe_links = sorted(
#                     safe_links,
#                     key=lambda lid: float(link_scores.get(lid, 0.0)),
#                     reverse=True
#                 )
#
#                 print(f">>> Safe links after sorting by decreasing score = {safe_links}")
#
#                 # ------------------------------------------------------------
#                 # Build INITIAL next-step upgrade decisions for all safe links
#                 # ------------------------------------------------------------
#                 if config['algorithm'] == 'greedy':
#                     upgrade_decisions = {
#                         link_id: [choose_upgrade_type_greedy_max(
#                             link_id,
#                             link_status_forward,
#                             link_status_backward
#                         )]
#                         for link_id in safe_links
#                     }
#                 elif config['algorithm'] == 'jump_to_max':
#                     upgrade_decisions = {
#                         link_id: algorithm.get_upgrade_decision(
#                             link_id,
#                             link_status_forward,
#                             link_status_backward
#                         )
#                         for link_id in safe_links
#                     }
#                 else:
#                     upgrade_decisions = {
#                         link_id: [choose_upgrade_type(
#                             link_id,
#                             link_status_forward,
#                             link_status_backward,
#                             algorithm
#                         )]
#                         for link_id in safe_links
#                     }
#
#
#                 # Clear SD stats for next cycle
#                 # Clear SD stats for next cycle
#
#                 metrics.sd_arrivals.clear()
#                 metrics.sd_blocked.clear()
#
#                 print(">>> Running PlanChecker...")
#                 plan_ok, final_links, final_upgrade_decisions = plan_checker(
#                     safe_links=safe_links,
#                     upgrade_decisions=upgrade_decisions,
#                     current_time=Current_global_time,
#                     PATHS=PATHS,
#                     current_traffic=traffic_rates,
#                     current_link_status_forward=copy.deepcopy(link_status_forward),
#                     current_link_status_backward=copy.deepcopy(link_status_backward),
#                     seed=seed,
#                     algorithm=algorithm,
#                     budget_params=budget_params,
#                 )
#                 if not plan_ok:
#                     print(">>> PlanChecker FAILED")
#                 else:
#                     print(">>> PlanChecker PASSED")
#
#                  path_stats.clear()
#
#                 safe_links = final_links
#                 upgrade_decisions = final_upgrade_decisions
#
#                 flat_final_decisions = flatten_upgrade_decisions(upgrade_decisions)
#
#                 upgrade_decision_history.append({
#                         "cycle": int(current_upgrade),
#                         "time_day": float(Current_global_time),
#                         "blocking_probability": float(metrics.get_blocking_probability()),
#                         "final_decisions": flat_final_decisions,
#                         "cycle_upgrade_level": get_cycle_upgrade_level(flat_final_decisions),
#                     })
#                 print_upgrade_decisions_summary(Current_global_time, safe_links, upgrade_decisions)
#
#                 # ------------------------------------------------------------
#                     # Build per-link execution plan
#                     #   - Immediate actions are applied globally first
#                     #   - Remaining downtime work is executed link-by-link
#                 # ------------------------------------------------------------
#                 immediate_actions = {}
#                 per_link_jobs = {}
#
#                 for lid in safe_links:
#                         decs = upgrade_decisions.get(lid, [])
#                         if isinstance(decs, dict):
#                             decs = [decs]
#                         else:
#                             decs = list(decs)
#
#                         immediate_actions[lid] = []
#                         per_link_jobs[lid] = []
#
#                         for dec in decs:
#                             if dec is None or dec.get("upgrade_type") is None:
#                                 continue
#
#                             utype = dec.get("upgrade_type")
#
#                             # Only truly immediate action
#                             if utype == "new_fiber_C":
#                                 immediate_actions[lid].append(dec)
#                             else:
#                                 # band_upgrade, new_fiber_CL, core_upgrade, and any post-core tail
#                                 per_link_jobs[lid].append(dec)
#
#                 # Clean empty entries
#                 immediate_actions = {
#                         lid: decs for lid, decs in immediate_actions.items() if decs
#                 }
#                 per_link_jobs = {
#                         lid: decs for lid, decs in per_link_jobs.items() if decs
#                 }
#
#                 # This queue preserves exact PlanChecker order
#                 link_execution_queue = [lid for lid in safe_links if lid in per_link_jobs]
#
#                 # ============================================================
#                 # ONE-SHOT COST COMPUTATION FOR THE WHOLE UPGRADE CYCLE
#                 # ============================================================
#                 cycle_actions = {}
#
#                 # Immediate actions
#                 for lid, decs in immediate_actions.items():
#                         cycle_actions.setdefault(lid, []).extend(decs)
#
#                 # Downtime actions
#                 for lid, decs in per_link_jobs.items():
#                         cycle_actions.setdefault(lid, []).extend(decs)
#
#                 cycle_links = sorted(cycle_actions.keys())
#
#                 total_cost_cycle, cost_summary, cost_details = compute_upgrade_costs(
#                         safe_links_for_upgrade=cycle_links,
#                         upgrade_decisions=cycle_actions,
#                         algorithm_name=algorithm.name
#                 )
#
#                 equip = float(cost_summary.get("equipment_total", 0.0))
#                 work = float(cost_summary.get("workforce_total", 0.0))
#                 capex_work_cycle = equip + work
#
#                 total_network_cost += capex_work_cycle
#
#                 day_marker = int(Current_global_time)
#
#                 if not upgrade_cost_timeline or int(upgrade_cost_timeline[-1].get("day", -1)) != day_marker:
#                         upgrade_cost_timeline.append({
#                             "day": float(day_marker),
#                             "equipment": 0.0,
#                             "workforce": 0.0,
#                             "opex": 0.0,
#                             "total": 0.0,
#                             "cumulative_total": float(total_network_cost),
#                 })
#
#                 row = upgrade_cost_timeline[-1]
#                 row["equipment"] = float(row.get("equipment", 0.0)) + equip
#                 row["workforce"] = float(row.get("workforce", 0.0)) + work
#                 row["total"] = (
#                         float(row.get("equipment", 0.0))
#                         + float(row.get("workforce", 0.0))
#                         + float(row.get("opex", 0.0))
#                     )
#                 row["cumulative_total"] = float(total_network_cost)
#
#                 print(
#                         f">>> ONE-SHOT upgrade cycle cost jump applied: "
#                         f"equip={equip:.2f}, workforce={work:.2f}, total={capex_work_cycle:.2f}"
#                     )
#
#                 # ------------------------------------------------------------
#                 # Apply IMMEDIATE upgrades (no downtime)
#                 # ------------------------------------------------------------
#                 if immediate_actions:
#                         immediate_links = sorted(immediate_actions.keys())
#                         print(f">>> Immediate (no-downtime) upgrades on links {immediate_links}")
#
#                         for link_id, decs in immediate_actions.items():
#                             for dec in decs:
#                                 log_upgrade_action(
#                                     upgrade_log_csv,
#                                     day=Current_global_time,
#                                     period_index=period_index,
#                                     scope="immediate",
#                                     link_id=link_id,
#                                     upgrade_type=dec.get("upgrade_type", ""),
#                                     fiber_id=dec.get("fiber_id", None),
#                                     core_type=dec.get("core_type", None),
#                                 )
#
#                         perform_upgrade(
#                             immediate_links,
#                             immediate_actions,
#                             link_status_forward,
#                             link_status_backward,
#                             C_BAND_SLOTS,
#                             TOTAL_SLOTS,
#                             Current_global_time,
#                             algorithm
#                         )
#
#                 link_execution_queue = list(safe_links)
#                 if link_execution_queue and not removed_links_info:
#
#                         lid = link_execution_queue.pop(0)
#                         decs = per_link_jobs.get(lid, [])
#
#                         if not decs:
#                             continue
#
#                         has_core = any(d["upgrade_type"] == "core_upgrade" for d in decs)
#
#                         if has_core:
#                             print(f">>> Starting LINK downtime job on link {lid}: {decs}")
#
#                             for dec in decs:
#                                 log_upgrade_action(...)
#
#                             (
#                                 working_topology,
#                                 link_status_forward,
#                                 link_status_backward,
#                                 removed_links_info,
#                                 ALL_DEMANDS,
#                                 ...
#                             ) = start_link_downtime_job(
#                                 link_id=lid,
#                                 decisions=decs,
#                                 pending_downtime_actions=None,
#                                 ...
#                             )
#
#                         else:
#                             # Fiber-only jobs (band + CL)
#                             for dec in decs:
#                                 print(f">>> Starting FIBER job on link {lid}: {dec}")
#
#                                 log_upgrade_action(
#                                     upgrade_log_csv,
#                                     day=Current_global_time,
#                                     period_index=period_index,
#                                     scope="fiber_downtime",
#                                     link_id=lid,
#                                     upgrade_type=dec.get("upgrade_type", ""),
#                                     fiber_id=dec.get("fiber_id", None),
#                                     core_type=dec.get("core_type", None),
#                                 )
#
#                                 (
#                                     working_topology,
#                                     link_status_forward,
#                                     link_status_backward,
#                                     removed_links_info,
#                                     ALL_DEMANDS,
#                                     ...
#                                 ) = start_downtime_upgrade_for_link(
#                                     scope="fiber",
#                                     link_id=lid,
#                                     decision=dec,
#                                     ...
#                                 )
#
#                     if pending_downtime_jobs and not removed_links_info:
#                         job = pending_downtime_jobs.pop(0)
#                         scope = job[0]
#                         lid = job[1]
#
#                         if scope == "fiber":
#                             dec = job[2]
#                             print(f">>> Starting FIBER downtime job on link {lid}: {dec}")
#
#                             log_upgrade_action(
#                                 upgrade_log_csv,
#                                 day=Current_global_time,
#                                 period_index=period_index,
#                                 scope="fiber_downtime",
#                                 link_id=lid,
#                                 upgrade_type=dec.get("upgrade_type", ""),
#                                 fiber_id=dec.get("fiber_id", None),
#                                 core_type=dec.get("core_type", None),
#                             )
#
#                             (
#                                 working_topology,
#                                 link_status_forward,
#                                 link_status_backward,
#                                 removed_links_info,
#                                 ALL_DEMANDS,
#                                 blocked_datarate_rerouting,
#                                 accepted_datarate_rerouting,
#                                 arrived_datarate_rerouting,
#                                 total_network_cost,
#                                 PATHS,
#                             ) = start_downtime_upgrade_for_link(
#                                 scope="fiber",
#                                 link_id=lid,
#                                 decision=dec,
#                                 pending_downtime_actions=None,  # ok if your function ignores it
#                                 ALL_DEMANDS=ALL_DEMANDS,
#                                 link_status_forward=link_status_forward,
#                                 link_status_backward=link_status_backward,
#                                 working_topology=working_topology,
#                                 removed_links_info=removed_links_info,
#                                 current_time=Current_global_time,
#                                 blocked_datarate_rerouting=0,
#                                 accepted_datarate_rerouting=0,
#                                 arrived_datarate_rerouting=0,
#                                 total_network_cost=total_network_cost,
#                                 upgrade_cost_timeline=upgrade_cost_timeline,
#                                 algorithm=algorithm
#                             )
#
#                         elif scope == "link":
#                             job_actions = job[2]  # list of decisions: core first, then band, then new fibers
#                             print(f">>> Starting LINK downtime job on link {lid}: {job_actions}")
#
#                             for dec in job_actions:
#                                 log_upgrade_action(
#                                     upgrade_log_csv,
#                                     day=Current_global_time,
#                                     period_index=period_index,
#                                     scope="link_downtime",
#                                     link_id=lid,
#                                     upgrade_type=dec.get("upgrade_type", ""),
#                                     fiber_id=dec.get("fiber_id", None),
#                                     core_type=dec.get("core_type", None),
#                                 )
#
#                             (
#                                 working_topology,
#                                 link_status_forward,
#                                 link_status_backward,
#                                 removed_links_info,
#                                 ALL_DEMANDS,
#                                 blocked_datarate_rerouting,
#                                 accepted_datarate_rerouting,
#                                 arrived_datarate_rerouting,
#                                 total_network_cost,
#                                 PATHS,
#                             ) = start_link_downtime_job(
#                                 link_id=lid,
#                                 decisions=job_actions,
#                                 pending_downtime_actions=pending_downtime_jobs,
#                                 ALL_DEMANDS=ALL_DEMANDS,
#                                 link_status_forward=link_status_forward,
#                                 link_status_backward=link_status_backward,
#                                 working_topology=working_topology,
#                                 removed_links_info=removed_links_info,
#                                 current_time=Current_global_time,
#                                 blocked_datarate_rerouting=0,
#                                 accepted_datarate_rerouting=0,
#                                 arrived_datarate_rerouting=0,
#                                 total_network_cost=total_network_cost,
#                                 upgrade_cost_timeline=upgrade_cost_timeline,
#                                 algorithm=algorithm
#                             )
#                     # >>> CHANGE 5 END
#
#
#         # ====================================================================
#         # Restore Downtime Links (when downtime expires) + start next job
#         # ====================================================================
#
#         restore_now = [
#             key
#             for key, info in removed_links_info.items()
#             if info.get("restore_time", math.inf) <= Current_global_time
#         ]
#
#         if restore_now:
#             print(f">>> Restoring downtime items at day {Current_global_time}")
#
#             for key in restore_now:
#                 info = removed_links_info[key]
#                 working_topology, link_status_forward, link_status_backward, removed_links_info = \
#                     restore_removed_links_after_downtime(
#                         key=key,
#                         info=info,
#                         current_time=Current_global_time,
#                         link_status_forward=link_status_forward,
#                         link_status_backward=link_status_backward,
#                         current_topology=working_topology,
#                         removed_links_info=removed_links_info
#                     )
#
#             PATHS = build_k_shortest_paths(working_topology)
#             if not PATHS:
#                 raise ValueError("PATHS empty after restoring downtime items.")
#
#             # >>> CHANGE 6 START: Start next queued JOB after restoration (job-based)
#             if pending_downtime_jobs and not removed_links_info:
#                 job = pending_downtime_jobs.pop(0)
#                 scope = job[0]
#                 lid = job[1]
#
#                 if scope == "fiber":
#                     dec = job[2]
#                     print(f">>> Starting NEXT FIBER downtime job on link {lid}: {dec}")
#
#                     log_upgrade_action(
#                         upgrade_log_csv,
#                         day=Current_global_time,
#                         period_index=period_index,
#                         scope="fiber_downtime",
#                         link_id=lid,
#                         upgrade_type=dec.get("upgrade_type", ""),
#                         fiber_id=dec.get("fiber_id", None),
#                         core_type=dec.get("core_type", None),
#                     )
#
#                     (
#                         working_topology,
#                         link_status_forward,
#                         link_status_backward,
#                         removed_links_info,
#                         ALL_DEMANDS,
#                         blocked_datarate_rerouting,
#                         accepted_datarate_rerouting,
#                         arrived_datarate_rerouting,
#                         total_network_cost,
#                         PATHS,
#                     ) = start_downtime_upgrade_for_link(
#                         scope="fiber",
#                         link_id=lid,
#                         decision=dec,
#                         pending_downtime_actions=None,
#                         ALL_DEMANDS=ALL_DEMANDS,
#                         link_status_forward=link_status_forward,
#                         link_status_backward=link_status_backward,
#                         working_topology=working_topology,
#                         removed_links_info=removed_links_info,
#                         current_time=Current_global_time,
#                         blocked_datarate_rerouting=0,
#                         accepted_datarate_rerouting=0,
#                         arrived_datarate_rerouting=0,
#                         total_network_cost=total_network_cost,
#                         upgrade_cost_timeline=upgrade_cost_timeline,
#                         algorithm=algorithm
#                     )
#
#                 elif scope == "link":
#                     job_actions = job[2]
#                     print(f">>> Starting NEXT LINK downtime job on link {lid}: {job_actions}")
#
#                     for dec in job_actions:
#                         log_upgrade_action(
#                             upgrade_log_csv,
#                             day=Current_global_time,
#                             period_index=period_index,
#                             scope="link_downtime",
#                             link_id=lid,
#                             upgrade_type=dec.get("upgrade_type", ""),
#                             fiber_id=dec.get("fiber_id", None),
#                             core_type=dec.get("core_type", None),
#                         )
#
#                     (
#                         working_topology,
#                         link_status_forward,
#                         link_status_backward,
#                         removed_links_info,
#                         ALL_DEMANDS,
#                         blocked_datarate_rerouting,
#                         accepted_datarate_rerouting,
#                         arrived_datarate_rerouting,
#                         total_network_cost,
#                         PATHS,
#                     ) = start_link_downtime_job(
#                         link_id=lid,
#                         decisions=job_actions,
#                         pending_downtime_actions=pending_downtime_jobs,
#                         ALL_DEMANDS=ALL_DEMANDS,
#                         link_status_forward=link_status_forward,
#                         link_status_backward=link_status_backward,
#                         working_topology=working_topology,
#                         removed_links_info=removed_links_info,
#                         current_time=Current_global_time,
#                         blocked_datarate_rerouting=0,
#                         accepted_datarate_rerouting=0,
#                         arrived_datarate_rerouting=0,
#                         total_network_cost=total_network_cost,
#                         upgrade_cost_timeline=upgrade_cost_timeline,
#                         algorithm=algorithm
#                     )
#             # >>> CHANGE 6 END
#
#
#         # ====================================================================
#         # Remove Expired Connections
#         # ====================================================================
#
#         index_to_remove = []
#         for idy, conn in enumerate(ALL_DEMANDS):
#             if conn.departure_time <= Current_global_time:
#                 index_to_remove.append(idy)
#
#         for idy in reversed(index_to_remove):
#             conn = ALL_DEMANDS[idy]
#
#             for i in range(len(conn.path) - 1):
#                 src_depart = conn.path[i]
#                 dest_depart = conn.path[i + 1]
#
#                 link = conn.link_ids[i]
#                 fiber = conn.fibers_used[i]
#                 core = conn.cores_used[i]
#
#                 if src_depart < dest_depart:
#                     if link in link_status_forward and fiber in link_status_forward[link]:
#                         fwd_obj = link_status_forward[link][fiber]
#                         if isinstance(fwd_obj, dict) and "slots" in fwd_obj:
#                             for s in range(conn.fs_index, conn.fs_index + conn.slice_window_size):
#                                 fwd_obj["slots"][s] = 0
#                         else:
#                             if core in fwd_obj:
#                                 for s in range(conn.fs_index, conn.fs_index + conn.slice_window_size):
#                                     fwd_obj[core]["slots"][s] = 0
#                 else:
#                     if link in link_status_backward and fiber in link_status_backward[link]:
#                         bwd_obj = link_status_backward[link][fiber]
#                         if isinstance(bwd_obj, dict) and "slots" in bwd_obj:
#                             for s in range(conn.fs_index, conn.fs_index + conn.slice_window_size):
#                                 bwd_obj["slots"][s] = 0
#                         else:
#                             if core in bwd_obj:
#                                 for s in range(conn.fs_index, conn.fs_index + conn.slice_window_size):
#                                     bwd_obj[core]["slots"][s] = 0
#
#             del ALL_DEMANDS[idy]
#
#         # ====================================================================
#         # Run RSA for New Connection
#         # ====================================================================
#         mf_index, fs_index, slice_window_size, path, fibers_used, cores_used, link_ids, attempted_paths_info = execute_first_fit(
#             src=src,
#             dest=dest,
#             datarate=datarate,
#             arrival_time=arrival_time,
#             departure_time=departure_time,
#             link_status_forward=link_status_forward,
#             link_status_backward=link_status_backward,
#             topology=working_topology,
#             PATHS=PATHS
#         )
#
#         accepted = (mf_index != float('inf') and fs_index != float('inf'))
#         status = "accepted" if accepted else "blocked"
#
#         connection_log_writer.writerow(
#             build_connection_request_row(
#                 connection_id=connection_id,
#                 src=src,
#                 dest=dest,
#                 datarate=datarate,
#                 arrival_time=arrival_time,
#                 holding_time=holding_time,
#                 departure_time=departure_time,
#                 status=status,
#                 mf_index=mf_index,
#                 fs_index=fs_index,
#                 slice_window_size=slice_window_size if accepted else None,
#                 path=path if accepted else None,
#                 link_ids=link_ids if accepted else None,
#                 fibers_used=fibers_used if accepted else None,
#                 cores_used=cores_used if accepted else None,
#             )
#         )
#         # ============================================================
#         # UPDATE PATH STATISTICS
#         # ============================================================
#
#         for info in (attempted_paths_info or []):
#
#             p_idx = int(info["path_index"])
#             status = info["status"]
#
#             # only count actual spectrum attempts
#             if status not in ("accepted", "blocked"):
#                 continue
#
#             sd_u = sd_key_undirected(src, dest)
#             key = (sd_u[0], sd_u[1], p_idx)
#
#             # ------------------------------------------------------------
#             # Current-cycle stats (used by link selection)
#             # ------------------------------------------------------------
#             if key not in path_stats:
#                 path_stats[key] = {
#                     "path_arrivals": 0,
#                     "path_accepted": 0,
#                     "path_blocked": 0
#                 }
#
#             path_stats[key]["path_arrivals"] += 1
#
#             if status == "accepted":
#                 path_stats[key]["path_accepted"] += 1
#             else:
#                 path_stats[key]["path_blocked"] += 1
#
#             # ------------------------------------------------------------
#             # Full-history stats (kept until end of simulation)
#             # ------------------------------------------------------------
#             if key not in path_stats_history:
#                 path_stats_history[key] = {
#                     "path_arrivals": 0,
#                     "path_accepted": 0,
#                     "path_blocked": 0
#                 }
#
#             path_stats_history[key]["path_arrivals"] += 1
#
#             if status == "accepted":
#                 path_stats_history[key]["path_accepted"] += 1
#             else:
#                 path_stats_history[key]["path_blocked"] += 1
#
#         if mf_index != float('inf') and fs_index != float('inf'):
#             ALL_DEMANDS.append(ConnectionData(
#                 path=path,
#                 link_ids=link_ids,
#                 fs_index=fs_index,
#                 slice_window_size=slice_window_size,
#                 mf_index=mf_index,
#                 arrival_time=arrival_time,
#                 holding_time=holding_time,
#                 departure_time=departure_time,
#                 datarate=datarate,
#                 fibers_used=fibers_used,
#                 cores_used=cores_used
#             ))
#
#         # ====================================================================
#         # Update Metrics (after warmup)
#         # ====================================================================
#
#         if connection_id > config['warmup_connections']:
#             sd_u = sd_key_undirected(src, dest)
#             metrics.sd_arrivals[sd_u] += 1
#
#             blocked = (mf_index == float('inf') or fs_index == float('inf'))
#             if blocked:
#                 metrics.sd_blocked[sd_u] += 1
#
#             metrics.record_connection(src, dest, datarate, blocked, Current_global_time)
#
#             if metrics.get_blocking_probability() >= config['blocking_threshold']:
#                 print(
#                     f"\nSimulation stopped: blocking threshold "
#                     f"({config['blocking_threshold']}) reached at day {Current_global_time:.1f}."
#                 )
#                 break
#
#     # ========================================================================
#     # Generate Results
#     # ========================================================================
#
#     print("\n" + "="*70)
#     print("SIMULATION COMPLETE")
#     print("="*70)
#
#     analysis.save_blocking_data(
#         metrics.time_points,
#         metrics.blocking_prob_points,
#         config['blocking_csv']
#     )
#
#     end_wall_time = time.time()
#     total_runtime_sec = end_wall_time - start_wall_time
#     account_running_opex_until(Current_global_time)
#
#     snapshot = {
#         "seed": seed,
#         "Current_global_time": Current_global_time,
#         "connection_id": connection_id,
#         "connection_count": metrics.connection_count,
#         "blocked_connection_count": metrics.blocked_connection_count,
#         "arrived_datarate": metrics.arrived_datarate,
#         "accepted_datarate": metrics.accepted_datarate,
#         "blocked_datarate": metrics.blocked_datarate,
#         "time_points": metrics.time_points,
#         "blocking_prob_points": metrics.blocking_prob_points,
#         "traffic_rates": traffic_rates,
#         "total_network_cost": total_network_cost,
#         "upgrade_cost_timeline": upgrade_cost_timeline,
#         "upgrade_decision_history": upgrade_decision_history,
#         "link_status_forward": link_status_forward,
#         "link_status_backward": link_status_backward,
#         "ALL_DEMANDS": ALL_DEMANDS,
#         "working_topology": working_topology,
#         "wall_clock_runtime_seconds": total_runtime_sec,
#         "budget_params": budget_params,
#         "path_stats_history": path_stats_history,
#     }
#     analysis.save_snapshot(snapshot, config['snapshot_file'])
#     save_path_stats_history_csv(path_stats_history, config['results_dir'])
#
#     analysis.print_simulation_summary(snapshot)
#     metrics.print_summary()
#     analysis.print_upgrade_timeline(upgrade_cost_timeline)
#
#     generate_plots(config, metrics, upgrade_cost_timeline, total_network_cost)
#
#     generate_budget_vs_cost_plot(
#         config=config,
#         upgrade_cost_timeline=upgrade_cost_timeline,
#         end_day=Current_global_time,
#         budget_params=budget_params
#     )
#
#     generate_blocking_vs_upgrade_plot(config, metrics, upgrade_decision_history)
#     connection_log_file.close()
#     print(f"\n✓ All results saved to: {config['results_dir']}")
#
#     end_wall_time = time.time()
#     total_runtime_sec = end_wall_time - start_wall_time
#     total_runtime_min = total_runtime_sec / 60.0
#     total_runtime_hr = total_runtime_min / 60.0
#
#     print("\n" + "-"*60)
#     print(f"Total wall-clock runtime:")
#     print(f"  {total_runtime_sec:.2f} seconds")
#     print(f"  {total_runtime_min:.2f} minutes")
#     print(f"  {total_runtime_hr:.2f} hours")
#     print("-"*60)
#
#     snapshot["wall_clock_runtime_seconds"] = total_runtime_sec
#
#
# # ============================================================================
# # Plotting Functions
# # ============================================================================
#
# def generate_plots(config, metrics, upgrade_cost_timeline, total_network_cost):
#     """Generate all simulation plots."""
#
#     print(f"\nGenerating plots and saving to {config['results_dir']}...")
#
#     plt.figure(figsize=(10, 6))
#     plt.plot(metrics.time_points, metrics.blocking_prob_points, label="Blocking Probability", linewidth=2)
#     plt.axhline(y=config['blocking_threshold'], linestyle="--", color='r', label="Threshold")
#     plt.xlabel("Time (days)", fontsize=12)
#     plt.ylabel("Blocking Probability", fontsize=12)
#     plt.yscale("log")
#     plt.title(f"Blocking Probability vs Time - {config['algorithm']} on {config['topology']}", fontsize=14)
#     plt.legend(fontsize=10)
#     plt.grid(True, which="both", alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(config['blocking_plot_file'], dpi=300, bbox_inches="tight")
#     plt.close()
#     print(f"✓ Saved blocking probability plot to {config['blocking_plot_file']}")
#
#     if upgrade_cost_timeline:
#         plt.figure(figsize=(12, 8))
#
#         # ---- Per-entry series (same length) ----
#         days_all = [e["day"] for e in upgrade_cost_timeline]
#         equipment_costs = [e.get("equipment", 0.0) for e in upgrade_cost_timeline]
#         workforce_costs = [e.get("workforce", 0.0) for e in upgrade_cost_timeline]
#         opex_costs = [e.get("opex", 0.0) for e in upgrade_cost_timeline]
#         total_costs = [e.get("total", 0.0) for e in upgrade_cost_timeline]
#
#         # ---- Cumulative series (use cumulative_total field) ----
#         days_cum = [e["day"] for e in upgrade_cost_timeline if "cumulative_total" in e]
#         cumulative_total = [e["cumulative_total"] for e in upgrade_cost_timeline if "cumulative_total" in e]
#
#         plt.subplot(2, 1, 1)
#         plt.plot(days_all, equipment_costs, 'o-', label='Equipment Cost', linewidth=2, markersize=4)
#         plt.plot(days_all, workforce_costs, 's-', label='Workforce Cost', linewidth=2, markersize=4)
#         plt.plot(days_all, opex_costs, '^-', label='OPEX Cost', linewidth=2, markersize=4)
#         plt.plot(days_all, total_costs, 'd-', label='Total Cost', linewidth=2, markersize=4, color='black')
#         plt.xlabel("Time (days)", fontsize=12)
#         plt.ylabel("Cost per Logged Point", fontsize=12)
#         plt.title(f"Cost Components vs Time - {config['algorithm']} on {config['topology']}", fontsize=14)
#         plt.legend(fontsize=10)
#         plt.grid(True, alpha=0.3)
#
#         plt.subplot(2, 1, 2)
#         plt.plot(days_cum, cumulative_total, 'o-', linewidth=2, markersize=3, color='darkblue')
#         plt.xlabel("Time (days)", fontsize=12)
#         plt.ylabel("Cumulative Total Cost", fontsize=12)
#         plt.title(f"Cumulative Cost vs Time - Total: ${total_network_cost:,.0f}", fontsize=14)
#         plt.grid(True, alpha=0.3)
#
#         plt.tight_layout()
#         plt.savefig(config['cost_plot_file'], dpi=300, bbox_inches="tight")
#         plt.close()
#         print(f"✓ Saved cost vs time plot to {config['cost_plot_file']}")
#     else:
#         print("✓ No upgrades performed, skipping cost plot")
#
#     plt.figure(figsize=(12, 10))
#
#     plt.subplot(2, 2, 1)
#     categories = ['Total', 'Accepted', 'Blocked']
#     counts = [metrics.connection_count,
#               metrics.connection_count - metrics.blocked_connection_count,
#               metrics.blocked_connection_count]
#     colors = ['blue', 'green', 'red']
#     plt.bar(categories, counts, color=colors, alpha=0.7)
#     plt.ylabel('Connection Count', fontsize=12)
#     plt.title('Connection Statistics', fontsize=12)
#     plt.grid(axis='y', alpha=0.3)
#
#     plt.subplot(2, 2, 2)
#     categories = ['Arrived', 'Accepted', 'Blocked']
#     datarates = [metrics.arrived_datarate, metrics.accepted_datarate, metrics.blocked_datarate]
#     colors = ['blue', 'green', 'red']
#     plt.bar(categories, datarates, color=colors, alpha=0.7)
#     plt.ylabel('Datarate (Gbps)', fontsize=12)
#     plt.title('Datarate Statistics', fontsize=12)
#     plt.grid(axis='y', alpha=0.3)
#
#     plt.subplot(2, 2, 3)
#     plt.plot(metrics.time_points, metrics.blocking_prob_points, linewidth=2, color='darkred')
#     plt.axhline(y=config['blocking_threshold'], linestyle="--", color='black', label="Threshold")
#     plt.xlabel("Time (days)", fontsize=12)
#     plt.ylabel("Blocking Probability", fontsize=12)
#     plt.title('Blocking Probability (Linear Scale)', fontsize=12)
#     plt.legend(fontsize=10)
#     plt.grid(True, alpha=0.3)
#
#     plt.subplot(2, 2, 4)
#     if metrics.arrived_datarate > 0:
#         throughput_efficiency = (metrics.accepted_datarate / metrics.arrived_datarate) * 100
#         blocking_rate = (metrics.blocked_datarate / metrics.arrived_datarate) * 100
#         categories = ['Accepted', 'Blocked']
#         percentages = [throughput_efficiency, blocking_rate]
#         colors = ['green', 'red']
#         plt.bar(categories, percentages, color=colors, alpha=0.7)
#         plt.ylabel('Percentage (%)', fontsize=12)
#         plt.title('Throughput Efficiency', fontsize=12)
#         plt.ylim(0, 100)
#         plt.grid(axis='y', alpha=0.3)
#
#     plt.suptitle(f"Performance Metrics - {config['algorithm']} on {config['topology']}",
#                  fontsize=14, y=0.995)
#     plt.tight_layout()
#     plt.savefig(config['metrics_plot_file'], dpi=300, bbox_inches="tight")
#     plt.close()
#     print(f"✓ Saved performance metrics plot to {config['metrics_plot_file']}")
#
# def build_budget_components_curve(
#         end_day: float,
#         period_days: int,
#         period_opex_budget_0: float,
#         period_capex_budget_0: float,
#         period_total_budget_0: float,
#         inflation_rate: float,
# ):
#     """
#     Build cumulative budget curves for the new fixed-period budget model.
#
#     Each period contributes:
#       - a fixed OPEX budget for that period
#       - a fixed CAPEX+workforce budget for that period
#       - total = OPEX + CAPEX/WF
#
#     Each of these grows by inflation every period.
#     """
#     max_day = int(math.ceil(end_day))
#     days = np.arange(0, max_day + 1, dtype=float)
#
#     cum_opex_budget = np.zeros(max_day + 1, dtype=float)
#     cum_upgrade_budget = np.zeros(max_day + 1, dtype=float)
#     cum_total_budget = np.zeros(max_day + 1, dtype=float)
#
#     opex_cum = 0.0
#     upgrade_cum = 0.0
#     total_cum = 0.0
#
#     current_period_idx = -1
#
#     for day in range(1, max_day + 1):
#         period_idx = (day - 1) // period_days
#
#         # Add the new period budget only once, at the start of each period
#         if period_idx != current_period_idx:
#             current_period_idx = period_idx
#             growth = (1 + inflation_rate) ** period_idx
#
#             period_opex_budget = period_opex_budget_0 * growth
#             period_upgrade_budget = period_capex_budget_0 * growth
#             period_total_budget = period_total_budget_0 * growth
#
#             opex_cum += period_opex_budget
#             upgrade_cum += period_upgrade_budget
#             total_cum += period_total_budget
#
#         cum_opex_budget[day] = opex_cum
#         cum_upgrade_budget[day] = upgrade_cum
#         cum_total_budget[day] = total_cum
#
#     return days, cum_opex_budget, cum_upgrade_budget, cum_total_budget
#
# def build_actual_cumulative_cost_components(upgrade_cost_timeline):
#     """
#     Build cumulative actual OPEX, cumulative actual CAPEX+workforce,
#     and cumulative actual total cost from upgrade_cost_timeline.
#     """
#     if not upgrade_cost_timeline:
#         return [], [], [], []
#
#     timeline_sorted = sorted(upgrade_cost_timeline, key=lambda x: x["day"])
#
#     days = []
#     cum_opex = []
#     cum_upgrade = []
#     cum_total = []
#
#     running_opex = 0.0
#     running_upgrade = 0.0
#
#     for row in timeline_sorted:
#         running_opex += float(row.get("opex", 0.0))
#         running_upgrade += float(row.get("equipment", 0.0)) + float(row.get("workforce", 0.0))
#
#         days.append(float(row["day"]))
#         cum_opex.append(running_opex)
#         cum_upgrade.append(running_upgrade)
#         cum_total.append(running_opex + running_upgrade)
#
#     return days, cum_opex, cum_upgrade, cum_total
#
#
# def generate_budget_vs_cost_plot(config, upgrade_cost_timeline, end_day: float, budget_params: dict):
#     """
#     Plot:
#       1) cumulative actual total cost vs cumulative total budget
#       2) cumulative actual OPEX vs OPEX budget
#       3) cumulative actual CAPEX+workforce vs CAPEX+workforce budget
#     """
#     if not config.get("budget_enabled", False):
#         print("✓ Budget plot disabled")
#         return
#
#     if not upgrade_cost_timeline:
#         print("✓ No cost timeline available, skipping budget plot")
#         return
#
#     actual_days, actual_cum_opex, actual_cum_upgrade, actual_cum_total = \
#         build_actual_cumulative_cost_components(upgrade_cost_timeline)
#
#     if not actual_days:
#         print("✓ No cumulative cost data available, skipping budget plot")
#         return
#
#     budget_days, budget_cum_opex, budget_cum_upgrade, budget_cum_total = \
#         build_budget_components_curve(
#             end_day=end_day,
#             period_days=budget_params["period_days"],
#             period_opex_budget_0=budget_params["period_opex_budget_0"],
#             period_capex_budget_0=budget_params["period_capex_budget_0"],
#             period_total_budget_0=budget_params["period_total_budget_0"],
#             inflation_rate=budget_params["inflation_rate"],
#         )
#
#     interp_actual_total = np.interp(
#         budget_days,
#         np.array(actual_days, dtype=float),
#         np.array(actual_cum_total, dtype=float),
#         left=0.0,
#         right=float(actual_cum_total[-1]),
#     )
#
#     over_budget_mask = interp_actual_total > budget_cum_total
#
#     plt.figure(figsize=(12, 10))
#
#     # 1) Total cost vs total budget
#     plt.subplot(3, 1, 1)
#     plt.plot(actual_days, actual_cum_total, linewidth=2, label="Actual cumulative total cost")
#     plt.plot(budget_days, budget_cum_total, linestyle="--", linewidth=2, label="Cumulative total budget")
#
#     if np.any(over_budget_mask):
#         plt.fill_between(
#             budget_days,
#             budget_cum_total,
#             interp_actual_total,
#             where=over_budget_mask,
#             alpha=0.25,
#             label="Over budget"
#         )
#
#     for d in range(budget_params["period_days"], int(math.ceil(end_day)) + 1, budget_params["period_days"]):
#         plt.axvline(d, linestyle=":", alpha=0.3)
#
#     plt.xlabel("Time (days)", fontsize=11)
#     plt.ylabel("Cumulative Cost", fontsize=11)
#     plt.title(f"Actual Cumulative Cost vs Budget - {config['algorithm']} on {config['topology']}", fontsize=13)
#     plt.grid(True, alpha=0.3)
#     plt.legend(fontsize=9)
#
#     # 2) OPEX vs OPEX budget
#     plt.subplot(3, 1, 2)
#     plt.plot(actual_days, actual_cum_opex, linewidth=2, label="Actual cumulative OPEX")
#     plt.plot(budget_days, budget_cum_opex, linestyle="--", linewidth=2, label="Cumulative OPEX budget")
#
#     for d in range(budget_params["period_days"], int(math.ceil(end_day)) + 1, budget_params["period_days"]):
#         plt.axvline(d, linestyle=":", alpha=0.3)
#
#     plt.xlabel("Time (days)", fontsize=11)
#     plt.ylabel("Cumulative OPEX", fontsize=11)
#     plt.title("OPEX Cost vs OPEX Budget", fontsize=12)
#     plt.grid(True, alpha=0.3)
#     plt.legend(fontsize=9)
#
#     # 3) CAPEX+workforce vs budget
#     plt.subplot(3, 1, 3)
#     plt.plot(actual_days, actual_cum_upgrade, linewidth=2, label="Actual cumulative CAPEX + workforce")
#     plt.plot(budget_days, budget_cum_upgrade, linestyle="--", linewidth=2, label="Cumulative CAPEX + workforce budget")
#
#     for d in range(budget_params["period_days"], int(math.ceil(end_day)) + 1, budget_params["period_days"]):
#         plt.axvline(d, linestyle=":", alpha=0.3)
#
#     plt.xlabel("Time (days)", fontsize=11)
#     plt.ylabel("Cumulative Upgrade Cost", fontsize=11)
#     plt.title("CAPEX + Workforce Cost vs Budget", fontsize=12)
#     plt.grid(True, alpha=0.3)
#     plt.legend(fontsize=9)
#
#     plt.tight_layout()
#     plt.savefig(config["budget_plot_file"], dpi=300, bbox_inches="tight")
#     plt.close()
#
#     print(f"✓ Saved cost vs budget plot to {config['budget_plot_file']}")
#
# def generate_blocking_vs_upgrade_plot(config, metrics, upgrade_decision_history):
#     """
#     Plot blocking probability and cycle-wise upgrade decision level together.
#     Left axis  -> blocking probability
#     Right axis -> cycle upgrade level
#     """
#
#     if not metrics.time_points:
#         print("✓ No blocking data available, skipping blocking vs upgrade plot")
#         return
#
#     plt.figure(figsize=(12, 6))
#     ax1 = plt.gca()
#
#     # Blocking probability line
#     ax1.plot(
#         metrics.time_points,
#         metrics.blocking_prob_points,
#         linewidth=2,
#         label="Blocking Probability"
#     )
#     ax1.axhline(
#         y=config['blocking_threshold'],
#         linestyle="--",
#         color="r",
#         label="Threshold"
#     )
#     ax1.set_xlabel("Time (days)", fontsize=12)
#     ax1.set_ylabel("Blocking Probability", fontsize=12)
#     ax1.grid(True, alpha=0.3)
#
#     # Upgrade level points on second axis
#     ax2 = ax1.twinx()
#
#     upgrade_colors = {
#         "band_upgrade": "blue",
#         "new_fiber_C": "green",
#         "new_fiber_CL": "orange",
#         "core_upgrade": "red"
#     }
#
#     upgrade_levels = {
#         "band_upgrade": 1,
#         "new_fiber_C": 2,
#         "new_fiber_CL": 3,
#         "core_upgrade": 4
#     }
#
#     if not upgrade_decision_history:
#         print("✓ No upgrade decisions recorded, skipping upgrade markers")
#     else:
#         for entry in upgrade_decision_history:
#             day = entry["time_day"]
#             decisions = entry.get("final_decisions", [])
#
#             for dec in decisions:
#                 upg_type = dec.get("upgrade_type")
#
#                 if upg_type not in upgrade_levels:
#                     continue
#
#                 ax2.scatter(
#                     day,
#                     upgrade_levels[upg_type],
#                     color=upgrade_colors[upg_type],
#                     s=80
#                 )
#     import matplotlib.patches as mpatches
#
#     legend_patches = [
#         mpatches.Patch(color="blue", label="Band Upgrade"),
#         mpatches.Patch(color="green", label="New Fiber C"),
#         mpatches.Patch(color="orange", label="New Fiber C+L"),
#         mpatches.Patch(color="red", label="Core Upgrade"),
#     ]
#
#     ax2.legend(handles=legend_patches, loc="upper right")
#
#     ax2.set_ylabel("Upgrade Decision Level", fontsize=12)
#     ax2.set_yticks([1, 2, 3, 4])
#     ax2.set_yticklabels([
#         "Band",
#         "New Fiber C",
#         "New Fiber C+L",
#         "Core"
#     ])
#
#     plt.title(
#         f"Blocking Probability and Upgrade Decisions - {config['algorithm']} on {config['topology']}",
#         fontsize=14
#     )
#     plt.tight_layout()
#
#     out_file = config['results_dir'] / 'blocking_vs_upgrade_decisions.png'
#     plt.savefig(out_file, dpi=300, bbox_inches="tight")
#     plt.close()
#
#     print(f"✓ Saved blocking vs upgrade decision plot to {out_file}")
#
# def save_path_stats_history_csv(path_stats_history: dict, results_dir: Path):
#     """
#     Save full-history path statistics to CSV.
#
#     Columns:
#       src, dest, path_index, arrivals, accepted, blocked, blocking_probability
#     """
#     out_file = results_dir / "path_stats_history.csv"
#
#     with open(out_file, "w", newline="", encoding="utf-8") as f:
#         writer = csv.writer(f)
#         writer.writerow([
#             "src", "dest", "path_index",
#             "path_arrivals", "path_accepted", "path_blocked",
#             "path_blocking_probability"
#         ])
#
#         for key in sorted(path_stats_history.keys()):
#             src, dest, path_idx = key
#             row = path_stats_history[key]
#
#             arrivals = row.get("path_arrivals", 0)
#             accepted = row.get("path_accepted", 0)
#             blocked = row.get("path_blocked", 0)
#             bp = (blocked / arrivals) if arrivals > 0 else 0.0
#
#             writer.writerow([
#                 src, dest, path_idx,
#                 arrivals, accepted, blocked,
#                 bp
#             ])
#
#     print(f"✓ Saved full path statistics to {out_file}")
#
#
# # ============================================================================
# # Entry Point
# # ============================================================================
#
# if __name__ == "__main__":
#     results_dir = create_results_directory(CONFIG['topology'], CONFIG['algorithm'])
#     log_file_path = results_dir / 'simulation_log.txt'
#     log_file = open(log_file_path, "w", encoding="utf-8", buffering=1)
#     sys.stdout = TeeOutput(sys.__stdout__, log_file)
#
#     try:
#         run_simulation(CONFIG)
#     finally:
#         try:
#             sys.stdout = sys.__stdout__
#         except Exception:
#             pass
#         try:
#             log_file.close()
#         except Exception:
#             pass
