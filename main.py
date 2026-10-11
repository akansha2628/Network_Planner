"""
Optical Network Simulator - Main Entry Point

A modular optical network simulator with network upgrade planning capabilities.
Configure simulation parameters in the CONFIG dictionary below.
"""

# ##############################################################################
# ##############################################################################
# ###                                                                        ###
# ###   ML-INTEGRATION CHANGE LOG — main.py (changes made by Claude)          ###
# ###                                                                        ###
# ###   Every change below is marked in the code with a box that starts      ###
# ###       ### ML-INTEGRATION CHANGE M<number> (Claude) ###                 ###
# ###   Search for "ML-INTEGRATION CHANGE" to jump between them. Anything    ###
# ###   WITHOUT such a box is your/Akansha's original code, unchanged.       ###
# ###   The commented-out legacy simulator at the bottom is untouched.       ###
# ###   plan_checker.py was NOT changed.                                     ###
# ###                                                                        ###
# ##############################################################################
# ##############################################################################
#
#  M1  Removed the top-level `import xgboost as xgb`. xgboost/sb3_contrib are now
#      imported only inside the ML loaders, so heuristic runs don't need them.
#  M2  Feature extraction = single source of truth (top of file):
#      - classify_fiber_state() moved here from inside run_single_cycle()
#      - new build_feature_row(): RAW features (what XGBoost was trained on)
#      - all flatten_*() helpers take an optional fixed link_ids list so the
#        vector is always 313 long, even while a link is down
#      - extract_313_features_standalone(): classifier arg now optional,
#        traffic-slice size computed from N instead of hard-coded 133
#  M3  CONFIG: added 'seed' (single-seed mode crashed without it) and the ML keys:
#      ml_mode ('off'|'train'|'deploy'), ml_base_algorithm, ml_fallback_algorithm,
#      ml_gatekeeper_path, ml_drl_policy_path, ml_gatekeeper_threshold,
#      ml_train_gatekeeper. ml_mode='off' = original heuristic behaviour.
#  M4  NEW block of ML helper functions (gatekeeper/DRL loading, gatekeeper
#      prediction, candidate links, O(1) action mask, DRL observation,
#      apply_drl_decision with the incremental fallback, commit_upgrade_plan,
#      run_heuristic_upgrade, ...).
#      commit_upgrade_plan() and run_heuristic_upgrade() are Akansha's original
#      code moved out of run_single_cycle() unchanged, so heuristics and ML share
#      exactly the same plan_checker/commit/physics path.
#  M5  initialize_simulation(): in ML modes loads ml_base_algorithm and writes
#      results to results/ml_<mode>_<topology>.
#  M6  initialize_simulation_state(): loads the gatekeeper, fixes the link order,
#      adds DRL bookkeeping keys to the state dict.
#  M7  run_single_cycle(): guard so it can't advance while a DRL decision is pending.
#  M8  run_single_cycle(): removed the nested classify_fiber_state /
#      log_pre_upgrade_per_link_state (now top level); added _sync_to_state /
#      _sync_from_state helpers.
#  M9  run_single_cycle() upgrade cycle:
#      - ml_mode='off': unchanged logic (now via run_heuristic_upgrade +
#        commit_upgrade_plan)
#      - ML modes: XGBoost gatekeeper decides IF; main.py's own link selection
#        gives the candidate links; a DRL decision point is created.
#  M10 BUG FIX: run_single_cycle() now writes ALL reassigned locals back into
#      state (the downtime job queue, per-link jobs, PATHS after restore,
#      topology, link status were being lost). Stop rule only checked after
#      warmup, as in the original single-loop simulator.
#  M11 run_simulation(): BUG FIX uses the extended config from state (was
#      KeyError 'blocking_csv' at the end of every run); deploy-mode loop
#      (gatekeeper -> DRL -> plan_checker -> fallback to incremental on FAIL).
#  M12 run_simulation(): saves drl_decisions.csv, closes files, returns state.
#  M13 __main__: log/results folder uses get_run_label() (ml_deploy etc.).
#  M14 MULTI-LINK DRL SELECTION: each cycle the DRL keeps picking candidate links
#      (one pick at a time) until no remaining candidate fits the remaining
#      CAPEX/TOTAL budget or the sequential-downtime limit; then the WHOLE set goes
#      to plan_checker once. (STOP rules were later changed by M16 - budget
#      saving within the year.) DRL observation = the
#      313 features + per-link "already picked" flags + remaining CAPEX/downtime.
#  M15 DRL CHOOSES THE UPGRADE LEVEL: each action is (link, level). Levels can
#      be Akansha's per-link rules (CONFIG['ml_upgrade_levels'] = a list such as
#      ['increment', 'greedy', 'jump_to_max']; levels that give the same upgrade
#      as a lower level are masked out). Default is now 'any_higher' (M17).
#      The per-link observation value encodes the level picked.
#  M16 BUDGET SAVING: the DRL may stop early or skip a cycle to keep money for a
#      later cycle of the same budget year (no incremental fallback overrides a
#      deliberate save). In the year's LAST cycle it must spend what fits.
#      Observation gains a "last cycle of the year" flag (336 values on dt12).
#      Optional: CONFIG['ml_year_end_override_gatekeeper'] lets the year's last
#      cycle reach the DRL even if the gatekeeper says no.
#  M17 ANY HIGHER UPGRADE: default upgrade choice is now 'any_higher': per link
#      the DRL picks any target technology state with more capacity than now
#      (18 targets; several upgrade steps on one link in one cycle allowed;
#      361 actions on dt12). Year-end gatekeeper override is ON by default, and
#      in that cycle the DRL decides whether to spend (it is not forced).
#  M18 Model files are also looked up in sim/ml/ (where gatekeeper.py and
#      DRL_code.py live in the repository and save their models).
#  M19 Optional CONFIG['ml_run_tag'] gives each parallel simulator (DRL training
#      with --n-envs, or --evaluate) its own results folder.
#  M20 BUG FIX: DRL input scaling. Every feature is now divided by its real
#      maximum (links with more than 1,000 slots used to all look the same, and
#      fiber counts were nearly invisible); traffic is on a log scale. XGBoost
#      input is unchanged (raw features).
#
#  Decision flow in ML modes:
#    XGBoost: upgrade needed? --no--> nothing this cycle
#        | yes
#    main.py link selection (select_links_for_upgrade + network_connectivity,
#    path ranks 0-2) -> candidate links -> DRL picks (link, upgrade level) one by
#    one until nothing else fits budget/downtime -> whole set goes to
#        -> plan_checker (budget, sequential downtime, projected BBP)
#             PASS -> commit
#             FAIL -> nothing spent, DRL penalised (training), cycle falls back
#                     to the incremental heuristic
# ##############################################################################


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

##############################################################################
# ### ML-INTEGRATION CHANGE M1 (Claude) ###
# REMOVED here: `import xgboost as xgb` (was line 23 of your file).
# Now imported lazily in load_gatekeeper()/load_drl_policy() (block M4).
##############################################################################
# NOTE: xgboost / sb3_contrib are imported lazily inside the ML helpers below,
# so heuristic runs (ml_mode = "off") do not require the ML packages.


##############################################################################
# ### ML-INTEGRATION CHANGE M2 (Claude) ###
# This whole feature-extraction section replaces your previous helper block.
# - classify_fiber_state(): MOVED here unchanged from inside run_single_cycle()
#   (DRL_code.py's old fallback version returned 'SC_C' for every fiber).
# - _iter_links() + optional link_ids param on every flatten_*(): fixed
#   link order/length, missing (downed) links are filled with zeros.
# - build_feature_row(): NEW, raw features for the XGBoost gatekeeper.
# - extract_313_features_standalone(): classify fn now optional; traffic
#   slice uses N*(N-1) instead of a hard-coded 1:133. Same normalization.
##############################################################################
# ============================================================================
# ML FEATURE EXTRACTION — SINGLE SOURCE OF TRUTH
# ----------------------------------------------------------------------------
# Every consumer (XGBoost data logging, XGBoost gatekeeper at deploy time,
# DRL training env, DRL deploy policy) builds features through these
# top-level functions so the feature vector is identical everywhere.
#
#   build_feature_row(...)              -> raw feature dict (XGBoost input)
#   extract_313_features_standalone(...) -> (raw dict, normalized vector) (DRL input)
#
# Feature layout (dt12: N=12 nodes, 20 links -> 1 + 132 + 4*20 + 5*20 = 313):
#   current_BBP
#   lambda_i_j                   for every ordered node pair i != j
#   link_<id>_{SC_C,SC_CL,MC_C,MC_CL}
#   link_<id>_{occupied_slots,total_slots,free_slots,slot_utilization,max_contiguous_free_slots}
# ============================================================================

def classify_fiber_state(fiber_obj):
    """
    Classify one fiber object as SC_C / SC_CL / MC_C / MC_CL (or None if dark/unknown).
    Moved to top level (was nested inside run_single_cycle) so DRL_code.py and the
    XGBoost logger can import the exact same classifier.
    """
    if not isinstance(fiber_obj, dict):
        return None

    if "lit" in fiber_obj and "bands" in fiber_obj:
        if not fiber_obj.get("lit", False):
            return None
        bands = set(fiber_obj.get("bands", []))
        if bands == {"C"}:
            return "SC_C"
        elif bands == {"C", "L"}:
            return "SC_CL"
        return None

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
        return "MC_CL" if "L" in all_bands else "MC_C"
    return None


def _iter_links(link_status_forward, link_ids=None):
    """
    Yield (link_id, fibers) in a FIXED order.
    If link_ids is given, every id is yielded even if the link is temporarily
    missing from link_status_forward (e.g. during downtime) -> fixed-length vector.
    """
    if link_ids is None:
        for link_id, fibers in sorted(link_status_forward.items()):
            yield link_id, fibers
    else:
        for link_id in link_ids:
            yield link_id, link_status_forward.get(link_id, {})


def flatten_traffic_matrix(traffic_rates):
    row = {}
    n = traffic_rates.shape[0]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            row[f"lambda_{i}_{j}"] = float(traffic_rates[i][j])
    return row


def flatten_link_technology_state(link_status_forward, classify_fiber_state_fn=None, link_ids=None):
    if classify_fiber_state_fn is None:
        classify_fiber_state_fn = classify_fiber_state
    row = {}
    for link_id, fibers in _iter_links(link_status_forward, link_ids):
        sc_c = sc_cl = mc_c = mc_cl = 0
        for _, fiber_obj in fibers.items():
            state = classify_fiber_state_fn(fiber_obj)
            if state == "SC_C": sc_c += 1
            elif state == "SC_CL": sc_cl += 1
            elif state == "MC_C": mc_c += 3
            elif state == "MC_CL": mc_cl += 3

        row[f"link_{link_id}_SC_C"] = sc_c
        row[f"link_{link_id}_SC_CL"] = sc_cl
        row[f"link_{link_id}_MC_C"] = mc_c
        row[f"link_{link_id}_MC_CL"] = mc_cl
    return row


def max_contiguous_free_slots(slots):
    max_run = current_run = 0
    for value in slots:
        if value == 0:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    return max_run


def summarize_slot_array(slots):
    if slots is None:
        return {"occupied_slots": 0, "total_slots": 0, "free_slots": 0, "max_contiguous_free_slots": 0}
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


def flatten_link_slot_occupancy_state(link_status_forward, link_ids=None):
    row = {}
    for link_id, fibers in _iter_links(link_status_forward, link_ids):
        occupied_slots = total_slots = free_slots = max_free_block = 0
        for _, fiber_obj in fibers.items():
            if not isinstance(fiber_obj, dict):
                continue
            if "lit" in fiber_obj and "slots" in fiber_obj:
                if not fiber_obj.get("lit", False):
                    continue
                summary = summarize_slot_array(fiber_obj.get("slots", []))
                occupied_slots += summary["occupied_slots"]
                total_slots += summary["total_slots"]
                free_slots += summary["free_slots"]
                max_free_block = max(max_free_block, summary["max_contiguous_free_slots"])
            else:
                core_keys = [k for k in fiber_obj.keys() if isinstance(k, int)]
                for core_id in core_keys:
                    core_obj = fiber_obj.get(core_id, {})
                    if not isinstance(core_obj, dict) or not core_obj.get("lit", False):
                        continue
                    summary = summarize_slot_array(core_obj.get("slots", []))
                    occupied_slots += summary["occupied_slots"]
                    total_slots += summary["total_slots"]
                    free_slots += summary["free_slots"]
                    max_free_block = max(max_free_block, summary["max_contiguous_free_slots"])

        slot_utilization = (occupied_slots / total_slots) if total_slots > 0 else 0.0
        row[f"link_{link_id}_occupied_slots"] = occupied_slots
        row[f"link_{link_id}_total_slots"] = total_slots
        row[f"link_{link_id}_free_slots"] = free_slots
        row[f"link_{link_id}_slot_utilization"] = slot_utilization
        row[f"link_{link_id}_max_contiguous_free_slots"] = max_free_block
    return row


def build_feature_row(current_bbp, traffic_rates, link_status_forward,
                      classify_fiber_state_fn=None, link_ids=None):
    """Raw (un-normalized) feature dict. This is what XGBoost was trained on."""
    row = {"current_BBP": float(current_bbp)}
    row.update(flatten_traffic_matrix(traffic_rates))
    row.update(flatten_link_technology_state(link_status_forward, classify_fiber_state_fn, link_ids))
    row.update(flatten_link_slot_occupancy_state(link_status_forward, link_ids))
    return row


def extract_313_features_standalone(current_bbp, traffic_rates, link_status_forward,
                                    classify_fiber_state_fn=None, link_ids=None):
    """
    Returns (raw_row_dict, normalized_vector) for the DRL agent.
    The normalized vector is ONLY for the DRL policy; XGBoost must get the raw row.
    """
    row = build_feature_row(current_bbp, traffic_rates, link_status_forward,
                            classify_fiber_state_fn, link_ids)

    ##############################################################################
    # ### ML-INTEGRATION CHANGE M20 (Claude) ###
    # BUG FIX (DRL input scaling). Was: every lambda and every per-link value
    # divided by 1000 and clipped at 1.0. A fully upgraded link has 8,640 slots,
    # so all links with more than 1,000 slots looked identical to the DRL (29% of
    # the total_slots values in combined_dataset.csv), and the fiber counts
    # (at most 3 or 9) became ~0.001-0.009, i.e. almost invisible.
    # Now each feature is divided by its real maximum (from sim/core/constants),
    # and traffic, which grows ~200x over a lifetime, is on a log scale.
    # Only the DRL vector changes; XGBoost still gets the raw row.
    ##############################################################################
    raw_vector = np.array(list(row.values()), dtype=np.float32)
    max_slots = float(FIBERS * Num_cores * TOTAL_SLOTS)        # 8,640: all fibers multi-core C+L
    per_link_max = {
        "_SC_C": float(FIBERS), "_SC_CL": float(FIBERS),
        "_MC_C": float(FIBERS * Num_cores), "_MC_CL": float(FIBERS * Num_cores),
        "_occupied_slots": max_slots, "_total_slots": max_slots, "_free_slots": max_slots,
        "_slot_utilization": 1.0, "_max_contiguous_free_slots": float(TOTAL_SLOTS),
    }
    normalized_vector = np.zeros_like(raw_vector, dtype=np.float32)
    for i, (name, value) in enumerate(row.items()):
        value = float(value)
        if name == "current_BBP":
            normalized_vector[i] = value / DRL_BBP_SCALE
        elif name.startswith("lambda_"):
            normalized_vector[i] = ((math.log1p(max(value, 0.0)) - math.log1p(DRL_LAMBDA_LOG_MIN))
                                    / (math.log1p(DRL_LAMBDA_LOG_MAX) - math.log1p(DRL_LAMBDA_LOG_MIN)))
        else:
            # longest suffix first: "_max_contiguous_free_slots" also ends in "_free_slots"
            suffix = max((sfx for sfx in per_link_max if name.endswith(sfx)), key=len)
            normalized_vector[i] = value / per_link_max[suffix]
    normalized_vector = np.clip(normalized_vector, 0.0, 1.0).astype(np.float32)
    return row, normalized_vector


# M20: DRL input scales. BBP up to the 0.1 lifetime threshold; traffic on a log scale
# from 10 to 20,000 (her datasets: 23 at the start, 4,582 at the end of a lifetime).
DRL_BBP_SCALE = 0.1
DRL_LAMBDA_LOG_MIN = 10.0
DRL_LAMBDA_LOG_MAX = 20000.0


# ============================================================================
# CONFIGURATION - Edit this dictionary to configure your simulation
# ============================================================================

CONFIG = {
    # Topology: 'test5' (5-node test network) or 'dt12' (12-node production network)
    'topology': 'dt12',

    ##############################################################################
    # ### ML-INTEGRATION CHANGE M3 (Claude) ###
    # Comment only: added 'random' to the list and noted it is used when ml_mode='off'.
    ##############################################################################
    # Algorithm: 'greedy', 'increment', 'max_initial', 'jump_to_max', 'no_upgrade', 'random'
    # (used when ml_mode = 'off')
    'algorithm': 'greedy',

    # Link selection mode:
    # "path_congestion" -> current logic
    # "most_congested" -> for each critical SD pair, select the highest-utilization link
    #                     from its most congested path
     'link_selection_mode': 'path_congestion',

    # Used only when link_selection_mode = "most_congested"
    'link_score_threshold': 0.02,

    # ---------------- Multi-seed running ----------------
    ##############################################################################
    # ### ML-INTEGRATION CHANGE M3 (Claude) ###
    # ADDED 'seed': the single-seed branch in __main__ reads CONFIG['seed'].
    ##############################################################################
    'seed': 0,                     # used when run_multiple_seeds = False
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
    ##############################################################################
    # ### ML-INTEGRATION CHANGE M3 (Claude) ###
    # Comment only. In ML modes, data collection is switched off automatically (M6).
    ##############################################################################
    # (only active when ml_mode = 'off'; labels come from the heuristic's check_need_for_upgrade)
    'collect_xgboost_data': True,
    'xgboost_csv_name': 'xgboost_need_upgrade_dataset.csv',
    'reset_xgboost_csv_before_run': True,

    ##############################################################################
    # ### ML-INTEGRATION CHANGE M3 (Claude) ###
    # ADDED: all ML configuration keys below. ml_mode='off' keeps the original behaviour.
    ##############################################################################
    # ---------------- ML pipeline ----------------
    # 'off'    -> Akansha's heuristic in CONFIG['algorithm'] (unchanged behaviour)
    # 'train'  -> backend physics engine for DRL_code.py (do not run main.py directly in this mode)
    # 'deploy' -> XGBoost gatekeeper decides IF, DRL policy decides WHICH link, plan_checker validates
    'ml_mode': 'off',

    # Heuristic whose per-link upgrade rule supplies the upgrade TYPE for the link the DRL
    # agent picks, and whose algorithm object plan_checker uses for escalation/cost naming.
    'ml_base_algorithm': 'increment',

    # Heuristic run for the cycle when the DRL's proposal fails plan_checker
    # (or the DRL passes while a feasible candidate existed).
    'ml_fallback_algorithm': 'increment',

    ##############################################################################
    # ### ML-INTEGRATION CHANGE M16 (Claude) ###
    # BUDGET SAVING: the DRL may stop picking early, or skip a cycle entirely, to
    # keep money for a LATER cycle of the SAME budget year (budgets do not carry
    # over to the next year). In the LAST cycle of a budget year it must keep
    # picking while anything still fits, so leftover money is used, not wasted.
    ##############################################################################
    'ml_allow_early_stop': True,               # DRL may save budget within the year
    'ml_spend_all_last_cycle_of_year': True,   # ...but must spend what fits in the year's last cycle
    # If True, the year's last cycle becomes a DRL decision point even when the
    # gatekeeper says "no upgrade", as long as budget is left (use it or lose it).
    # False = the gatekeeper's "no" always skips the cycle.
    'ml_year_end_override_gatekeeper': True,
    # In such an override cycle: False = the DRL DECIDES whether to spend (it may skip);
    # True = it must spend whatever fits.
    'ml_year_end_override_forces_spend': False,

    ##############################################################################
    # ### ML-INTEGRATION CHANGE M15 (Claude) ###
    # ADDED: upgrade levels the DRL can choose from for each link.
    ##############################################################################
    # 'any_higher' (M17, default): the DRL picks any higher target technology state per
    #     link (18 targets -> 20 links x 18 + STOP = 361 actions on dt12).
    # A list of rule names instead, e.g. ['increment', 'greedy', 'jump_to_max'], uses
    #     her per-link rules as the options (M15); ['increment'] = links-only.
    'ml_upgrade_levels': 'any_higher',

    # Model files (relative paths are looked up in: current folder, next to main.py, sim/ml/)
    'ml_gatekeeper_path': 'xgboost_gatekeeper_model.json',
    'ml_drl_policy_path': 'drl_optical_policy.zip',
    'ml_gatekeeper_threshold': 0.5,

    # Who decides IF an upgrade is needed while TRAINING the DRL agent:
    # 'auto'    -> XGBoost gatekeeper if its JSON exists, otherwise 'always'
    # 'xgboost' -> XGBoost gatekeeper (error if JSON missing)
    # 'always'  -> every upgrade cycle is a decision point (agent can still pass)
    'ml_train_gatekeeper': 'auto',

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


##############################################################################
# ### ML-INTEGRATION CHANGE M4 (Claude) ###
# ENTIRE BLOCK IS NEW, down to the 'M4 — END' box. Contents:
# - get_ml_mode / get_run_label / resolve_model_path
# - load_gatekeeper / load_drl_policy / gatekeeper_for_mode (cached loaders)
# - gatekeeper_predict: XGBoost on RAW features, columns matched by name
# - heuristic_safe_links / run_heuristic_upgrade: your heuristic link selection
#   + plan_checker loop, moved here unchanged (used by 'off' AND by the fallback)
# - drl_candidate_links: candidates = union of safe links for path ranks 0-2
# - get_fallback_algorithm: 'increment' by default
# - build_link_upgrade_proposal: DRL link -> fully formed decision dicts
#   (upgrade type + fiber_id from the base algorithm's rule)
# - remaining_budget_snapshot / _selected_decisions / selection_usage /
#   compute_drl_action_mask / drl_observation: O(1) multi-pick mask (M14)
# - commit_upgrade_plan: your 'if plan_ok:' block moved here unchanged
# - apply_drl_decision / _finalize_drl_cycle: DRL picks links one by one; the
#   whole set -> plan_checker -> commit, or FAIL -> incremental (M14)
# - drl_policy_action (deploy inference), close_state_files
##############################################################################
# ============================================================================
# ML PIPELINE HELPERS (XGBoost gatekeeper + DRL link selection)
# ============================================================================

ML_MODES = ("off", "train", "deploy")
_ML_MODEL_CACHE: Dict[str, Any] = {}


def get_ml_mode(config: Dict[str, Any]) -> str:
    mode = str(config.get("ml_mode", "off")).lower()
    if mode not in ML_MODES:
        raise ValueError(f"CONFIG['ml_mode'] must be one of {ML_MODES}, got {mode!r}")
    return mode


def get_run_label(config: Dict[str, Any]) -> str:
    """Results-folder label: heuristic name, or ml_train / ml_deploy.
    M19: an optional CONFIG['ml_run_tag'] is appended so parallel training / evaluation
    simulators each write to their own results folder instead of overwriting each other."""
    mode = get_ml_mode(config)
    label = config["algorithm"] if mode == "off" else f"ml_{mode}"
    tag = config.get("ml_run_tag")
    return f"{label}_{tag}" if tag else label


##############################################################################
# ### ML-INTEGRATION CHANGE M18 (Claude) ###
# FIX for the repository layout: gatekeeper.py and DRL_code.py live in sim/ml/
# and save their models there, while main.py is at the top level. Model files
# are now looked up in: the current folder, next to main.py, and sim/ml/.
##############################################################################
def resolve_model_path(path_like) -> Path:
    """
    Absolute path -> as is. Relative path -> first existing of:
      1) current working directory, 2) next to main.py, 3) <main.py folder>/sim/ml/
    If none exists, returns the sim/ml/ location (where the training scripts save).
    """
    p = Path(path_like)
    if p.is_absolute():
        return p
    main_dir = Path(__file__).resolve().parent
    candidates = [Path.cwd() / p, main_dir / p, main_dir / "sim" / "ml" / p]
    for c in candidates:
        if c.exists():
            return c
    return candidates[-1]


def load_gatekeeper(path_like):
    """Load (and cache) the XGBoost gatekeeper saved by gatekeeper.py."""
    path = resolve_model_path(path_like)
    key = f"xgb::{path}"
    if key in _ML_MODEL_CACHE:
        return _ML_MODEL_CACHE[key]
    if not path.exists():
        raise FileNotFoundError(
            f"XGBoost gatekeeper model not found (looked in the current folder, next to "
            f"main.py, and in sim/ml/). Run sim/ml/gatekeeper.py first (it writes "
            f"xgboost_gatekeeper_model.json next to itself), or point "
            f"CONFIG['ml_gatekeeper_path'] at the file."
        )
    import xgboost as xgb
    model = xgb.XGBClassifier()
    model.load_model(str(path))
    feature_names = model.get_booster().feature_names
    if not feature_names:
        raise ValueError(
            f"{path} has no stored feature names. Re-train with gatekeeper.py "
            f"(it trains on a pandas DataFrame, which stores them)."
        )
    _ML_MODEL_CACHE[key] = model
    print(f"✓ Loaded XGBoost gatekeeper: {path} ({len(feature_names)} features)")
    return model


def load_drl_policy(path_like):
    """Load (and cache) the MaskablePPO policy saved by DRL_code.py."""
    path = resolve_model_path(path_like)
    key = f"drl::{path}"
    if key in _ML_MODEL_CACHE:
        return _ML_MODEL_CACHE[key]
    if not path.exists():
        raise FileNotFoundError(
            f"DRL policy not found (looked in the current folder, next to main.py, and in "
            f"sim/ml/). Train it with sim/ml/DRL_code.py first, or point "
            f"CONFIG['ml_drl_policy_path'] at the .zip file."
        )
    from sb3_contrib import MaskablePPO
    policy = MaskablePPO.load(str(path), device="cpu")
    _ML_MODEL_CACHE[key] = policy
    print(f"✓ Loaded DRL policy: {path}")
    return policy


def gatekeeper_for_mode(config: Dict[str, Any]):
    """Return the gatekeeper model to use, or None for 'always upgrade-check'."""
    mode = get_ml_mode(config)
    if mode == "off":
        return None
    if mode == "deploy":
        return load_gatekeeper(config["ml_gatekeeper_path"])

    choice = str(config.get("ml_train_gatekeeper", "auto")).lower()
    if choice == "always":
        return None
    if choice == "xgboost":
        return load_gatekeeper(config["ml_gatekeeper_path"])
    if choice == "auto":
        if resolve_model_path(config["ml_gatekeeper_path"]).exists():
            return load_gatekeeper(config["ml_gatekeeper_path"])
        print("⚠️ ml_train_gatekeeper='auto': no gatekeeper JSON found -> every upgrade cycle is a DRL decision point.")
        return None
    raise ValueError(f"CONFIG['ml_train_gatekeeper'] must be 'auto', 'xgboost' or 'always', got {choice!r}")


def gatekeeper_predict(gatekeeper, feature_row: Dict[str, float], threshold: float = 0.5):
    """
    XGBoost decision on the RAW feature row (same scale as the training CSV).
    Columns are re-ordered to the model's stored feature names; any name the model
    expects but the simulator did not produce raises immediately (no silent misalignment).
    Returns (upgrade_needed: bool, probability: float).
    """
    if gatekeeper is None:
        return True, 1.0

    import pandas as pd
    expected = list(gatekeeper.get_booster().feature_names)
    missing = [c for c in expected if c not in feature_row]
    if missing:
        raise KeyError(
            f"Gatekeeper expects {len(missing)} feature(s) the simulator did not produce, e.g. "
            f"{missing[:5]}. The training CSV columns and build_feature_row() must use the same names."
        )
    X = pd.DataFrame([[float(feature_row[c]) for c in expected]], columns=expected)
    prob = float(gatekeeper.predict_proba(X)[0][1])
    return bool(prob >= threshold), prob


def heuristic_safe_links(state: Dict[str, Any], algorithm, sd_last_block_prob, path_rank):
    """
    main.py's hard-coded link selection for one path rank (the exact calls the heuristics
    make): select_links_for_upgrade -> network_connectivity -> sort by link score.
    Returns the ordered list of safe candidate links (possibly empty).
    """
    config = state["config"]
    upgrade_links, link_scores, top_contrib = algorithm.select_links_for_upgrade(
        traffic_rates=state["traffic_rates"],
        PATHS=state["PATHS"],
        sd_block_prob=sd_last_block_prob,
        block_threshold=PER_SD_BLOCK_THRESHOLD,
        working_topology=state["working_topology"],
        LINK_INDEX=state["LINK_INDEX"],
        link_status_forward=state["link_status_forward"],
        link_status_backward=state["link_status_backward"],
        current_time=state["Current_global_time"],
        path_stats=state["path_stats"],
        path_rank=path_rank,
        link_selection_mode=config.get("link_selection_mode", "path_congestion"),
        link_score_threshold=config.get("link_score_threshold", PER_SD_BLOCK_THRESHOLD),
    )
    if not upgrade_links:
        return []
    safe_links = network_connectivity(upgrade_links, state["working_topology"], state["LINK_INDEX"])
    return sorted(safe_links, key=lambda lid: float(link_scores.get(lid, 0.0)), reverse=True)


def run_heuristic_upgrade(state: Dict[str, Any], algo_key, algorithm, sd_last_block_prob):
    """
    Akansha's heuristic upgrade cycle (path ranks 0..2, per-link rule, plan_checker),
    unchanged. Used by ml_mode='off' and as the incremental fallback when the DRL's
    proposal fails plan_checker. Does NOT commit; returns (plan_ok, final_links, final_decisions).
    """
    config = state["config"]
    for path_rank in range(3):
        safe_links = heuristic_safe_links(state, algorithm, sd_last_block_prob, path_rank)
        if not safe_links:
            continue

        upgrade_decisions = {
            lid: _algorithm_upgrade_decision(algo_key, algorithm, lid,
                                             state["link_status_forward"], state["link_status_backward"])
            for lid in safe_links
        }

        plan_ok, candidate_links, candidate_upgrade_decisions = plan_checker(
            safe_links=safe_links,
            upgrade_decisions=upgrade_decisions,
            current_time=state["Current_global_time"],
            PATHS=state["PATHS"],
            current_traffic=state["traffic_rates"],
            current_link_status_forward=copy.deepcopy(state["link_status_forward"]),
            current_link_status_backward=copy.deepcopy(state["link_status_backward"]),
            seed=state["seed"],
            algorithm=algorithm,
            budget_params=state["budget_params"],
            budget_selection_mode=config.get("budget_selection_mode", "budget_aware"),
            budget_tracker=state["budget_tracker"],
        )
        if plan_ok:
            return True, candidate_links, candidate_upgrade_decisions
    return False, [], {}


def drl_candidate_links(state: Dict[str, Any], sd_last_block_prob):
    """
    Links the DRL may choose from: union (in rank order, no duplicates) of the safe links
    main.py's hard-coded selection produces for path ranks 0..2.
    """
    seen, out = set(), []
    for path_rank in range(3):
        for lid in heuristic_safe_links(state, state["algorithm"], sd_last_block_prob, path_rank):
            if lid not in seen:
                seen.add(lid)
                out.append(lid)
    return out


def get_fallback_algorithm(state: Dict[str, Any]):
    """(algo_key, algorithm_object) used when the DRL proposal fails. Default: increment."""
    key = state["config"].get("ml_fallback_algorithm", "increment")
    if key == state["algorithm"].name:
        return key, state["algorithm"]
    return key, AlgorithmSelector.get_algorithm(key)


def get_drl_link_ids(state: Dict[str, Any]):
    """Fixed, sorted link ids captured at initialization (action layout: see action_to_choice)."""
    return list(state["drl_link_ids"])


def state_features(state: Dict[str, Any]):
    """(raw_row, normalized_vector) for the current simulator state."""
    return extract_313_features_standalone(
        current_bbp=state["metrics"].get_blocking_probability(),
        traffic_rates=state["traffic_rates"],
        link_status_forward=state["link_status_forward"],
        link_ids=get_drl_link_ids(state),
    )


def _algorithm_upgrade_decision(algo_key, algorithm, lid, link_status_forward, link_status_backward):
    """Same per-link upgrade rule dispatch main.py uses for the heuristics."""
    if algo_key == 'greedy':
        dec = choose_upgrade_type_greedy_max(lid, link_status_forward, link_status_backward)
    elif algo_key == 'jump_to_max':
        dec = algorithm.get_upgrade_decision(lid, link_status_forward, link_status_backward)
    elif algo_key == 'random':
        dec = algorithm.get_random_upgrade_decision(lid, link_status_forward)
    else:
        dec = choose_upgrade_type(lid, link_status_forward, link_status_backward, algorithm)

    if dec is None:
        return []
    if isinstance(dec, dict):
        dec = [dec]
    return [d for d in dec if d is not None and d.get("upgrade_type") is not None]


##############################################################################
# ### ML-INTEGRATION CHANGE M15 (Claude) ###
# NEW (DRL chooses the upgrade LEVEL): each level is one of Akansha's per-link
# upgrade rules (default: increment = next single step, greedy = greedy-max,
# jump_to_max = straight to max technology). The DRL picks (link, level), and
# the decision dicts come from that rule, so they stay well-formed for
# perform_upgrade / plan_checker. CONFIG['ml_upgrade_levels'] = ['increment']
# reproduces the previous links-only behaviour. (The DEFAULT is now
# 'any_higher', see M17 below; a list of rule names still works.)
##############################################################################
##############################################################################
# ### ML-INTEGRATION CHANGE M17 (Claude) ###
# NEW: 'ANY HIGHER UPGRADE'. A link's technology state is its core mode
# (single-core SC / multi-core MC) plus how many fibers are lit C and lit C+L.
# With FIBERS=3 that gives 18 possible target states. The DRL picks a TARGET
# state for a link; it is allowed only if it has MORE capacity than now and is
# reachable without un-lighting a fiber or removing a band (SC->MC allowed via a
# core upgrade, MC->SC never). The exact step list (several steps on one link
# in one cycle is fine) is built here from the same moves Akansha's code allows:
# band_upgrade on a lit C fiber, new_fiber_C / new_fiber_CL on a dark fiber,
# core_upgrade from single-core. Each step is priced once by her cost model.
##############################################################################
ANY_HIGHER = "any_higher"


def _any_higher_target_list():
    out = []
    for mode in ("SC", "MC"):
        for total in range(1, FIBERS + 1):
            for n_cl in range(0, total + 1):
                out.append((mode, total - n_cl, n_cl))
    return out


ANY_HIGHER_TARGETS = _any_higher_target_list()            # 18 targets with FIBERS = 3


def target_key(target) -> str:
    mode, n_c, n_cl = target
    return f"{mode}:{n_c}C+{n_cl}CL"


TARGET_BY_KEY = {target_key(t): t for t in ANY_HIGHER_TARGETS}


def link_capacity_slots(mode: str, n_c: int, n_cl: int) -> int:
    """Total spectrum slots of a link state (per direction)."""
    cores = Num_cores if mode == "MC" else 1
    return cores * (n_c * C_BAND_SLOTS + n_cl * TOTAL_SLOTS)


def link_tech_state(fibers: dict):
    """(mode, {fiber_id: 'D' | 'C' | 'CL'}) using Akansha's own fiber helpers."""
    from sim.upgrade.upgrade_manager import fiber_mode, is_fiber_lit, band_set_of_fiber
    multi = any(isinstance(f, dict) and fiber_mode(f) == "multi" for f in fibers.values())
    techs = {}
    for fid, f in sorted(fibers.items()):
        if not is_fiber_lit(f):
            techs[fid] = "D"
        else:
            techs[fid] = "CL" if band_set_of_fiber(f) == {"C", "L"} else "C"
    return ("MC" if multi else "SC"), techs


def build_target_decisions(fibers: dict, target):
    """
    Step list that takes one link from its current state to `target`, or None if the
    target is not higher / not reachable. Rule for which fibers change: band-upgrade
    existing C fibers first (lowest fiber ids, matching her incremental order), then
    light dark fibers as C+L for any remaining C+L needed, then as C.
    """
    mode_now, techs = link_tech_state(fibers)
    mode_t, n_c_t, n_cl_t = target
    n_c_now = sum(1 for t in techs.values() if t == "C")
    n_cl_now = sum(1 for t in techs.values() if t == "CL")

    if mode_now == "MC" and mode_t == "SC":
        return None
    if link_capacity_slots(mode_t, n_c_t, n_cl_t) <= link_capacity_slots(mode_now, n_c_now, n_cl_now):
        return None

    decs = []
    if mode_now == "SC" and mode_t == "MC":
        # her perform_upgrade: core_upgrade makes every fiber 3-core, only fiber 0 lit (C)
        decs.append({"upgrade_type": "core_upgrade", "fiber_id": None, "core_type": 1})
        techs = {fid: ("C" if fid == 0 else "D") for fid in techs}
        core_type = 3
    else:
        core_type = 1 if mode_now == "SC" else 3

    c_ids = sorted(f for f, t in techs.items() if t == "C")
    dark_ids = sorted(f for f, t in techs.items() if t == "D")
    cur_c, cur_cl = len(c_ids), sum(1 for t in techs.values() if t == "CL")
    tot_now, tot_t = cur_c + cur_cl, n_c_t + n_cl_t
    if n_cl_t < cur_cl or tot_t < tot_now:
        return None                                   # would need removing a band / fiber

    k = min(cur_c, n_cl_t - cur_cl)                   # existing C fibers -> C+L
    new_cl = n_cl_t - cur_cl - k                      # dark fibers lit as C+L
    new_c = (tot_t - tot_now) - new_cl                # dark fibers lit as C
    if new_c < 0 or new_cl + new_c > len(dark_ids):
        return None

    for fid in c_ids[:k]:
        decs.append({"upgrade_type": "band_upgrade", "fiber_id": fid, "core_type": core_type})
    for fid in dark_ids[:new_cl]:
        decs.append({"upgrade_type": "new_fiber_CL", "fiber_id": fid, "core_type": core_type})
    for fid in dark_ids[new_cl:new_cl + new_c]:
        decs.append({"upgrade_type": "new_fiber_C", "fiber_id": fid, "core_type": core_type})
    return decs or None


def get_upgrade_levels(state: Dict[str, Any]):
    """Ordered list of upgrade-level names (= algorithm keys whose per-link rule defines the level)."""
    levels = state["config"].get("ml_upgrade_levels", ["increment"])
    if levels == ANY_HIGHER:
        return [target_key(t) for t in ANY_HIGHER_TARGETS]
    levels = list(levels)
    if not levels:
        raise ValueError("CONFIG['ml_upgrade_levels'] must list at least one level")
    return levels


def _level_algorithm(state: Dict[str, Any], level_key):
    """Algorithm object whose per-link rule generates this level's decisions (cached)."""
    if level_key == state["algorithm"].name:
        return state["algorithm"]
    cache = state.setdefault("level_algorithms", {})
    if level_key not in cache:
        cache[level_key] = AlgorithmSelector.get_algorithm(level_key)
    return cache[level_key]


def build_link_upgrade_proposal(state: Dict[str, Any], link_id, level_key=None):
    """
    Turn a (link, level) choice into main.py's (safe_links, upgrade_decisions) format.
    The upgrade decisions (type, fiber_id, core_type) come from the per-link rule named by
    level_key (default: CONFIG['ml_base_algorithm']), so perform_upgrade / plan_checker
    receive fully-formed decision dicts. Returns None if no upgrade is physically possible.
    """
    if level_key is None:
        level_key = state["config"].get("ml_base_algorithm", "increment")
    decs = _algorithm_upgrade_decision(
        level_key,
        _level_algorithm(state, level_key),
        link_id,
        state["link_status_forward"],
        state["link_status_backward"],
    )
    if not decs:
        return None
    return [link_id], {link_id: decs}


def remaining_budget_snapshot(state: Dict[str, Any]) -> Dict[str, float]:
    """
    O(1), read-only view of the remaining annual budget (same arithmetic as
    plan_checker._compute_cycle_budget_components, but no prints and no mutation).
    """
    budget_params = state["budget_params"]
    budget_tracker = state["budget_tracker"]
    year_idx = int(state["Current_global_time"] // 365)

    if year_idx in budget_tracker:
        y = budget_tracker[year_idx]
        return {
            "opex": max(y["opex_budget"] - y["used_opex"], 0.0),
            "capex": max(y["capex_budget"] - y["used_capex"], 0.0),
            "total": max(y["total_budget"] - y["used_total"], 0.0),
        }

    growth = (1.0 + float(budget_params["inflation_rate"])) ** year_idx
    return {
        "opex": float(budget_params["period_opex_budget_0"]) * growth,
        "capex": float(budget_params["period_capex_budget_0"]) * growth,
        "total": float(budget_params["period_total_budget_0"]) * growth,
    }


##############################################################################
# ### ML-INTEGRATION CHANGE M14 (Claude) ###
# NEW (multi-link selection): proposal for the links picked so far this cycle,
# and their cost/downtime computed with the SAME formulas plan_checker uses
# (compute_upgrade_costs and plan_checker._compute_total_downtime; read-only,
# plan_checker.py itself is not modified).
##############################################################################
##############################################################################
# ### ML-INTEGRATION CHANGE M16 (Claude) ###
# NEW: is this upgrade cycle the last one before the budget year resets?
##############################################################################
def is_last_cycle_of_budget_year(current_time: float) -> bool:
    """True if the NEXT upgrade cycle falls in a different budget year than this one,
    i.e. any budget not used now is lost (budgets reset every 365 days, no carry-over)."""
    next_cycle_time = float(current_time) + float(Upgrade_initiation_days)
    return int(next_cycle_time // 365) != int(float(current_time) // 365)


def _link_level_decisions(state: Dict[str, Any], pending, link_id, level_key):
    """Decisions for one (link, level), cached for the whole cycle. None if impossible."""
    cache = pending.setdefault("proposal_cache", {})
    key = (link_id, level_key)
    if key not in cache:
        if level_key in TARGET_BY_KEY:                     # M17: any-higher target state
            cache[key] = build_target_decisions(state["link_status_forward"][link_id],
                                                TARGET_BY_KEY[level_key])
        else:                                              # M15: one of her per-link rules
            proposal = build_link_upgrade_proposal(state, link_id, level_key)
            cache[key] = None if proposal is None else proposal[1][link_id]
    return cache[key]


def _selected_decisions(state: Dict[str, Any], pending, extra=None):
    """
    {link: [decisions]} for the (link, level) picks made so far this cycle, plus optionally
    one more pick `extra` = (link, level). Returns None if `extra` has no possible upgrade.
    """
    picks = [(lid, pending["selected_levels"][lid]) for lid in pending["selected_links"]]
    if extra is not None:
        picks.append(tuple(extra))
    out = {}
    for lid, level in picks:
        decs = _link_level_decisions(state, pending, lid, level)
        if decs is None:
            return None
        out[lid] = decs
    return out


def selection_usage(state: Dict[str, Any], decisions) -> Dict[str, float]:
    """
    O(1) cost/downtime of a candidate selection, using the SAME formulas plan_checker uses:
      - CAPEX+workforce: compute_upgrade_costs over the whole selection
      - downtime: plan_checker._compute_total_downtime (per-link max, summed over links,
        because downtime jobs run one link at a time)
    """
    # Load the plan_checker MODULE by full name. `from sim.upgrade import plan_checker`
    # would return the plan_checker FUNCTION, because sim/upgrade/__init__.py does
    # `from .plan_checker import *`, which shadows the module name.
    import importlib
    _pc_module = importlib.import_module("sim.upgrade.plan_checker")
    if not decisions:
        return {"capex": 0.0, "downtime": 0.0}
    _, cost_summary, _ = compute_upgrade_costs(
        safe_links_for_upgrade=list(decisions.keys()),
        upgrade_decisions=decisions,
        algorithm_name=state["algorithm"].name,
    )
    capex = float(cost_summary.get("equipment_total", 0.0)) + float(cost_summary.get("workforce_total", 0.0))
    downtime = float(_pc_module._compute_total_downtime(decisions, state["algorithm"].name))
    return {"capex": capex, "downtime": downtime}


##############################################################################
# ### ML-INTEGRATION CHANGE M14 (Claude) ###
# CHANGED (multi-link): a link is selectable only if the WHOLE selection incl.
# that link still fits the CAPEX/TOTAL budget and the downtime limit.
# STOP only when nothing else fits.
# M15: the mask is now over (link, upgrade level) pairs; a level that gives the
# same upgrade as a lower level for that link is masked out.
##############################################################################
def drl_action_count(state: Dict[str, Any]) -> int:
    """Number of DRL actions = n_links * n_levels + 1 (STOP)."""
    return len(get_drl_link_ids(state)) * len(get_upgrade_levels(state)) + 1


def compute_drl_action_mask(state: Dict[str, Any]) -> np.ndarray:
    """
    Mask for the NEXT pick of the current cycle (O(1) per action, no traffic simulation).
    Action layout: index = link_index * n_levels + level_index; last index = STOP.
    A (link, level) action is selectable if ALL hold
      - the link is one of main.py's hard-coded candidate links for this cycle
      - the link has not already been picked this cycle (one level per link per cycle)
      - that level's rule gives a physically possible upgrade for the link, and it is not
        identical to a lower level's upgrade (duplicates are masked out)
      - it has no upgrade job already queued / in progress
      - taking it down keeps the network connected (same check heuristics use)
      - the WHOLE selection incl. this link fits the remaining annual CAPEX and TOTAL
        budget (budget_aware only)
      - the WHOLE selection's sequential downtime fits Upgrade_initiation_days
    STOP (last index) = finish this cycle's selection. Valid when no link is selectable, or
    (M16) when saving is allowed and this is NOT the last cycle of the budget year. In the
    year's last cycle the DRL must keep picking while anything fits. OPEX and projected BBP
    are left to plan_checker.
    """
    link_ids = get_drl_link_ids(state)
    levels = get_upgrade_levels(state)
    mask = np.zeros(drl_action_count(state), dtype=bool)

    pending = state.get("pending_drl_decision")
    if pending is None:
        mask[-1] = True
        return mask

    candidates = set(pending["candidate_links"])
    selected = set(pending["selected_links"])
    busy = set(state["link_execution_queue"]) | {lid for lid, d in state["per_link_jobs"].items() if d}
    budget_aware = (state["config"].get("budget_selection_mode", "budget_aware") == "budget_aware")
    remaining = remaining_budget_snapshot(state) if budget_aware else None
    downtime_limit = float(Upgrade_initiation_days)

    for link_idx, lid in enumerate(link_ids):
        if lid not in candidates or lid in selected or lid in busy or lid not in state["link_status_forward"]:
            continue
        safe = network_connectivity([lid], state["working_topology"], state["LINK_INDEX"])
        if lid not in list(safe or []):
            continue
        seen_decisions = []
        for level_idx, level in enumerate(levels):
            own = _link_level_decisions(state, pending, lid, level)
            if own is None or own in seen_decisions:
                continue                      # impossible, or same upgrade as a lower level
            seen_decisions.append(own)
            decisions = _selected_decisions(state, pending, extra=(lid, level))
            usage = selection_usage(state, decisions)
            if usage["downtime"] > downtime_limit:
                continue
            if budget_aware and (usage["capex"] > remaining["capex"] or usage["capex"] > remaining["total"]):
                continue
            mask[link_idx * len(levels) + level_idx] = True

    # STOP rule (M16): always allowed when nothing else fits. Otherwise allowed only
    # when saving is enabled AND this is not the last cycle of the budget year.
    cfg = state["config"]
    must_spend = (bool(cfg.get("ml_spend_all_last_cycle_of_year", True))
                  and is_last_cycle_of_budget_year(state["Current_global_time"]))
    if pending.get("gatekeeper_override") and not cfg.get("ml_year_end_override_forces_spend", False):
        must_spend = False      # gatekeeper said "no": the DRL decides whether to spend at all
    allow_stop = bool(cfg.get("ml_allow_early_stop", False)) and not must_spend
    mask[-1] = (not mask[:-1].any()) or allow_stop
    return mask


##############################################################################
# ### ML-INTEGRATION CHANGE M14 (Claude) ###
# NEW: DRL observation (335 values on dt12) = 313 features + 20 picked values
# (0 = not picked, else which level it was picked at, M15)
# + remaining CAPEX fraction + remaining downtime fraction.
##############################################################################
def drl_observation(state: Dict[str, Any]) -> np.ndarray:
    """
    DRL observation = [ the 313 normalized network features (identical to everywhere else),
                        one value per link: 0 = not picked this cycle, otherwise
                        (level_index + 1) / n_levels for the level it was picked at,
                        remaining CAPEX fraction after the current picks,
                        remaining downtime fraction after the current picks,
                        1 if this is the last cycle of the budget year else 0 (M16) ].
    The extra values tell the agent what it has already chosen and how much room is left,
    which it needs to choose sensibly within a cycle. XGBoost still gets only the 313 raw
    features.
    """
    _, base = state_features(state)
    link_ids = get_drl_link_ids(state)
    picked = np.zeros(len(link_ids), dtype=np.float32)
    capex_frac = downtime_frac = 1.0

    pending = state.get("pending_drl_decision")
    if pending is not None:
        levels = get_upgrade_levels(state)
        chosen = pending["selected_levels"]
        picked = np.array([(levels.index(chosen[lid]) + 1) / len(levels) if lid in chosen else 0.0
                           for lid in link_ids], dtype=np.float32)
        usage = selection_usage(state, _selected_decisions(state, pending) or {})
        downtime_frac = 1.0 - usage["downtime"] / float(Upgrade_initiation_days)
        if state["config"].get("budget_selection_mode", "budget_aware") == "budget_aware":
            year_idx = int(state["Current_global_time"] // 365)
            growth = (1.0 + float(state["budget_params"]["inflation_rate"])) ** year_idx
            annual_capex = float(state["budget_params"]["period_capex_budget_0"]) * growth
            remaining = remaining_budget_snapshot(state)
            capex_frac = (min(remaining["capex"], remaining["total"]) - usage["capex"]) / max(annual_capex, 1e-9)

    # M16: 1.0 if this is the last cycle before the budget resets (use it or lose it)
    last_cycle_flag = 1.0 if is_last_cycle_of_budget_year(state["Current_global_time"]) else 0.0
    extra = np.clip(np.array([capex_frac, downtime_frac, last_cycle_flag], dtype=np.float32), 0.0, 1.0)
    return np.concatenate([base, picked, extra]).astype(np.float32)


def action_to_choice(state: Dict[str, Any], action):
    """Discrete action index -> (link_id, level_name), or None for STOP (last index)."""
    link_ids, levels = get_drl_link_ids(state), get_upgrade_levels(state)
    a, n = int(action), drl_action_count(state)
    if a < 0 or a >= n:
        raise ValueError(f"Action {a} out of range [0, {n - 1}]")
    if a == n - 1:
        return None
    return link_ids[a // len(levels)], levels[a % len(levels)]


def log_pre_upgrade_per_link_state(writer, link_status_forward, cycle_idx, time_day):
    for link_id, fibers in link_status_forward.items():
        sc_c = sc_cl = mc_c = mc_cl = 0
        for fiber_id, fiber_obj in fibers.items():
            f_state = classify_fiber_state(fiber_obj)
            if f_state == "SC_C":
                sc_c += 1
            elif f_state == "SC_CL":
                sc_cl += 1
            elif f_state == "MC_C":
                mc_c += 3
            elif f_state == "MC_CL":
                mc_cl += 3

        writer.writerow([
            int(cycle_idx),
            float(time_day),
            int(link_id),
            sc_c,
            sc_cl,
            mc_c,
            mc_cl,
        ])


def commit_upgrade_plan(state: Dict[str, Any], final_links, final_upgrade_decisions, period_index):
    """
    Execute an ACCEPTED plan_checker plan: logging, cost accounting, immediate
    new_fiber_C upgrades, and downtime job queue. Shared by the heuristics and the ML
    pipeline so both use identical physics/accounting. Reads and writes `state` in place.
    """
    metrics = state["metrics"]
    algorithm = state["algorithm"]
    Current_global_time = state["Current_global_time"]
    current_upgrade = state["current_upgrade"]
    upgrade_cost_timeline = state["upgrade_cost_timeline"]
    upgrade_log_csv = state["upgrade_log_csv"]
    total_network_cost = state["total_network_cost"]
    link_status_forward = state["link_status_forward"]
    link_status_backward = state["link_status_backward"]

    state["path_stats"].clear()
    safe_links = final_links
    upgrade_decisions = final_upgrade_decisions
    flat_final_decisions = flatten_upgrade_decisions(upgrade_decisions)

    log_pre_upgrade_per_link_state(
        state["pre_upgrade_per_link_writer"], link_status_forward,
        cycle_idx=current_upgrade, time_day=Current_global_time,
    )

    state["upgrade_decision_history"].append({
        "cycle": int(current_upgrade),
        "time_day": float(Current_global_time),
        "blocking_probability": float(metrics.get_blocking_probability()),
        "final_decisions": flat_final_decisions,
        "cycle_upgrade_level": get_cycle_upgrade_level(flat_final_decisions),
    })

    state["upgrade_cycle_writer"].writerow([
        int(current_upgrade),
        float(Current_global_time),
        float(metrics.get_blocking_probability()),
        len(safe_links),
        str(list(safe_links)),
        str(flat_final_decisions),
        get_cycle_upgrade_level(flat_final_decisions),
    ])
    print_upgrade_decisions_summary(Current_global_time, safe_links, upgrade_decisions)

    immediate_actions = {}
    per_link_jobs = {}

    for lid in safe_links:
        decs = upgrade_decisions.get(lid, [])
        decs = [decs] if isinstance(decs, dict) else list(decs)
        has_core = any(d.get("upgrade_type") == "core_upgrade" for d in decs)

        immediate_actions[lid] = []
        per_link_jobs[lid] = []

        if has_core:
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

    immediate_actions = {lid: decs for lid, decs in immediate_actions.items() if decs}
    per_link_jobs = {lid: decs for lid, decs in per_link_jobs.items() if decs}
    link_execution_queue = [lid for lid in safe_links if lid in per_link_jobs]

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

    if immediate_actions:
        immediate_links = sorted(immediate_actions.keys())
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
        working_topology=state["working_topology"],
        link_status_forward=link_status_forward,
        link_status_backward=link_status_backward,
        removed_links_info=state["removed_links_info"],
        ALL_DEMANDS=state["ALL_DEMANDS"],
        total_network_cost=total_network_cost,
        PATHS=state["PATHS"],
        upgrade_log_csv=upgrade_log_csv,
        upgrade_cost_timeline=upgrade_cost_timeline,
        algorithm=algorithm,
    )

    state["link_execution_queue"] = link_execution_queue
    state["per_link_jobs"] = per_link_jobs
    state["working_topology"] = working_topology
    state["link_status_forward"] = link_status_forward
    state["link_status_backward"] = link_status_backward
    state["removed_links_info"] = removed_links_info
    state["ALL_DEMANDS"] = ALL_DEMANDS
    state["total_network_cost"] = total_network_cost
    state["PATHS"] = PATHS


##############################################################################
# ### ML-INTEGRATION CHANGE M14 (Claude) ###
# NEW/CHANGED (multi-link): apply_drl_decision adds one (link, level) pick at a time;
# _finalize_drl_cycle sends the whole selection to plan_checker once and
# falls back to incremental on FAIL. Replaces the old single-link version.
##############################################################################
def _finalize_drl_cycle(state: Dict[str, Any], pending, stop_reason) -> Dict[str, Any]:
    """
    Send the DRL's whole selection for this cycle to plan_checker once.
      PASS -> commit.
      FAIL (or nothing selected) -> nothing spent (plan_checker only commits on PASS);
                                    the cycle FALLS BACK to the incremental heuristic.
      Deliberate save (STOP with nothing picked, M16) -> nothing spent, NO fallback.
    """
    selected = list(pending["selected_links"])
    decisions = _selected_decisions(state, pending) or {}

    result = {
        "cycle": pending["cycle"],
        "time_day": float(state["Current_global_time"]),
        "n_candidates": len(pending["candidate_links"]),
        "selected_links": selected,
        "selected_levels": {lid: pending["selected_levels"][lid] for lid in selected},
        "n_selected": len(selected),
        "stop_reason": stop_reason,
        "status": None,
        "plan_ok": False,
        "fallback": None,
        "final_links": [],
        "final_decisions": {},
        "gatekeeper_prob": pending.get("gatekeeper_prob"),
        "last_cycle_of_year": is_last_cycle_of_budget_year(state["Current_global_time"]),
        "gatekeeper_override": bool(pending.get("gatekeeper_override", False)),
        "cycle_complete": True,
    }
    state["pending_drl_decision"] = None

    if not selected and stop_reason == "early_stop":
        # M16: the DRL deliberately saved this cycle's budget for later in the year.
        # Respected: no incremental fallback (that would spend the money anyway).
        result["status"] = "saved_budget"
    elif not selected:
        result["status"] = "nothing_selected"
    else:
        config = state["config"]
        plan_ok, final_links, final_upgrade_decisions = plan_checker(
            safe_links=selected,
            upgrade_decisions=decisions,
            current_time=state["Current_global_time"],
            PATHS=state["PATHS"],
            current_traffic=state["traffic_rates"],
            current_link_status_forward=copy.deepcopy(state["link_status_forward"]),
            current_link_status_backward=copy.deepcopy(state["link_status_backward"]),
            seed=state["seed"],
            algorithm=state["algorithm"],
            budget_params=state["budget_params"],
            budget_selection_mode=config.get("budget_selection_mode", "budget_aware"),
            budget_tracker=state["budget_tracker"],
        )
        if plan_ok:
            commit_upgrade_plan(state, final_links, final_upgrade_decisions, pending["period_index"])
            result.update(status="executed", plan_ok=True,
                          final_links=list(final_links), final_decisions=final_upgrade_decisions)
        else:
            result["status"] = "rejected_by_plan_checker"

    if result["status"] not in ("executed", "saved_budget"):
        fb_key, fb_algorithm = get_fallback_algorithm(state)
        print(f">>> DRL {result['status']} -> falling back to '{fb_key}' for this cycle")
        fb_ok, fb_links, fb_decisions = run_heuristic_upgrade(
            state, fb_key, fb_algorithm, pending["sd_last_block_prob"]
        )
        if fb_ok:
            commit_upgrade_plan(state, fb_links, fb_decisions, pending["period_index"])
            result.update(fallback="executed", final_links=list(fb_links), final_decisions=fb_decisions)
        else:
            result["fallback"] = "failed"

    picked_txt = ", ".join(f"{lid}:{result['selected_levels'][lid]}" for lid in selected)
    print(f">>> DRL cycle {result['cycle']}: selected [{picked_txt}] ({stop_reason}) -> {result['status']}"
          + (f" | fallback: {result['fallback']}" if result["fallback"] else ""))
    state["drl_decision_log"].append({k: v for k, v in result.items()
                                      if k not in ("final_decisions", "cycle_complete")})
    state["last_drl_result"] = result
    return result


def apply_drl_decision(state: Dict[str, Any], action) -> Dict[str, Any]:
    """
    Apply ONE pick of the DRL at the pending decision point.
      action = (link, level) index -> add that link at that upgrade level to this cycle's
               selection. If no further pick
               fits the remaining budget/downtime, the selection is finalized right away.
      action = STOP (last index; only unmasked when nothing else fits, or when
               CONFIG['ml_allow_early_stop'] is True) -> finalize.
    Finalizing = plan_checker on the whole selection; FAIL -> incremental fallback.
    Returns {"cycle_complete": False, ...} while the cycle is still being built, or the
    final cycle result (cycle_complete=True) from _finalize_drl_cycle().
    """
    pending = state.get("pending_drl_decision")
    if pending is None:
        raise RuntimeError("apply_drl_decision() called but no DRL decision is pending.")

    mask = compute_drl_action_mask(state)
    action = int(action)
    choice = action_to_choice(state, action)

    if not mask[action]:
        print(f"⚠️ DRL chose masked-out action {action} ({choice}); treating it as STOP.")
        return _finalize_drl_cycle(state, pending, "invalid_action")

    if choice is None:
        reason = "no_feasible_link_left" if not mask[:-1].any() else "early_stop"
        return _finalize_drl_cycle(state, pending, reason)

    link_id, level = choice
    pending["selected_links"].append(link_id)
    pending["selected_levels"][link_id] = level
    next_mask = compute_drl_action_mask(state)
    if not next_mask[:-1].any():
        return _finalize_drl_cycle(state, pending, "no_feasible_link_left")

    return {"cycle_complete": False, "picked_link": link_id, "picked_level": level,
            "selected_links": list(pending["selected_links"]),
            "selected_levels": dict(pending["selected_levels"])}


def drl_policy_action(state: Dict[str, Any], policy) -> int:
    """Deploy-time inference: one masked, deterministic pick from the trained policy."""
    obs = drl_observation(state)
    mask = compute_drl_action_mask(state)

    if tuple(policy.observation_space.shape) != tuple(obs.shape):
        raise ValueError(
            f"DRL policy expects observation shape {policy.observation_space.shape}, simulator "
            f"produces {obs.shape}. Was the policy trained with a different topology or an "
            f"older DRL_code.py? Retrain with the current DRL_code.py."
        )
    if int(policy.action_space.n) != len(mask):
        raise ValueError(
            f"DRL policy has {policy.action_space.n} actions, simulator needs {len(mask)} "
            f"(links x upgrade levels + STOP). Was it trained with a different topology or "
            f"different CONFIG['ml_upgrade_levels']?"
        )

    action, _ = policy.predict(obs, action_masks=mask, deterministic=True)
    return int(np.asarray(action).item())


def close_state_files(state: Dict[str, Any]):
    """Close every CSV handle opened by initialize_simulation_state (safe to call twice)."""
    for key in ("connection_log_file", "daily_network_file", "upgrade_cycle_file", "pre_upgrade_per_link_file"):
        f = state.get(key)
        try:
            if f is not None and not f.closed:
                f.close()
        except Exception:
            pass


##############################################################################
# ### ML-INTEGRATION CHANGE M4 — END ###
# End of the new ML helper block. Original code continues below.
##############################################################################
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

    ##############################################################################
    # ### ML-INTEGRATION CHANGE M5 (Claude) ###
    # Was: algorithm = AlgorithmSelector.get_algorithm(config['algorithm'])
    #      results_dir = create_results_directory(config['topology'], config['algorithm'])
    # Now: in ML modes load CONFIG['ml_base_algorithm'] and label the results
    #      folder ml_train / ml_deploy. ml_mode='off' is identical to before.
    ##############################################################################
    # Load algorithm.
    # In ML modes the algorithm object is the BASE algorithm: it supplies the per-link
    # upgrade rule for the DRL's chosen link and is what plan_checker / perform_upgrade
    # use internally (escalation, cost naming). Heuristic link selection is NOT used.
    ml_mode = get_ml_mode(config)
    algo_key = config['algorithm'] if ml_mode == "off" else config.get('ml_base_algorithm', 'increment')
    algorithm = AlgorithmSelector.get_algorithm(algo_key)

    # Create results directory
    results_dir = create_results_directory(config['topology'], get_run_label(config))
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

# ============================================================================
# Modularized Simulation Core
# ============================================================================

def initialize_simulation_state(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Initializes topology, traffic matrices, data loggers, and metrics.
    Returns a unified state dictionary for DRL training or standalone evaluation.
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

    print("=" * 70)
    print("OPTICAL NETWORK SIMULATOR")
    print("=" * 70)
    print(f"Topology: {config['topology']}")
    print(f"Algorithm: {algorithm.name}")
    print(f"Seed: {config['seed']}")
    print("=" * 70 + "\n")

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

    path_stats = {}
    path_stats_history = {}

    # Initialize network state
    total_network_cost = 0.0
    upgrade_cost_timeline = []
    connection_id = 0
    upgrade_decision_history = []
    budget_tracker = {}
    removed_links_info = {}
    Current_global_time = 0.0
    current_week = 0
    current_upgrade = 0
    last_cost_time = 0.0
    last_logged_day = -1

    link_execution_queue = []
    per_link_jobs = {}

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

    initial_daily_opex = float(compute_network_opex_per_day(link_status_forward))

    base_budget_params = {
        "period_days": int(Upgrade_initiation_days),
        "period_total_budget_0": 750_0000.0,
        "period_opex_budget_0": 100_0000.0,
        "period_capex_budget_0": 650_0000.0,
        "inflation_rate": 0.03,
    }

    if config["topology"] == "test5":
        budget_params = base_budget_params
    elif config["topology"] == "dt12":
        dt12_multiplier = 3.0
        budget_params = {
            "period_days": base_budget_params["period_days"],
            "period_total_budget_0": dt12_multiplier * base_budget_params["period_total_budget_0"],
            "period_opex_budget_0": dt12_multiplier * base_budget_params["period_opex_budget_0"],
            "period_capex_budget_0": dt12_multiplier * base_budget_params["period_capex_budget_0"],
            "inflation_rate": base_budget_params["inflation_rate"],
        }
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

    traffic_generator = NetworkTrafficGenerator(
        number_of_nodes=N,
        datarates=DATARATE,
        lambda_0=lambda_0,
        mean_holding_time=MEAN_HOLDING_TIME,
        Current_global_time=Current_global_time,
        rng=main_rng
    )

    PATHS = build_k_shortest_paths(TOPOLOGY)
    if not PATHS:
        raise ValueError("build_k_shortest_paths() returned empty structure")

    ##############################################################################
    # ### ML-INTEGRATION CHANGE M6 (Claude) ###
    # ADDED: turn off XGBoost data logging in ML modes, load the gatekeeper,
    # freeze the link order used for DRL actions/features.
    ##############################################################################
    # ML pipeline state
    ml_mode = get_ml_mode(config)
    if ml_mode != "off":
        # Gatekeeper/DRL decide; the heuristic's check_need_for_upgrade labels are not computed.
        config["collect_xgboost_data"] = False
    gatekeeper = gatekeeper_for_mode(config)
    drl_link_ids = sorted(link_status_forward.keys())   # fixed action/feature order for the whole run
    print(f"ML mode: {ml_mode}" + (f" (base algorithm: {algorithm.name}, {len(drl_link_ids)} links)"
                                   if ml_mode != "off" else ""))

    print("✓ Simulation initialized\n")

    # Return unified state dictionary
    return {
        ##############################################################################
        # ### ML-INTEGRATION CHANGE M6 (Claude) ###
        # ADDED state keys: gatekeeper, drl_link_ids, pending_drl_decision,
        # drl_decision_log, last_drl_result. All original keys are unchanged below.
        ##############################################################################
        "gatekeeper": gatekeeper,
        "drl_link_ids": drl_link_ids,
        "pending_drl_decision": None,
        "drl_decision_log": [],
        "last_drl_result": None,
        "config": config,
        "algorithm": algorithm,
        "results_dir": results_dir,
        "topology_data": topology_data,
        "TOPOLOGY": TOPOLOGY,
        "TOPOLOGY_LINK_LENGTHS": TOPOLOGY_LINK_LENGTHS,
        "LINK_INDEX": LINK_INDEX,
        "N": N,
        "LINKS": LINKS,
        "seed": seed,
        "main_rng": main_rng,
        "metrics": metrics,
        "analysis": analysis,
        "traffic_rates": traffic_rates,
        "ALL_DEMANDS": ALL_DEMANDS,
        "working_topology": working_topology,
        "link_status_forward": link_status_forward,
        "link_status_backward": link_status_backward,
        "PATHS": PATHS,
        "traffic_generator": traffic_generator,
        "budget_params": budget_params,
        "path_stats": path_stats,
        "path_stats_history": path_stats_history,
        "total_network_cost": total_network_cost,
        "upgrade_cost_timeline": upgrade_cost_timeline,
        "connection_id": connection_id,
        "upgrade_decision_history": upgrade_decision_history,
        "budget_tracker": budget_tracker,
        "removed_links_info": removed_links_info,
        "Current_global_time": Current_global_time,
        "current_week": current_week,
        "current_upgrade": current_upgrade,
        "last_cost_time": last_cost_time,
        "last_logged_day": last_logged_day,
        "current_month": current_month,
        "MONTH_DAYS": MONTH_DAYS,
        "link_execution_queue": link_execution_queue,
        "per_link_jobs": per_link_jobs,
        "start_wall_time": start_wall_time,
        "util_csv": util_csv,
        "frag_csv": frag_csv,
        "daily_network_file": daily_network_file,
        "daily_network_writer": daily_network_writer,
        "pre_upgrade_per_link_file": pre_upgrade_per_link_file,
        "pre_upgrade_per_link_writer": pre_upgrade_per_link_writer,
        "upgrade_cycle_file": upgrade_cycle_file,
        "upgrade_cycle_writer": upgrade_cycle_writer,
        "connection_log_file": connection_log_file,
        "connection_log_writer": connection_log_writer,
        "upgrade_log_csv": upgrade_log_csv,
    }

    
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
        upgrade_log_csv,         
        upgrade_cost_timeline,   
        algorithm,
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

def run_single_cycle(state: Dict[str, Any], action: Optional[Any] = None) -> Tuple[Dict[str, Any], float, bool]:
    """
    Executes exactly ONE event step (one connection arrival) of the simulation loop.
    Returns: (updated_state, reward, done)

    In ML modes, when an upgrade cycle triggers and the gatekeeper says an upgrade is
    needed, state["pending_drl_decision"] is set and the caller must resolve it with
    apply_drl_decision(state, action) before calling run_single_cycle again.
    The `action` argument is unused and kept only for backward compatibility.
    """
    ##############################################################################
    # ### ML-INTEGRATION CHANGE M7 (Claude) ###
    # ADDED (plus the docstring text above): refuse to advance while a DRL
    # decision is pending. It must be resolved with apply_drl_decision().
    ##############################################################################
    if state.get("pending_drl_decision") is not None:
        raise RuntimeError(
            "A DRL decision is pending. Call apply_drl_decision(state, action) "
            "before advancing the simulation."
        )
    # --- 1. UNPACK ENVIRONMENT STATE ---
    config = state["config"]
    algorithm = state["algorithm"]
    TOPOLOGY = state["TOPOLOGY"]
    LINK_INDEX = state["LINK_INDEX"]
    N = state["N"]
    LINKS = state["LINKS"]
    seed = state["seed"]
    main_rng = state["main_rng"]
    metrics = state["metrics"]
    traffic_rates = state["traffic_rates"]
    ALL_DEMANDS = state["ALL_DEMANDS"]
    working_topology = state["working_topology"]
    link_status_forward = state["link_status_forward"]
    link_status_backward = state["link_status_backward"]
    PATHS = state["PATHS"]
    traffic_generator = state["traffic_generator"]
    budget_params = state["budget_params"]
    path_stats = state["path_stats"]
    path_stats_history = state["path_stats_history"]
    total_network_cost = state["total_network_cost"]
    upgrade_cost_timeline = state["upgrade_cost_timeline"]
    connection_id = state["connection_id"]
    upgrade_decision_history = state["upgrade_decision_history"]
    budget_tracker = state["budget_tracker"]
    removed_links_info = state["removed_links_info"]
    Current_global_time = state["Current_global_time"]
    current_week = state["current_week"]
    current_upgrade = state["current_upgrade"]
    last_cost_time = state["last_cost_time"]
    last_logged_day = state["last_logged_day"]
    current_month = state["current_month"]
    MONTH_DAYS = state["MONTH_DAYS"]
    link_execution_queue = state["link_execution_queue"]
    per_link_jobs = state["per_link_jobs"]
    util_csv = state["util_csv"]
    frag_csv = state["frag_csv"]
    daily_network_writer = state["daily_network_writer"]
    pre_upgrade_per_link_writer = state["pre_upgrade_per_link_writer"]
    upgrade_cycle_writer = state["upgrade_cycle_writer"]
    connection_log_writer = state["connection_log_writer"]
    upgrade_log_csv = state["upgrade_log_csv"]

    # --- 2. INNER HELPER FUNCTIONS ---
    def account_running_opex_until(new_time: float):
        nonlocal total_network_cost, last_cost_time, last_logged_day
        new_time = float(new_time)
        t = float(last_cost_time)

        if new_time <= t:
            return

        while t < new_time:
            next_boundary = float(int(t) + 1)
            seg_end = min(new_time, next_boundary)
            dt = seg_end - t
            if dt <= 0:
                break

            opex_inc = float(compute_running_opex_interval(link_status_forward, dt))
            total_network_cost += opex_inc

            day_marker = int(seg_end)
            if day_marker > last_logged_day:
                opex_rate = float(compute_network_opex_per_day(link_status_forward))

                upgrade_cost_timeline.append({
                    "day": float(day_marker),
                    "equipment": 0.0,
                    "workforce": 0.0,
                    "opex": float(opex_inc),
                    "opex_rate": float(opex_rate),
                    "total": float(opex_inc),
                    "cumulative_total": float(total_network_cost),
                })
                last_logged_day = day_marker
                log_daily_network_state(day_marker)
            else:
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

    ##############################################################################
    # ### ML-INTEGRATION CHANGE M8 (Claude) ###
    # REMOVED here: nested classify_fiber_state() and log_pre_upgrade_per_link_state()
    # (now top level, M2 / M4).
    # ADDED: _sync_to_state() / _sync_from_state() so locals and the state dict
    # stay consistent around state-based helpers (commit_upgrade_plan etc.).
    ##############################################################################
    def _sync_to_state():
        """Write every (possibly reassigned) local back into state."""
        state["Current_global_time"] = Current_global_time
        state["current_week"] = current_week
        state["current_upgrade"] = current_upgrade
        state["current_month"] = current_month
        state["last_cost_time"] = last_cost_time
        state["last_logged_day"] = last_logged_day
        state["total_network_cost"] = total_network_cost
        state["connection_id"] = connection_id
        state["working_topology"] = working_topology
        state["link_status_forward"] = link_status_forward
        state["link_status_backward"] = link_status_backward
        state["removed_links_info"] = removed_links_info
        state["ALL_DEMANDS"] = ALL_DEMANDS
        state["PATHS"] = PATHS
        state["link_execution_queue"] = link_execution_queue
        state["per_link_jobs"] = per_link_jobs

    def _sync_from_state():
        """Re-read locals that a state-based helper (commit_upgrade_plan) may have replaced."""
        nonlocal total_network_cost, working_topology, link_status_forward, link_status_backward
        nonlocal removed_links_info, ALL_DEMANDS, PATHS, link_execution_queue, per_link_jobs
        total_network_cost = state["total_network_cost"]
        working_topology = state["working_topology"]
        link_status_forward = state["link_status_forward"]
        link_status_backward = state["link_status_backward"]
        removed_links_info = state["removed_links_info"]
        ALL_DEMANDS = state["ALL_DEMANDS"]
        PATHS = state["PATHS"]
        link_execution_queue = state["link_execution_queue"]
        per_link_jobs = state["per_link_jobs"]

    # --- 3. UNROLLED SINGLE CYCLE EVENT EXECUTION ---
    # Weekly traffic growth
    while (Current_global_time // Traffic_growth_days) > current_week:
        print(f">>> Weekly traffic update at day {Current_global_time}")
        current_week += 1
        for i in range(N):
            for j in range(N):
                if i != j:
                    growth_factor = 1 + main_rng.uniform(0, alpha / 100)
                    traffic_rates[i][j] *= growth_factor
                    traffic_rates[i][j] *= (1 + delta / 100)

    # Generate next request
    src, dest, datarate = traffic_generator.generate_connection_data()
    lambda_sd = traffic_rates[src][dest]

    if lambda_sd <= 0:
        raise ValueError("Invalid lambda_sd: must be non-zero and positive.")

    arrival_time, holding_time = traffic_generator.get_connection(lambda_sd)
    Current_global_time = arrival_time
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

    # Upgrade Cycle Check (every 3 months)
    if (Current_global_time // Upgrade_initiation_days) > current_upgrade:
        print(f"\n=== Upgrade cycle triggered at day {Current_global_time} ===")
        current_upgrade += 1
        ##############################################################################
        # ### ML-INTEGRATION CHANGE M9 (Claude) ###
        # CHANGED upgrade check:
        # - ml_mode='off': algorithm.check_need_for_upgrade(...) exactly as before
        # - ML modes: XGBoost gatekeeper on the RAW feature row decides upgrade_needed
        ##############################################################################
        ml_mode = get_ml_mode(config)

        print(">>> Checking whether upgrade is needed...")
        gatekeeper_prob = None
        gatekeeper_overridden = False                      # M16/M17 year-end override
        if ml_mode == "off":
            upgrade_needed = algorithm.check_need_for_upgrade(
                current_traffic=traffic_rates,
                current_time=Current_global_time,
                PATHS=PATHS,
                current_link_status_forward=copy.deepcopy(link_status_forward),
                current_link_status_backward=copy.deepcopy(link_status_backward),
                seed=seed
            )
        else:
            # XGBoost gatekeeper on the RAW feature row (same scale as its training CSV)
            feature_row = build_feature_row(
                current_bbp=metrics.get_blocking_probability(),
                traffic_rates=traffic_rates,
                link_status_forward=link_status_forward,
                link_ids=state["drl_link_ids"],
            )
            upgrade_needed, gatekeeper_prob = gatekeeper_predict(
                state["gatekeeper"], feature_row,
                threshold=float(config.get("ml_gatekeeper_threshold", 0.5)),
            )
            print(f">>> Gatekeeper P(upgrade) = {gatekeeper_prob:.4f}")

            # M16: optional "use it or lose it" in the last cycle of the budget year
            if (not upgrade_needed
                    and config.get("ml_year_end_override_gatekeeper", False)
                    and is_last_cycle_of_budget_year(Current_global_time)):
                _sync_to_state()
                left = remaining_budget_snapshot(state)
                if min(left["capex"], left["total"]) > 0:
                    print(">>> Last cycle of the budget year with budget left -> DRL decides anyway")
                    upgrade_needed = True
                    gatekeeper_overridden = True

        print(f">>> Upgrade needed = {upgrade_needed}")

        if config.get("collect_xgboost_data", False):
            xgb_csv_path = config["results_dir"] / config.get("xgboost_csv_name", "xgboost_need_upgrade_dataset.csv")
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

        if upgrade_needed:
            sd_last_block_prob = copy.deepcopy(metrics.get_sd_blocking_probability())
            metrics.sd_arrivals.clear()
            metrics.sd_blocked.clear()

            ##############################################################################
            # ### ML-INTEGRATION CHANGE M9 (Claude) ###
            # ADDED (ML modes): candidate links from main.py's own link selection, then a
            # DRL decision point is stored in state['pending_drl_decision'].
            # If there are no candidate links, nothing is upgraded this cycle.
            # REMOVED: the old `config['algorithm'] == 'drl'` branch.
            ##############################################################################
            if ml_mode != "off":
                # Candidate links come from main.py's hard-coded selection (same as heuristics).
                _sync_to_state()
                candidates = drl_candidate_links(state, sd_last_block_prob)
                if candidates:
                    # DRL decision point. The caller (run_simulation in deploy mode, or the
                    # Gym env in DRL_code.py) must call apply_drl_decision(state, action)
                    # before the next run_single_cycle call.
                    state["pending_drl_decision"] = {
                        "cycle": int(current_upgrade),
                        "time_day": float(Current_global_time),
                        "period_index": int(period_index),
                        "gatekeeper_prob": gatekeeper_prob,
                        "candidate_links": candidates,
                        "selected_links": [],        # DRL picks accumulate here (M14)
                        "gatekeeper_override": gatekeeper_overridden,   # M17
                        "selected_levels": {},       # link -> upgrade level picked (M15)
                        "sd_last_block_prob": sd_last_block_prob,
                    }
                else:
                    print(">>> No candidate links from link selection -> no upgrade this cycle")
            else:
                ##############################################################################
                # ### ML-INTEGRATION CHANGE M9 (Claude) ###
                # CHANGED (ml_mode='off'): your path-rank loop and the 'if plan_ok:' block
                # now live in run_heuristic_upgrade() and commit_upgrade_plan() (M4). The
                # logic is identical; verified connection-by-connection against your
                # original main.py on the same seed.
                ##############################################################################
                _sync_to_state()
                plan_ok, final_links, final_upgrade_decisions = run_heuristic_upgrade(
                    state, config['algorithm'], algorithm, sd_last_block_prob
                )
                if plan_ok:
                    commit_upgrade_plan(state, final_links, final_upgrade_decisions, period_index)
                _sync_from_state()

    # Restore Downtime Links
    restore_now = [
        key
        for key, info in removed_links_info.items()
        if info.get("restore_time", math.inf) <= Current_global_time
    ]

    if restore_now:
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
            upgrade_log_csv=upgrade_log_csv,        
            upgrade_cost_timeline=upgrade_cost_timeline,  
            algorithm=algorithm,
        )

    # Remove Expired Connections
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

    # Run RSA for New Connection
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

    # Update Path Statistics
    for info in (attempted_paths_info or []):
        p_idx = int(info["path_index"])
        status = info["status"]

        if status not in ("accepted", "blocked"):
            continue

        sd_u = sd_key_undirected(src, dest)
        key = (sd_u[0], sd_u[1], p_idx)

        if key not in path_stats:
            path_stats[key] = {"path_arrivals": 0, "path_accepted": 0, "path_blocked": 0}
        path_stats[key]["path_arrivals"] += 1
        if status == "accepted":
            path_stats[key]["path_accepted"] += 1
        else:
            path_stats[key]["path_blocked"] += 1

        if key not in path_stats_history:
            path_stats_history[key] = {"path_arrivals": 0, "path_accepted": 0, "path_blocked": 0}
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

    # Update Metrics
    if connection_id > config['warmup_connections']:
        sd_u = sd_key_undirected(src, dest)
        metrics.sd_arrivals[sd_u] += 1
        blocked = (mf_index == float('inf') or fs_index == float('inf'))
        if blocked:
            metrics.sd_blocked[sd_u] += 1
        metrics.record_connection(src, dest, datarate, blocked, Current_global_time)

    ##############################################################################
    # ### ML-INTEGRATION CHANGE M10 (Claude) ###
    # BUG FIX: previously only 8 scalars were written back, so link_execution_queue,
    # per_link_jobs, PATHS (after a downtime restore), working_topology,
    # removed_links_info, ALL_DEMANDS and link status were lost between calls.
    # Stop rule now only applies after warmup, like the original single loop.
    ##############################################################################
    # --- 4. PACK UPDATED STATE BACK ---
    # ALL reassigned locals are written back (previously only the scalars were, so the
    # downtime job queue, per-link jobs, restored PATHS, etc. were silently lost).
    _sync_to_state()

    # Same stop rule as the original single-loop simulator: only checked after warmup.
    done = bool(
        connection_id > config['warmup_connections']
        and metrics.get_blocking_probability() >= config['blocking_threshold']
    )
    if done:
        print(
            f"\nSimulation stopped: blocking threshold "
            f"({config['blocking_threshold']}) reached at day {Current_global_time:.1f}."
        )
    reward = -float(metrics.get_blocking_probability())

    return state, reward, done

def run_simulation(config: Dict[str, Any]):
    """
    Run the optical network simulation with given configuration using the modular loop.
    """
    ##############################################################################
    # ### ML-INTEGRATION CHANGE M11 (Claude) ###
    # ADDED: ml_mode='train' is driven by DRL_code.py; deploy loads the policy
    # up front so a missing file fails fast.
    ##############################################################################
    ml_mode = get_ml_mode(config)
    if ml_mode == "train":
        raise RuntimeError(
            "ml_mode='train' makes main.py the backend for DRL_code.py. "
            "Run DRL_code.py to train, or set ml_mode to 'off' or 'deploy'."
        )

    # Load the DRL policy before building the state, so a missing file fails fast.
    drl_policy = load_drl_policy(config["ml_drl_policy_path"]) if ml_mode == "deploy" else None

    # 1. Initialize simulation state
    state = initialize_simulation_state(config)
    ##############################################################################
    # ### ML-INTEGRATION CHANGE M11 (Claude) ###
    # BUG FIX: `config` was the raw CONFIG, so config['blocking_csv'] raised a
    # KeyError at the end of every run.
    ##############################################################################
    # Use the EXTENDED config (results_dir, blocking_csv, plot paths, ...) from here on.
    # The raw CONFIG passed in does not contain those keys.
    config = state["config"]
    done = False

    # 2. Main Event Loop
    while not done:
        state, reward, done = run_single_cycle(state)

        ##############################################################################
        # ### ML-INTEGRATION CHANGE M11 (Claude) ###
        # ADDED: deploy-mode decision loop. The DRL keeps picking links until nothing
        # else fits the budget/downtime; then the whole set goes to plan_checker once
        # (incremental fallback on FAIL happens inside apply_drl_decision).
        ##############################################################################
        # ML deploy: gatekeeper said "upgrade" -> DRL picks links -> plan_checker validates
        if state.get("pending_drl_decision") is not None:
            if done:
                state["pending_drl_decision"] = None
            else:
                while state.get("pending_drl_decision") is not None:
                    action = drl_policy_action(state, drl_policy)
                    apply_drl_decision(state, action)

    # 3. Post-Simulation Results, Plots, and Reporting
    metrics = state["metrics"]
    analysis = state["analysis"]
    Current_global_time = state["Current_global_time"]
    total_network_cost = state["total_network_cost"]
    upgrade_cost_timeline = state["upgrade_cost_timeline"]
    upgrade_decision_history = state["upgrade_decision_history"]
    budget_params = state["budget_params"]
    path_stats_history = state["path_stats_history"]
    start_wall_time = state["start_wall_time"]

    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)

    analysis.save_blocking_data(
        metrics.time_points,
        metrics.blocking_prob_points,
        config['blocking_csv']
    )

    end_wall_time = time.time()
    total_runtime_sec = end_wall_time - start_wall_time

    snapshot = {
        "seed": state["seed"],
        "Current_global_time": Current_global_time,
        "connection_id": state["connection_id"],
        "connection_count": metrics.connection_count,
        "blocked_connection_count": metrics.blocked_connection_count,
        "arrived_datarate": metrics.arrived_datarate,
        "accepted_datarate": metrics.accepted_datarate,
        "blocked_datarate": metrics.blocked_datarate,
        "time_points": metrics.time_points,
        "blocking_prob_points": metrics.blocking_prob_points,
        "traffic_rates": state["traffic_rates"],
        "total_network_cost": total_network_cost,
        "upgrade_cost_timeline": upgrade_cost_timeline,
        "upgrade_decision_history": upgrade_decision_history,
        "link_status_forward": state["link_status_forward"],
        "link_status_backward": state["link_status_backward"],
        "ALL_DEMANDS": state["ALL_DEMANDS"],
        "working_topology": state["working_topology"],
        "wall_clock_runtime_seconds": total_runtime_sec,
        "budget_params": budget_params,
        "path_stats_history": path_stats_history,
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

    try:
        from sim.modules.simulation_analysis import plot_technology_mix_by_upgrade_index
        plot_technology_mix_by_upgrade_index(
            folder=config["results_dir"],
            outdir=config["results_dir"],
            algo_name=config["algorithm"]
        )
    except Exception as e:
        print(f"[WARN] Technology mix plot failed: {e}")

    ##############################################################################
    # ### ML-INTEGRATION CHANGE M12 (Claude) ###
    # ADDED: save drl_decisions.csv; close_state_files() replaces the four
    # individual .close() calls; run_simulation now returns the final state.
    ##############################################################################
    if state["drl_decision_log"]:
        drl_log_path = config["results_dir"] / "drl_decisions.csv"
        with open(drl_log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(state["drl_decision_log"][0].keys()))
            writer.writeheader()
            writer.writerows(state["drl_decision_log"])
        print(f"✓ Saved DRL decision log to {drl_log_path}")

    close_state_files(state)

    print(f"\n✓ All results saved to: {config['results_dir']}")
    return state


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
        ##############################################################################
        # ### ML-INTEGRATION CHANGE M13 (Claude) ###
        # CHANGED: CONFIG['algorithm'] -> get_run_label(CONFIG) (ml_deploy folder in ML modes).
        ##############################################################################
        results_dir = create_results_directory(
            CONFIG["topology"],
            get_run_label(CONFIG)
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

            ##############################################################################
            # ### ML-INTEGRATION CHANGE M13 (Claude) ###
            # CHANGED: run_config['algorithm'] -> get_run_label(run_config).
            ##############################################################################
            results_dir = create_results_directory(
                run_config["topology"],
                get_run_label(run_config)
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
        ##############################################################################
        # ### ML-INTEGRATION CHANGE M13 (Claude) ###
        # CHANGED: CONFIG['algorithm'] -> get_run_label(CONFIG).
        ##############################################################################
        results_dir = create_results_directory(
            CONFIG["topology"],
            get_run_label(CONFIG)
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