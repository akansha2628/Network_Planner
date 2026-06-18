import copy
import csv
import random
from pathlib import Path

from sim.core import topology as Topology
from sim.upgrade.upgrade_manager import (
    reset_all_slots_empty,
    perform_upgrade,
    choose_upgrade_type,
)
from sim.upgrade.cost_model import compute_upgrade_costs
from sim.routing.rsa import execute_first_fit
from sim.core.traffic_generator import NetworkTrafficGenerator
from sim.core.constants import *
from sim.upgrade.opex_model import compute_network_opex_per_day
from sim.core import blocked_connection_prob_threshold_plan_checker


# =====================================================================
# Logging helpers
# =====================================================================

def _get_plan_checker_log_path() -> Path:
    return Path("results") / "plan_checker_log.csv"


def _init_plan_checker_log_if_needed():
    log_path = _get_plan_checker_log_path()
    if not log_path.exists():
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "time_day",
                "status",
                "reason",
                "baseline_bp",
                "threshold",
                "selected_links",
                "kept_links",
                "projected_cycle_opex",
                "total_capex_workforce",
                "total_downtime",
                "opex_ok",
                "capex_ok",
                "total_ok",
                "time_ok",
            ])


def _append_plan_checker_log_row(
    current_time,
    status,
    reason,
    baseline_bp,
    threshold,
    selected_links,
    kept_links,
    projected_cycle_opex="",
    total_capex_workforce="",
    total_downtime="",
    opex_ok="",
    capex_ok="",
    total_ok="",
    time_ok="",
):
    _init_plan_checker_log_if_needed()
    with open(_get_plan_checker_log_path(), "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            current_time,
            status,
            reason,
            baseline_bp,
            threshold,
            list(selected_links) if selected_links is not None else [],
            list(kept_links) if kept_links is not None else [],
            projected_cycle_opex,
            total_capex_workforce,
            total_downtime,
            opex_ok,
            capex_ok,
            total_ok,
            time_ok,
        ])


# =====================================================================
# Downtime / cost / budget helpers
# =====================================================================

def _get_upgrade_downtime_days_local(dec, algorithm_name):
    """
    Local downtime mapping.
    """
    if dec is None:
        return 0.0

    utype = dec.get("upgrade_type")

    if utype == "new_fiber_C":
        return float(t_C1)
    if utype == "new_fiber_CL":
        return float(t_CL1)
    if utype == "band_upgrade":
        return float(t_b)
    if utype == "core_upgrade":
        return float(t_3C)

    return 0.0

def _compute_cycle_budget_components(current_time, budget_params=None, budget_tracker=None):
    """
    Annual budget logic:
      - Budget grows once per year using inflation
      - Entire remaining yearly budget can be used at any cycle
      - No forced 50/50 split between upgrade cycles
      - No carry-forward across years
    """

    if budget_params is None:
        raise ValueError("budget_params cannot be None")

    if budget_tracker is None:
        raise ValueError("budget_tracker cannot be None")

    inflation_rate = float(budget_params["inflation_rate"])

    # Current year index
    year_idx = int(current_time // 365)

    # Initialize yearly budget once
    if year_idx not in budget_tracker:

        growth = (1.0 + inflation_rate) ** year_idx

        budget_tracker[year_idx] = {
            "total_budget": float(budget_params["period_total_budget_0"]) * growth,
            "opex_budget": float(budget_params["period_opex_budget_0"]) * growth,
            "capex_budget": float(budget_params["period_capex_budget_0"]) * growth,

            "used_total": 0.0,
            "used_opex": 0.0,
            "used_capex": 0.0,
        }

    year_data = budget_tracker[year_idx]

    # ============================================================
    # NEW LOGIC:
    # Entire remaining yearly budget is available
    # ============================================================

    remaining_total = (
        year_data["total_budget"]
        - year_data["used_total"]
    )

    remaining_opex = (
        year_data["opex_budget"]
        - year_data["used_opex"]
    )

    remaining_capex = (
        year_data["capex_budget"]
        - year_data["used_capex"]
    )

    print(
        f"[BUDGET] Year={year_idx}, "
        f"Used={year_data['used_total']:.2f}, "
        f"Remaining={remaining_total:.2f}"
    )

    return (
        max(remaining_opex, 0.0),
        max(remaining_capex, 0.0),
        max(remaining_total, 0.0),
    )


def _commit_budget_usage(current_time, budget_tracker, projected_cycle_opex, total_capex_workforce, label="FINAL COMMIT"):
    """
    Commit budget usage exactly once, after the final accepted plan is chosen.
    Never call this from trial checks, otherwise budget is double-counted.
    """
    if budget_tracker is None:
        return

    year_idx = int(current_time // 365)
    if year_idx not in budget_tracker:
        return

    projected_cycle_opex = float(projected_cycle_opex or 0.0)
    total_capex_workforce = float(total_capex_workforce or 0.0)
    committed = projected_cycle_opex + total_capex_workforce

    budget_tracker[year_idx]["used_total"] += committed
    budget_tracker[year_idx]["used_opex"] += projected_cycle_opex
    budget_tracker[year_idx]["used_capex"] += total_capex_workforce

    print(
        f"[BUDGET {label}] Year={year_idx}, "
        f"Committed={committed:.2f}, "
        f"New Used={budget_tracker[year_idx]['used_total']:.2f}"
    )


def _rebuild_state_from_decisions(
    base_forward,
    base_backward,
    decisions,
    current_time,
    algorithm,
):
    """
    Rebuild a clean network state by applying a candidate decision set
    on top of the current real network state.
    """
    trial_forward = copy.deepcopy(base_forward)
    trial_backward = copy.deepcopy(base_backward)
    trial_forward, trial_backward = reset_all_slots_empty(trial_forward, trial_backward)

    decisions = _normalize_upgrade_decisions(decisions)
    trial_links = list(decisions.keys())

    if trial_links:
        perform_upgrade(
            trial_links,
            decisions,
            trial_forward,
            trial_backward,
            C_BAND_SLOTS,
            TOTAL_SLOTS,
            current_time,
            algorithm,
        )

    return trial_forward, trial_backward


def _try_strengthen_kept_decisions_with_remaining_budget(
    kept_decisions,
    link_rows,
    current_time,
    current_link_status_forward,
    current_link_status_backward,
    PATHS,
    traffic_base,
    seed,
    algorithm,
    budget_params,
    budget_selection_mode="budget_aware",
    budget_tracker=None,
    require_threshold_pass=True,
):
    """
    Try to spend remaining same-year budget on stronger upgrades for the SAME kept links.

    If require_threshold_pass=True, accept a stronger step only when the trial BP
    remains below the PlanChecker threshold. This is used after the kept set already passes.

    If require_threshold_pass=False, accept a stronger step whenever budget/time
    constraints pass. This is used before fallback when the kept set fails; the
    final strengthened plan is still checked against the threshold before returning PASS.

    Rules:
      - Do not add new links here.
      - For each kept link, repeatedly try the next technology step.
      - Accept a stronger step only if budget/time constraints still pass.
      - Recheck performance to guard against unexpected simulation side effects.
      - Budget is NOT committed here; it is only checked. Commit happens once
        after the final strengthened plan is selected.
    """
    if budget_selection_mode != "budget_aware":
        return None

    if budget_tracker is None:
        return None

    strengthened = _normalize_upgrade_decisions(copy.deepcopy(kept_decisions))
    if not strengthened:
        return None

    # Prefer links that were most useful in the minimum feasible solution.
    ordered_links = [row["link"] for row in link_rows if row["link"] in strengthened]
    if not ordered_links:
        ordered_links = list(strengthened.keys())

    best_forward, best_backward = _rebuild_state_from_decisions(
        current_link_status_forward,
        current_link_status_backward,
        strengthened,
        current_time,
        algorithm,
    )

    best_bp = evaluate_blocking_probability(
        forward_status_pc=copy.deepcopy(best_forward),
        backward_status_pc=copy.deepcopy(best_backward),
        traffic_matrix_pc=traffic_base.copy(),
        PATHS=PATHS,
        start_time_pc=current_time,
        seed=seed,
    )

    (
        best_opex_ok,
        best_capex_ok,
        best_total_ok,
        best_time_ok,
        best_projected_cycle_opex,
        best_total_capex_workforce,
        best_total_downtime,
        best_cost_summary,
        best_cost_details,
    ) = _constraints_ok(
        cum_upgrade_decisions=strengthened,
        cum_forward=best_forward,
        current_time=current_time,
        current_link_status_forward=current_link_status_forward,
        algorithm_name=algorithm.name,
        budget_params=budget_params,
        budget_selection_mode=budget_selection_mode,
        budget_tracker=budget_tracker,
    )

    if not (best_opex_ok and best_capex_ok and best_total_ok and best_time_ok):
        print("⚠️ Strengthening skipped: current kept plan no longer satisfies constraints.")
        return None

    print("\n" + "=" * 80)
    print("↑ EXTRA-BUDGET STRENGTHENING MODE")
    if require_threshold_pass:
        print("   Kept plan already passes. Trying stronger upgrades using remaining annual budget.")
    else:
        print("   Kept plan fails threshold. Trying stronger upgrades BEFORE fallback.")
    print("=" * 80)

    max_accepts = max(1, 10 * len(ordered_links))
    strengthen_round = 0
    accepted_count = 0

    while accepted_count < max_accepts:
        strengthen_round += 1
        print(f"\n   🔁 Strengthening round #{strengthen_round}")
        accepted_this_round = False

        for link in ordered_links:
            # Rebuild the current accepted state, then ask for the next upgrade on that link.
            current_forward, current_backward = _rebuild_state_from_decisions(
                current_link_status_forward,
                current_link_status_backward,
                strengthened,
                current_time,
                algorithm,
            )

            next_upg = choose_upgrade_type(
                link,
                current_forward,
                current_backward,
                algorithm,
            )

            if next_upg is None or next_upg.get("upgrade_type") is None:
                continue

            trial_decisions = copy.deepcopy(strengthened)
            if link not in trial_decisions:
                trial_decisions[link] = []
            elif isinstance(trial_decisions[link], dict):
                trial_decisions[link] = [trial_decisions[link]]
            elif trial_decisions[link] is None:
                trial_decisions[link] = []

            trial_decisions[link].append(next_upg)
            trial_decisions[link] = _canonicalize_link_decisions(trial_decisions[link])
            trial_decisions = _normalize_upgrade_decisions(trial_decisions)

            # Avoid accepting a no-op/canonicalization duplicate.
            if trial_decisions == strengthened:
                continue

            trial_forward, trial_backward = _rebuild_state_from_decisions(
                current_link_status_forward,
                current_link_status_backward,
                trial_decisions,
                current_time,
                algorithm,
            )

            (
                opex_ok,
                capex_ok,
                total_ok,
                time_ok,
                projected_cycle_opex,
                total_capex_workforce,
                total_downtime,
                cost_summary,
                cost_details,
            ) = _constraints_ok(
                cum_upgrade_decisions=trial_decisions,
                cum_forward=trial_forward,
                current_time=current_time,
                current_link_status_forward=current_link_status_forward,
                algorithm_name=algorithm.name,
                budget_params=budget_params,
                budget_selection_mode=budget_selection_mode,
                budget_tracker=budget_tracker,
            )

            if not (opex_ok and capex_ok and total_ok and time_ok):
                print(
                    f"   → Cannot strengthen link {link} with {next_upg.get('upgrade_type')}: "
                    f"OPEX_OK={opex_ok}, CAPEX_OK={capex_ok}, TOTAL_OK={total_ok}, TIME_OK={time_ok}, "
                    f"downtime={total_downtime:.2f} days, "
                    f"CAPEX/WF={total_capex_workforce:.2f}, "
                    f"OPEX={projected_cycle_opex:.2f}"
                )

                continue

            trial_bp = evaluate_blocking_probability(
                forward_status_pc=copy.deepcopy(trial_forward),
                backward_status_pc=copy.deepcopy(trial_backward),
                traffic_matrix_pc=traffic_base.copy(),
                PATHS=PATHS,
                start_time_pc=current_time,
                seed=seed,
            )

            # Acceptance policy:
            #   - If the kept plan already passed the threshold, keep the final
            #     strengthened plan within the PlanChecker threshold.
            #   - If the kept plan failed and we are strengthening before fallback,
            #     do NOT require immediate BP improvement for each individual step.
            #     Accept any stronger step that satisfies budget/time constraints,
            #     then check the final strengthened plan against the threshold.
            if require_threshold_pass:
                accept_trial = trial_bp <= blocked_connection_prob_threshold_plan_checker
            else:
                accept_trial = True

            if accept_trial:
                old_link_decisions = strengthened.get(link, [])
                new_link_decisions = trial_decisions.get(link, [])

                old_downtime = best_total_downtime
                new_downtime = total_downtime

                print("\n   🔍 STRENGTHENING DETAIL")
                print(f"   Round             : {strengthen_round}")
                print(f"   Link              : {link}")
                print(f"   Previous decisions: {old_link_decisions}")
                print(f"   Added decision    : {next_upg}")
                print(f"   New decisions     : {new_link_decisions}")
                print(f"   Previous downtime : {old_downtime:.2f} days")
                print(f"   New downtime      : {new_downtime:.2f} days")
                print(f"   Downtime increase : {new_downtime - old_downtime:.2f} days")
                print(f"   Trial BP          : {trial_bp:.6f}")
                print(f"   Trial CAPEX/WF    : {total_capex_workforce:.2f}")
                print(f"   Trial OPEX        : {projected_cycle_opex:.2f}")

                # ============================================================
                # COMMIT ACCEPTED STRENGTHENING
                # ============================================================

                strengthened = trial_decisions

                best_forward = trial_forward
                best_backward = trial_backward

                best_bp = trial_bp

                best_projected_cycle_opex = projected_cycle_opex
                best_total_capex_workforce = total_capex_workforce
                best_total_downtime = total_downtime

                best_opex_ok = opex_ok
                best_capex_ok = capex_ok
                best_total_ok = total_ok
                best_time_ok = time_ok

                best_cost_summary = cost_summary
                best_cost_details = cost_details

                accepted_count += 1
                accepted_this_round = True

                print("   ✅ Strengthening accepted.")

        if not accepted_this_round:
            print("\n   ⛔ No further feasible strengthening found in this round.")
            break
    print("=" * 80)
    print(f"✓ Strengthening complete. Accepted extra steps: {accepted_count}")
    print(f"✓ Final strengthened BP: {best_bp:.6f}")
    print("=" * 80 + "\n")

    return (
        list(strengthened.keys()),
        strengthened,
        best_forward,
        best_backward,
        best_bp,
        best_projected_cycle_opex,
        best_total_capex_workforce,
        best_total_downtime,
        best_opex_ok,
        best_capex_ok,
        best_total_ok,
        best_time_ok,
        best_cost_summary,
        best_cost_details,
    )

def _compute_total_downtime(cum_upgrade_decisions, algorithm_name):
    """
    Rule per link:
      - one decision  -> use that decision time
      - multiple decisions -> use max(decision times)

    Total downtime = sum across links
    """
    total_downtime = 0.0

    for _, decs in cum_upgrade_decisions.items():
        if decs is None:
            continue

        if isinstance(decs, dict):
            decs = [decs]

        valid_decs = [d for d in decs if d is not None and d.get("upgrade_type") is not None]
        if not valid_decs:
            continue

        link_times = [
            _get_upgrade_downtime_days_local(dec, algorithm_name)
            for dec in valid_decs
        ]

        if len(link_times) == 1:
            link_downtime = link_times[0]
        else:
            link_downtime = max(link_times)

        total_downtime += link_downtime

    return total_downtime


def _compute_total_capex_workforce(cum_upgrade_decisions, algorithm_name):
    """
    Compute total equipment + workforce cost for the cumulative selected set.
    compute_upgrade_costs expects each link's decisions to be a list.
    """
    normalized = {}

    for link_id, decs in cum_upgrade_decisions.items():
        if decs is None:
            continue

        if isinstance(decs, dict):
            normalized[link_id] = [decs]
        elif isinstance(decs, list):
            valid = [d for d in decs if d is not None and d.get("upgrade_type") is not None]
            if valid:
                normalized[link_id] = valid
        else:
            raise ValueError(f"Unexpected decision format for link {link_id}: {type(decs)}")

    safe_links_for_upgrade = list(normalized.keys())

    if not safe_links_for_upgrade:
        return 0.0, {"equipment_total": 0.0, "workforce_total": 0.0}, {}

    total_cost, cost_summary, cost_details = compute_upgrade_costs(
        safe_links_for_upgrade=safe_links_for_upgrade,
        upgrade_decisions=normalized,
        algorithm_name=algorithm_name
    )

    total_capex_workforce = (
        float(cost_summary.get("equipment_total", 0.0))
        + float(cost_summary.get("workforce_total", 0.0))
    )

    return total_capex_workforce, cost_summary, cost_details


def _compute_projected_cycle_opex(cum_forward):
    """
    Compute projected OPEX for one upgrade-initiation cycle under the
    currently selected upgraded state.
    """
    daily_opex = float(compute_network_opex_per_day(cum_forward))
    period_opex = daily_opex * float(Upgrade_initiation_days)
    return daily_opex, period_opex


def _constraints_ok(
    cum_upgrade_decisions,
    cum_forward,
    current_time,
    current_link_status_forward,
    algorithm_name,
    budget_params=None,
    budget_selection_mode="budget_aware",
    budget_tracker=None
):
    """
    Constraint check used by PlanChecker.

    budget_aware:
        Checks OPEX, CAPEX+workforce, total budget, and downtime.

    budget_unaware:
        Skips all budget/cost checks completely and checks only downtime.
    """
    total_downtime = _compute_total_downtime(cum_upgrade_decisions, algorithm_name)
    time_ok = total_downtime <= Upgrade_initiation_days

    if budget_selection_mode == "budget_unaware":
        print(f"   → Constraint mode: {budget_selection_mode}")
        print("   → Budget checks skipped: budget-unaware mode")
        print(f"   → Total downtime: {total_downtime:.2f} days")
        print(f"   → Time constraint: {'PASS' if time_ok else 'FAIL'}")

        return (
            True,    # opex_ok
            True,    # capex_ok
            True,    # total_ok
            time_ok,
            0.0,     # projected_cycle_opex not used in budget-unaware mode
            0.0,     # total_capex_workforce not used in budget-unaware mode
            total_downtime,
            {"equipment_total": 0.0, "workforce_total": 0.0},
            {}
        )

    elif budget_selection_mode != "budget_aware":
        raise ValueError(f"Unknown budget_selection_mode: {budget_selection_mode}")

    total_capex_workforce, cost_summary, cost_details = _compute_total_capex_workforce(
        cum_upgrade_decisions,
        algorithm_name
    )

    daily_opex, projected_cycle_opex = _compute_projected_cycle_opex(cum_forward)

    current_opex_budget, current_capex_budget, current_total_budget = _compute_cycle_budget_components(
        current_time=current_time,
        budget_params=budget_params,
        budget_tracker=budget_tracker
    )

    total_projected_spending = projected_cycle_opex + total_capex_workforce

    opex_ok = projected_cycle_opex <= current_opex_budget
    capex_ok = total_capex_workforce <= current_capex_budget
    total_ok = total_projected_spending <= current_total_budget

    print(f"   → Constraint mode: {budget_selection_mode}")
    print(f"   → Daily OPEX under current upgraded state: {daily_opex:.2f}")
    print(f"   → Projected OPEX for this {Upgrade_initiation_days}-day cycle: {projected_cycle_opex:.2f}")
    print(f"   → OPEX budget for cycle: {current_opex_budget:.2f}")
    print(f"   → CAPEX + Workforce spent: {total_capex_workforce:.2f}")
    print(f"   → CAPEX + Workforce budget for cycle: {current_capex_budget:.2f}")
    print(f"   → Total projected spending this cycle: {total_projected_spending:.2f}")
    print(f"   → Total budget for cycle: {current_total_budget:.2f}")
    print(f"   → Total downtime: {total_downtime:.2f} days")
    print(f"   → OPEX constraint: {'PASS' if opex_ok else 'FAIL'}")
    print(f"   → CAPEX/WF constraint: {'PASS' if capex_ok else 'FAIL'}")
    print(f"   → TOTAL budget constraint: {'PASS' if total_ok else 'FAIL'}")
    print(f"   → Time constraint: {'PASS' if time_ok else 'FAIL'}")

    return (
        opex_ok,
        capex_ok,
        total_ok,
        time_ok,
        projected_cycle_opex,
        total_capex_workforce,
        total_downtime,
        cost_summary,
        cost_details
    )


def _canonicalize_link_decisions(dec_list):
    """
    Clean one link's ordered decision list.

    Rules:
      1) remove invalid decisions
      2) merge:
            new_fiber_C(f) + band_upgrade(f) -> new_fiber_CL(f)
         on the same fiber
      3) if a core_upgrade exists, drop everything before the first core_upgrade
         and keep only from core_upgrade onward
    """
    if not dec_list:
        return []

    cleaned = [d for d in dec_list if d is not None and d.get("upgrade_type") is not None]
    if not cleaned:
        return []

    # If core_upgrade exists, everything before it becomes irrelevant
    core_idx = None
    for idx, d in enumerate(cleaned):
        if d.get("upgrade_type") == "core_upgrade":
            core_idx = idx
            break

    if core_idx is not None:
        cleaned = cleaned[core_idx:]

    # Merge new_fiber_C + band_upgrade on same fiber into new_fiber_CL
    merged = []
    i = 0
    while i < len(cleaned):
        d1 = cleaned[i]

        if i + 1 < len(cleaned):
            d2 = cleaned[i + 1]

            if (
                d1.get("upgrade_type") == "new_fiber_C"
                and d2.get("upgrade_type") == "band_upgrade"
                and d1.get("fiber_id") == d2.get("fiber_id")
            ):
                merged.append({
                    "upgrade_type": "new_fiber_CL",
                    "fiber_id": d1.get("fiber_id"),
                    "core_type": d1.get("core_type"),
                })
                i += 2
                continue

        merged.append(d1)
        i += 1

    return merged


def _normalize_upgrade_decisions(upgrade_decisions):
    """
    Normalize and canonicalize per-link decision lists.
    """
    normalized = {}

    for link, decs in upgrade_decisions.items():
        if decs is None:
            continue

        if isinstance(decs, dict):
            decs = [decs]

        canonical = _canonicalize_link_decisions(list(decs))

        if canonical:
            normalized[link] = canonical

    return normalized


def _compute_single_link_capex_workforce(link_id, link_decisions, algorithm_name):
    """
    Compute equipment + workforce for one link's already-decided upgrade list.
    """
    total_cost, cost_summary, cost_details = compute_upgrade_costs(
        safe_links_for_upgrade=[link_id],
        upgrade_decisions={link_id: link_decisions},
        algorithm_name=algorithm_name
    )

    return (
        float(cost_summary.get("equipment_total", 0.0))
        + float(cost_summary.get("workforce_total", 0.0))
    )

def _get_link_heaviest_upgrade_level(decs):
    """
    Classify a link by the heaviest upgrade present in its decision list.
    Lower value = lighter technology, preferred in fallback.
      band_upgrade -> 1
      new_fiber_C  -> 2
      new_fiber_CL -> 3
      core_upgrade -> 4
    """
    if not decs:
        return 999

    if isinstance(decs, dict):
        decs = [decs]

    utypes = [d.get("upgrade_type") for d in decs if d is not None and d.get("upgrade_type") is not None]

    if "core_upgrade" in utypes:
        return 4
    if "band_upgrade" in utypes:
        return 3
    if "new_fiber_CL" in utypes:
        return 2
    if "new_fiber_C" in utypes:
        return 1

    return 999


def _get_link_max_downtime(decs, algorithm_name):
    """
    For one link, return the max downtime among its decisions.
    Used only for fallback ranking.
    """
    if not decs:
        return float("inf")

    if isinstance(decs, dict):
        decs = [decs]

    vals = [
        _get_upgrade_downtime_days_local(d, algorithm_name)
        for d in decs
        if d is not None and d.get("upgrade_type") is not None
    ]

    return max(vals) if vals else float("inf")


def _run_fallback_light_links_first(
    link_rows,
    current_time,
    current_link_status_forward,
    current_link_status_backward,
    PATHS,
    traffic_base,
    seed,
    algorithm,
    budget_params,
    budget_selection_mode="budget_aware",
    budget_tracker=None,
):
    """
    Fallback strategy:
      - do NOT change per-link decisions
      - prefer links with lighter upgrade types
      - among same type, prefer lower downtime, then lower cost
      - greedily keep maximum number of feasible links
      - run final performance check
    """
    fallback_rows = sorted(
        link_rows,
        key=lambda x: (
            _get_link_heaviest_upgrade_level(x["decisions"]),
            _get_link_max_downtime(x["decisions"], algorithm.name),
            x["cost"],
        )
    )

    print("\n" + "=" * 80)
    print("↺ FALLBACK MODE: TRYING LIGHTER-UPGRADE LINKS FIRST")
    print("=" * 80)
    for row in fallback_rows:
        print(
            f"   link {row['link']}: "
            f"heaviest={_get_link_heaviest_upgrade_level(row['decisions'])}, "
            f"max_dt={_get_link_max_downtime(row['decisions'], algorithm.name):.2f}, "
            f"cost={row['cost']:.2f}"
        )
    print("=" * 80 + "\n")

    fallback_kept = {}

    for row in fallback_rows:
        trial_kept = copy.deepcopy(fallback_kept)
        trial_kept[row["link"]] = row["decisions"]

        trial_links = list(trial_kept.keys())

        trial_forward = copy.deepcopy(current_link_status_forward)
        trial_backward = copy.deepcopy(current_link_status_backward)
        trial_forward, trial_backward = reset_all_slots_empty(trial_forward, trial_backward)

        perform_upgrade(
            trial_links,
            trial_kept,
            trial_forward,
            trial_backward,
            C_BAND_SLOTS,
            TOTAL_SLOTS,
            current_time,
            algorithm
        )

        (
            opex_ok,
            capex_ok,
            total_ok,
            time_ok,
            projected_cycle_opex,
            total_capex_workforce,
            total_downtime,
            cost_summary,
            cost_details
        ) = _constraints_ok(
            cum_upgrade_decisions=trial_kept,
            cum_forward=trial_forward,
            current_time=current_time,
            current_link_status_forward=current_link_status_forward,
            algorithm_name=algorithm.name,
            budget_params=budget_params,
            budget_selection_mode=budget_selection_mode,
            budget_tracker=budget_tracker
        )

        if budget_selection_mode == "budget_aware":
            keep_link = opex_ok and capex_ok and total_ok and time_ok
        elif budget_selection_mode == "budget_unaware":
            keep_link = time_ok
        else:
            raise ValueError(f"Unknown mode: {budget_selection_mode}")

        if keep_link:
            fallback_kept = trial_kept
            print(f"   → FALLBACK KEEP link {row['link']}")
        else:
            print(f"   → FALLBACK REJECT link {row['link']}")

    if not fallback_kept:
        print("❌ FALLBACK: no links remain after budget/time pruning.")
        return False, [], {}, None, None, None, None, None, None, None

    fallback_links = list(fallback_kept.keys())

    final_forward = copy.deepcopy(current_link_status_forward)
    final_backward = copy.deepcopy(current_link_status_backward)
    final_forward, final_backward = reset_all_slots_empty(final_forward, final_backward)

    perform_upgrade(
        fallback_links,
        fallback_kept,
        final_forward,
        final_backward,
        C_BAND_SLOTS,
        TOTAL_SLOTS,
        current_time,
        algorithm
    )

    final_bp = evaluate_blocking_probability(
        forward_status_pc=copy.deepcopy(final_forward),
        backward_status_pc=copy.deepcopy(final_backward),
        traffic_matrix_pc=traffic_base.copy(),
        PATHS=PATHS,
        start_time_pc=current_time,
        seed=seed
    )

    print(f"✓ FALLBACK final blocking probability = {final_bp:.6f}")

    (
        opex_ok,
        capex_ok,
        total_ok,
        time_ok,
        projected_cycle_opex,
        total_capex_workforce,
        total_downtime,
        cost_summary,
        cost_details
    ) = _constraints_ok(
        cum_upgrade_decisions=fallback_kept,
        cum_forward=final_forward,
        current_time=current_time,
        current_link_status_forward=current_link_status_forward,
        algorithm_name=algorithm.name,
        budget_params=budget_params,
        budget_selection_mode=budget_selection_mode,
        budget_tracker=budget_tracker
    )

    return (
        final_bp <= blocked_connection_prob_threshold_plan_checker,
        fallback_links,
        fallback_kept,
        final_bp,
        projected_cycle_opex,
        total_capex_workforce,
        total_downtime,
        opex_ok,
        capex_ok,
        total_ok,
        time_ok,
    )


# =====================================================================
# Blocking simulation
# =====================================================================

def run_planchecker_simulation(
    forward_status_pc,
    backward_status_pc,
    traffic_matrix_pc,
    PATHS,
    start_time_pc,
    seed
):
    """
    Boolean pass/fail wrapper retained for compatibility.
    """
    bp = evaluate_blocking_probability(
        forward_status_pc=forward_status_pc,
        backward_status_pc=backward_status_pc,
        traffic_matrix_pc=traffic_matrix_pc,
        PATHS=PATHS,
        start_time_pc=start_time_pc,
        seed=seed
    )
    return bp <= blocked_connection_prob_threshold_plan_checker


def evaluate_blocking_probability(
    forward_status_pc,
    backward_status_pc,
    traffic_matrix_pc,
    PATHS,
    start_time_pc,
    seed
):
    """
    Runs a projection-window traffic simulation and returns blocking probability.
    """
    testing_topology = Topology.TOPOLOGY

    start_pc = start_time_pc + Upgrade_initiation_days
    current_time_pc = start_pc
    SIM_END_pc = start_pc + Traffic_growth_days

    pc_rng = random.Random(seed + 5555)

    tg_pc = NetworkTrafficGenerator(
        number_of_nodes=Topology.N,
        datarates=DATARATE,
        lambda_0=lambda_0,
        mean_holding_time=MEAN_HOLDING_TIME,
        Current_global_time=start_pc,
        rng=pc_rng
    )

    ALL_DEMANDS_PLAN_CHECKER = []
    blocked_pc = 0
    total_pc = 0
    next_growth_time_pc = start_pc + Traffic_growth_days

    while True:
        while current_time_pc >= next_growth_time_pc:
            for i in range(Topology.N):
                for j in range(Topology.N):
                    if i != j:
                        growth = 1 + pc_rng.uniform(0, alpha / 100)
                        traffic_matrix_pc[i][j] *= growth
                        traffic_matrix_pc[i][j] *= (1 + delta / 100)
            next_growth_time_pc += Traffic_growth_days

        src_pc, dest_pc, rate_pc = tg_pc.generate_connection_data()
        lam = traffic_matrix_pc[src_pc][dest_pc]
        arrival_time_pc, holding_time_pc = tg_pc.get_connection(lam)
        current_time_pc = arrival_time_pc

        if current_time_pc >= SIM_END_pc:
            break

        departure_time_pc = arrival_time_pc + holding_time_pc
        total_pc += 1

        index_to_remove_plan_checker = []

        for idp, conn_pc in enumerate(ALL_DEMANDS_PLAN_CHECKER):
            if conn_pc.departure_time_pc <= current_time_pc:
                index_to_remove_plan_checker.append(idp)

        for idp in reversed(index_to_remove_plan_checker):
            conn_pc = ALL_DEMANDS_PLAN_CHECKER[idp]

            for i in range(len(conn_pc.path_pc) - 1):
                src_depart = conn_pc.path_pc[i]
                dest_depart = conn_pc.path_pc[i + 1]

                link_pc = conn_pc.link_ids_pc[i]
                fiber_pc = conn_pc.fibers_used_pc[i]
                core_pc = conn_pc.cores_used_pc[i]

                if src_depart < dest_depart:
                    if link_pc in forward_status_pc and fiber_pc in forward_status_pc[link_pc]:
                        fwd_obj = forward_status_pc[link_pc][fiber_pc]

                        if isinstance(fwd_obj, dict) and "slots" in fwd_obj:
                            for s in range(conn_pc.fs_pc, conn_pc.fs_pc + conn_pc.sw_pc):
                                fwd_obj["slots"][s] = 0
                        else:
                            if core_pc in fwd_obj:
                                for s in range(conn_pc.fs_pc, conn_pc.fs_pc + conn_pc.sw_pc):
                                    fwd_obj[core_pc]["slots"][s] = 0
                else:
                    if link_pc in backward_status_pc and fiber_pc in backward_status_pc[link_pc]:
                        bwd_obj = backward_status_pc[link_pc][fiber_pc]

                        if isinstance(bwd_obj, dict) and "slots" in bwd_obj:
                            for s in range(conn_pc.fs_pc, conn_pc.fs_pc + conn_pc.sw_pc):
                                bwd_obj["slots"][s] = 0
                        else:
                            if core_pc in bwd_obj:
                                for s in range(conn_pc.fs_pc, conn_pc.fs_pc + conn_pc.sw_pc):
                                    bwd_obj[core_pc]["slots"][s] = 0

            del ALL_DEMANDS_PLAN_CHECKER[idp]

        mf_pc, fs_pc, sw_pc, path_pc, fibers_used_pc, cores_used_pc, link_ids_pc, attempted_paths_info_pc = execute_first_fit(
            src=src_pc,
            dest=dest_pc,
            datarate=rate_pc,
            arrival_time=arrival_time_pc,
            departure_time=departure_time_pc,
            link_status_forward=forward_status_pc,
            link_status_backward=backward_status_pc,
            topology=testing_topology,
            PATHS=PATHS
        )

        if mf_pc == float("inf") or fs_pc == float("inf"):
            blocked_pc += 1
        else:
            ALL_DEMANDS_PLAN_CHECKER.append(
                ConnectionData_plan_checker(
                    path_pc, link_ids_pc, fs_pc, sw_pc, mf_pc,
                    arrival_time_pc, holding_time_pc, departure_time_pc,
                    rate_pc, fibers_used_pc, cores_used_pc
                )
            )

    blocking_probability_pc = (blocked_pc / total_pc) if total_pc > 0 else 1.0

    print(f"   → Total requests tested: {total_pc}")
    print(f"   → Blocked connections: {blocked_pc}")
    print(f"   → Blocking probability: {blocking_probability_pc:.4f}")

    return blocking_probability_pc


def plan_checker(
        safe_links,
        upgrade_decisions,
        current_time,
        PATHS,
        current_traffic,
        current_link_status_forward,
        current_link_status_backward,
        seed,
        algorithm,
        budget_params=None,
        budget_selection_mode="budget_aware",
        budget_tracker=None,
):
    print("\n▶ RUNNING PLAN CHECKER...")

    # ------------------------------------------------------------
    # Step 0: normalize decisions from main.py
    # ------------------------------------------------------------
    full_decisions = _normalize_upgrade_decisions(upgrade_decisions)

    print(f"PLAN CHECKER MODE: {budget_selection_mode}")

    if not full_decisions:
        print("❌ No valid upgrade decisions received.")
        _append_plan_checker_log_row(
            current_time=current_time,
            status="FAIL",
            reason="No valid upgrade decisions received",
            baseline_bp="",
            threshold=blocked_connection_prob_threshold_plan_checker,
            selected_links=[],
            kept_links=[]
        )
        return False, [], {}

    # Keep input ordering from main.py
    full_links = [lid for lid in safe_links if lid in full_decisions]
    if not full_links:
        full_links = list(full_decisions.keys())

    # ------------------------------------------------------------
    # Step 1: create cumulative state from initial decisions
    # ------------------------------------------------------------
    cum_forward = copy.deepcopy(current_link_status_forward)
    cum_backward = copy.deepcopy(current_link_status_backward)
    cum_forward, cum_backward = reset_all_slots_empty(cum_forward, cum_backward)

    perform_upgrade(
        full_links,
        full_decisions,
        cum_forward,
        cum_backward,
        C_BAND_SLOTS,
        TOTAL_SLOTS,
        current_time,
        algorithm
    )
    print("✓ Applied full candidate upgrade set from main.py")

    traffic_base = current_traffic.copy()
    steps = int(Upgrade_initiation_days / Traffic_growth_days)
    growth_factor_pc = ((1 + alpha / 100) * (1 + delta / 100)) ** steps
    traffic_base *= growth_factor_pc

    print(f"✓ Traffic projected +{Upgrade_initiation_days} days ahead")

    # cumulative decisions that may grow in progressive rounds
    cum_upgrade_decisions = copy.deepcopy(full_decisions)

    # ------------------------------------------------------------
    # Step 2: progressive multi-round escalation if performance fails
    # ------------------------------------------------------------
    round_idx = 0

    while True:
        baseline_bp = evaluate_blocking_probability(
            forward_status_pc=copy.deepcopy(cum_forward),
            backward_status_pc=copy.deepcopy(cum_backward),
            traffic_matrix_pc=traffic_base.copy(),
            PATHS=PATHS,
            start_time_pc=current_time,
            seed=seed
        )

        print(f"✓ Current cumulative blocking probability = {baseline_bp:.6f}")

        if baseline_bp <= blocked_connection_prob_threshold_plan_checker:
            print("\n" + "-" * 80)
            print("✓ PLAN CHECKER: PERFORMANCE PASSES")
            print("-" * 80)
            print(f"→ Current day        : {current_time}")
            print(f"→ Blocking           : {baseline_bp:.6f}")
            print(f"→ Threshold          : {blocked_connection_prob_threshold_plan_checker:.6f}")
            print(f"→ Candidate links    : {list(cum_upgrade_decisions.keys())}")
            print("-" * 80 + "\n")
            break

        round_idx += 1
        print("\n" + "=" * 80)
        print(f"❌ PLAN CHECKER PERFORMANCE FAILED — STARTING ROUND #{round_idx} ESCALATION")
        print("=" * 80)
        print(f"→ Current simulation day           : {current_time}")
        print(f"→ Blocking threshold               : {blocked_connection_prob_threshold_plan_checker:.6f}")
        print(f"→ Achieved blocking                : {baseline_bp:.6f}")
        print("→ Escalating next upgrade decision for all eligible links")
        print("=" * 80 + "\n")

        round_upgrades = {}

        for link in full_links:
            upg = choose_upgrade_type(
                link,
                cum_forward,
                cum_backward,
                algorithm
            )

            if upg is None or upg.get("upgrade_type") is None:
                continue

            round_upgrades[link] = upg

        if not round_upgrades:
            print("⚠️ No further upgrades possible on any selected link.")
            _append_plan_checker_log_row(
                current_time=current_time,
                status="FAIL",
                reason="Performance failed and no further upgrades possible",
                baseline_bp=baseline_bp,
                threshold=blocked_connection_prob_threshold_plan_checker,
                selected_links=list(cum_upgrade_decisions.keys()),
                kept_links=[]
            )
            return False, list(cum_upgrade_decisions.keys()), cum_upgrade_decisions

        print(f"✓ Selected next upgrades for {len(round_upgrades)} links in escalation round #{round_idx}")

        # Append then canonicalize immediately
        for link, upg in round_upgrades.items():
            if link not in cum_upgrade_decisions:
                cum_upgrade_decisions[link] = []
            elif isinstance(cum_upgrade_decisions[link], dict):
                cum_upgrade_decisions[link] = [cum_upgrade_decisions[link]]
            elif cum_upgrade_decisions[link] is None:
                cum_upgrade_decisions[link] = []

            cum_upgrade_decisions[link].append(upg)
            cum_upgrade_decisions[link] = _canonicalize_link_decisions(cum_upgrade_decisions[link])

        # rebuild cumulative network state cleanly from scratch
        cum_upgrade_decisions = _normalize_upgrade_decisions(cum_upgrade_decisions)

        cum_forward = copy.deepcopy(current_link_status_forward)
        cum_backward = copy.deepcopy(current_link_status_backward)
        cum_forward, cum_backward = reset_all_slots_empty(cum_forward, cum_backward)

        perform_upgrade(
            list(cum_upgrade_decisions.keys()),
            cum_upgrade_decisions,
            cum_forward,
            cum_backward,
            C_BAND_SLOTS,
            TOTAL_SLOTS,
            current_time,
            algorithm
        )

    # ------------------------------------------------------------
    # Step 3: canonicalize once more before contribution scoring
    # ------------------------------------------------------------
    cum_upgrade_decisions = _normalize_upgrade_decisions(cum_upgrade_decisions)

    passed_links = list(cum_upgrade_decisions.keys())
    passed_decisions = copy.deepcopy(cum_upgrade_decisions)

    link_rows = []

    if budget_selection_mode == "budget_aware":
        total_cost_all, total_summary_all, total_cost_details_all = compute_upgrade_costs(
            safe_links_for_upgrade=passed_links,
            upgrade_decisions=passed_decisions,
            algorithm_name=algorithm.name
        )
    else:
        total_cost_all, total_summary_all, total_cost_details_all = 0.0, {}, {}

    for link in passed_links:
        reduced_decisions = copy.deepcopy(passed_decisions)
        removed_link_decisions = reduced_decisions.pop(link, None)

        if not removed_link_decisions:
            continue

        reduced_links = list(reduced_decisions.keys())

        reduced_forward = copy.deepcopy(current_link_status_forward)
        reduced_backward = copy.deepcopy(current_link_status_backward)
        reduced_forward, reduced_backward = reset_all_slots_empty(reduced_forward, reduced_backward)

        if reduced_links:
            perform_upgrade(
                reduced_links,
                reduced_decisions,
                reduced_forward,
                reduced_backward,
                C_BAND_SLOTS,
                TOTAL_SLOTS,
                current_time,
                algorithm
            )

        bp_without_link = evaluate_blocking_probability(
            forward_status_pc=copy.deepcopy(reduced_forward),
            backward_status_pc=copy.deepcopy(reduced_backward),
            traffic_matrix_pc=traffic_base.copy(),
            PATHS=PATHS,
            start_time_pc=current_time,
            seed=seed
        )

        contribution = bp_without_link - baseline_bp

        if budget_selection_mode == "budget_aware":
            if link in total_cost_details_all:
                link_cost = float(total_cost_details_all[link].get("link_total_cost", 0.0))
            else:
                link_cost = _compute_single_link_capex_workforce(
                    link_id=link,
                    link_decisions=removed_link_decisions,
                    algorithm_name=algorithm.name
                )

            if link_cost <= 0:
                continue

            efficiency = contribution / link_cost
        else:
            # In budget-unaware mode, cost is not used for selection.
            link_cost = 0.0
            efficiency = contribution

        link_rows.append({
            "link": link,
            "decisions": removed_link_decisions,
            "bp_without_link": bp_without_link,
            "contribution": contribution,
            "cost": link_cost,
            "efficiency": efficiency,
        })

    if not link_rows:
        print("❌ No valid link-level contribution rows could be computed.")
        _append_plan_checker_log_row(
            current_time=current_time,
            status="FAIL",
            reason="No valid link-level contribution rows",
            baseline_bp=baseline_bp,
            threshold=blocked_connection_prob_threshold_plan_checker,
            selected_links=passed_links,
            kept_links=[]
        )
        return False, [], {}

    # ------------------------------------------------------------
    # Step 4: performance-first ranking
    # ------------------------------------------------------------
    # Primary objective: maximize lifetime/performance contribution.
    # Cost is still enforced by the budget constraints and is used only
    # as a secondary tie-breaker when two links have similar contribution.
    link_rows.sort(
        key=lambda x: (x["contribution"], -x["cost"]),
        reverse=True
    )

    print(f"✓ Ranked links using PERFORMANCE-FIRST mode (budget mode={budget_selection_mode}):")
    for row in link_rows:
        print(
            f"   link {row['link']}: "
            f"bp_without={row['bp_without_link']:.6f}, "
            f"contribution={row['contribution']:.6f}, "
            f"cost={row['cost']:.2f}"
        )

    # ------------------------------------------------------------
    # Step 5: keep links greedily under budget/time
    # ------------------------------------------------------------
    kept_decisions = {}

    for row in link_rows:
        trial_kept = copy.deepcopy(kept_decisions)
        trial_kept[row["link"]] = row["decisions"]

        trial_links = list(trial_kept.keys())

        trial_forward = copy.deepcopy(current_link_status_forward)
        trial_backward = copy.deepcopy(current_link_status_backward)
        trial_forward, trial_backward = reset_all_slots_empty(trial_forward, trial_backward)

        perform_upgrade(
            trial_links,
            trial_kept,
            trial_forward,
            trial_backward,
            C_BAND_SLOTS,
            TOTAL_SLOTS,
            current_time,
            algorithm
        )

        (
            opex_ok,
            capex_ok,
            total_ok,
            time_ok,
            projected_cycle_opex,
            total_capex_workforce,
            total_downtime,
            cost_summary,
            cost_details
        ) = _constraints_ok(
            cum_upgrade_decisions=trial_kept,
            cum_forward=trial_forward,
            current_time=current_time,
            current_link_status_forward=current_link_status_forward,
            algorithm_name=algorithm.name,
            budget_params=budget_params,
            budget_selection_mode=budget_selection_mode,
            budget_tracker=budget_tracker

        )

        if budget_selection_mode == "budget_aware":
            keep_link = opex_ok and capex_ok and total_ok and time_ok

        elif budget_selection_mode == "budget_unaware":
            keep_link = time_ok

        else:
            raise ValueError(f"Unknown mode: {budget_selection_mode}")

        if keep_link:
            kept_decisions = trial_kept
            print(f"   → Keeping link {row['link']}")
        else:
            print(f"\n   ❌ REJECTING LINK {row['link']}")
            print(f"      → Cost        : {row['cost']:.2f}")
            print(f"      → Efficiency  : {row['efficiency']:.10f}")
            print(f"      → OPEX OK     : {opex_ok}")
            print(f"      → CAPEX OK    : {capex_ok}")
            print(f"      → TOTAL OK    : {total_ok}")
            print(f"      → TIME OK     : {time_ok}")

    if not kept_decisions:
        print("❌ No links remain after budget/time pruning.")
        _append_plan_checker_log_row(
            current_time=current_time,
            status="FAIL",
            reason="All links rejected by budget/time pruning",
            baseline_bp=baseline_bp,
            threshold=blocked_connection_prob_threshold_plan_checker,
            selected_links=passed_links,
            kept_links=[]
        )
        return False, [], {}

    # ------------------------------------------------------------
    # Step 6: final performance check on kept set
    # ------------------------------------------------------------
    kept_links = list(kept_decisions.keys())

    final_forward = copy.deepcopy(current_link_status_forward)
    final_backward = copy.deepcopy(current_link_status_backward)
    final_forward, final_backward = reset_all_slots_empty(final_forward, final_backward)

    perform_upgrade(
        kept_links,
        kept_decisions,
        final_forward,
        final_backward,
        C_BAND_SLOTS,
        TOTAL_SLOTS,
        current_time,
        algorithm
    )

    final_bp = evaluate_blocking_probability(
        forward_status_pc=copy.deepcopy(final_forward),
        backward_status_pc=copy.deepcopy(final_backward),
        traffic_matrix_pc=traffic_base.copy(),
        PATHS=PATHS,
        start_time_pc=current_time,
        seed=seed
    )

    print(f"✓ Final kept-set blocking probability = {final_bp:.6f}")

    (
        opex_ok,
        capex_ok,
        total_ok,
        time_ok,
        projected_cycle_opex,
        total_capex_workforce,
        total_downtime,
        cost_summary,
        cost_details
    ) = _constraints_ok(
        cum_upgrade_decisions=kept_decisions,
        cum_forward=final_forward,
        current_time=current_time,
        current_link_status_forward=current_link_status_forward,
        algorithm_name=algorithm.name,
        budget_params=budget_params,
        budget_selection_mode=budget_selection_mode,
        budget_tracker=budget_tracker
    )

    if final_bp <= blocked_connection_prob_threshold_plan_checker:
        print("✅ PLAN CHECKER PASSED — minimum feasible kept set satisfies performance and constraints.")

        # ------------------------------------------------------------
        # Step 7: use remaining same-year budget to strengthen upgrades
        # ------------------------------------------------------------
        if budget_selection_mode == "budget_aware":
            strengthened_result = _try_strengthen_kept_decisions_with_remaining_budget(
                kept_decisions=kept_decisions,
                link_rows=link_rows,
                current_time=current_time,
                current_link_status_forward=current_link_status_forward,
                current_link_status_backward=current_link_status_backward,
                PATHS=PATHS,
                traffic_base=traffic_base,
                seed=seed,
                algorithm=algorithm,
                budget_params=budget_params,
                budget_selection_mode=budget_selection_mode,
                budget_tracker=budget_tracker,
                require_threshold_pass=True,
            )

            if strengthened_result is not None:
                (
                    kept_links,
                    kept_decisions,
                    final_forward,
                    final_backward,
                    final_bp,
                    projected_cycle_opex,
                    total_capex_workforce,
                    total_downtime,
                    opex_ok,
                    capex_ok,
                    total_ok,
                    time_ok,
                    cost_summary,
                    cost_details,
                ) = strengthened_result

        print("✅ PLAN CHECKER PASSED — final strengthened plan selected.")

        _append_plan_checker_log_row(
            current_time=current_time,
            status="PASS",
            reason="Final plan satisfies performance and uses remaining feasible annual budget",
            baseline_bp=final_bp,
            threshold=blocked_connection_prob_threshold_plan_checker,
            selected_links=passed_links,
            kept_links=kept_links,
            projected_cycle_opex=projected_cycle_opex,
            total_capex_workforce=total_capex_workforce,
            total_downtime=total_downtime,
            opex_ok=opex_ok,
            capex_ok=capex_ok,
            total_ok=total_ok,
            time_ok=time_ok,
        )

        if budget_selection_mode == "budget_aware":
            _commit_budget_usage(
                current_time=current_time,
                budget_tracker=budget_tracker,
                projected_cycle_opex=projected_cycle_opex,
                total_capex_workforce=total_capex_workforce,
                label="FINAL COMMIT"
            )

        return True, list(kept_decisions.keys()), kept_decisions

    print("❌ Final kept set is under budget/time but does not satisfy threshold.")

    # ------------------------------------------------------------
    # Step 7B: Before fallback, try stronger upgrades on the kept set.
    # This handles cases where pruning links satisfies budget/time but
    # makes performance fail, while additional budget is still available.
    # ------------------------------------------------------------
    if budget_selection_mode == "budget_aware":
        print("↗ Trying stronger upgrades on kept set before fallback...")

        strengthened_result = _try_strengthen_kept_decisions_with_remaining_budget(
            kept_decisions=kept_decisions,
            link_rows=link_rows,
            current_time=current_time,
            current_link_status_forward=current_link_status_forward,
            current_link_status_backward=current_link_status_backward,
            PATHS=PATHS,
            traffic_base=traffic_base,
            seed=seed,
            algorithm=algorithm,
            budget_params=budget_params,
            budget_selection_mode=budget_selection_mode,
            budget_tracker=budget_tracker,
            require_threshold_pass=False,
        )

        if strengthened_result is not None:
            (
                strengthened_links,
                strengthened_decisions,
                strengthened_forward,
                strengthened_backward,
                strengthened_bp,
                strengthened_projected_cycle_opex,
                strengthened_total_capex_workforce,
                strengthened_total_downtime,
                strengthened_opex_ok,
                strengthened_capex_ok,
                strengthened_total_ok,
                strengthened_time_ok,
                strengthened_cost_summary,
                strengthened_cost_details,
            ) = strengthened_result

            if strengthened_bp <= blocked_connection_prob_threshold_plan_checker:
                print("✅ PLAN CHECKER PASSED — strengthened kept set satisfies performance and constraints.")

                _append_plan_checker_log_row(
                    current_time=current_time,
                    status="PASS",
                    reason="Strengthened kept set satisfies performance before fallback",
                    baseline_bp=strengthened_bp,
                    threshold=blocked_connection_prob_threshold_plan_checker,
                    selected_links=passed_links,
                    kept_links=strengthened_links,
                    projected_cycle_opex=strengthened_projected_cycle_opex,
                    total_capex_workforce=strengthened_total_capex_workforce,
                    total_downtime=strengthened_total_downtime,
                    opex_ok=strengthened_opex_ok,
                    capex_ok=strengthened_capex_ok,
                    total_ok=strengthened_total_ok,
                    time_ok=strengthened_time_ok,
                )

                _commit_budget_usage(
                    current_time=current_time,
                    budget_tracker=budget_tracker,
                    projected_cycle_opex=strengthened_projected_cycle_opex,
                    total_capex_workforce=strengthened_total_capex_workforce,
                    label="PRE-FALLBACK STRENGTHEN COMMIT"
                )

                return True, strengthened_links, strengthened_decisions

            print(
                f"↘ Strengthening improved/used feasible options but still failed threshold: "
                f"BP={strengthened_bp:.6f}, threshold={blocked_connection_prob_threshold_plan_checker:.6f}"
            )

    print("❌ PlanChecker failed for this selected path/link set.")
    print("↩ Returning failure to main.py so main can try the next congested path rank.")

    _append_plan_checker_log_row(
        current_time=current_time,
        status="FAIL",
        reason="Selected path/link set failed after pruning and strengthening; returning to main for next path-rank retry",
        baseline_bp=final_bp,
        threshold=blocked_connection_prob_threshold_plan_checker,
        selected_links=passed_links,
        kept_links=kept_links,
        projected_cycle_opex=projected_cycle_opex,
        total_capex_workforce=total_capex_workforce,
        total_downtime=total_downtime,
        opex_ok=opex_ok,
        capex_ok=capex_ok,
        total_ok=total_ok,
        time_ok=time_ok,
    )

    return False, list(kept_decisions.keys()), kept_decisions

    _append_plan_checker_log_row(
        current_time=current_time,
        status="FAIL",
        reason="Final kept set and fallback lighter-link set both fail performance",
        baseline_bp=final_bp,
        threshold=blocked_connection_prob_threshold_plan_checker,
        selected_links=passed_links,
        kept_links=kept_links,
        projected_cycle_opex=projected_cycle_opex,
        total_capex_workforce=total_capex_workforce,
        total_downtime=total_downtime,
        opex_ok=opex_ok,
        capex_ok=capex_ok,
        total_ok=total_ok,
        time_ok=time_ok,
    )
    return False, list(kept_decisions.keys()), kept_decisions


