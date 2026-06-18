# sim/upgrade/opex_model.py
from __future__ import annotations
from typing import Dict, Tuple, Any

from sim.core.constants import O_C1, O_CL1, O_C3, O_CL3
from sim.core import topology as Topology   # ✅ FIX: missing import


def _is_single_core_fiber(fiber_obj: Any) -> bool:
    return isinstance(fiber_obj, dict) and ("slots" in fiber_obj) and ("cores" in fiber_obj)


def _is_multi_core_fiber(fiber_obj: Any) -> bool:
    if not isinstance(fiber_obj, dict):
        return False
    int_core_keys = [k for k, v in fiber_obj.items() if isinstance(k, int) and isinstance(v, dict)]
    return len(int_core_keys) > 0


def _fiber_is_lit_and_has_L(fiber_obj: dict) -> Tuple[bool, bool]:
    if _is_single_core_fiber(fiber_obj):
        if not fiber_obj.get("lit", False):
            return False, False
        bands = set(fiber_obj.get("bands", []))
        return True, ("L" in bands)

    if _is_multi_core_fiber(fiber_obj):
        is_lit = False
        has_L = False
        for core_id, core in fiber_obj.items():
            if not isinstance(core_id, int) or not isinstance(core, dict):
                continue
            if core.get("lit", False):
                is_lit = True
                bands = set(core.get("bands", []))
                if "L" in bands:
                    has_L = True
        return is_lit, has_L

    return False, False

def count_network_fibers_by_technology(link_status_forward: dict):
    """
    Returns network-wide counts of lit fibers by technology class:
      SC_C, SC_CL, MC_C, MC_CL
    """
    sc_c = sc_cl = mc_c = mc_cl = 0

    if not isinstance(link_status_forward, dict):
        return {
            "SC_C": 0,
            "SC_CL": 0,
            "MC_C": 0,
            "MC_CL": 0,
        }

    for link_id, fibers in link_status_forward.items():
        if not isinstance(fibers, dict):
            continue

        for fiber_id, fiber_obj in fibers.items():
            if not isinstance(fiber_obj, dict):
                continue

            cores = _fiber_core_count(fiber_obj)
            if cores <= 0:
                continue

            is_lit, has_L = _fiber_is_lit_and_has_L(fiber_obj)
            if not is_lit:
                continue

            if cores == 1:
                if has_L:
                    sc_cl += 1
                else:
                    sc_c += 1
            else:
                if has_L:
                    mc_cl += 1
                else:
                    mc_c += 1

    return {
        "SC_C": sc_c,
        "SC_CL": sc_cl,
        "MC_C": mc_c,
        "MC_CL": mc_cl,
    }


def count_link_fibers_by_technology(link_status_forward: dict, link_id: int):
    """
    Returns per-link counts of lit fibers by technology class.
    """
    sc_c = sc_cl = mc_c = mc_cl = 0

    fibers = link_status_forward.get(link_id, {})
    if not isinstance(fibers, dict):
        return {
            "SC_C": 0,
            "SC_CL": 0,
            "MC_C": 0,
            "MC_CL": 0,
        }

    for fiber_id, fiber_obj in fibers.items():
        if not isinstance(fiber_obj, dict):
            continue

        cores = _fiber_core_count(fiber_obj)
        if cores <= 0:
            continue

        is_lit, has_L = _fiber_is_lit_and_has_L(fiber_obj)
        if not is_lit:
            continue

        if cores == 1:
            if has_L:
                sc_cl += 1
            else:
                sc_c += 1
        else:
            if has_L:
                mc_cl += 1
            else:
                mc_c += 1

    return {
        "SC_C": sc_c,
        "SC_CL": sc_cl,
        "MC_C": mc_c,
        "MC_CL": mc_cl,
    }


def _fiber_core_count(fiber_obj: dict) -> int:
    if _is_single_core_fiber(fiber_obj):
        return int(fiber_obj.get("cores", 1))

    if _is_multi_core_fiber(fiber_obj):
        for k, core in fiber_obj.items():
            if isinstance(k, int) and isinstance(core, dict) and "cores" in core:
                return int(core["cores"])
        return len([k for k in fiber_obj.keys() if isinstance(k, int)])

    return 0


def compute_network_opex_per_day(link_status_forward: dict) -> float:
    """
    Correct OPEX:
      per-link fiber count × (cost/km/day) × (link length)
    """

    total_opex = 0.0

    if not isinstance(link_status_forward, dict):
        return total_opex

    for link_id, fibers in link_status_forward.items():

        if not isinstance(fibers, dict):
            continue

        # -------------------------------------
        # Get link length
        # -------------------------------------
        src = dst = None
        for i in range(len(Topology.LINK_INDEX)):
            for j in range(len(Topology.LINK_INDEX[i])):
                if Topology.LINK_INDEX[i][j] == link_id:
                    src, dst = i, j
                    break
            if src is not None:
                break

        if src is None or dst is None:
            raise ValueError(f"Invalid LINK_INDEX mapping for link_id {link_id}")

        L_e = Topology.TOPOLOGY_LINK_LENGTHS[src][dst]

        # -------------------------------------
        # Count fibers for this link
        # -------------------------------------
        sc_c = sc_cl = mc_c = mc_cl = 0

        for fiber_id, fiber_obj in fibers.items():

            if not isinstance(fiber_obj, dict):
                continue

            cores = _fiber_core_count(fiber_obj)
            if cores <= 0:
                continue

            is_lit, has_L = _fiber_is_lit_and_has_L(fiber_obj)
            if not is_lit:
                continue

            if cores == 1:
                if has_L:
                    sc_cl += 1
                else:
                    sc_c += 1
            else:
                if has_L:
                    mc_cl += 1
                else:
                    mc_c += 1

        # -------------------------------------
        # Apply OPEX (KEY FIX)
        # -------------------------------------
        link_opex = (
            (sc_c * O_C1) +
            (sc_cl * O_CL1) +
            (mc_c * O_C3) +
            (mc_cl * O_CL3)
        ) * L_e

        total_opex += link_opex

    return total_opex


def compute_running_opex_interval(link_status_forward: dict, delta_days: float) -> float:
    if delta_days <= 0:
        return 0.0

    return compute_network_opex_per_day(link_status_forward) * float(delta_days)

