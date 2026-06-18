# sim/upgrade/fallback_light_fiber_sym.py

from typing import Dict, Any, List, Tuple, Optional


def _is_singlecore_fiber_obj(obj: Any) -> bool:
    return isinstance(obj, dict) and isinstance(obj.get("slots", None), list)


def _get_link_core_type_from_forward(
    link_status_forward: Dict[int, Dict[int, Any]],
    link_id: int
) -> int:
    """
    Returns 1 if link appears to be single-core format,
    Returns 3 if link appears to be three-core format.
    Uses forward direction only.
    """
    fwd_fibers = link_status_forward.get(link_id, {})
    if not isinstance(fwd_fibers, dict) or not fwd_fibers:
        return 1

    # Look at fiber 0 if present, else first fiber
    sample = fwd_fibers.get(0)
    if sample is None:
        sample = next(iter(fwd_fibers.values()))

    # single-core: fiber dict has "slots"
    if _is_singlecore_fiber_obj(sample):
        return 1

    # three-core: fiber dict is keyed by core_id dicts
    return 3


def _find_first_dark_fiber_forward_only(
    link_status_forward: Dict[int, Dict[int, Any]],
    link_id: int,
) -> Optional[int]:
    """
    Find a dark fiber_id in FORWARD direction.
    Supports BOTH formats:
      - single-core: fiber_obj["lit"] == False
      - three-core: all cores have lit == False
    """
    fwd_fibers = link_status_forward.get(link_id, {})
    if not isinstance(fwd_fibers, dict):
        return None

    for fiber_id, fiber_obj in fwd_fibers.items():
        # single-core
        if _is_singlecore_fiber_obj(fiber_obj):
            if not bool(fiber_obj.get("lit", False)):
                return fiber_id

        # three-core (fiber_obj is dict keyed by core_id)
        elif isinstance(fiber_obj, dict):
            any_lit = False
            for core_id, core_info in fiber_obj.items():
                if isinstance(core_id, int) and isinstance(core_info, dict):
                    if bool(core_info.get("lit", False)):
                        any_lit = True
                        break
            if not any_lit:
                return fiber_id

    return None


def build_symmetric_new_fiber_actions_all_links(
    LINKS: int,
    link_status_forward: Dict[int, Dict[int, Any]],
    max_new_fibers_total: Optional[int] = None,
) -> Dict[int, List[Dict[str, Any]]]:
    """
    For ALL links, choose one dark fiber from FORWARD direction and create decision:
        {"upgrade_type":"new_fiber_C", "fiber_id": <that fiber>, "core_type": 1|3}
    """
    actions: Dict[int, List[Dict[str, Any]]] = {}
    count = 0

    for link_id in range(LINKS):
        fiber_id = _find_first_dark_fiber_forward_only(link_status_forward, link_id)
        if fiber_id is None:
            continue

        core_type = _get_link_core_type_from_forward(link_status_forward, link_id)

        actions.setdefault(link_id, []).append({
            "upgrade_type": "new_fiber_C",
            "fiber_id": fiber_id,
            "core_type": core_type
        })
        count += 1

        if max_new_fibers_total is not None and count >= max_new_fibers_total:
            break

    return actions


def apply_symmetric_light_fiber_fallback(
    *,
    LINKS: int,
    working_topology,
    PATHS,
    link_status_forward: Dict[int, Dict[int, Any]],
    link_status_backward: Dict[int, Dict[int, Any]],
    Current_global_time: float,
    C_BAND_SLOTS: int,
    TOTAL_SLOTS: int,
    perform_upgrade_fn,
    build_k_paths_fn=None,
    max_new_fibers_total: Optional[int] = None,
) -> Tuple[
    Dict[int, Dict[int, Any]],
    Dict[int, Dict[int, Any]],
    Any,
    Any,
    Dict[int, List[Dict[str, Any]]]
]:
    actions = build_symmetric_new_fiber_actions_all_links(
        LINKS=LINKS,
        link_status_forward=link_status_forward,
        max_new_fibers_total=max_new_fibers_total,
    )

    if not actions:
        return link_status_forward, link_status_backward, working_topology, PATHS, {}

    immediate_links = sorted(actions.keys())

    perform_upgrade_fn(
        immediate_links,
        actions,
        link_status_forward,
        link_status_backward,
        C_BAND_SLOTS,
        TOTAL_SLOTS,
        Current_global_time
    )

    if build_k_paths_fn is not None:
        PATHS = build_k_paths_fn(working_topology)

    return link_status_forward, link_status_backward, working_topology, PATHS, actions
