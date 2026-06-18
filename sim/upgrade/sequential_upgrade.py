from typing import Any

from sim.routing.rsa import execute_first_fit, ConnectionData
from sim.routing.shortest_path import build_k_shortest_paths
from sim.upgrade.upgrade_manager import perform_upgrade
from sim.core.constants import C_BAND_SLOTS, TOTAL_SLOTS, t_b, t_CL1, t_3C
from sim.upgrade.upgrade_manager import remove_links_from_topology
from sim.core import topology as Topology


def _conn_uses_link_fiber(conn, link_id: int, fiber_id: int) -> bool:
    """
    Returns True if this connection uses (link_id, fiber_id) on ANY hop.
    Assumes conn.link_ids and conn.fibers_used align per hop.
    """
    for hop in range(len(conn.link_ids)):
        if conn.link_ids[hop] == link_id and conn.fibers_used[hop] == fiber_id:
            return True
    return False


def start_downtime_upgrade_for_link(
    scope: str,                      # "fiber" or "link"
    link_id: int,
    decision: dict,                  # SINGLE decision dict (one action)
    pending_downtime_actions: list,  # kept for signature compatibility; may be unused
    ALL_DEMANDS: list,
    link_status_forward: dict,
    link_status_backward: dict,
    working_topology: list,
    removed_links_info: dict,
    current_time: float,
    blocked_datarate_rerouting: float,
    accepted_datarate_rerouting: float,
    arrived_datarate_rerouting: float,
    total_network_cost: float,
    upgrade_cost_timeline: list,     # kept for signature compatibility; NOT USED
    algorithm: Any,
):
    """
    Runs ONE downtime ACTION (NO COST HERE):
      1) find connections using this link/fiber
      2) free its spectrum and remove those connections
      3) perform the upgrade (apply decision)
      4) enforce downtime (remove link or fiber)
      5) recompute PATHS
      6) reroute the freed connections
    """

    if not isinstance(decision, dict):
        raise ValueError("start_downtime_upgrade_for_link expects decision as a SINGLE dict")

    upgrade_type = decision.get("upgrade_type")

    if scope == "fiber" and upgrade_type not in ("band_upgrade", "new_fiber_CL"):
        raise ValueError(
            f"scope='fiber' expects upgrade_type in ('band_upgrade', 'new_fiber_CL'), got {upgrade_type}"
        )

    if scope == "link" and upgrade_type != "core_upgrade":
        raise ValueError(f"scope='link' expects upgrade_type='core_upgrade', got {upgrade_type}")

    # -------------------------------------------------
    # 1) Identify connections to reroute
    # -------------------------------------------------
    indices_to_remove = []
    connections_to_reroute = []

    if scope == "link":
        for idx, conn in enumerate(ALL_DEMANDS):
            if link_id in conn.link_ids:
                indices_to_remove.append(idx)
                connections_to_reroute.append(conn)

    elif scope == "fiber":
        fiber_id = decision.get("fiber_id", None)
        if fiber_id is None:
            raise ValueError(f"{upgrade_type} decision must include 'fiber_id'")

        for idx, conn in enumerate(ALL_DEMANDS):
            if _conn_uses_link_fiber(conn, link_id, fiber_id):
                indices_to_remove.append(idx)
                connections_to_reroute.append(conn)

    else:
        raise ValueError(f"Unknown scope: {scope}")

    print(f">>> {len(connections_to_reroute)} connections will be rerouted due to {scope}-downtime on link {link_id}")

    # -------------------------------------------------
    # 2) Free spectrum for affected connections
    # -------------------------------------------------
    for conn in connections_to_reroute:
        for i in range(len(conn.path) - 1):
            src_depart = conn.path[i]
            dest_depart = conn.path[i + 1]

            link = conn.link_ids[i]
            fiber = conn.fibers_used[i]
            core = conn.cores_used[i]

            # -------- FORWARD -----------
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

            # -------- BACKWARD ----------
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

    # -------------------------------------------------
    # 3) Remove those connections from ALL_DEMANDS
    # -------------------------------------------------
    for idx in reversed(indices_to_remove):
        conn = ALL_DEMANDS[idx]
        arrived_datarate_rerouting += conn.datarate
        accepted_datarate_rerouting += conn.datarate
        del ALL_DEMANDS[idx]

    # -------------------------------------------------
    # 4) Apply upgrade in link status (ONLY this decision)
    # -------------------------------------------------
    perform_upgrade(
        [link_id],
        {link_id: [decision]},
        link_status_forward,
        link_status_backward,
        C_BAND_SLOTS,
        TOTAL_SLOTS,
        current_time,
        algorithm
    )

    # -------------------------------------------------
    # 5) Enforce downtime
    # -------------------------------------------------
    if scope == "link":
        working_topology, link_status_forward, link_status_backward, removed_links_info = \
            remove_links_from_topology(
                topology=working_topology,
                topology_link_length=Topology.TOPOLOGY_LINK_LENGTHS,
                link_index=Topology.LINK_INDEX,
                links_to_remove={link_id: "core_upgrade"},
                link_status_forward=link_status_forward,
                link_status_backward=link_status_backward,
                removed_links_info=removed_links_info,
                current_time=current_time,
                algorithm_name=getattr(algorithm, "name", str(algorithm)),
            )

    elif scope == "fiber":
        fiber_id = decision.get("fiber_id", None)
        if fiber_id is None:
            raise ValueError(f"{upgrade_type} decision must include 'fiber_id' for fiber-level downtime")

        if link_id not in link_status_forward or link_id not in link_status_backward:
            raise ValueError(f"Fiber downtime: link {link_id} not found in status dicts")

        stored_forward_fiber = link_status_forward[link_id].get(fiber_id)
        stored_backward_fiber = link_status_backward[link_id].get(fiber_id)

        if stored_forward_fiber is None or stored_backward_fiber is None:
            raise ValueError(f"Fiber downtime: fiber {fiber_id} not found on link {link_id}")

        # remove fiber from live structure
        link_status_forward[link_id].pop(fiber_id, None)
        link_status_backward[link_id].pop(fiber_id, None)

        # choose downtime duration by upgrade type
        if upgrade_type == "band_upgrade":
            restore_time = current_time + float(t_b)
        elif upgrade_type == "new_fiber_CL":
            restore_time = current_time + float(t_CL1)
        else:
            raise ValueError(f"Unsupported fiber downtime upgrade_type: {upgrade_type}")

        removed_links_info[("fiber", link_id, fiber_id)] = {
            "scope": "fiber",
            "link_id": link_id,
            "fiber_id": fiber_id,
            "start_time": current_time,
            "restore_time": restore_time,
            "upgrade_type": upgrade_type,
            "stored_forward_fiber": stored_forward_fiber,
            "stored_backward_fiber": stored_backward_fiber,
        }

    # -------------------------------------------------
    # 6) Recompute PATHS
    # -------------------------------------------------
    PATHS = build_k_shortest_paths(working_topology)
    if not PATHS:
        raise ValueError("PATHS empty after downtime enforcement")

    # -------------------------------------------------
    # 7) Reroute freed connections
    # -------------------------------------------------
    for conn in connections_to_reroute:
        mf_new, fs_new, sw_new, path_new, fibers_used_rerouting, cores_used_rerouting, link_ids_rerouting, attempted_paths_info = \
            execute_first_fit(
                src=conn.path[0],
                dest=conn.path[-1],
                datarate=conn.datarate,
                arrival_time=current_time,
                departure_time=current_time + conn.holding_time,
                link_status_forward=link_status_forward,
                link_status_backward=link_status_backward,
                topology=working_topology,
                PATHS=PATHS
            )

        if mf_new != float("inf") and fs_new != float("inf"):
            new_conn = ConnectionData(
                path=path_new,
                link_ids=link_ids_rerouting,
                fs_index=fs_new,
                slice_window_size=sw_new,
                mf_index=mf_new,
                arrival_time=current_time,
                holding_time=conn.holding_time,
                departure_time=current_time + conn.holding_time,
                datarate=conn.datarate,
                fibers_used=fibers_used_rerouting,
                cores_used=cores_used_rerouting
            )
            ALL_DEMANDS.append(new_conn)
            accepted_datarate_rerouting += conn.datarate
            arrived_datarate_rerouting += conn.datarate
        else:
            blocked_datarate_rerouting += conn.datarate

    return (
        working_topology,
        link_status_forward,
        link_status_backward,
        removed_links_info,
        ALL_DEMANDS,
        blocked_datarate_rerouting,
        accepted_datarate_rerouting,
        arrived_datarate_rerouting,
        total_network_cost,
        PATHS,
    )

def start_combined_fiber_downtime_job(
    link_id: int,
    decisions: list,                  # list of fiber-level decision dicts
    ALL_DEMANDS: list,
    link_status_forward: dict,
    link_status_backward: dict,
    working_topology: list,
    removed_links_info: dict,
    current_time: float,
    blocked_datarate_rerouting: float,
    accepted_datarate_rerouting: float,
    arrived_datarate_rerouting: float,
    total_network_cost: float,
    upgrade_cost_timeline: list,
    algorithm: Any,
):
    """
    Execute ALL fiber-level upgrades for one link together.

    Rule:
      - apply all fiber upgrades now
      - all affected fibers are unavailable together
      - restore all of them at the SAME time
      - common restore time = current_time + max(decision downtimes)
    """

    if not isinstance(decisions, list) or not decisions:
        raise ValueError("start_combined_fiber_downtime_job expects non-empty list of decisions")

    for d in decisions:
        if not isinstance(d, dict):
            raise ValueError("Each item in decisions must be a dict")
        if d.get("upgrade_type") not in ("band_upgrade", "new_fiber_CL"):
            raise ValueError(
                f"Combined fiber job only supports band_upgrade/new_fiber_CL, got {d.get('upgrade_type')}"
            )
        if d.get("fiber_id") is None:
            raise ValueError("Fiber-level decision must include fiber_id")

    # -------------------------------------------------
    # 1) Identify all affected fibers and connections
    # -------------------------------------------------
    affected_fibers = {d["fiber_id"] for d in decisions}

    indices_to_remove = []
    connections_to_reroute = []

    for idx, conn in enumerate(ALL_DEMANDS):
        affected = False
        for fiber_id in affected_fibers:
            if _conn_uses_link_fiber(conn, link_id, fiber_id):
                affected = True
                break
        if affected:
            indices_to_remove.append(idx)
            connections_to_reroute.append(conn)

    print(
        f">>> {len(connections_to_reroute)} connections will be rerouted due to "
        f"COMBINED fiber-downtime on link {link_id}, fibers={sorted(affected_fibers)}"
    )

    # -------------------------------------------------
    # 2) Free spectrum for affected connections
    # -------------------------------------------------
    for conn in connections_to_reroute:
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

    # -------------------------------------------------
    # 3) Remove affected connections from ALL_DEMANDS
    # -------------------------------------------------
    for idx in reversed(indices_to_remove):
        conn = ALL_DEMANDS[idx]
        arrived_datarate_rerouting += conn.datarate
        accepted_datarate_rerouting += conn.datarate
        del ALL_DEMANDS[idx]

    # -------------------------------------------------
    # 4) Apply ALL fiber decisions immediately
    # -------------------------------------------------
    perform_upgrade(
        [link_id],
        {link_id: decisions},
        link_status_forward,
        link_status_backward,
        C_BAND_SLOTS,
        TOTAL_SLOTS,
        current_time,
        algorithm
    )

    # -------------------------------------------------
    # 5) Remove all affected fibers from live state
    # -------------------------------------------------
    stored_fibers = []

    for d in decisions:
        fiber_id = d["fiber_id"]

        if link_id not in link_status_forward or link_id not in link_status_backward:
            raise ValueError(f"Combined fiber downtime: link {link_id} not found in status dicts")

        stored_forward_fiber = link_status_forward[link_id].get(fiber_id)
        stored_backward_fiber = link_status_backward[link_id].get(fiber_id)

        if stored_forward_fiber is None or stored_backward_fiber is None:
            raise ValueError(f"Combined fiber downtime: fiber {fiber_id} not found on link {link_id}")

        link_status_forward[link_id].pop(fiber_id, None)
        link_status_backward[link_id].pop(fiber_id, None)

        stored_fibers.append((fiber_id, stored_forward_fiber, stored_backward_fiber, d.get("upgrade_type")))

    # -------------------------------------------------
    # 6) Common restore time = MAX of all fiber decision times
    # -------------------------------------------------
    times = []
    for d in decisions:
        u = d.get("upgrade_type")
        if u == "band_upgrade":
            times.append(float(t_b))
        elif u == "new_fiber_CL":
            times.append(float(t_CL1))
        else:
            times.append(0.0)

    common_restore_time = current_time + (max(times) if times else 0.0)

    for fiber_id, stored_forward_fiber, stored_backward_fiber, upgrade_type in stored_fibers:
        removed_links_info[("fiber", link_id, fiber_id)] = {
            "scope": "fiber",
            "link_id": link_id,
            "fiber_id": fiber_id,
            "start_time": current_time,
            "restore_time": common_restore_time,
            "upgrade_type": upgrade_type,
            "stored_forward_fiber": stored_forward_fiber,
            "stored_backward_fiber": stored_backward_fiber,
        }

    # -------------------------------------------------
    # 7) Recompute PATHS
    # -------------------------------------------------
    PATHS = build_k_shortest_paths(working_topology)
    if not PATHS:
        raise ValueError("PATHS empty after combined fiber downtime enforcement")

    # -------------------------------------------------
    # 8) Reroute freed connections
    # -------------------------------------------------
    for conn in connections_to_reroute:
        mf_new, fs_new, sw_new, path_new, fibers_used_rerouting, cores_used_rerouting, link_ids_rerouting, attempted_paths_info = \
            execute_first_fit(
                src=conn.path[0],
                dest=conn.path[-1],
                datarate=conn.datarate,
                arrival_time=current_time,
                departure_time=current_time + conn.holding_time,
                link_status_forward=link_status_forward,
                link_status_backward=link_status_backward,
                topology=working_topology,
                PATHS=PATHS
            )

        if mf_new != float("inf") and fs_new != float("inf"):
            new_conn = ConnectionData(
                path=path_new,
                link_ids=link_ids_rerouting,
                fs_index=fs_new,
                slice_window_size=sw_new,
                mf_index=mf_new,
                arrival_time=current_time,
                holding_time=conn.holding_time,
                departure_time=current_time + conn.holding_time,
                datarate=conn.datarate,
                fibers_used=fibers_used_rerouting,
                cores_used=cores_used_rerouting
            )
            ALL_DEMANDS.append(new_conn)
            accepted_datarate_rerouting += conn.datarate
            arrived_datarate_rerouting += conn.datarate
        else:
            blocked_datarate_rerouting += conn.datarate

    return (
        working_topology,
        link_status_forward,
        link_status_backward,
        removed_links_info,
        ALL_DEMANDS,
        blocked_datarate_rerouting,
        accepted_datarate_rerouting,
        arrived_datarate_rerouting,
        total_network_cost,
        PATHS,
    )


def start_link_downtime_job(
    link_id: int,
    decisions: list,                  # list of decision dicts
    pending_downtime_actions: list,   # kept for signature compatibility; not used here
    ALL_DEMANDS: list,
    link_status_forward: dict,
    link_status_backward: dict,
    working_topology: list,
    removed_links_info: dict,
    current_time: float,
    blocked_datarate_rerouting: float,
    accepted_datarate_rerouting: float,
    arrived_datarate_rerouting: float,
    total_network_cost: float,
    upgrade_cost_timeline: list,      # kept for signature compatibility; NOT USED
    algorithm: Any,
):
    """
    Runs ONE LINK-level downtime JOB (NO COST HERE):
      1) find all connections using link_id
      2) free spectrum and remove them from ALL_DEMANDS
      3) perform ALL upgrades in 'decisions'
      4) enforce LINK downtime by removing link from topology + status dicts
      5) recompute PATHS
      6) reroute freed connections
    """

    if not isinstance(decisions, list) or not decisions:
        raise ValueError("start_link_downtime_job expects 'decisions' as a non-empty list of dicts")

    for d in decisions:
        if not isinstance(d, dict):
            raise ValueError("Each item in 'decisions' must be a dict")

    print(f">>> LINK downtime job on link {link_id}: {decisions}")

    # 1) Identify connections to reroute
    indices_to_remove = []
    connections_to_reroute = []

    for idx, conn in enumerate(ALL_DEMANDS):
        if link_id in conn.link_ids:
            indices_to_remove.append(idx)
            connections_to_reroute.append(conn)

    print(f">>> {len(connections_to_reroute)} connections will be rerouted due to LINK-downtime on link {link_id}")

    # 2) Free spectrum
    for conn in connections_to_reroute:
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

    # 3) Remove from ALL_DEMANDS
    for idx in reversed(indices_to_remove):
        conn = ALL_DEMANDS[idx]
        arrived_datarate_rerouting += conn.datarate
        accepted_datarate_rerouting += conn.datarate
        del ALL_DEMANDS[idx]

    # 4) Apply ALL upgrades for this job
    perform_upgrade(
        [link_id],
        {link_id: decisions},
        link_status_forward,
        link_status_backward,
        C_BAND_SLOTS,
        TOTAL_SLOTS,
        current_time,
        algorithm
    )

    # 5) Enforce LINK downtime
    # Rule:
    #   - one decision  -> use that decision time
    #   - multiple decisions -> use max(decision times)

    downtime_candidates = []

    for d in decisions:
        u = d.get("upgrade_type")

        if u == "core_upgrade":
            downtime_candidates.append(float(t_3C))
        elif u == "band_upgrade":
            downtime_candidates.append(float(t_b))
        elif u == "new_fiber_CL":
            downtime_candidates.append(float(t_CL1))
        elif u == "new_fiber_C":
            downtime_candidates.append(0.0)

    if not downtime_candidates:
        total_downtime = 0.0
    elif len(downtime_candidates) == 1:
        total_downtime = downtime_candidates[0]
    else:
        total_downtime = max(downtime_candidates)

    # Remove the link using normal link-removal path
    working_topology, link_status_forward, link_status_backward, removed_links_info = \
        remove_links_from_topology(
            topology=working_topology,
            topology_link_length=Topology.TOPOLOGY_LINK_LENGTHS,
            link_index=Topology.LINK_INDEX,
            links_to_remove={link_id: "core_upgrade"},
            link_status_forward=link_status_forward,
            link_status_backward=link_status_backward,
            removed_links_info=removed_links_info,
            current_time=current_time,
            algorithm_name=getattr(algorithm, "name", str(algorithm)),
        )

    # Override restore time so combined link job uses SUM of all downtimes
    if ("link", link_id) in removed_links_info:
        removed_links_info[("link", link_id)]["restore_time"] = current_time + total_downtime
    elif link_id in removed_links_info:
        removed_links_info[link_id]["restore_time"] = current_time + total_downtime
    else:
        raise KeyError(f"Could not find removed_links_info entry for link {link_id}")

    # 6) Recompute PATHS
    PATHS = build_k_shortest_paths(working_topology)
    if not PATHS:
        raise ValueError("PATHS empty after link downtime enforcement")

    # 7) Reroute freed connections
    for conn in connections_to_reroute:
        mf_new, fs_new, sw_new, path_new, fibers_used_rerouting, cores_used_rerouting, link_ids_rerouting, attempted_paths_info = \
            execute_first_fit(
                src=conn.path[0],
                dest=conn.path[-1],
                datarate=conn.datarate,
                arrival_time=current_time,
                departure_time=current_time + conn.holding_time,
                link_status_forward=link_status_forward,
                link_status_backward=link_status_backward,
                topology=working_topology,
                PATHS=PATHS
            )

        if mf_new != float("inf") and fs_new != float("inf"):
            new_conn = ConnectionData(
                path=path_new,
                link_ids=link_ids_rerouting,
                fs_index=fs_new,
                slice_window_size=sw_new,
                mf_index=mf_new,
                arrival_time=current_time,
                holding_time=conn.holding_time,
                departure_time=current_time + conn.holding_time,
                datarate=conn.datarate,
                fibers_used=fibers_used_rerouting,
                cores_used=cores_used_rerouting
            )
            ALL_DEMANDS.append(new_conn)
            accepted_datarate_rerouting += conn.datarate
            arrived_datarate_rerouting += conn.datarate
        else:
            blocked_datarate_rerouting += conn.datarate

    return (
        working_topology,
        link_status_forward,
        link_status_backward,
        removed_links_info,
        ALL_DEMANDS,
        blocked_datarate_rerouting,
        accepted_datarate_rerouting,
        arrived_datarate_rerouting,
        total_network_cost,
        PATHS,
    )

