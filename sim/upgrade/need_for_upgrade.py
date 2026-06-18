import copy, math, numpy as np, random
from sim.core import topology as Topology
from sim.upgrade.upgrade_manager import *
from sim.routing.rsa import execute_first_fit
from sim.core.traffic_generator import NetworkTrafficGenerator
from sim.core.constants import *

def check_need_for_upgrade(current_time, PATHS, current_traffic, current_link_status_forward, current_link_status_backward,
                           seed):
    """
    Check if network upgrade is needed by projecting traffic 3 months ahead
    and testing for 30 days to measure blocking probability.
    
    Algorithm:
    1. Project traffic 3 months forward (deterministic growth)
    2. Simulate 30 days starting at month 3
    3. Return True if blocking probability exceeds threshold
    
    Args:
        current_time: Current simulation time (days)
        PATHS: Pre-computed k-shortest paths dictionary
        current_traffic: Current traffic matrix (numpy array)
        current_link_status_forward: Forward link state (dict)
        current_link_status_backward: Backward link state (dict)
        seed: Random seed for reproducible projection
    
    Returns:
        bool: True if upgrade needed, False otherwise
    """

    print("\nCHECKING NEED FOR UPGRADE (NEXT 6 MONTHS)")

    # ---- Traffic COPY (safe) ----
    traffic_nu = current_traffic.copy()

    # ---- Link status COPIES (SAFE: PlanChecker must not affect live network) ----
    forward_nu = copy.deepcopy(current_link_status_forward)
    backward_nu = copy.deepcopy(current_link_status_backward)

    forward_nu, backward_nu = reset_all_slots_empty(forward_nu, backward_nu)

    # --------------------------------------------------
    # 1️⃣ Scale traffic forward 3 months (deterministic)
    # --------------------------------------------------
    projection_period_days_nu = Upgrade_initiation_days
    steps_nu = int(projection_period_days_nu / Traffic_growth_days)
    growth_factor_nu = ((1 + alpha / 100) * (1 + delta / 100)) ** steps_nu

    print(f"Traffic growth factor for 6 months = {growth_factor_nu:.3f}")

    traffic_nu *= growth_factor_nu

    # --------------------------------------------------
    # 2️⃣ Setup 30-day projection window
    # --------------------------------------------------
    # test_duration_days_nu = 30
    current_time_nu = current_time + Upgrade_initiation_days     # start of projection
    SIM_END_nu = current_time_nu + Traffic_growth_days

    nu_rng = random.Random(seed + 9999)  # separate stream; any constant offset is fine

    tg_nu = NetworkTrafficGenerator(
        number_of_nodes=Topology.N,
        datarates=DATARATE,
        lambda_0=lambda_0,
        mean_holding_time=MEAN_HOLDING_TIME,
        Current_global_time=current_time_nu,
        rng=nu_rng
    )

    blocked_connections_nu = 0
    total_connections_nu = 0
    ALL_DEMANDS_NEED_FOR_UPGRADE =  []
    blocking_probability_nu = 0
    
    # Minimum connections to test before considering blocking probability
    # MIN_CONNECTIONS_FOR_BLOCKING_CHECK = 100

    # --------------------------------------------------
    # NEW: Correct periodic growth scheduling
    # --------------------------------------------------
    next_growth_time_nu = current_time_nu + Traffic_growth_days

    # --------------------------------------------------
    # 3️⃣ Begin 30-day simulation
    # --------------------------------------------------
    while True:
        # ---- Apply traffic growth exactly every Traffic_growth_days ----
        while current_time_nu >= next_growth_time_nu:
            # Apply per-node traffic growth (alpha)
            for i in range(Topology.N):
                for j in range(Topology.N):
                    if i != j:
                        growth = 1 + nu_rng.uniform(0, alpha / 100)
                        traffic_nu[i][j] *= growth

            # Apply global traffic growth (delta)
            for i in range(Topology.N):
                for j in range(Topology.N):
                    if i != j:
                        growth_factor_nu = (1 +  delta / 100)
                        traffic_nu[i][j] *= growth_factor_nu

            next_growth_time_nu += Traffic_growth_days

        # ---- New request ----
        src_nu, dest_nu, rate_nu = tg_nu.generate_connection_data()
        lambda_sd_nu = traffic_nu[src_nu][dest_nu]

        if lambda_sd_nu <= 0:
            raise ValueError("Invalid lambda_sd inside need-for-upgrade")

        arrival_time_nu, holding_time_nu = tg_nu.get_connection(lambda_sd_nu)
        current_time_nu = arrival_time_nu

        # end test window
        if current_time_nu >= SIM_END_nu:
            break

        departure_time_nu = arrival_time_nu + holding_time_nu
        total_connections_nu += 1

        # Remove expired connections
        index_to_remove_need_upgrade = []

        # -------------------------------------------------
        # 1) Identify expired connections
        # -------------------------------------------------
        for idn, conn_nu in enumerate(ALL_DEMANDS_NEED_FOR_UPGRADE):
            if conn_nu.departure_time_nu <= current_time_nu:
                index_to_remove_need_upgrade.append(idn)

        for idn in reversed(index_to_remove_need_upgrade):
            conn_nu = ALL_DEMANDS_NEED_FOR_UPGRADE[idn]

            # ========= FREE SPECTRUM (shape-safe) ========
            for i in range(len(conn_nu.path_nu) - 1):
                src_depart = conn_nu.path_nu[i]
                dest_depart = conn_nu.path_nu[i + 1]

                link_nu = conn_nu.link_ids_nu[i]
                fiber_nu = conn_nu.fibers_used_nu[i]
                core_nu = conn_nu.cores_used_nu[i]

                # -------- FORWARD -----------
                if src_depart < dest_depart:
                    if link_nu in forward_nu and fiber_nu in forward_nu[link_nu]:
                        fwd_obj = forward_nu[link_nu][fiber_nu]

                        # single-core layout: has "slots"
                        if isinstance(fwd_obj, dict) and "slots" in fwd_obj:
                            for s in range(conn_nu.fs_nu, conn_nu.fs_nu + conn_nu.sw_nu):
                                fwd_obj["slots"][s] = 0

                        # 3-core layout: dict keyed by core index
                        else:
                            if core_nu in fwd_obj:
                                for s in range(conn_nu.fs_nu, conn_nu.fs_nu + conn_nu.sw_nu):
                                    fwd_obj[core_nu]["slots"][s] = 0

                # -------- BACKWARD ----------
                else:
                    if link_nu in backward_nu and fiber_nu in backward_nu[link_nu]:
                        bwd_obj = backward_nu[link_nu][fiber_nu]

                        # single-core layout
                        if isinstance(bwd_obj, dict) and "slots" in bwd_obj:
                            for s in range(conn_nu.fs_nu, conn_nu.fs_nu + conn_nu.sw_nu):
                                bwd_obj["slots"][s] = 0

                        # 3-core layout
                        else:
                            if core_nu in bwd_obj:
                                for s in range(conn_nu.fs_nu, conn_nu.fs_nu + conn_nu.sw_nu):
                                    bwd_obj[core_nu]["slots"][s] = 0

            # ====== Finally remove the connection ======
            del ALL_DEMANDS_NEED_FOR_UPGRADE[idn]

        # ---- RSA on temporary copies ----
        mf_nu, fs_nu, sw_nu, path_nu, fibers_used_nu, cores_used_nu, link_ids_nu, attempted_paths_info_nu = execute_first_fit(
            src=src_nu,
            dest=dest_nu,
            datarate=rate_nu,
            arrival_time=arrival_time_nu,
            departure_time=departure_time_nu,
            link_status_forward=forward_nu,
            link_status_backward=backward_nu,
            topology=Topology.TOPOLOGY,
            PATHS=PATHS
        )

        # ---- Blocking check ----
        if mf_nu == float("inf") or fs_nu == float("inf"):
            blocked_connections_nu += 1
            blocking_probability_nu = blocked_connections_nu/total_connections_nu
            
            # Exit simulation if blocking threshold exceeded (after minimum sample size)
            if (blocking_probability_nu > blocked_connection_prob_threshold_need_for_upgrade):
                break

        else:
            ALL_DEMANDS_NEED_FOR_UPGRADE.append(ConnectionData_need_upgrade(
                path_nu, link_ids_nu, fs_nu, sw_nu, mf_nu,
                arrival_time_nu, holding_time_nu, departure_time_nu,
                rate_nu, fibers_used_nu, cores_used_nu
            ))

    # --------------------------------------------------
    # 4️⃣ Print results and determine upgrade decision
    # --------------------------------------------------
    print(f" → Total connections tested: {total_connections_nu}")
    print(f" → Blocked connections: {blocked_connections_nu}")
    
    # Handle edge case: no connections tested
    if total_connections_nu == 0:
        print("⚠️  Warning: No connections tested in projection window")
        print("✅ Network stable — NO upgrade needed (no traffic).")
        return False
    
    print(f" → Blocking probability: {blocking_probability_nu:.4f}")
    print(f" → Threshold: {blocked_connection_prob_threshold_need_for_upgrade}")

    if blocking_probability_nu <= blocked_connection_prob_threshold_need_for_upgrade:
        print("✅ Network stable — NO upgrade needed.")
        return False
    else:
        print("🚨 Upgrade REQUIRED.")
        return True


