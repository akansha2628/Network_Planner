# sim/upgrade/cost_model.py

from sim.core.constants import *
from sim.core import topology as Topology
import math

K1 = 1
K2 = 1
K = 1


def compute_upgrade_costs(safe_links_for_upgrade: list, upgrade_decisions: dict, algorithm_name: str = ""):
    """
    Upgrade-event costs only:
      - Equipment (CAPEX)
      - Workforce

    OPEX is computed separately in sim/upgrade/opex_model.py and added in main.py over time.

    Jump-to-max behavior (as implemented here):
      core_upgrade also includes:
        • L-band add on fiber 0 (all 3 cores)
        • Two new_fiber_CL (fibers 1 and 2)

    Greedy behavior (added as requested):
      core_upgrade also includes:
        • L-band add on fiber 0 (all 3 cores)
    """

    print("Starting compute_upgrade_costs...")

    total_equipment = 0.0
    total_workforce = 0.0
    cost_details = {}

    for link_id in safe_links_for_upgrade:

        if link_id not in upgrade_decisions:
            raise ValueError(f"Missing upgrade_decision for link_id {link_id}")

        decision_list = upgrade_decisions[link_id]

        if not isinstance(decision_list, list):
            raise ValueError(f"upgrade_decisions[{link_id}] must be a list")

        link_equipment = 0.0
        link_workforce = 0.0
        link_entries = []

        # -------------------------------------
        # Find link length
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
        # Evaluate decisions
        # -------------------------------------
        for decision in decision_list:

            if decision is None:
                continue

            upgrade_type = decision.get("upgrade_type")

            if upgrade_type is None:
                continue

            c = decision.get("core_type")

            if c is None:
                raise ValueError(f"core_type missing for link_id {link_id}")

            E_C = 0.0
            W_F = 0.0

            # -------------------------------------
            # New fiber C
            # -------------------------------------
            if upgrade_type == "new_fiber_C":

                E_fc = 2 * (alpha_c + w_c + beta_c) + 2 * (math.ceil(L_e / L_s) * alpha_c + f_l * L_e)
                E_C = c * 1 * K1 * E_fc * (1 - d) ** (m / 12)
                W_F =  K1 * W_fc * (0.4 + 0.6 * (L_e/L_s)) * (1 + I) ** (m / 12)

            # -------------------------------------
            # New fiber C+L
            # -------------------------------------
            elif upgrade_type == "new_fiber_CL":

                E_fcl = (
                    2 * (alpha_c + w_c + beta_c)
                    + 2 * (alpha_l + w_l + beta_l)
                    + 2 * (math.ceil(L_e / L_s) * (alpha_c + alpha_l + mu + w) + f_l * L_e)
                )

                E_C = c * K2 * E_fcl * (1 - d) ** (m / 12)
                W_F =  K2 * W_fcl *  (0.4 + 0.6 * (L_e/L_s)) * (1 + I) ** (m / 12)

            # -------------------------------------
            # Band upgrade (your corrected formula)
            # -------------------------------------
            elif upgrade_type == "band_upgrade":

                E_b = 2 * math.ceil(L_e / L_s) * (alpha_l + mu + w) + 2 * (alpha_l + w_l + beta_l)
                E_C = c * K * E_b * (1 - d) ** (m / 12)
                W_F = K * W_b *  (0.4 + 0.6 * (L_e/L_s))* (1 + I) ** (m / 12)

            # -------------------------------------
            # Core upgrade
            # -------------------------------------
            elif upgrade_type == "core_upgrade":

                E_cu = C_3c * L_e
                E_fc = 2 * (alpha_c + w_c + beta_c) + 2 * (math.ceil(L_e / L_s) * alpha_c)
                E_C = (E_cu + c * E_fc) * (1 - d) ** (m / 12) + K_d * L_e * (1 + I) ** (m / 12)
                W_F = ((W_c * (0.4 + 0.6 * (L_e/L_s))) + (W_fc * (0.4 + 0.6 * (L_e/L_s)))) * (1 + I) ** (m / 12)

                # -------------------------------------
                # GREEDY behavior:
                # After core upgrade, add L-band on fiber 0 for ALL 3 cores
                # -------------------------------------
                if algorithm_name == "greedy":
                    E_b = 2 * math.ceil(L_e / L_s) * (alpha_l + mu + w) + 2 * (alpha_l + w_l + beta_l)
                    E_C += c * K * E_b * (1 - d) ** (m / 12)
                    W_F += K * W_b *  (0.4 + 0.6 * (L_e/L_s))* (1 + I) ** (m / 12)

                # -------------------------------------
                # Jump-to-max extra costs
                # -------------------------------------
                if algorithm_name == "jump_to_max":

                    # ---- L band add on fiber 0 ----
                    E_b = 2 * math.ceil(L_e / L_s) * (alpha_l + mu + w) + 2 * (alpha_l + w_l + beta_l)
                    E_C += c * K * E_b * (1 - d) ** (m / 12)
                    W_F += K * W_b *  (0.4 + 0.6 * (L_e/L_s))* (1 + I) ** (m / 12)

                    # ---- two new fibers with C+L ----
                    E_fcl = (
                            2 * (alpha_c + w_c + beta_c)
                            + 2 * (alpha_l + w_l + beta_l)
                            + 2 * (math.ceil(L_e / L_s) * (alpha_c + alpha_l + mu + w) + f_l * L_e)
                    )

                    E_C += 2 * (c * K2 * E_fcl * (1 - d) ** (m / 12))
                    W_F += 2 * (K2 * W_fcl * (0.4 + 0.6 * (L_e / L_s)) * (1 + I) ** (m / 12))

            else:
                raise ValueError(f"Unknown upgrade type '{upgrade_type}' for link {link_id}")

            link_equipment += E_C
            link_workforce += W_F

            link_entries.append({
                "upgrade_type": upgrade_type,
                "equipment_cost": float(E_C),
                "workforce_cost": float(W_F),
                "total_cost": float(E_C + W_F),
            })

        cost_details[link_id] = {
            "link_total_equipment": float(link_equipment),
            "link_total_workforce": float(link_workforce),
            "link_total_cost": float(link_equipment + link_workforce),
            "individual_upgrades": link_entries,
        }

        total_equipment += link_equipment
        total_workforce += link_workforce

    summary = {
        "equipment_total": float(total_equipment),
        "workforce_total": float(total_workforce),
        "total_cycle_cost": float(total_equipment + total_workforce),
    }

    return float(total_equipment + total_workforce), summary, cost_details

