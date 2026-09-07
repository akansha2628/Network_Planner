import sys
from pathlib import Path

# Add project root directory to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ------------------------------------------------------------------
# Embedded Feature Extraction Helper Functions
# ------------------------------------------------------------------
def flatten_traffic_matrix(traffic_rates):
    row = {}
    n = traffic_rates.shape[0]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            row[f"lambda_{i}_{j}"] = float(traffic_rates[i][j])
    return row

def flatten_link_technology_state(link_status_forward, classify_fiber_state):
    row = {}
    for link_id, fibers in sorted(link_status_forward.items()):
        sc_c = sc_cl = mc_c = mc_cl = 0
        for _, fiber_obj in fibers.items():
            state = classify_fiber_state(fiber_obj)
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

def flatten_link_slot_occupancy_state(link_status_forward):
    row = {}
    for link_id, fibers in sorted(link_status_forward.items()):
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

# Try importing classify_fiber_state if defined in main.py, else define a fallback
try:
    from main import classify_fiber_state
except ImportError:
    def classify_fiber_state(fiber_obj):
        return "SC_C"


# ------------------------------------------------------------------
# Environment Class
# ------------------------------------------------------------------
class OpticalNetworkEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, sim_config, gatekeeper_model, plan_checker):
        super(OpticalNetworkEnv, self).__init__()
        self.sim_config = sim_config
        self.gatekeeper = gatekeeper_model
        self.plan_checker = plan_checker
        self.bbp_threshold = sim_config.get("bbp_threshold", 0.01)
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(313,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(19)

    def _extract_313_features(self, current_bbp, traffic_rates, link_status_forward):
        row = {"current_BBP": float(current_bbp)}
        row.update(flatten_traffic_matrix(traffic_rates))
        row.update(flatten_link_technology_state(link_status_forward, classify_fiber_state))
        row.update(flatten_link_slot_occupancy_state(link_status_forward))
        
        feature_vector = np.array(list(row.values()), dtype=np.float32)
        return row, feature_vector

    def get_19_link_capacities(self, link_status_forward):
        capacities = {}
        for link_id, fibers in sorted(link_status_forward.items()):
            active_count = sum(
                1 for _, f_obj in fibers.items() 
                if isinstance(f_obj, dict) and f_obj.get("lit", False)
            )
            capacities[link_id] = active_count
        return capacities

    def _calculate_reward(self, current_bbp, action_executed, plan_valid):
        if not plan_valid:
            return -100.0
        if current_bbp > self.bbp_threshold:
            return -50.0
        cost_penalty = 5.0 if action_executed is not None else 0.0
        return (1.0 - current_bbp) * 10.0 - cost_penalty

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        from main import initialize_simulation
        self.sim_state = initialize_simulation(self.sim_config, seed=seed)
        state_dict, obs_vector = self._extract_313_features(
            current_bbp=self.sim_state["current_bbp"],
            traffic_rates=self.sim_state["traffic_rates"],
            link_status_forward=self.sim_state["link_status_forward"],
        )
        return obs_vector, {"state_dict": state_dict}

    def step(self, action):
        from main import run_single_cycle
        link_capacities = self.get_19_link_capacities(self.sim_state["link_status_forward"])
        state_dict, obs_vector = self._extract_313_features(
            current_bbp=self.sim_state["current_bbp"],
            traffic_rates=self.sim_state["traffic_rates"],
            link_status_forward=self.sim_state["link_status_forward"],
        )

        gatekeeper_prediction = self.gatekeeper.predict([obs_vector])[0]
        chosen_action = action if gatekeeper_prediction == 1 else None
        plan_valid = True

        if chosen_action is not None:
            plan_valid = self.plan_checker.verify_action(chosen_action, link_capacities)
            if not plan_valid:
                chosen_action = None

        self.sim_state = run_single_cycle(state=self.sim_state, upgrade_action=chosen_action)
        new_bbp = self.sim_state["current_bbp"]
        _, new_obs_vector = self._extract_313_features(
            current_bbp=new_bbp,
            traffic_rates=self.sim_state["traffic_rates"],
            link_status_forward=self.sim_state["link_status_forward"],
        )
        done = bool(new_bbp > self.bbp_threshold or self.sim_state.get("is_finished", False))
        reward = self._calculate_reward(new_bbp, chosen_action, plan_valid)
        return new_obs_vector, reward, done, False, {"bbp": new_bbp, "action_applied": chosen_action}


# =====================================================================
# TEST BLOCK
# =====================================================================
if __name__ == "__main__":

    class MockGatekeeper:
        def predict(self, obs):
            return [1]

    class MockPlanChecker:
        def verify_action(self, action, capacities):
            return True

    dummy_config = {"bbp_threshold": 0.01}

    print("--- Running DRL Environment Sanity Test ---")
    env = OpticalNetworkEnv(
        sim_config=dummy_config,
        gatekeeper_model=MockGatekeeper(),
        plan_checker=MockPlanChecker(),
    )

    print("1. Environment created successfully.")
    sample_obs = env.observation_space.sample()
    print(f"2. Observation space shape verified: {sample_obs.shape}")
    print(f"3. Action space size verified: {env.action_space.n}")
    print("--- Sanity Test Complete ---")