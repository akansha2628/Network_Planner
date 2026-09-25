import sys
import os
from pathlib import Path

# Add project root directory to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3.common.monitor import Monitor
from sb3_contrib import MaskablePPO

# ------------------------------------------------------------------
# STRICT TOP-LEVEL IMPORTS
# ------------------------------------------------------------------
try:
    from main import (
        initialize_simulation_state,  # FIXED: Now imports the correct state builder
        run_single_cycle,
        CONFIG,
        plan_checker
    )
except ImportError as e:
    raise ImportError(
        f"CRITICAL: Failed to import required components from main.py: {e}.\n"
        "Ensure main.py is in the root project directory and defines "
        "'initialize_simulation_state', 'run_single_cycle', and 'plan_checker'."
    )

# ------------------------------------------------------------------
# Fiber Classifier Fallback Helper
# ------------------------------------------------------------------
def classify_fiber_state(fiber_obj):
    if isinstance(fiber_obj, str):
        return fiber_obj
    if isinstance(fiber_obj, dict):
        if "type" in fiber_obj:
            return fiber_obj["type"]
        if "state" in fiber_obj:
            return fiber_obj["state"]
        return "SC_C"
    return "SC_C"

# ------------------------------------------------------------------
# Feature Extraction Helper Functions
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
            low=0.0, high=1.0, shape=(313,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(20)

    def _extract_313_features(self, current_bbp, traffic_rates, link_status_forward):
        row = {"current_BBP": float(current_bbp)}
        row.update(flatten_traffic_matrix(traffic_rates))
        row.update(flatten_link_technology_state(link_status_forward, classify_fiber_state))
        row.update(flatten_link_slot_occupancy_state(link_status_forward))
        
        raw_vector = np.array(list(row.values()), dtype=np.float32)
        
        scale_max = np.ones_like(raw_vector, dtype=np.float32)
        scale_max[0] = 0.1         
        scale_max[1:133] = 1000.0  
        scale_max[133:] = 1000.0   

        normalized_vector = np.clip(raw_vector / scale_max, 0.0, 1.0)
        return row, normalized_vector

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
        
        # FIXED: Inject the seed into the config so main.py doesn't crash
        self.sim_config["seed"] = seed if seed is not None else 0
        
        # Call initialize_simulation_state directly, which returns a dict
        self.sim_state = initialize_simulation_state(self.sim_config)

        # Extract BBP from the metrics object
        current_bbp = self.sim_state["metrics"].get_blocking_probability()

        state_dict, obs_vector = self._extract_313_features(
            current_bbp=current_bbp,
            traffic_rates=self.sim_state["traffic_rates"],
            link_status_forward=self.sim_state["link_status_forward"],
        )
        return obs_vector, {"state_dict": state_dict}
        
    def step(self, action):
        # 1. Action 19 = Pass / Do Nothing. Actions 0..18 = Upgrade Link ID
        if action < 19:
            # Format action as the tuple main.py expects: (safe_links, upgrade_decisions)
            chosen_action = ([action], {action: [{"upgrade_type": "new_fiber_C"}]})
            plan_valid = True
        else:
            plan_valid = True
            chosen_action = None

        # 2. Advance simulation engine safely
        cycle_output = run_single_cycle(state=self.sim_state, action=chosen_action)
        
        if isinstance(cycle_output, tuple):
            self.sim_state = cycle_output[0]
        else:
            self.sim_state = cycle_output

        # 3. Extract new observation & calculate reward
        new_bbp = self.sim_state["metrics"].get_blocking_probability()
        _, new_obs_vector = self._extract_313_features(
            current_bbp=new_bbp,
            traffic_rates=self.sim_state["traffic_rates"],
            link_status_forward=self.sim_state["link_status_forward"],
        )
        
        done = bool(new_bbp > self.bbp_threshold or self.sim_state.get("is_finished", False))
        reward = self._calculate_reward(new_bbp, chosen_action, plan_valid)

        return new_obs_vector, reward, done, False, {"bbp": new_bbp, "action_applied": chosen_action}

    def action_masks(self) -> np.ndarray:
        mask = np.ones(self.action_space.n, dtype=bool)
        candidate_links = self.sim_state.get("candidate_links", None)

        # 1. Candidate link filter
        if candidate_links is not None and len(candidate_links) > 0:
            for action_id in range(19):
                if action_id not in candidate_links:
                    mask[action_id] = False

        # Action 19 ("Do Nothing / Pass") is ALWAYS valid
        mask[19] = True
        return mask

# =====================================================================
# DRL TRAINING ROUTINE
# =====================================================================
if __name__ == "__main__":

    class MockGatekeeper:
        def predict(self, obs):
            return [1]

    log_dir = "./drl_logs/"
    os.makedirs(log_dir, exist_ok=True)

    print("--- Starting Optical DRL Training Pipeline ---")
    
    raw_env = OpticalNetworkEnv(
        sim_config=CONFIG,
        gatekeeper_model=MockGatekeeper(),
        plan_checker=plan_checker,
    )

    env = Monitor(raw_env, log_dir)
    model = MaskablePPO("MlpPolicy", env, verbose=1, learning_rate=3e-4, gamma=0.99)

    total_timesteps = 50000
    #for above line, in the real simulation, make sure the value is set to 50000, for sample tests, use 500
    print(f"Training MaskablePPO for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps)
    
    model_path = "drl_optical_policy.zip"
    model.save(model_path)
    print(f"✅ Trained model saved as '{model_path}'!")

    monitor_file = os.path.join(log_dir, "monitor.csv")
    if os.path.exists(monitor_file):
        results_df = pd.read_csv(monitor_file, skiprows=1)
        
        if len(results_df) > 0:
            results_df['timesteps'] = results_df['l'].cumsum()
            
            window_size = min(20, max(1, len(results_df) // 5))
            results_df['smoothed_reward'] = results_df['r'].rolling(window=window_size, min_periods=1).mean()
            results_df['smoothed_lifetime'] = results_df['l'].rolling(window=window_size, min_periods=1).mean()

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

            ax1.plot(results_df['timesteps'], results_df['r'], color='lightskyblue', alpha=0.4, label='Raw Episode Reward')
            ax1.plot(results_df['timesteps'], results_df['smoothed_reward'], color='navy', linewidth=2, label='Moving Average Reward')
            ax1.set_title('DRL Reward & Penalty Improvement Curve Over Training', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Total Episode Reward', fontsize=10)
            ax1.grid(True, linestyle='--', alpha=0.6)
            ax1.legend(loc='lower right')

            ax2.plot(results_df['timesteps'], results_df['l'], color='lightgreen', alpha=0.4, label='Raw Episode Lifetime')
            ax2.plot(results_df['timesteps'], results_df['smoothed_lifetime'], color='darkgreen', linewidth=2, label='Moving Average Lifetime')
            ax2.set_title('Agent Lifetime (Cycles Survived Until BBP Crosses Threshold)', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Total Training Timesteps', fontsize=10)
            ax2.set_ylabel('Lifetime (Steps Survived)', fontsize=10)
            ax2.grid(True, linestyle='--', alpha=0.6)
            ax2.legend(loc='lower right')

            plt.tight_layout()
            plot_output_path = os.path.join(log_dir, "drl_training_curves.png")
            plt.savefig(plot_output_path, dpi=300)
            print(f"✅ Training visuals successfully saved to '{plot_output_path}'!")
            plt.show()
        else:
            print("⚠️ Monitor CSV exists but contains no completed episodes yet.")
    else:
        print("⚠️ Warning: Could not find monitor log file to render training graphs.")