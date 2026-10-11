"""
DRL link-selection agent (MaskablePPO) for the optical network upgrade simulator.

How one RL step maps onto the simulator
---------------------------------------
main.run_single_cycle() advances ONE connection arrival. Upgrade decisions only happen
every Upgrade_initiation_days (180), so the environment works like this:
    XGBoost gatekeeper says "upgrade needed" (or it is the last cycle of the budget
    year) -> main.py's own link selection gives the candidate links
    -> ONE RL step = ONE pick of (link, target technology state), or STOP.
       The agent keeps picking until nothing else fits the remaining budget /
       sequential downtime, or it STOPs early to save budget for later in the
       same year (not allowed in the year's last cycle).
    -> main.apply_drl_decision() sends the WHOLE selection to plan_checker once:
           PASS -> commit;  FAIL -> nothing spent, penalty, cycle falls back to incremental
    -> run connection events until the next decision point
       or until the network's lifetime ends (BBP threshold reached)
    -> observe, reward (given once per cycle; intermediate picks get 0)
The agent never calls plan_checker itself. The action mask (main.compute_drl_action_mask)
only allows picks that keep the whole selection within the CAPEX/TOTAL budget and the
downtime limit and keep the network connected.

Usage (run from the repository root or from sim/ml)
-----
    python sim/ml/DRL_code.py --check                      # quick smoke test (no training)
    python sim/ml/DRL_code.py --timesteps 50000 --n-envs 4 # train with 4 simulators in parallel
    python sim/ml/DRL_code.py --resume sim/ml/drl_optical_policy.zip --timesteps 20000
    python sim/ml/DRL_code.py --evaluate sim/ml/drl_optical_policy.zip   # vs random + incremental
"""

# ##############################################################################
# ##############################################################################
# ###                                                                        ###
# ###   ML-INTEGRATION CHANGE LOG — DRL_code.py (changes made by Claude)      ###
# ###                                                                        ###
# ###   Most of this file was rewritten, so nearly every section has a box   ###
# ###       ### ML-INTEGRATION CHANGE D<number> (Claude) ###                 ###
# ###   Search for "ML-INTEGRATION CHANGE" to jump between them.             ###
# ###   KEPT from your version: the OpticalNetworkEnv/MaskablePPO structure, ###
# ###   ActionMasker -> Monitor wrapping order, your reward shape            ###
# ###   ((1-BBP)*10, -5 per upgrade, -10000 terminal, -100 penalties), and   ###
# ###   the training-curve plotting.                                         ###
# ###                                                                        ###
# ##############################################################################
# ##############################################################################
#
#  D1  Module docstring ABOVE this log (new): explains how one RL step maps onto
#      the simulator and how to run the script.
#  D2  Imports: removed the duplicated `from main import ...` blocks and the
#      plan_checker import; now imports the new helpers from main.py
#      (apply_drl_decision, compute_drl_action_mask, drl_observation, ...).
#      REMOVED the local classify_fiber_state fallback (it returned 'SC_C' for
#      every fiber) and the local flatten_* helpers -> main.py is the single
#      source of truth for features.
#  D3  __init__: action/observation spaces sized from the real topology
#      (dt12: 20 links x 18 target states + STOP = 361 actions, 336 observation values) instead of a
#      hard-coded Discrete(20) where action 19 collided with link 19. Training
#      seeds are a parameter (0-11 since D17); penalties are parameters.
#  D4  _sim(): silences simulator prints during training (speed).
#  D5  _advance_to_decision(): THE KEY FIX. One RL step used to be one
#      connection arrival (the action was ignored ~99.9% of the time). Now a
#      step runs the simulator until the next real decision point. Cycles
#      where no candidate is feasible are resolved automatically.
#  D6  _calculate_reward(): same values as yours, driven by the result status
#      from main.apply_drl_decision (executed / rejected / pass).
#  D7  reset(): closes the previous episode's 4 CSV handles (was leaking),
#      cycles through training seeds (SB3 passes seed=None after the first
#      reset, so every episode used seed 0).
#  D8  step(): NO plan_checker call and no deepcopy here anymore. The action
#      goes to main.apply_drl_decision(): plan_checker validates in the
#      backend; FAIL -> nothing spent, penalty, incremental fallback.
#      REMOVED get_19_link_capacities() and _get_incremental_fallback() (it
#      called plan_checker with 2 of its 12 arguments -> crash).
#  D9  action_masks(): candidate links from main.py's link selection, filtered
#      by O(1) checks (physical headroom, no queued job, connectivity,
#      CAPEX/TOTAL budget). Previously it read 'candidate_links', which main.py
#      never set, so the mask allowed everything.
#  D10 close() (new): releases file handles.
#  D11 run_smoke_test() + `--check` flag (new): quick test without training.
#  D12 __main__: argparse (--timesteps, --check, --verbose-sim, --out),
#      MockGatekeeper removed (main.py loads the real XGBoost gatekeeper),
#      Monitor logs bbp/day, PPO n_steps=256 (steps are expensive: every
#      finished selection runs a 180-day cycle + plan_checker), model saved next to this script, lifetime plotted in days,
#      plt.show() removed (headless runs).
#  D13 MULTI-LINK SELECTION: one RL step = one link pick. The agent keeps
#      picking candidate links until nothing else fits the remaining budget /
#      downtime; then the whole set goes to plan_checker once. Intermediate picks
#      get reward 0, the cycle reward comes when the selection is finalized.
#      Hoarding penalty removed (STOP rules: see D16).
#      Observation = 313 features + picked flags + remaining CAPEX/downtime
#      (+ the D16 last-cycle flag = 336 on dt12).
#  D15 UPGRADE LEVELS: each action is (link, upgrade level), so the agent also
#      decides how far to upgrade each link (levels = CONFIG['ml_upgrade_levels']
#      in main.py). Action space = links x options + STOP. Default options are
#      main.py's 'any_higher' target states (M17): 20 x 18 + 1 = 361 on dt12.
#  D16 BUDGET SAVING (logic lives in main.py, M16): STOP is now allowed before
#      the budget is used up, except in the last cycle of a budget year. A cycle
#      the agent deliberately skips is reported as "saved_budget" (no penalty,
#      no incremental fallback). Observation is now 336 values on dt12 (adds a
#      "last cycle of the year" flag).
#  D14 CHECKPOINTS: every --save-every steps (default 1000) the policy is saved
#      to drl_logs/checkpoints/ AND the main policy zip is overwritten with the
#      latest, and drl_training_curves.png is refreshed, so you can watch for the
#      plateau while it trains. Ctrl+C stops cleanly and saves. --resume <zip>
#      continues a stopped/crashed run (curves continue in the same monitor.csv).
#      Each run gets its own checkpoint folder; a fresh run backs up an existing
#      trained zip before training, so nothing is overwritten by accident.
#      Your plotting code is unchanged, just moved into plot_training_curves().
#  D17 PARALLEL TRAINING + VALIDATION:
#      - Seeds: train 0-11, validation 12-14, test 15-19 (TRAIN/VALIDATION/TEST_SEEDS).
#      - --n-envs N trains with N simulators at once (one per CPU core,
#        SubprocVecEnv). Each simulator writes to its own results folder
#        (run tag, main.py M19) and its own drl_logs/env<k>.monitor.csv; the
#        curves merge all of them.
#      - --evaluate <policy.zip> runs that policy, a random agent (same choices
#        and masks as the DRL) and incremental on the validation seeds in
#        parallel, prints a table and appends drl_logs/validation_results.csv.
#        --eval-max-days caps each run for a quicker comparison.
#        Use it on checkpoints to decide whether training is worth continuing.
#  D18 --seed (default 0) makes training reproducible. Give each machine /
#      independent run a different seed.
# ##############################################################################

import sys
import os
import copy
import argparse
import contextlib
from pathlib import Path

# Add project root directory to Python path (main.py lives there)
##############################################################################
# ### ML-INTEGRATION CHANGE D2 (Claude) ###
# CHANGED: path setup also adds this script's folder (your parent.parent.parent
# path is kept), so `from main import ...` works from either location.
##############################################################################
_THIS_DIR = Path(__file__).resolve().parent
for _p in (_THIS_DIR, _THIS_DIR.parent.parent):
    if str(_p) not in sys.path:
        sys.path.append(str(_p))

import numpy as np
import gymnasium as gym
from gymnasium import spaces

##############################################################################
# ### ML-INTEGRATION CHANGE D2 (Claude) ###
# CHANGED: single import block (yours imported from main twice). Imports the new
# main.py helpers; plan_checker is no longer imported here.
# REMOVED below this point: local classify_fiber_state fallback and the local
# flatten_traffic_matrix / flatten_link_* / summarize_slot_array helpers.
# They now live only in main.py.
##############################################################################
try:
    from main import (
        CONFIG,
        initialize_simulation_state,
        run_single_cycle,
        apply_drl_decision,
        compute_drl_action_mask,
        drl_policy_action,
        drl_observation,
        drl_action_count,
        get_drl_link_ids,
        close_state_files,
    )
except ImportError as e:
    raise ImportError(
        f"CRITICAL: Failed to import required components from main.py: {e}.\n"
        "Make sure you are using the updated main.py (it defines apply_drl_decision, "
        "compute_drl_action_mask, drl_observation, ...)."
    )


##############################################################################
# ### ML-INTEGRATION CHANGE D17 (Claude) ###
# SEED SPLIT: DRL trains on seeds 0-11 and is VALIDATED on 12-14 (unseen during
# training; same validation seeds gatekeeper.py uses). Seeds 15-19 stay the
# untouched TEST seeds for the final results.
##############################################################################
TRAIN_SEEDS = tuple(range(0, 12))
VALIDATION_SEEDS = (12, 13, 14)
TEST_SEEDS = (15, 16, 17, 18, 19)


# =====================================================================
# Environment
# =====================================================================
class OpticalNetworkEnv(gym.Env):
    metadata = {"render_modes": []}

    ##############################################################################
    # ### ML-INTEGRATION CHANGE D3 (Claude) ###
    # CHANGED: removed gatekeeper_model/plan_checker args (main.py handles both);
    # added train_seeds, quiet, max_decisions_per_episode and the penalty values.
    # Spaces are sized from a probe simulator state (see end of __init__).
    # One RL step = ONE (link, upgrade level) pick (D15). The agent keeps picking links within a cycle
    # until nothing else fits the budget/downtime; then the whole set goes to
    # plan_checker. STOP is allowed early to SAVE budget for later in the same
    # year, except in the year's last cycle (D16 / main.py M16).
    ##############################################################################
    def __init__(
        self,
        sim_config,
        train_seeds=TRAIN_SEEDS,             # D17: 0-11 (validation 12-14, test 15-19)
        quiet=True,                          # silence simulator prints during training
        run_tag=None,                        # D17: own results folder per parallel simulator
        max_decisions_per_episode=None,      # optional truncation
        terminal_penalty=-10000.0,
        action_cost_penalty=5.0,
        invalid_action_penalty=-100.0,       # DRL's selection failed plan_checker (cycle falls back to incremental)
    ):
        super().__init__()
        self.sim_config = copy.deepcopy(sim_config)
        self.sim_config["ml_mode"] = "train"
        if run_tag:
            self.sim_config["ml_run_tag"] = str(run_tag)
        self.train_seeds = tuple(int(s) for s in train_seeds)
        if not self.train_seeds:
            raise ValueError("train_seeds must not be empty")
        self.quiet = quiet
        self.max_decisions_per_episode = max_decisions_per_episode
        self.terminal_penalty = float(terminal_penalty)
        self.action_cost_penalty = float(action_cost_penalty)
        self.invalid_action_penalty = float(invalid_action_penalty)

        self.sim_state = None
        self.decisions_this_episode = 0
        self._current_mask = None

        # Probe one simulator state to size the spaces from the real topology
        # (dt12: 20 links -> obs = 313 features + 20 picked values + 3 extra values = 336,
        #  actions = 20 links x 18 'any higher' target states + STOP = 361 by default).
        probe = self._new_sim_state(self.train_seeds[0])
        try:
            self.link_ids = get_drl_link_ids(probe)
            obs = drl_observation(probe)
            n_actions = drl_action_count(probe)
        finally:
            close_state_files(probe)

        self.n_links = len(self.link_ids)
        self.stop_action = n_actions - 1
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=obs.shape, dtype=np.float32)
        self.action_space = spaces.Discrete(n_actions)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    ##############################################################################
    # ### ML-INTEGRATION CHANGE D4 (Claude) ###
    # NEW: silence simulator stdout while training.
    ##############################################################################
    def _sim(self):
        """Context manager that silences simulator prints when quiet=True."""
        if not self.quiet:
            return contextlib.nullcontext()
        if getattr(self, "_devnull", None) is None or self._devnull.closed:
            self._devnull = open(os.devnull, "w")
        return contextlib.redirect_stdout(self._devnull)

    ##############################################################################
    # ### ML-INTEGRATION CHANGE D4 (Claude) ###
    # NEW: builds a fresh simulator state for one training seed (forces ml_mode='train').
    ##############################################################################
    def _new_sim_state(self, sim_seed):
        cfg = copy.deepcopy(self.sim_config)
        cfg["seed"] = int(sim_seed)
        with self._sim():
            return initialize_simulation_state(cfg)

    ##############################################################################
    # ### ML-INTEGRATION CHANGE D5 (Claude) ###
    # NEW (key fix): advance the simulator to the next DRL decision point instead of
    # a single connection arrival.
    ##############################################################################
    def _advance_to_decision(self):
        """Run connection events until a DRL decision with a real choice is pending, or the
        lifetime ends. Returns True if the episode terminated (BBP threshold reached).
        Decision points where no candidate link is feasible are resolved automatically
        (STOP -> incremental fallback) because the agent has nothing to choose."""
        with self._sim():
            while True:
                self.sim_state, _, done = run_single_cycle(self.sim_state)
                if done:
                    self.sim_state["pending_drl_decision"] = None
                    return True
                if self.sim_state.get("pending_drl_decision") is not None:
                    mask = compute_drl_action_mask(self.sim_state)
                    if mask[:-1].any():
                        self._current_mask = mask
                        return False
                    apply_drl_decision(self.sim_state, self.stop_action)

    ##############################################################################
    # ### ML-INTEGRATION CHANGE D5 (Claude) ###
    # NEW: observation = main.drl_observation(): the same 313 features as everywhere
    # else + which links are already picked this cycle + remaining CAPEX/downtime room.
    ##############################################################################
    def _observe(self):
        with self._sim():
            return drl_observation(self.sim_state)

    def _bbp(self):
        return float(self.sim_state["metrics"].get_blocking_probability())

    ##############################################################################
    # ### ML-INTEGRATION CHANGE D6 (Claude) ###
    # CHANGED: same reward values as your version; uses the status returned by
    # main.apply_drl_decision and the simulator's own termination (lifetime end)
    # instead of a separate 0.01 bbp_threshold. Given once per CYCLE (when the
    # selection is finalized); intermediate picks get 0.
    ##############################################################################
    def _calculate_reward(self, bbp, result, terminated):
        if terminated:
            return self.terminal_penalty
        reward = (1.0 - bbp) * 10.0
        status = result["status"]
        if status == "executed":
            reward -= self.action_cost_penalty
        elif status.startswith("rejected"):
            reward += self.invalid_action_penalty
        return reward

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    ##############################################################################
    # ### ML-INTEGRATION CHANGE D7 (Claude) ###
    # CHANGED: closes previous episode files, cycles training seeds, advances to
    # the first decision point.
    ##############################################################################
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.sim_state is not None:
            close_state_files(self.sim_state)     # avoid leaking 4 file handles per episode

        options = options or {}
        if "sim_seed" in options:
            seed_order = [int(options["sim_seed"])]
        else:
            start = int(self.np_random.integers(len(self.train_seeds)))
            seed_order = [self.train_seeds[(start + k) % len(self.train_seeds)]
                          for k in range(len(self.train_seeds))]

        for sim_seed in seed_order:
            self.sim_state = self._new_sim_state(sim_seed)
            self.decisions_this_episode = 0
            terminated = self._advance_to_decision()
            if not terminated:
                return self._observe(), {"sim_seed": sim_seed, "bbp": self._bbp(),
                                         "day": float(self.sim_state["Current_global_time"])}
            close_state_files(self.sim_state)

        raise RuntimeError(
            "The network reached its BBP threshold before any upgrade decision point for every "
            "training seed. Check Upgrade_initiation_days / the gatekeeper threshold."
        )

    ##############################################################################
    # ### ML-INTEGRATION CHANGE D8 (Claude) ###
    # CHANGED: no plan_checker / fallback logic here anymore. main.apply_drl_decision
    # does plan_checker validation and the incremental fallback in the backend.
    # REMOVED: get_19_link_capacities() and _get_incremental_fallback().
    # One step = one pick. While the cycle is still being built the simulator does
    # not advance and the reward is 0; when the selection is finalized the
    # simulator runs to the next decision point and the cycle reward is given.
    ##############################################################################
    def step(self, action):
        if self.sim_state is None or self.sim_state.get("pending_drl_decision") is None:
            raise RuntimeError("step() called without a pending decision. Call reset() first.")

        action = int(action)
        with self._sim():
            result = apply_drl_decision(self.sim_state, action)
        self._current_mask = None

        if not result["cycle_complete"]:
            # Still building this cycle's selection: no time passes, no reward yet.
            info = {"bbp": self._bbp(), "day": float(self.sim_state["Current_global_time"]),
                    "cycle_complete": False, "selected_links": result["selected_links"],
                    "selected_levels": result["selected_levels"],
                    "status": "picking", "fallback": None, "final_links": []}
            return self._observe(), 0.0, False, False, info

        self.decisions_this_episode += 1
        terminated = self._advance_to_decision()
        bbp = self._bbp()
        reward = self._calculate_reward(bbp, result, terminated)

        truncated = bool(
            not terminated
            and self.max_decisions_per_episode is not None
            and self.decisions_this_episode >= self.max_decisions_per_episode
        )
        info = {
            "bbp": bbp,
            "day": float(self.sim_state["Current_global_time"]),
            "cycle_complete": True,
            "selected_links": result["selected_links"],
            "selected_levels": result["selected_levels"],
            "status": result["status"],
            "fallback": result["fallback"],
            "final_links": result["final_links"],
        }
        return self._observe(), float(reward), terminated, truncated, info

    ##############################################################################
    # ### ML-INTEGRATION CHANGE D9 (Claude) ###
    # CHANGED: mask = main.compute_drl_action_mask(): candidates not yet picked whose
    # addition keeps the WHOLE selection within CAPEX/TOTAL budget and the
    # sequential-downtime limit. STOP (last action): see D16 (budget saving).
    ##############################################################################
    def action_masks(self) -> np.ndarray:
        if self.sim_state is None or self.sim_state.get("pending_drl_decision") is None:
            mask = np.zeros(self.action_space.n, dtype=bool)
            mask[self.stop_action] = True
            return mask
        if self._current_mask is None:
            with self._sim():                     # cost model prints while pricing
                self._current_mask = compute_drl_action_mask(self.sim_state)
        return self._current_mask.copy()

    ##############################################################################
    # ### ML-INTEGRATION CHANGE D10 (Claude) ###
    # NEW: close simulator CSV handles.
    ##############################################################################
    def close(self):
        if self.sim_state is not None:
            close_state_files(self.sim_state)
            self.sim_state = None


# =====================================================================
# Smoke test (no training): random masked actions for a few decisions
# =====================================================================
##############################################################################
# ### ML-INTEGRATION CHANGE D11 (Claude) ###
# NEW: used by `python DRL_code.py --check`.
##############################################################################
def run_smoke_test(env, n_decisions=5):
    """Random masked picks until n_decisions upgrade CYCLES have been completed."""
    print("--- Smoke test ---")
    print(f"links={env.n_links}, obs_shape={env.observation_space.shape}, actions={env.action_space.n}")
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs), "observation outside observation_space"
    print(f"reset ok: sim_seed={info['sim_seed']}, day={info['day']:.1f}, bbp={info['bbp']:.5f}")
    rng = np.random.default_rng(0)
    cycles = picks = 0
    while cycles < n_decisions:
        mask = env.action_masks()
        action = int(rng.choice(np.flatnonzero(mask)))
        obs, reward, terminated, truncated, info = env.step(action)
        picks += 1
        assert env.observation_space.contains(obs), "observation outside observation_space"
        if not info["cycle_complete"]:
            continue
        cycles += 1
        picked = ", ".join(f"{l}:{info['selected_levels'][l]}" for l in info["selected_links"])
        print(f"  cycle {cycles}: picked [{picked}] ({picks} picks) -> {info['status']}"
              f"{' | fallback ' + info['fallback'] if info['fallback'] else ''}, "
              f"reward={reward:.2f}, bbp={info['bbp']:.5f}, day={info['day']:.1f}")
        picks = 0
        if terminated or truncated:
            print("  episode ended")
            obs, info = env.reset()
    env.close()
    print("--- Smoke test passed ---")


# =====================================================================
# Training curves + checkpointing helpers
# =====================================================================
##############################################################################
# ### ML-INTEGRATION CHANGE D14 (Claude) ###
# NEW: your training-curve plotting moved into a function (same plot) so it can
# be refreshed DURING training, plus a checkpoint callback.
##############################################################################
def _read_monitor_logs(log_dir):
    """All episode rows from monitor.csv / env*.monitor.csv (one file per parallel
    simulator, D17), merged in the order the episodes finished."""
    import glob, json
    import pandas as pd
    frames = []
    for path in sorted(glob.glob(os.path.join(log_dir, "*monitor.csv"))):
        with open(path) as f:
            header = f.readline()
        try:
            t_start = float(json.loads(header.lstrip("#")).get("t_start", 0.0))
        except Exception:
            t_start = 0.0
        df = pd.read_csv(path, skiprows=1)
        if len(df):
            # "t" restarts from 0 when a run is resumed (--resume appends to the same file):
            # make it keep increasing so the file's own order is preserved.
            t = df["t"].to_numpy(dtype=float)
            offset, prev, adj = 0.0, 0.0, []
            for v in t:
                if v < prev:
                    offset += prev
                prev = v
                adj.append(offset + v)
            df["finished_at"] = t_start + np.asarray(adj)
            frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True).sort_values("finished_at").reset_index(drop=True)


def plot_training_curves(log_dir):
    """Re-draw drl_training_curves.png from the monitor logs in drl_logs/. Returns the path or None."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results_df = _read_monitor_logs(log_dir)
    if results_df is None or len(results_df) == 0:
        print("⚠️ No completed training episodes logged yet (nothing to plot).")
        return None

    results_df['timesteps'] = results_df['l'].cumsum()

    window_size = min(20, max(1, len(results_df) // 5))
    results_df['smoothed_reward'] = results_df['r'].rolling(window=window_size, min_periods=1).mean()
    # Lifetime is plotted in simulated days (episode length is now number of picks,
    # not connection arrivals).
    results_df['smoothed_lifetime'] = results_df['day'].rolling(window=window_size, min_periods=1).mean()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(results_df['timesteps'], results_df['r'], color='lightskyblue', alpha=0.4, label='Raw Episode Reward')
    ax1.plot(results_df['timesteps'], results_df['smoothed_reward'], color='navy', linewidth=2, label='Moving Average Reward')
    ax1.set_title('DRL Reward & Penalty Improvement Curve Over Training', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Total Episode Reward', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='lower right')

    ax2.plot(results_df['timesteps'], results_df['day'], color='lightgreen', alpha=0.4, label='Raw Network Lifetime')
    ax2.plot(results_df['timesteps'], results_df['smoothed_lifetime'], color='darkgreen', linewidth=2, label='Moving Average Lifetime')
    ax2.set_title('Network Lifetime (days until BBP crosses threshold)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Total Training Steps (link picks)', fontsize=10)
    ax2.set_ylabel('Lifetime (days)', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='lower right')

    plt.tight_layout()
    plot_output_path = os.path.join(log_dir, "drl_training_curves.png")
    plt.savefig(plot_output_path, dpi=150)
    plt.close(fig)
    return plot_output_path


def make_checkpoint_callback(save_every, checkpoint_dir, latest_model_path, log_dir):
    """
    Every `save_every` training steps:
      - saves drl_logs/checkpoints/drl_policy_<steps>_steps.zip   (history)
      - overwrites the main policy zip (latest_model_path)          (always usable by main.py)
      - refreshes drl_training_curves.png                           (watch for the plateau)
    """
    from stable_baselines3.common.callbacks import BaseCallback

    class CheckpointAndPlotCallback(BaseCallback):
        def __init__(self):
            super().__init__(verbose=0)
            self._last_save = None      # set on the first step (works for fresh and resumed runs)

        def _save(self):
            os.makedirs(checkpoint_dir, exist_ok=True)
            ckpt = os.path.join(checkpoint_dir, f"drl_policy_{self.num_timesteps}_steps.zip")
            self.model.save(ckpt)
            self.model.save(latest_model_path)
            plot = plot_training_curves(log_dir)
            print(f"💾 Checkpoint at {self.num_timesteps} steps -> {ckpt} "
                  f"(also {latest_model_path}){' | curves updated' if plot else ''}")

        def _on_step(self) -> bool:
            if self._last_save is None:
                self._last_save = self.num_timesteps - 1
            if self.num_timesteps - self._last_save >= save_every:
                self._last_save = self.num_timesteps
                self._save()
            return True

    return CheckpointAndPlotCallback()


##############################################################################
# ### ML-INTEGRATION CHANGE D17 (Claude) ###
# NEW: parallel training environments and the --evaluate (validation) mode.
#   make_training_env(): one simulator, with its own results folder and its own
#       monitor log, so --n-envs N runs N simulators on N CPU cores at once.
#   run_evaluation(): runs a saved policy, a RANDOM agent (same freedoms as the
#       DRL) and the INCREMENTAL heuristic on the validation seeds, in parallel,
#       and writes drl_logs/validation_results.csv. Baselines are cached in
#       drl_logs/validation_baselines.csv so they are only computed once.
##############################################################################
def _mask_fn(env):
    return env.action_masks()


def make_training_env(rank, quiet, log_dir, override_existing, n_envs):
    """One training simulator: OpticalNetworkEnv -> ActionMasker -> Monitor (top level so it
    can be sent to worker processes)."""
    from stable_baselines3.common.monitor import Monitor
    from sb3_contrib.common.wrappers import ActionMasker
    env = OpticalNetworkEnv(sim_config=CONFIG, quiet=quiet,
                            run_tag=(f"train_env{rank}" if n_envs > 1 else None))
    env = ActionMasker(env, _mask_fn)
    monitor_path = (os.path.join(log_dir, "monitor.csv") if n_envs == 1
                    else os.path.join(log_dir, f"env{rank}.monitor.csv"))
    # Append when resuming, but a log that does not exist yet must be created with its header.
    override_existing = override_existing or not os.path.exists(monitor_path)
    return Monitor(env, monitor_path, info_keywords=("bbp", "day"),
                   override_existing=override_existing)


def _evaluate_one(job):
    """Run ONE full network lifetime for (method, seed). method: 'drl' | 'random' | 'incremental'."""
    method, seed, policy_path, max_days, quiet = job
    import time
    cfg = copy.deepcopy(CONFIG)
    cfg.update(seed=int(seed), collect_xgboost_data=False, ml_run_tag=f"eval_{method}_s{seed}")
    if method == "incremental":
        cfg.update(ml_mode="off", algorithm="increment")
    else:
        cfg["ml_mode"] = "train"          # same gatekeeper setting the DRL trains with
    policy = None
    if method == "drl":
        from sb3_contrib import MaskablePPO
        policy = MaskablePPO.load(policy_path, device="cpu")
    rng = np.random.default_rng(int(seed))
    t0 = time.time()
    devnull = open(os.devnull, "w")
    ctx = contextlib.redirect_stdout(devnull) if quiet else contextlib.nullcontext()
    with ctx:
        st = initialize_simulation_state(cfg)
        done = False
        while not done:
            st, _, done = run_single_cycle(st)
            while st.get("pending_drl_decision") is not None:
                if done:
                    st["pending_drl_decision"] = None
                    break
                mask = compute_drl_action_mask(st)
                if not mask[:-1].any():
                    action = len(mask) - 1          # no feasible link: STOP (same as training)
                elif method == "drl":
                    action = drl_policy_action(st, policy)
                else:
                    action = int(rng.choice(np.flatnonzero(mask)))
                apply_drl_decision(st, action)
            if max_days and st["Current_global_time"] >= float(max_days):
                break
    close_state_files(st)
    devnull.close()
    return {
        "method": method, "seed": int(seed), "max_days": max_days or "",
        "lifetime_reached_threshold": bool(done),
        "days_simulated": round(float(st["Current_global_time"]), 1),
        "final_bbp": round(float(st["metrics"].get_blocking_probability()), 6),
        "total_cost": round(float(st["total_network_cost"]), 2),
        "upgrades_committed": len(st["upgrade_decision_history"]),
        "wall_seconds": round(time.time() - t0, 1),
    }


def run_evaluation(policy_path, seeds, max_days, workers, log_dir, quiet=True):
    """Policy vs random vs incremental on `seeds`; writes CSVs and prints a summary table."""
    import csv, time
    import multiprocessing as mp
    os.makedirs(log_dir, exist_ok=True)
    base_file = os.path.join(log_dir, "validation_baselines.csv")
    res_file = os.path.join(log_dir, "validation_results.csv")
    key = lambda r: (r["method"], int(r["seed"]), str(r["max_days"]))

    cached = {}
    if os.path.exists(base_file):
        with open(base_file, newline="") as f:
            for r in csv.DictReader(f):
                cached[key(r)] = r
    jobs = [("drl", s, policy_path, max_days, quiet) for s in seeds]
    for m in ("incremental", "random"):
        jobs += [(m, s, None, max_days, quiet) for s in seeds
                 if (m, int(s), str(max_days or "")) not in cached]

    workers = max(1, min(workers or os.cpu_count() or 1, len(jobs)))
    print(f"Evaluating on seeds {list(seeds)}: {len(jobs)} simulations on {workers} worker(s) "
          f"(cached baselines reused: {len(cached)})...")
    if workers == 1:
        results = [_evaluate_one(j) for j in jobs]
    else:
        with mp.get_context("spawn").Pool(workers) as pool:
            results = pool.map(_evaluate_one, jobs)

    fields = list(results[0].keys())
    new_base = [r for r in results if r["method"] != "drl"]
    if new_base:
        exists = os.path.exists(base_file)
        with open(base_file, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if not exists:
                w.writeheader()
            w.writerows(new_base)
    rows = [r for r in results if r["method"] == "drl"] + [
        cached[(m, int(s), str(max_days or ""))] for m in ("incremental", "random") for s in seeds
        if (m, int(s), str(max_days or "")) in cached] + new_base
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    exists = os.path.exists(res_file)
    with open(res_file, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["evaluated_at", "policy"] + fields)
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow({"evaluated_at": stamp, "policy": os.path.basename(policy_path), **r})

    print(f"\n{'seed':>5} | {'method':<12} | {'days':>8} | {'hit threshold':>13} | {'final BBP':>9} | {'total cost':>14}")
    for s in seeds:
        for m in ("drl", "incremental", "random"):
            r = next(x for x in rows if x["method"] == m and int(x["seed"]) == int(s))
            print(f"{int(s):>5} | {m:<12} | {float(r['days_simulated']):>8.0f} | {str(r['lifetime_reached_threshold']):>13} | "
                  f"{float(r['final_bbp']):>9.4f} | {float(r['total_cost']):>14,.0f}")
    print("(If a run did not hit the lifetime threshold, it was cut off by --eval-max-days: "
          "compare final BBP and cost at the same day instead of lifetime.)")
    print(f"Saved: {res_file}")
    return rows


# =====================================================================
# DRL TRAINING ROUTINE
# =====================================================================
if __name__ == "__main__":
    ##############################################################################
    # ### ML-INTEGRATION CHANGE D12 (Claude) ###
    # NEW: command-line options. MockGatekeeper REMOVED (main.py loads the real one).
    ##############################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=50000,
                        help="number of RL steps (link picks) to train on")
    parser.add_argument("--check", action="store_true",
                        help="run the SB3 env checker + a short smoke test, then exit")
    parser.add_argument("--verbose-sim", action="store_true",
                        help="show simulator prints during training")
    parser.add_argument("--out", default="drl_optical_policy.zip")
    ##############################################################################
    # ### ML-INTEGRATION CHANGE D14 (Claude) ###
    # NEW: checkpointing / resume options.
    ##############################################################################
    parser.add_argument("--save-every", type=int, default=1000,
                        help="save a checkpoint + refresh the training curves every N steps")
    parser.add_argument("--resume", default=None,
                        help="continue training from a saved .zip (e.g. the main policy zip "
                             "or a file in drl_logs/checkpoints/)")
    ##############################################################################
    # ### ML-INTEGRATION CHANGE D17 (Claude) ###
    # NEW: parallel environments and validation (--evaluate) options.
    ##############################################################################
    parser.add_argument("--n-envs", type=int, default=1,
                        help="number of simulators to train with in parallel (one per CPU core)")
    parser.add_argument("--evaluate", default=None, metavar="POLICY_ZIP",
                        help="do not train: compare this saved policy with a random agent and "
                             "incremental on the validation seeds, then exit")
    parser.add_argument("--val-seeds", default=",".join(str(s) for s in VALIDATION_SEEDS),
                        help="comma-separated validation seeds for --evaluate (default 12,13,14)")
    parser.add_argument("--eval-max-days", type=float, default=None,
                        help="optional cap on simulated days per evaluation run (faster, "
                             "compares BBP/cost at the cap instead of full lifetime)")
    parser.add_argument("--eval-workers", type=int, default=None,
                        help="parallel simulations for --evaluate (default: all CPU cores)")
    ##############################################################################
    # ### ML-INTEGRATION CHANGE D18 (Claude) ###
    # NEW: --seed makes a training run reproducible (network weights, PPO sampling and
    # which training seed each simulator starts on). Use a different value per machine.
    ##############################################################################
    parser.add_argument("--seed", type=int, default=0,
                        help="random seed for training (use a different one for each independent run)")
    args = parser.parse_args()

    if args.evaluate:
        run_evaluation(args.evaluate, [int(s) for s in args.val_seeds.split(",") if s.strip()],
                       args.eval_max_days, args.eval_workers, str(_THIS_DIR / "drl_logs"),
                       quiet=not args.verbose_sim)
        sys.exit(0)

    from stable_baselines3.common.vec_env import SubprocVecEnv
    from sb3_contrib import MaskablePPO

    ##############################################################################
    # ### ML-INTEGRATION CHANGE D12 (Claude) ###
    # NEW: SB3 env checker + smoke test, then exit.
    ##############################################################################
    if args.check:
        from stable_baselines3.common.env_checker import check_env
        raw_env = OpticalNetworkEnv(sim_config=CONFIG, quiet=not args.verbose_sim)
        check_env(raw_env, warn=True)
        print("(smoke test simulates 2 upgrade cycles of real traffic: about 10 minutes)")
        run_smoke_test(raw_env, n_decisions=2)
        sys.exit(0)

    import shutil, time as _time
    run_stamp = _time.strftime("%Y%m%d-%H%M%S")
    log_dir = str(_THIS_DIR / "drl_logs")
    checkpoint_dir = os.path.join(log_dir, "checkpoints", f"run_{run_stamp}")   # one folder per run
    os.makedirs(log_dir, exist_ok=True)
    model_path = str(_THIS_DIR / args.out) if not os.path.isabs(args.out) else args.out

    # A FRESH run (no --resume) must not silently destroy a previously trained policy.
    if args.resume is None and os.path.exists(model_path):
        backup = model_path.replace(".zip", f".backup_{run_stamp}.zip")
        shutil.copy2(model_path, backup)
        print(f"⚠️ Existing {os.path.basename(model_path)} backed up to {os.path.basename(backup)} "
              f"(use --resume to continue training it instead of starting over).")
    print("--- Starting Optical DRL Training Pipeline ---")

    # ActionMasker first, then Monitor (MaskablePPO finds action_masks through the wrappers).
    # When resuming, append to the existing monitor log(s) so the curves continue.
    # D17: with --n-envs N > 1, N simulators run in separate processes (one per CPU core).
    import functools
    n_envs = max(1, int(args.n_envs))
    quiet = not args.verbose_sim
    if n_envs == 1:
        env = make_training_env(0, quiet, log_dir, args.resume is None, 1)
    else:
        env = SubprocVecEnv([functools.partial(make_training_env, rank, quiet, log_dir,
                                               args.resume is None, n_envs)
                             for rank in range(n_envs)])
        print(f"Training with {n_envs} parallel simulators.")

    ##############################################################################
    # ### ML-INTEGRATION CHANGE D12 (Claude) ###
    # CHANGED: n_steps=256, batch_size=64 (PPO default 2048 is too many full
    # upgrade cycles per update). Monitor above now also logs bbp and day.
    ##############################################################################
    if args.resume:
        print(f"Resuming from {args.resume}")
        model = MaskablePPO.load(args.resume, env=env, device="cpu")
        model.set_random_seed(args.seed + int(model.num_timesteps))   # D18: new stream, still reproducible
    else:
        model = MaskablePPO("MlpPolicy", env, verbose=1, learning_rate=3e-4, gamma=0.99,
                            n_steps=256, batch_size=64, seed=args.seed)   # D18: seed

    ##############################################################################
    # ### ML-INTEGRATION CHANGE D14 (Claude) ###
    # NEW: checkpoints during training; Ctrl+C stops cleanly and still saves.
    # The policy zip next to this script is always the latest checkpoint, so you
    # can stop as soon as drl_logs/drl_training_curves.png has plateaued.
    ##############################################################################
    callback = make_checkpoint_callback(args.save_every, checkpoint_dir, model_path, log_dir)
    print(f"Training MaskablePPO for {args.timesteps} steps "
          f"(checkpoint + curves every {args.save_every}; Ctrl+C to stop and save)...")
    try:
        model.learn(total_timesteps=args.timesteps, callback=callback,
                    reset_num_timesteps=args.resume is None)
    except KeyboardInterrupt:
        print("\n⏹ Training stopped by user (Ctrl+C) — saving current model...")

    ##############################################################################
    # ### ML-INTEGRATION CHANGE D12 (Claude) ###
    # CHANGED: policy saved next to this script (main.py looks there too).
    ##############################################################################
    model.save(model_path)
    print(f"✅ Trained model saved as '{model_path}' ({model.num_timesteps} steps total)!")
    env.close()

    plot_path = plot_training_curves(log_dir)
    if plot_path:
        print(f"✅ Training visuals successfully saved to '{plot_path}'!")