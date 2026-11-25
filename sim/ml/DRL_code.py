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
           PASS -> commit;  FAIL -> nothing spent, small penalty, cycle falls back to incremental
    -> run connection events until the next decision point
       or until the network's lifetime ends (BBP threshold reached)
    -> observe, reward (given once per cycle; intermediate picks get 0; see CycleReward, D20)
The agent never calls plan_checker itself. The action mask (main.compute_drl_action_mask)
only allows picks that keep the whole selection within the CAPEX/TOTAL budget and the
downtime limit and keep the network connected.

Usage (run from the repository root or from sim/ml)
-----
    python sim/ml/DRL_code.py --check                      # quick smoke test (no training)
    python sim/ml/DRL_code.py --n-envs 4 --max-decisions 4 --time-limit-minutes 60   # timing test
    python sim/ml/DRL_code.py --timesteps 50000 --n-envs 4 # train with 4 simulators in parallel
    python sim/ml/DRL_code.py --resume sim/ml/drl_optical_policy.zip --timesteps 20000
    python sim/ml/DRL_code.py --evaluate sim/ml/drl_optical_policy.zip   # vs random + incremental
    python sim/ml/DRL_code.py --evaluate sim/ml/drl_optical_policy.zip --eval-heuristics greedy,jump_to_max
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
#      from main.apply_drl_decision (executed / rejected / pass). Replaced by
#      CycleReward in D20 (your values kept as --reward legacy).
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
#  D19 FIRST-RUN FIXES (after Akansha's first run on real packages):
#      - --n-steps (default 32): PPO learns after every n_steps x n_envs steps.
#        The old fixed 256 meant 256 x 7 = 1,792 steps before the FIRST update,
#        which takes days once the simulated network is older. --batch-size is
#        picked automatically to divide the rollout evenly.
#      - --max-cycles N: each training episode stops just before upgrade cycle
#        N+1 starts (main.py's M21 limit, so training, main.py and --evaluate
#        all cut at the same point). It counts as truncated, not as the end of
#        the lifetime, so there is no terminal penalty. Episodes finish, curves
#        appear, and early training uses the fast early years. Raise it over
#        time (e.g. 4 -> 8 -> no limit) with --resume.
#      - Progress line every --progress-minutes (default 10): steps, steps/hour,
#        learning updates so far and steps until the next one, episodes finished.
#      - --time-limit-minutes N: stop cleanly (and save) after N minutes. Used
#        for the timing test and for overnight runs.
#      - Curves: a third panel with BBP at episode end; the lifetime panel says
#        when episodes were cut off by --max-cycles (Monitor logs lifetime_ended).
#      - --evaluate: --eval-max-cycles N, and "lifetime_reached_threshold" is now
#        False for runs stopped by a day or cycle cap.
#      - Old monitor / validation CSVs with the previous columns are renamed
#        (*.pre_D19) instead of being appended to with mismatched columns.
#  D20 NEW REWARD (CycleReward) + FAIR COMPARISON SETTINGS:
#      - Old reward: 10 x (1 - cumulative BBP) - 5 per upgrade, -100 on a
#        plan_checker reject, -10000 at end of life. Cumulative BBP is ~0.003 for
#        most of the lifetime, so every cycle scored ~5 whatever the agent did,
#        and the -10000 spike was 2000x a normal cycle. Kept as --reward legacy
#        (ablation only).
#      - New reward, once per finished cycle (picks inside a cycle still get 0):
#          r = (days / 180) * max(1 - w / threshold, floor)
#              - lambda * (cost added / B) - kappa * [fallback used]
#        w = blocking in THIS window only (blocked / arrived since the agent's
#        last decision); threshold = CONFIG['blocking_threshold'] (0.1);
#        floor = -3; cost added = change in total_network_cost (CAPEX +
#        workforce + OPEX, including any fallback spending); B = half of year 0's
#        total budget (one cycle's share); fallback = the cycle was handed to
#        incremental (rejected or nothing selected). Summed over an episode, the
#        first term is the network's lifetime in cycles (minus blocking), so the
#        agent maximises what the paper reports. No extra end-of-life penalty:
#        ending early already loses every later +1.
#      - Defaults = the conference setting: lambda 0 (year-end spending is on, as
#        for the heuristics, so leftover money is use-it-or-lose-it and spending it
#        costs nothing extra), kappa 0.5 (big enough that handing cycles to
#        incremental on purpose does not pay, small enough not to make the agent
#        timid), floor -3. JOCN: --no-spend-all-last-cycle with --reward-lambda.
#      - Discount 0.999 (was 0.99): gamma applies per PICK, so with 0.99 a
#        15-pick cycle shrank the future 14% and a 2-pick cycle 2%; the number of
#        picks should not change how much the future counts. Entropy bonus 0.01
#        so the policy keeps exploring the 361 actions early on.
#      - Spending settings passed to every simulator (also the parallel ones):
#        --year-end-override (main.py M22 turned it off by default: the
#        heuristics never get that extra chance) and --no-spend-all-last-cycle.
#      - --max-decisions N: a training episode ends after the agent's N-th upgrade
#        decision. With the year-end override off, the gatekeeper can say "no" for
#        the first several calendar cycles, so --max-cycles 4 could end every
#        episode before the agent ever acts. --max-decisions always waits for the
#        first decision; use it for the curriculum (4 -> 8 -> none).
#      - Training settings are saved next to the policy (<policy>.settings.json);
#        --evaluate warns if its settings differ.
#      - --evaluate scores EVERY method with the same reward (DRL, random agent,
#        incremental, plus any --eval-heuristics), 180-day windows, so you can
#        check the reward ranks the heuristics sensibly before training. Also
#        reports fallbacks and days per $1M. Runs to the lifetime threshold
#        unless --eval-max-days / --eval-max-cycles are given.
#  D21 ABLATIONS, PARALLEL RUNS, GATEKEEPER TIMING:
#      - Every conference ablation is a flag (sets main.py CONFIG for this run and
#        all its worker processes; nothing to edit by hand):
#          --trigger heuristic        your trigger decides IF, DRL decides WHAT
#          --gatekeeper-threshold 0   DRL alone (every cycle reaches it; it may save)
#          --upgrade-levels increment DRL with a fixed upgrade step
#          --no-early-stop            budget saving off
#        and --eval-gatekeeper-incremental adds the "gatekeeper + incremental"
#        baseline to --evaluate (main.py M23 ml_selector='heuristic').
#      - --out-dir DIR: the policy, logs and checkpoints of this run go under DIR,
#        and its simulator result folders (in results/) are named after DIR, so
#        several runs can train at once on the cluster without overwriting each
#        other. --baseline-cache shares the
#        heuristic results between runs; --eval-json writes the results table.
#      - Heuristic baselines (incremental, --eval-heuristics) no longer depend on
#        the ML settings in the cache key, so every run reuses them.
#      - --time-gatekeeper: runs incremental on the given seeds and, at every
#        upgrade check, also asks the gatekeeper (never obeys it). Prints how often
#        they agree and how long each takes (the gatekeeper's speed claim).
#      - run_experiments.py (next to this file) chains everything per variant.
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


##############################################################################
# ### ML-INTEGRATION CHANGE D20 (Claude) ###
# NEW: the reward, in one place, used by training AND by --evaluate (so every
# method is scored the same way). See the D20 log at the top for the reasoning.
##############################################################################
REWARD_DEFAULTS = {"mode": "window", "lam": 0.0, "kappa": 0.5, "floor": -3.0}


class CycleReward:
    """Per-cycle reward.

    window (default):
        r = (days / cycle_days) * max(1 - w / threshold, floor)
            - lam * (cost added / B) - kappa * [fallback used]
    legacy (ablation only): the original 10 * (1 - cumulative BBP) - 5 per upgrade,
        -100 on a plan_checker reject, -10000 at the end of the lifetime.

    start(state) marks the beginning of a window; window(state) measures everything
    since then and starts the next window.
    """

    def __init__(self, mode="window", lam=0.0, kappa=0.5, floor=-3.0,
                 terminal_penalty=-10000.0, action_cost_penalty=5.0, invalid_action_penalty=-100.0):
        if mode not in ("window", "legacy"):
            raise ValueError(f"unknown reward mode {mode!r} (use 'window' or 'legacy')")
        self.mode = mode
        self.lam = float(lam)
        self.kappa = float(kappa)
        self.floor = float(floor)
        self.terminal_penalty = float(terminal_penalty)
        self.action_cost_penalty = float(action_cost_penalty)
        self.invalid_action_penalty = float(invalid_action_penalty)
        self._snap = None

    def settings(self):
        return {"reward": self.mode, "reward_lambda": self.lam,
                "reward_kappa": self.kappa, "reward_floor": self.floor}

    @staticmethod
    def budget_ref(state):
        """B: one cycle's share of year 0's total budget (the budget is per year, 2 cycles a year)."""
        return float(state["budget_params"]["period_total_budget_0"]) / 2.0

    @staticmethod
    def _read(state):
        m = state["metrics"]
        return (float(state["Current_global_time"]), int(m.connection_count),
                int(m.blocked_connection_count), float(state["total_network_cost"]))

    def start(self, state):
        self._snap = self._read(state)

    def window(self, state):
        """Everything since the last start()/window(), then start the next window."""
        now = self._read(state)
        t0, n0, b0, c0 = self._snap if self._snap is not None else now
        self._snap = now
        days = max(now[0] - t0, 0.0)
        arrivals, blocked, spent = now[1] - n0, now[2] - b0, now[3] - c0
        threshold = float(state["config"]["blocking_threshold"])
        cycle_days = float(state["budget_params"]["period_days"])
        w = blocked / arrivals if arrivals > 0 else 0.0
        life = (days / cycle_days) * max(1.0 - w / threshold, self.floor)
        return {"days": days, "window_bbp": w, "life": life,
                "cost": spent / self.budget_ref(state), "spent": spent}

    def total(self, life, cost, fallbacks):
        """Reward for already-measured parts (also used to score whole evaluation runs)."""
        return float(life) - self.lam * float(cost) - self.kappa * float(fallbacks)

    def cycle_reward(self, state, result, terminated):
        """Reward for the cycle the agent just finished (result = apply_drl_decision's result)."""
        comp = self.window(state)
        comp["fallback"] = result.get("fallback") is not None
        if self.mode == "legacy":
            if terminated:
                return self.terminal_penalty, comp
            r = (1.0 - float(state["metrics"].get_blocking_probability())) * 10.0
            if result["status"] == "executed":
                r -= self.action_cost_penalty
            elif result["status"].startswith("rejected"):
                r += self.invalid_action_penalty
            return r, comp
        return self.total(comp["life"], comp["cost"], int(comp["fallback"])), comp


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
        max_decisions_per_episode=None,      # optional truncation (counts DRL decisions)
        max_cycles=None,                     # D19: stop the episode after upgrade cycle N
        reward_mode=REWARD_DEFAULTS["mode"],    # D20: 'window' (new) or 'legacy' (old, ablation)
        reward_lambda=REWARD_DEFAULTS["lam"],   # D20: weight on money spent (0 = conference setting)
        reward_kappa=REWARD_DEFAULTS["kappa"],  # D20: penalty when the cycle falls back to incremental
        reward_floor=REWARD_DEFAULTS["floor"],  # D20: lowest value of the lifetime term per cycle
        terminal_penalty=-10000.0,           # legacy reward only
        action_cost_penalty=5.0,             # legacy reward only
        invalid_action_penalty=-100.0,       # legacy reward only
    ):
        super().__init__()
        self.sim_config = copy.deepcopy(sim_config)
        self.sim_config["ml_mode"] = "train"
        # D19: the cycle limit uses main.py's own limit (M21), so a training episode stops
        # exactly where main.py and --evaluate stop: just before cycle N+1 starts. The env
        # reports that as TRUNCATED (no terminal penalty), not as the end of the lifetime.
        self.sim_config["max_upgrade_cycles_for_debug"] = int(max_cycles) if max_cycles else None
        if run_tag:
            self.sim_config["ml_run_tag"] = str(run_tag)
        self.train_seeds = tuple(int(s) for s in train_seeds)
        if not self.train_seeds:
            raise ValueError("train_seeds must not be empty")
        self.quiet = quiet
        self.max_decisions_per_episode = max_decisions_per_episode
        self.max_cycles = int(max_cycles) if max_cycles else None
        self.reward = CycleReward(mode=reward_mode, lam=reward_lambda, kappa=reward_kappa,
                                  floor=reward_floor, terminal_penalty=terminal_penalty,
                                  action_cost_penalty=action_cost_penalty,
                                  invalid_action_penalty=invalid_action_penalty)    # D20

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
        lifetime ends. Returns True if the episode ended: either the BBP threshold was reached
        (self._cut_by_cycle_limit False) or the --max-cycles limit was hit (True, D19).
        Decision points where no candidate link is feasible are resolved automatically
        (STOP -> incremental fallback) because the agent has nothing to choose."""
        self._cut_by_cycle_limit = False
        with self._sim():
            while True:
                self.sim_state, _, done = run_single_cycle(self.sim_state)
                if done:
                    self.sim_state["pending_drl_decision"] = None
                    self._cut_by_cycle_limit = bool(self.sim_state.get("stopped_by_cycle_limit"))
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
    # ### ML-INTEGRATION CHANGE D6 / D20 (Claude) ###
    # The reward now lives in CycleReward (above the class), shared with --evaluate.
    # REMOVED: _calculate_reward() with the old values (still available as
    # reward_mode='legacy').
    ##############################################################################

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
            ended = self._advance_to_decision()
            if not ended:
                self.reward.start(self.sim_state)          # D20: first reward window starts here
                return self._observe(), {"sim_seed": sim_seed, "bbp": self._bbp(),
                                         "day": float(self.sim_state["Current_global_time"])}
            close_state_files(self.sim_state)

        raise RuntimeError(
            "No upgrade decision point was reached for any training seed (the network hit its BBP "
            "threshold, or the --max-cycles limit came first because the gatekeeper said 'no' "
            "every cycle). Use --max-decisions instead of --max-cycles (it always waits for the "
            "first decision), or a larger --max-cycles."
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
                    "status": "picking", "fallback": None, "final_links": [],
                    "lifetime_ended": False,
                    "total_cost": float(self.sim_state["total_network_cost"])}
            return self._observe(), 0.0, False, False, info

        self.decisions_this_episode += 1
        ended = self._advance_to_decision()
        cut = ended and self._cut_by_cycle_limit           # D19: --max-cycles reached
        terminated = ended and not cut                       # real end of the network's lifetime
        bbp = self._bbp()
        reward, parts = self.reward.cycle_reward(self.sim_state, result, terminated)   # D20

        truncated = bool(
            not terminated
            and self.max_decisions_per_episode is not None
            and self.decisions_this_episode >= self.max_decisions_per_episode
        )
        truncated = truncated or cut                         # D19
        info = {
            "bbp": bbp,
            "day": float(self.sim_state["Current_global_time"]),
            "cycle_complete": True,
            "selected_links": result["selected_links"],
            "selected_levels": result["selected_levels"],
            "status": result["status"],
            "fallback": result["fallback"],
            "final_links": result["final_links"],
            "lifetime_ended": bool(terminated),        # D19: False when cut off by --max-cycles
            "total_cost": float(self.sim_state["total_network_cost"]),            # D20
            "window_bbp": parts["window_bbp"], "reward_life": parts["life"],      # D20
            "reward_cost": parts["cost"], "spent": parts["spent"],
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
              f"reward={reward:.3f} (lifetime {info['reward_life']:.3f}, cost {info['reward_cost']:.3f}, "
              f"window BBP {info['window_bbp']:.5f}), bbp={info['bbp']:.5f}, day={info['day']:.1f}")
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
        try:
            df = pd.read_csv(path, skiprows=1)
        except Exception:            # D19: empty / half-written file (e.g. a killed run): skip it
            continue
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
    # D19: BBP at episode end, and whether episodes were cut off by --max-cycles.
    has_bbp = 'bbp' in results_df.columns
    if has_bbp:
        results_df['smoothed_bbp'] = results_df['bbp'].rolling(window=window_size, min_periods=1).mean()
    cut_off = False
    if 'lifetime_ended' in results_df.columns:
        ended = results_df['lifetime_ended'].astype(str).str.lower().isin(['true', '1'])
        cut_off = bool((~ended).any())

    fig, axes = plt.subplots(3 if has_bbp else 2, 1, figsize=(10, 11 if has_bbp else 8), sharex=True)
    ax1, ax2 = axes[0], axes[1]

    ax1.plot(results_df['timesteps'], results_df['r'], color='lightskyblue', alpha=0.4, label='Raw Episode Reward')
    ax1.plot(results_df['timesteps'], results_df['smoothed_reward'], color='navy', linewidth=2, label='Moving Average Reward')
    ax1.set_title('DRL Reward & Penalty Improvement Curve Over Training', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Total Episode Reward', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='lower right')

    ax2.plot(results_df['timesteps'], results_df['day'], color='lightgreen', alpha=0.4, label='Raw Network Lifetime')
    ax2.plot(results_df['timesteps'], results_df['smoothed_lifetime'], color='darkgreen', linewidth=2, label='Moving Average Lifetime')
    if cut_off:   # D19
        ax2.set_title('Day at episode end (episodes cut off by --max-decisions/--max-cycles, not full lifetimes)',
                      fontsize=12, fontweight='bold')
    else:
        ax2.set_title('Network Lifetime (days until BBP crosses threshold)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Lifetime (days)', fontsize=10)
    ax2.ticklabel_format(axis='y', useOffset=False, style='plain')   # D19: no '+5.4e2' offset
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='lower right')
    if has_bbp:   # D19
        ax3 = axes[2]
        ax3.plot(results_df['timesteps'], results_df['bbp'], color='navajowhite', alpha=0.6, label='Raw BBP at episode end')
        ax3.plot(results_df['timesteps'], results_df['smoothed_bbp'], color='darkorange', linewidth=2, label='Moving Average BBP')
        ax3.set_title('Blocking probability at episode end (lower is better)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('BBP', fontsize=10)
        ax3.grid(True, linestyle='--', alpha=0.6)
        ax3.legend(loc='upper right')
    axes[-1].set_xlabel('Total Training Steps (link picks)', fontsize=10)

    plt.tight_layout()
    plot_output_path = os.path.join(log_dir, "drl_training_curves.png")
    plt.savefig(plot_output_path, dpi=150)
    plt.close(fig)
    return plot_output_path


def make_checkpoint_callback(save_every, checkpoint_dir, latest_model_path, log_dir,
                             progress_minutes=10.0, time_limit_minutes=None, steps_per_update=None):
    """
    Every `save_every` training steps:
      - saves drl_logs/checkpoints/drl_policy_<steps>_steps.zip   (history)
      - overwrites the main policy zip (latest_model_path)          (always usable by main.py)
      - refreshes drl_training_curves.png                           (watch for the plateau)
    D19: also prints a progress line every `progress_minutes` and stops training cleanly
    once `time_limit_minutes` have passed (the caller then saves the model).
    """
    import time
    from stable_baselines3.common.callbacks import BaseCallback

    def _hms(sec):
        sec = int(sec)
        return f"{sec // 3600}h{(sec % 3600) // 60:02d}m{sec % 60:02d}s"

    class CheckpointAndPlotCallback(BaseCallback):
        def __init__(self):
            super().__init__(verbose=0)
            self._last_save = None      # set on the first step (works for fresh and resumed runs)
            self._t0 = None             # D19 progress bookkeeping
            self._start_steps = 0
            self._last_report_t = None
            self._last_report_steps = 0
            self._episodes = 0

        def _progress(self, final=False):
            now = time.time()
            run_steps = self.num_timesteps - self._start_steps
            elapsed = max(now - self._t0, 1e-9)
            recent = max(now - self._last_report_t, 1e-9)
            rate_all = run_steps / elapsed * 3600.0
            rate_recent = (self.num_timesteps - self._last_report_steps) / recent * 3600.0
            rates = (f"{rate_all:,.0f} steps/h overall" if final else
                     f"{rate_recent:,.0f} steps/h recently, {rate_all:,.0f} steps/h overall")
            line = (f"[progress{' - final' if final else ''}] {_hms(elapsed)} | "
                    f"{self.num_timesteps} steps total ({run_steps} this run) | {rates} | "
                    f"episodes finished: {self._episodes}")
            if steps_per_update:
                done_updates = run_steps // steps_per_update
                to_next = steps_per_update - (run_steps % steps_per_update)
                line += f" | learning updates: {done_updates} (next after {to_next} more steps)"
            print(line, flush=True)
            self._last_report_t, self._last_report_steps = now, self.num_timesteps

        def _save(self):
            os.makedirs(checkpoint_dir, exist_ok=True)
            ckpt = os.path.join(checkpoint_dir, f"drl_policy_{self.num_timesteps}_steps.zip")
            self.model.save(ckpt)
            self.model.save(latest_model_path)
            try:                     # D19: a plotting problem must never stop training
                plot = plot_training_curves(log_dir)
            except Exception as e:
                plot = None
                print(f"⚠️ Could not refresh the training curves ({e}); training continues.")
            print(f"💾 Checkpoint at {self.num_timesteps} steps -> {ckpt} "
                  f"(also {latest_model_path}){' | curves updated' if plot else ''}")

        def _on_step(self) -> bool:
            if self._last_save is None:
                self._last_save = self.num_timesteps - 1
            if self._t0 is None:                                   # D19
                self._t0 = self._last_report_t = time.time()
                n_envs_now = int(getattr(self.model, "n_envs", 1) or 1)
                self._start_steps = self._last_report_steps = self.num_timesteps - n_envs_now
            dones = self.locals.get("dones") if isinstance(self.locals, dict) else None
            if dones is not None:
                self._episodes += int(np.sum(dones))
            if self.num_timesteps - self._last_save >= save_every:
                self._last_save = self.num_timesteps
                self._save()
            now = time.time()
            if progress_minutes and now - self._last_report_t >= 60.0 * progress_minutes:
                self._progress()
            if time_limit_minutes and now - self._t0 >= 60.0 * time_limit_minutes:
                print(f"⏱ Time limit of {time_limit_minutes} min reached: stopping training "
                      f"(the model is saved next).", flush=True)
                return False
            return True

        def _on_training_end(self) -> None:                        # D19
            if self._t0 is not None:
                self._progress(final=True)

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


MONITOR_KEYWORDS = ("bbp", "day", "lifetime_ended", "total_cost")   # D19: + lifetime_ended; D20: + total_cost


def _set_aside_if_columns_differ(path, expected_columns):
    """D19: a CSV written by an older version has different columns; appending to it
    would corrupt it. Rename it to <name>.old (kept, but no longer read) instead."""
    if not os.path.exists(path):
        return
    try:
        with open(path) as f:
            first = f.readline()
            header = f.readline() if first.startswith("#") else first
        columns = [c.strip() for c in header.strip().split(",")]
    except Exception:
        columns = []
    if columns != list(expected_columns):
        old = path + ".old"
        if os.path.exists(old):
            os.remove(old)
        os.replace(path, old)
        print(f"Note: {os.path.basename(path)} had older columns; kept as {os.path.basename(old)}.")


def make_training_env(rank, quiet, log_dir, override_existing, n_envs, max_cycles=None,
                      sim_overrides=None, reward_kwargs=None, max_decisions=None, tag_prefix=""):
    """One training simulator: OpticalNetworkEnv -> ActionMasker -> Monitor (top level so it
    can be sent to worker processes). D20: sim_overrides (spending settings) and the reward
    settings are passed in explicitly, because a worker process re-imports main.py and would
    otherwise see the default CONFIG."""
    from stable_baselines3.common.monitor import Monitor
    from sb3_contrib.common.wrappers import ActionMasker
    sim_config = dict(CONFIG, **(sim_overrides or {}))
    env = OpticalNetworkEnv(sim_config=sim_config, quiet=quiet,
                            run_tag=(f"{tag_prefix}train_env{rank}" if (n_envs > 1 or tag_prefix) else None),
                            max_cycles=max_cycles,                       # D19
                            max_decisions_per_episode=max_decisions,     # D20
                            **(reward_kwargs or {}))                     # D20
    env = ActionMasker(env, _mask_fn)
    monitor_path = (os.path.join(log_dir, "monitor.csv") if n_envs == 1
                    else os.path.join(log_dir, f"env{rank}.monitor.csv"))
    if not override_existing:                                           # D19
        _set_aside_if_columns_differ(monitor_path, ("r", "l", "t") + MONITOR_KEYWORDS)
    # Append when resuming, but a log that does not exist yet must be created with its header.
    override_existing = override_existing or not os.path.exists(monitor_path)
    return Monitor(env, monitor_path, info_keywords=MONITOR_KEYWORDS,
                   override_existing=override_existing)


##############################################################################
# ### ML-INTEGRATION CHANGE D17 / D19 / D20 (Claude) ###
# D20: every method is scored with the SAME reward (CycleReward) over fixed 180-day
# windows, so the reward can be checked on the heuristics before any training.
# Extra heuristics via --eval-heuristics; spending settings passed to every worker.
##############################################################################
def _settings_tag(sim_overrides, floor, method=None):
    """Cache key for a baseline row. D21: the heuristic baselines (incremental, heuristic_*)
    do not use any ML setting, so only the reward floor is part of their key."""
    if method is not None and (method == "incremental" or method.startswith("heuristic_")):
        return f"floor={float(floor):g}"
    merged = {k: CONFIG.get(k) for k in ML_SETTING_KEYS}
    merged.update(sim_overrides or {})
    return ";".join(f"{k.replace('ml_', '')}={merged[k]}" for k in ML_SETTING_KEYS) + f";floor={float(floor):g}"


ML_SETTING_KEYS = ("ml_year_end_override_gatekeeper", "ml_spend_all_last_cycle_of_year", "ml_trigger",
                   "ml_gatekeeper_threshold", "ml_upgrade_levels", "ml_allow_early_stop")


def _evaluate_one(job):
    """Run ONE network lifetime for job['method'] on job['seed'].
    method: 'drl' | 'random' (random agent, same choices and masks as the DRL) |
            'incremental' | 'heuristic_<name>' (any of Akansha's algorithms) |
            'gatekeeper_incremental' (D21 ablation)."""
    import math
    import time
    method, seed = job["method"], int(job["seed"])
    max_days, max_cycles = job.get("max_days"), job.get("max_cycles")
    cfg = dict(copy.deepcopy(CONFIG), **(job.get("sim_overrides") or {}))
    cfg.update(seed=seed, collect_xgboost_data=False,
               ml_run_tag=f"{job.get('tag_prefix', '')}eval_{method}_s{seed}")
    cfg["max_upgrade_cycles_for_debug"] = int(max_cycles) if max_cycles else None   # D19 (main.py M21)
    if method == "incremental":
        cfg.update(ml_mode="off", algorithm="increment")
    elif method.startswith("heuristic_"):
        cfg.update(ml_mode="off", algorithm=method[len("heuristic_"):])
    elif method == "gatekeeper_incremental":          # D21: gatekeeper decides IF, incremental WHAT
        cfg.update(ml_mode="train", ml_selector="heuristic")
    else:
        cfg["ml_mode"] = "train"          # same gatekeeper setting the DRL trains with
    policy = None
    if method == "drl":
        from sb3_contrib import MaskablePPO
        policy = MaskablePPO.load(job["policy_path"], device="cpu")
    rng = np.random.default_rng(seed)
    meter = CycleReward(floor=job.get("reward_floor", REWARD_DEFAULTS["floor"]))
    t0 = time.time()
    devnull = open(os.devnull, "w")
    ctx = contextlib.redirect_stdout(devnull) if job.get("quiet", True) else contextlib.nullcontext()
    life = cost_units = 0.0
    with ctx:
        st = initialize_simulation_state(cfg)
        meter.start(st)
        cycle_days = float(st["budget_params"]["period_days"])
        next_edge = cycle_days
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
            now = float(st["Current_global_time"])
            if now >= next_edge:                    # D20: close a 180-day reward window
                part = meter.window(st)
                life += part["life"]
                cost_units += part["cost"]
                next_edge = (math.floor(now / cycle_days) + 1) * cycle_days
            if max_days and now >= float(max_days):
                break
        part = meter.window(st)                     # last (partial) window
        life += part["life"]
        cost_units += part["cost"]
    close_state_files(st)
    devnull.close()
    total_cost = float(st["total_network_cost"])
    days = float(st["Current_global_time"])
    return {
        "method": method, "seed": seed, "max_days": max_days or "",
        "max_cycles": max_cycles or "",                                    # D19
        "settings": _settings_tag(job.get("sim_overrides"), meter.floor, method),  # D20/D21
        # D19: a run stopped by --eval-max-cycles also returns done=True; that is not a lifetime end.
        "lifetime_reached_threshold": bool(done) and not st.get("stopped_by_cycle_limit", False),
        "days_simulated": round(days, 1),
        "final_bbp": round(float(st["metrics"].get_blocking_probability()), 6),
        "total_cost": round(total_cost, 2),
        "days_per_million": round(days / (total_cost / 1e6), 2) if total_cost > 0 else "",   # D20
        "upgrades_committed": len(st["upgrade_decision_history"]),
        "fallbacks": sum(1 for d in st.get("drl_decision_log", []) if d.get("fallback") is not None),
        "reward_life": round(life, 4),                                     # D20
        "reward_cost_units": round(cost_units, 4),                         # D20 (cost / B)
        "wall_seconds": round(time.time() - t0, 1),
    }


EVAL_FIELDS = ["method", "seed", "max_days", "max_cycles", "settings", "lifetime_reached_threshold",
               "days_simulated", "final_bbp", "total_cost", "days_per_million", "upgrades_committed",
               "fallbacks", "reward_life", "reward_cost_units", "wall_seconds"]
RESULT_FIELDS = ["evaluated_at", "policy", "reward_lambda", "reward_kappa", "reward_return"] + EVAL_FIELDS


def _settings_path(policy_path):
    return os.path.splitext(str(policy_path))[0] + ".settings.json"


def run_evaluation(policy_path, seeds, max_days, workers, log_dir, quiet=True, max_cycles=None,
                   heuristics=(), sim_overrides=None, scorer=None, gatekeeper_incremental=False,
                   baseline_file=None, json_out=None, tag_prefix=""):
    """Policy vs random agent vs incremental (+ any extra heuristics) on `seeds`.
    Every method is scored with the same reward. Writes CSVs and prints a summary table."""
    import csv, json, time
    import multiprocessing as mp
    scorer = scorer or CycleReward(**REWARD_DEFAULTS)
    sim_overrides = dict(sim_overrides or {})
    os.makedirs(log_dir, exist_ok=True)
    base_file = baseline_file or os.path.join(log_dir, "validation_baselines.csv")   # D21: shareable
    if os.path.dirname(base_file):
        os.makedirs(os.path.dirname(base_file), exist_ok=True)
    res_file = os.path.join(log_dir, "validation_results.csv")
    _set_aside_if_columns_differ(base_file, EVAL_FIELDS)
    _set_aside_if_columns_differ(res_file, RESULT_FIELDS)

    no_drl = str(policy_path).lower() == "none"          # D21: --evaluate none = baselines only
    # D20: warn when the policy was trained with different settings than this evaluation uses.
    eval_settings = dict(scorer.settings(),
                         **{k: sim_overrides.get(k, CONFIG.get(k)) for k in ML_SETTING_KEYS})
    try:
        with open(_settings_path(policy_path)) as f:
            trained = json.load(f)
        # D21: a setting missing from the file was the main.py default at training time
        diff = {k: (trained.get(k, CONFIG.get(k)), v) for k, v in eval_settings.items()
                if trained.get(k, CONFIG.get(k)) != v}
        if diff:
            print("WARNING: this evaluation uses different settings than training "
                  "(setting: trained -> now): " + ", ".join(f"{k}: {a} -> {b}" for k, (a, b) in diff.items()))
    except (OSError, ValueError):
        pass

    tag = _settings_tag(sim_overrides, scorer.floor)
    key = lambda r: (r["method"], int(r["seed"]), str(r["max_days"]), str(r["max_cycles"]), str(r["settings"]))
    baselines = ["incremental", "random"] + [f"heuristic_{h}" for h in heuristics if h != "increment"]
    if gatekeeper_incremental:
        baselines.append("gatekeeper_incremental")                             # D21
    _caps = lambda m: (str(max_days or ""), str(max_cycles or ""), _settings_tag(sim_overrides, scorer.floor, m))

    cached = {}
    if os.path.exists(base_file):
        with open(base_file, newline="") as f:
            for r in csv.DictReader(f):
                cached[key(r)] = r
    common = dict(max_days=max_days, max_cycles=max_cycles, quiet=quiet,
                  sim_overrides=sim_overrides, reward_floor=scorer.floor, tag_prefix=tag_prefix)
    jobs = [] if no_drl else [dict(common, method="drl", seed=s, policy_path=policy_path) for s in seeds]
    for m in baselines:
        jobs += [dict(common, method=m, seed=s) for s in seeds if (m, int(s)) + _caps(m) not in cached]

    workers = max(1, min(workers or os.cpu_count() or 1, len(jobs)))
    print(f"Evaluating on seeds {list(seeds)}: {len(jobs)} simulations on {workers} worker(s) "
          f"(cached baselines reused: {sum(1 for m in baselines for s in seeds if (m, int(s)) + _caps(m) in cached)})...")
    print(f"Scoring every method with reward lambda={scorer.lam:g}, kappa={scorer.kappa:g}, "
          f"floor={scorer.floor:g}; settings {tag}.")
    if workers == 1:
        results = [_evaluate_one(j) for j in jobs]
    else:
        with mp.get_context("spawn").Pool(workers) as pool:
            results = pool.map(_evaluate_one, jobs)

    new_base = [r for r in results if r["method"] != "drl"]
    if new_base:
        exists = os.path.exists(base_file)
        with open(base_file, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=EVAL_FIELDS)
            if not exists:
                w.writeheader()
            w.writerows(new_base)
    rows = [r for r in results if r["method"] == "drl"] + new_base + [
        cached[(m, int(s)) + _caps(m)] for m in baselines for s in seeds if (m, int(s)) + _caps(m) in cached]
    for r in rows:
        r["reward_return"] = round(scorer.total(r["reward_life"], r["reward_cost_units"], r["fallbacks"]), 4)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    exists = os.path.exists(res_file)
    with open(res_file, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow({"evaluated_at": stamp, "policy": os.path.basename(policy_path),
                        "reward_lambda": scorer.lam, "reward_kappa": scorer.kappa,
                        **{k: r[k] for k in EVAL_FIELDS + ["reward_return"]}})

    methods = ([] if no_drl else ["drl"]) + baselines
    head = (f"{'seed':>5} | {'method':<22} | {'days':>7} | {'hit thr.':>8} | {'final BBP':>9} | "
            f"{'total cost':>13} | {'days/$1M':>8} | {'fallbk':>6} | {'reward':>8}")
    print("\n" + head + "\n" + "-" * len(head))

    def _line(label, r):
        dpm = r["days_per_million"]
        return (f"{label} | {float(r['days_simulated']):>7.0f} | {str(r['lifetime_reached_threshold']):>8} | "
                f"{float(r['final_bbp']):>9.4f} | {float(r['total_cost']):>13,.0f} | "
                f"{(f'{float(dpm):.1f}' if dpm not in ('', None) else '-'):>8} | {int(float(r['fallbacks'])):>6} | "
                f"{float(r['reward_return']):>8.3f}")
    for s in seeds:
        for m in methods:
            r = next(x for x in rows if x["method"] == m and int(x["seed"]) == int(s))
            print(_line(f"{int(s):>5} | {m:<22}", r))
    print("-" * len(head))
    for m in methods:
        rs = [x for x in rows if x["method"] == m]
        mean = {k: sum(float(x[k]) for x in rs) / len(rs)
                for k in ("days_simulated", "final_bbp", "total_cost", "fallbacks", "reward_return")}
        dpms = [float(x["days_per_million"]) for x in rs if x["days_per_million"] not in ("", None)]
        mean["days_per_million"] = sum(dpms) / len(dpms) if dpms else ""
        mean["lifetime_reached_threshold"] = f"{sum(str(x['lifetime_reached_threshold']) == 'True' for x in rs)}/{len(rs)}"
        print(_line(f"{'mean':>5} | {m:<22}", mean))
    print("(days/$1M and lifetime only compare fairly when every run hit the threshold. "
          "If runs were cut off by --eval-max-days or --eval-max-cycles, compare final BBP and "
          "cost at the same point instead.)")
    print(f"Saved: {res_file}")
    if json_out:                                                             # D21
        with open(json_out, "w") as f:
            json.dump({"policy": str(policy_path), "seeds": [int(x) for x in seeds],
                       "settings": eval_settings, "rows": rows}, f, indent=2, default=str)
    return rows


##############################################################################
# ### ML-INTEGRATION CHANGE D21 (Claude) ###
# NEW: gatekeeper timing check (the paper's speed claim). Runs your incremental
# heuristic unchanged; at every upgrade check main.py (M23) also asks the gatekeeper.
##############################################################################
def _time_gatekeeper_one(job):
    import time
    cfg = copy.deepcopy(CONFIG)
    cfg.update(seed=int(job["seed"]), ml_mode="off", algorithm="increment", ml_time_gatekeeper=True,
               collect_xgboost_data=False, ml_run_tag=f"{job.get('tag_prefix', '')}gktime_s{job['seed']}")
    if job.get("threshold") is not None:
        cfg["ml_gatekeeper_threshold"] = float(job["threshold"])
    devnull = open(os.devnull, "w")
    ctx = contextlib.redirect_stdout(devnull) if job.get("quiet", True) else contextlib.nullcontext()
    with ctx:
        st = initialize_simulation_state(cfg)
        done = False
        while not done:
            st, _, done = run_single_cycle(st)
            if job.get("max_days") and float(st["Current_global_time"]) >= float(job["max_days"]):
                break
    close_state_files(st)
    devnull.close()
    return [dict(r, seed=int(job["seed"])) for r in st["gatekeeper_timing"]]


def run_gatekeeper_timing(seeds, max_days, workers, log_dir, quiet=True, threshold=None, tag_prefix=""):
    import csv
    import multiprocessing as mp
    os.makedirs(log_dir, exist_ok=True)
    jobs = [dict(seed=s, max_days=max_days, quiet=quiet, threshold=threshold, tag_prefix=tag_prefix)
            for s in seeds]
    workers = max(1, min(workers or os.cpu_count() or 1, len(jobs)))
    print(f"Timing the gatekeeper against incremental's own check on seeds {list(seeds)} "
          f"({'to day ' + str(max_days) if max_days else 'full lifetimes'}, {workers} worker(s))...")
    if workers == 1:
        parts = [_time_gatekeeper_one(j) for j in jobs]
    else:
        with mp.get_context("spawn").Pool(workers) as pool:
            parts = pool.map(_time_gatekeeper_one, jobs)
    rows = [r for p in parts for r in p]
    out = os.path.join(log_dir, "gatekeeper_timing.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "cycle", "day", "heuristic_says", "gatekeeper_says",
                                          "gatekeeper_prob", "heuristic_seconds", "gatekeeper_seconds"])
        w.writeheader()
        w.writerows(rows)
    if not rows:
        print("No upgrade checks happened (runs too short?).")
        return rows
    n = len(rows)
    agree = sum(r["heuristic_says"] == r["gatekeeper_says"] for r in rows)
    tp = sum(r["heuristic_says"] and r["gatekeeper_says"] for r in rows)
    fn = sum(r["heuristic_says"] and not r["gatekeeper_says"] for r in rows)
    fp = sum(r["gatekeeper_says"] and not r["heuristic_says"] for r in rows)
    h = sum(r["heuristic_seconds"] for r in rows) / n
    g = sum(r["gatekeeper_seconds"] for r in rows) / n
    print(f"\nUpgrade checks: {n}")
    print(f"Agreement with incremental's check: {agree}/{n} = {100.0 * agree / n:.1f}% "
          f"(gatekeeper missed {fn} upgrades, added {fp} extra; both said yes {tp} times)")
    print(f"Average time per check: incremental {h:.3f} s, gatekeeper {1000.0 * g:.2f} ms "
          f"(about {h / g:,.0f}x faster)" if g > 0 else f"Average time: incremental {h:.3f} s")
    print(f"Saved: {out}")
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
                             "incremental on the validation seeds, then exit ('none' = baselines only)")
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
    ##############################################################################
    # ### ML-INTEGRATION CHANGE D19 (Claude) ###
    # NEW: options that make the first training runs practical (see the D19 log above).
    ##############################################################################
    parser.add_argument("--n-steps", type=int, default=None,
                        help="steps per simulator between learning updates (default 32 for a new run; "
                             "a resumed run keeps its saved value unless you pass this)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="PPO minibatch size (default: chosen to divide n_steps x n_envs evenly)")
    parser.add_argument("--max-cycles", type=int, default=None,
                        help="end each training episode after calendar upgrade cycle N (default: full lifetime)")
    parser.add_argument("--max-decisions", type=int, default=None,
                        help="end each training episode after the agent's N-th upgrade decision "
                             "(D20; recommended for the curriculum: 4 -> 8 -> none)")
    parser.add_argument("--progress-minutes", type=float, default=10.0,
                        help="print a progress line every N minutes (0 = off)")
    parser.add_argument("--time-limit-minutes", type=float, default=None,
                        help="stop training cleanly and save after N minutes")
    parser.add_argument("--eval-max-cycles", type=int, default=None,
                        help="with --evaluate: stop each run after upgrade cycle N")
    ##############################################################################
    # ### ML-INTEGRATION CHANGE D20 (Claude) ###
    # NEW: reward, discount and spending options (defaults = the conference setting).
    # Use the SAME values for training and --evaluate.
    ##############################################################################
    parser.add_argument("--reward", choices=("window", "legacy"), default=REWARD_DEFAULTS["mode"],
                        help="'window' = new reward (default); 'legacy' = the original reward (ablation)")
    parser.add_argument("--reward-lambda", type=float, default=REWARD_DEFAULTS["lam"],
                        help="weight on money spent (default 0, the conference setting; JOCN sweep 0.25-2 "
                             "together with --no-spend-all-last-cycle)")
    parser.add_argument("--reward-kappa", type=float, default=REWARD_DEFAULTS["kappa"],
                        help="penalty when a cycle falls back to incremental (default 0.5)")
    parser.add_argument("--reward-floor", type=float, default=REWARD_DEFAULTS["floor"],
                        help="lowest value of the lifetime term per cycle (default -3)")
    parser.add_argument("--gamma", type=float, default=None,
                        help="discount per pick (default 0.999 for a new run; a resumed run keeps its "
                             "saved value unless you pass this)")
    parser.add_argument("--ent-coef", type=float, default=None,
                        help="entropy bonus that keeps the policy exploring (default 0.01 for a new run)")
    parser.add_argument("--year-end-override", action="store_true",
                        help="ablation: the year's last cycle reaches the DRL even when the gatekeeper "
                             "says no (heuristics never get this chance; off by default)")
    parser.add_argument("--no-spend-all-last-cycle", action="store_true",
                        help="JOCN setting: the DRL may also save money in the year's last cycle "
                             "(default: it must spend what fits, like the heuristics' plan_checker top-up)")
    parser.add_argument("--eval-heuristics", default="",
                        help="with --evaluate: extra heuristics to compare, comma-separated "
                             "(e.g. greedy,jump_to_max,random,no_upgrade)")
    ##############################################################################
    # ### ML-INTEGRATION CHANGE D21 (Claude) ###
    # NEW: ablation flags, per-run output folder, shared baseline cache, JSON results,
    # gatekeeper timing. Use the same ablation flags for training and --evaluate.
    ##############################################################################
    parser.add_argument("--trigger", choices=("gatekeeper", "heuristic"), default=None,
                        help="who decides IF: the XGBoost gatekeeper (default) or your heuristic's "
                             "own check (ablation 'trigger + DRL')")
    parser.add_argument("--gatekeeper-threshold", type=float, default=None,
                        help="gatekeeper probability needed to upgrade (default 0.5 from main.py); "
                             "0 = every cycle reaches the DRL (ablation 'DRL alone')")
    parser.add_argument("--upgrade-levels", choices=("any_higher", "increment"), default=None,
                        help="any_higher (default) or increment = one fixed step per link "
                             "(ablation 'fixed upgrade step')")
    parser.add_argument("--no-early-stop", action="store_true",
                        help="ablation 'budget saving off': the DRL may not stop early to save money")
    parser.add_argument("--eval-gatekeeper-incremental", action="store_true",
                        help="with --evaluate: also run the 'gatekeeper + incremental' ablation")
    parser.add_argument("--out-dir", default=None,
                        help="put this run's policy, logs and checkpoints in this folder and name its "
                             "simulator result folders after it (lets several runs train at once)")
    parser.add_argument("--baseline-cache", default=None,
                        help="CSV of saved baseline results to reuse (default: <logs>/validation_baselines.csv)")
    parser.add_argument("--eval-json", default=None,
                        help="with --evaluate: also write the results to this JSON file")
    parser.add_argument("--time-gatekeeper", action="store_true",
                        help="do not train: run incremental on --val-seeds and compare the gatekeeper's "
                             "answer and speed with incremental's own check at every upgrade check")
    args = parser.parse_args()
    for _name in ("n_steps", "batch_size", "max_cycles", "max_decisions", "eval_max_cycles"):
        _v = getattr(args, _name)
        if _v is not None and _v < 1:
            parser.error(f"--{_name.replace('_', '-')} must be at least 1")
    if args.gamma is not None and not 0.0 < args.gamma <= 1.0:
        parser.error("--gamma must be in (0, 1]")
    if args.reward_lambda < 0 or args.reward_kappa < 0:
        parser.error("--reward-lambda and --reward-kappa must be 0 or more")
    if args.reward_floor > 0:
        parser.error("--reward-floor must be 0 or less")
    if args.gatekeeper_threshold is not None and not 0.0 <= args.gatekeeper_threshold <= 1.0:
        parser.error("--gatekeeper-threshold must be between 0 and 1")

    # D20: settings every simulator gets (explicitly, so the parallel workers see them too)
    sim_overrides = {"ml_year_end_override_gatekeeper": bool(args.year_end_override),
                     "ml_spend_all_last_cycle_of_year": not args.no_spend_all_last_cycle}
    if args.trigger:                                                          # D21 ablations
        sim_overrides["ml_trigger"] = args.trigger
    if args.gatekeeper_threshold is not None:
        sim_overrides["ml_gatekeeper_threshold"] = float(args.gatekeeper_threshold)
    if args.upgrade_levels:
        sim_overrides["ml_upgrade_levels"] = "any_higher" if args.upgrade_levels == "any_higher" else ["increment"]
    if args.no_early_stop:
        sim_overrides["ml_allow_early_stop"] = False

    # D21: one folder per run (policy, logs, checkpoints, simulator results)
    if args.out_dir:
        out_root = os.path.abspath(args.out_dir)
        log_dir = os.path.join(out_root, "drl_logs")
        tag_prefix = os.path.basename(out_root.rstrip(os.sep)) + "_"
    else:
        out_root = str(_THIS_DIR)
        log_dir = str(_THIS_DIR / "drl_logs")
        tag_prefix = ""
    os.makedirs(log_dir, exist_ok=True)
    reward_kwargs = {"reward_mode": args.reward, "reward_lambda": args.reward_lambda,
                     "reward_kappa": args.reward_kappa, "reward_floor": args.reward_floor}
    scorer = CycleReward(mode=args.reward, lam=args.reward_lambda, kappa=args.reward_kappa,
                         floor=args.reward_floor)
    print(f"Reward: {args.reward} (lambda={args.reward_lambda:g}, kappa={args.reward_kappa:g}, "
          f"floor={args.reward_floor:g}); year-end override "
          f"{'ON' if args.year_end_override else 'off'}; spend all in the year's last cycle "
          f"{'off' if args.no_spend_all_last_cycle else 'ON'}.")
    _abl = {k: v for k, v in sim_overrides.items()
            if k not in ("ml_year_end_override_gatekeeper", "ml_spend_all_last_cycle_of_year")}
    if _abl:
        print(f"Ablation settings: {_abl}")

    if args.time_gatekeeper:                                                   # D21
        run_gatekeeper_timing([int(s) for s in args.val_seeds.split(",") if s.strip()],
                              args.eval_max_days, args.eval_workers, log_dir,
                              quiet=not args.verbose_sim, threshold=args.gatekeeper_threshold,
                              tag_prefix=tag_prefix)
        sys.exit(0)

    if args.evaluate:
        run_evaluation(args.evaluate, [int(s) for s in args.val_seeds.split(",") if s.strip()],
                       args.eval_max_days, args.eval_workers, log_dir,
                       quiet=not args.verbose_sim, max_cycles=args.eval_max_cycles,
                       heuristics=[h.strip() for h in args.eval_heuristics.split(",") if h.strip()],
                       sim_overrides=sim_overrides, scorer=scorer,                     # D20
                       gatekeeper_incremental=args.eval_gatekeeper_incremental,         # D21
                       baseline_file=args.baseline_cache, json_out=args.eval_json,
                       tag_prefix=tag_prefix)
        sys.exit(0)

    from stable_baselines3.common.vec_env import SubprocVecEnv
    from sb3_contrib import MaskablePPO

    ##############################################################################
    # ### ML-INTEGRATION CHANGE D12 (Claude) ###
    # NEW: SB3 env checker + smoke test, then exit.
    ##############################################################################
    if args.check:
        from stable_baselines3.common.env_checker import check_env
        raw_env = OpticalNetworkEnv(sim_config=dict(CONFIG, **sim_overrides), quiet=not args.verbose_sim,
                                    **reward_kwargs)                                    # D20
        check_env(raw_env, warn=True)
        print("(smoke test simulates 2 upgrade cycles of real traffic: about 10 minutes)")
        run_smoke_test(raw_env, n_decisions=2)
        sys.exit(0)

    import shutil, time as _time
    run_stamp = _time.strftime("%Y%m%d-%H%M%S")
    checkpoint_dir = os.path.join(log_dir, "checkpoints", f"run_{run_stamp}")   # one folder per run
    model_path = os.path.join(out_root, args.out) if not os.path.isabs(args.out) else args.out   # D21

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
        env = make_training_env(0, quiet, log_dir, args.resume is None, 1, args.max_cycles,
                                sim_overrides, reward_kwargs, args.max_decisions, tag_prefix)   # D20/D21
    else:
        env = SubprocVecEnv([functools.partial(make_training_env, rank, quiet, log_dir,
                                               args.resume is None, n_envs, args.max_cycles,
                                               sim_overrides, reward_kwargs, args.max_decisions,
                                               tag_prefix)                              # D20/D21
                             for rank in range(n_envs)])
        print(f"Training with {n_envs} parallel simulators.")
    if args.max_cycles:
        print(f"Episodes end after upgrade cycle {args.max_cycles} (--max-cycles).")
    if args.max_decisions:
        print(f"Episodes end after {args.max_decisions} upgrade decisions by the agent (--max-decisions).")

    ##############################################################################
    # ### ML-INTEGRATION CHANGE D19 (Claude) ###
    # PPO learns once every n_steps x n_envs steps. Pick a minibatch size that divides
    # that rollout evenly (about 64), so no data is dropped and SB3 does not warn.
    ##############################################################################
    def _pick_batch(buffer_size, wanted=None):
        if wanted:
            return max(2, min(int(wanted), buffer_size))
        n_batches = max(1, round(buffer_size / 64))
        while buffer_size % n_batches:
            n_batches += 1
        return max(2, buffer_size // n_batches)

    ##############################################################################
    # ### ML-INTEGRATION CHANGE D12 (Claude) ###
    # CHANGED: n_steps=256, batch_size=64 (PPO default 2048 is too many full
    # upgrade cycles per update). Monitor above now also logs bbp and day.
    ##############################################################################
    if args.resume:
        print(f"Resuming from {args.resume}")
        custom = {}
        if args.n_steps is not None or args.batch_size is not None:      # D19: change rollout size on resume
            from sb3_contrib import MaskablePPO as _PPO
            saved = _PPO.load(args.resume, device="cpu")
            n_steps = args.n_steps or int(saved.n_steps)
            del saved
            custom.update(n_steps=n_steps, batch_size=_pick_batch(n_steps * n_envs, args.batch_size))
        if args.gamma is not None:                                       # D20
            custom["gamma"] = float(args.gamma)
        if args.ent_coef is not None:                                    # D20
            custom["ent_coef"] = float(args.ent_coef)
        model = MaskablePPO.load(args.resume, env=env, device="cpu", custom_objects=custom or None)
        model.set_random_seed(args.seed + int(model.num_timesteps))   # D18: new stream, still reproducible
    else:
        n_steps = args.n_steps or 32                                    # D19: was a fixed 256
        model = MaskablePPO("MlpPolicy", env, verbose=1, learning_rate=3e-4,
                            gamma=args.gamma if args.gamma is not None else 0.999,          # D20: was 0.99
                            ent_coef=args.ent_coef if args.ent_coef is not None else 0.01,  # D20: was 0
                            n_steps=n_steps, batch_size=_pick_batch(n_steps * n_envs, args.batch_size),
                            seed=args.seed)   # D18: seed
    steps_per_update = int(model.n_steps) * n_envs                     # D19
    print(f"PPO learns after every {int(model.n_steps)} x {n_envs} = {steps_per_update} steps "
          f"(minibatch {int(model.batch_size)}, gamma {float(model.gamma):g}, "
          f"entropy bonus {float(model.ent_coef):g}).")

    # D20: record the settings next to the policy so --evaluate can check it uses the same ones.
    import json as _json
    _settings = dict(scorer.settings(), **sim_overrides, gamma=float(model.gamma),
                     ent_coef=float(model.ent_coef), max_cycles=args.max_cycles,
                     max_decisions=args.max_decisions,
                     n_envs=n_envs, n_steps=int(model.n_steps), seed=args.seed,
                     resumed_from=args.resume, started=run_stamp)
    with open(_settings_path(model_path), "w") as _f:
        _json.dump(_settings, _f, indent=2)

    ##############################################################################
    # ### ML-INTEGRATION CHANGE D14 (Claude) ###
    # NEW: checkpoints during training; Ctrl+C stops cleanly and still saves.
    # The policy zip next to this script is always the latest checkpoint, so you
    # can stop as soon as drl_logs/drl_training_curves.png has plateaued.
    ##############################################################################
    callback = make_checkpoint_callback(args.save_every, checkpoint_dir, model_path, log_dir,
                                        progress_minutes=args.progress_minutes,
                                        time_limit_minutes=args.time_limit_minutes,
                                        steps_per_update=steps_per_update)       # D19
    print(f"Training MaskablePPO for {args.timesteps} steps "
          f"(checkpoint + curves every {args.save_every}; Ctrl+C to stop and save)"
          f"{f'; stops after {args.time_limit_minutes:g} min' if args.time_limit_minutes else ''}...")
    if args.timesteps < steps_per_update:
        print(f"Note: PPO always collects a full rollout first, so at least {steps_per_update} steps "
              f"will run even though --timesteps is {args.timesteps}.")
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