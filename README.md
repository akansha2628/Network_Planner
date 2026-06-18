# Optical Network Simulator with Upgrade Planning

This repository contains a modular elastic optical network simulator for studying long-term capacity upgrades in brownfield networks. It supports traffic growth, routing and spectrum assignment, blocking-probability tracking, cost accounting, budget-aware planning, and multiple upgrade strategies.

## Quick Start

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Open `main.py` and edit the `CONFIG` dictionary.
3. Run the simulator:

   ```bash
   python main.py
   ```

4. Results are written to the `results/` directory. The output folder name includes the algorithm, budget mode, and topology, for example:

   ```text
   results/random_budget_aware_dt12/
   ```

## Main Configuration

Most experiments can be configured from the `CONFIG` dictionary in `main.py`.

```python
CONFIG = {
    "topology": "dt12",
    "algorithm": "random",
    "link_selection_mode": "path_congestion",
    "seed": 0,
    "budget_enabled": True,
    "budget_selection_mode": "budget_aware",
}
```

### Topology Options

- `test5`: small 5-node test topology for debugging.
- `dt12`: 12-node production topology used for the main experiments.

### Algorithm Options

- `increment`: adaptive incremental upgrade strategy.
- `greedy`: aggressive upgrade strategy toward higher-capacity states.
- `random`: randomly selects eligible links and upgrade actions.
- `jump_to_max`: directly upgrades selected links to the maximum feasible capacity state.
- `max_initial`: applies a maximum-capacity upgrade at the beginning of the simulation.
- `no_upgrade`: baseline with no capacity upgrades.

### Link Selection Modes

- `path_congestion`: selects links along the most congested path for a critical source-destination pair.
- `most_congested`: selects the highest-utilization link from the most congested path.

### Budget Modes

- `budget_aware`: checks both performance and budget feasibility before accepting an upgrade plan.
- `budget_unaware`: checks performance only; cost is recorded but not used as a constraint.
- `cost_only`: checks budget/cost feasibility without enforcing the blocking-performance pass condition.

For the DT12 topology, the current budget envelope is:

- Annual total budget: `$22.5M`
- Annual OPEX budget: `$3M`
- Annual CAPEX + workforce budget: `$19.5M`
- Annual inflation rate: `3%`
- Upgrade-check interval: `180 days`

Within a year, unused budget from the first 180-day cycle remains available for the second cycle. The annual budget is then reset and inflated for the next year.

## Project Structure

```text
.
├── main.py                         # Main entry point and experiment configuration
├── sim/
│   ├── algorithms/                 # Upgrade algorithms and algorithm selector
│   ├── core/                       # Constants, topology state, traffic generator
│   ├── routing/                    # K-shortest paths and routing/spectrum assignment
│   ├── upgrade/                    # Upgrade decisions, Plan Checker, costs, downtime
│   ├── topologies/                 # Topology definitions and topology manager
│   └── modules/                    # Performance metrics, analysis, and plotting helpers
├── results/                        # Simulation outputs generated after each run
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

## Important Output Files

Each run creates an output folder under `results/`. The most useful files are:

| File | Description |
|---|---|
| `simulation_log.txt` | Full console log of the run. |
| `blocking_results.csv` | Time-series blocking probability data. |
| `connection_requests.csv` | Per-request data, including source, destination, accepted/blocked status, route, fiber/core, and slot allocation. |
| `upgrade_log.csv` | Upgrade actions applied during the simulation. |
| `upgrade_cycle_summary.csv` | Summary of each upgrade cycle and selected upgrade decisions. |
| `daily_network_technology_summary.csv` | Daily technology-state counts and OPEX. |
| `pre_upgrade_per_link_technology.csv` | Per-link technology state before each upgrade cycle. |
| `path_stats_history.csv` | Full-history path-level arrivals, accepted requests, blocked requests, and path blocking probability. |
| `final_state.pkl` | Snapshot of the final simulation state. |
| `blocking_probability_vs_time.png` | Blocking-probability plot. |
| `cost_vs_time.png` | Cost-component and cumulative-cost plot. |
| `budget_vs_cost.png` | Actual cumulative cost versus budget. |
| `blocking_vs_upgrade_decisions.png` | Blocking probability with upgrade-decision markers. |
| `performance_metrics.png` | Summary performance dashboard. |

## Notes for Machine Learning Work

The simulator can be used to generate supervised learning datasets for upgrade planning. Useful input features may include:

- source-destination blocking probability,
- path-level blocking contribution,
- link utilization,
- link fragmentation,
- current link technology state,
- available remaining budget,
- traffic growth level,
- previous upgrade decisions.

Possible prediction targets include:

- whether an upgrade is needed,
- which source-destination pair is most critical,
- which link should be upgraded,
- which upgrade action should be selected,
- whether the Plan Checker passes,
- resulting blocking probability after the upgrade.

Recommended CSV files for ML dataset construction:

- `connection_requests.csv`
- `upgrade_cycle_summary.csv`
- `daily_network_technology_summary.csv`
- `pre_upgrade_per_link_technology.csv`
- `path_stats_history.csv`

## Reproducibility

Set `seed` in `CONFIG` to make repeated runs comparable. For experiments, record the following together with the results:

- topology,
- algorithm,
- link-selection mode,
- budget-selection mode,
- seed,
- key constants from `sim/core/constants.py`.

## Advanced Configuration

For physical-layer, traffic, spectrum, cost, and upgrade parameters, edit:

```text
sim/core/constants.py
```

For upgrade-cost formulas, edit:

```text
sim/upgrade/cost_model.py
```

For Plan Checker behavior, edit:

```text
sim/upgrade/plan_checker.py
```
