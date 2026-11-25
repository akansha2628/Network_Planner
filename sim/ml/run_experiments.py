"""
sim/ml/run_experiments.py  —  one command per experiment for the conference paper
================================================================================
(ML-integration change D21, Claude.)  Chains DRL_code.py so each run needs one
command: train in 3 stages -> evaluate each stage on the validation seeds ->
pick the best stage -> evaluate it once on the test seeds -> write a summary.
Every step is saved, so if a cluster job is killed, run the SAME command again
and it continues where it stopped.

Commands (run from the repository root):

  python sim/ml/run_experiments.py baselines --eval-workers 8
      Heuristics (incremental, greedy, jump_to_max, random, no_upgrade) and the
      "gatekeeper + incremental" ablation on the validation and test seeds, once.
      Every run below reuses these results.

  python sim/ml/run_experiments.py train --variant full --seed 1 --n-envs 32 --stage-minutes 480,480,960
      One training run. Variants (conference): full, trigger_drl, drl_alone,
      fixed_step, no_saving. (JOCN): legacy_reward, year_end_override.
      Runs go to sim/ml/runs/<variant>_seed<N>/ and can run at the same time.

  python sim/ml/run_experiments.py gatekeeper-timing --eval-workers 3
      The gatekeeper's agreement with incremental's own check and its speed.

  python sim/ml/run_experiments.py summary
      Table of every finished run (mean and spread over training seeds) next to
      the baselines on the test seeds. Also written to sim/ml/runs/summary.csv.
"""
import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DRL = HERE / "DRL_code.py"
RUNS = HERE / "runs"
CACHE = RUNS / "baselines.csv"
VAL_SEEDS = "12,13,14"
TEST_SEEDS = "15,16,17,18,19"
TEST_HEURISTICS = "greedy,jump_to_max"

VARIANTS = {
    # conference
    "full":        ([], "gatekeeper decides IF, DRL decides WHAT"),
    "trigger_drl": (["--trigger", "heuristic"], "your trigger decides IF, DRL decides WHAT"),
    "drl_alone":   (["--gatekeeper-threshold", "0"], "no gatekeeper: every cycle reaches the DRL, it may save"),
    "fixed_step":  (["--upgrade-levels", "increment"], "DRL picks links, one fixed upgrade step each"),
    "no_saving":   (["--no-early-stop"], "DRL may not stop early to save money"),
    # JOCN
    "legacy_reward":     (["--reward", "legacy"], "the original reward (to show the new one matters)"),
    "year_end_override": (["--year-end-override"], "extra year-end decision even when the gatekeeper says no"),
}
STAGE_DECISIONS = (4, 8, None)       # episode length per stage: 4 decisions, 8, full lifetime


def run(cmd, log_path):
    """Run a command, show its output live and save it to log_path. Stops on failure."""
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    with open(log_path, "a") as log:
        log.write("\n$ " + " ".join(str(c) for c in cmd) + "\n")
        proc = subprocess.Popen([str(c) for c in cmd], cwd=str(REPO), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        code = proc.wait()
    if code != 0:
        sys.exit(f"\nStopped: the command above failed (exit code {code}). Full output: {log_path}\n"
                 f"Fix the problem and run the same run_experiments.py command again to continue.")


def mean_of(rows, method, key):
    vals = [float(r[key]) for r in rows if r["method"] == method and r.get(key) not in ("", None)]
    return sum(vals) / len(vals) if vals else None


def common_eval_args(args):
    extra = ["--eval-workers", str(args.eval_workers), "--baseline-cache", str(CACHE)]
    if args.eval_max_days:
        extra += ["--eval-max-days", str(args.eval_max_days)]
    return extra


def cmd_baselines(args):
    RUNS.mkdir(parents=True, exist_ok=True)
    log = RUNS / "baselines_log.txt"
    for seeds, label in ((args.val_seeds, "validation"), (args.test_seeds, "test")):
        print(f"\n=== Baselines on the {label} seeds ({seeds}) ===")
        run([sys.executable, DRL, "--evaluate", "none", "--val-seeds", seeds,
             "--eval-heuristics", "greedy,jump_to_max,random,no_upgrade", "--eval-gatekeeper-incremental",
             "--out-dir", RUNS / "baselines", "--eval-json", RUNS / f"baselines_{label}.json"]
            + common_eval_args(args), log)
    print(f"\nBaselines saved in {CACHE}. Check that the reward ranks the heuristics sensibly "
          f"(longer lifetime = higher reward; no_upgrade last).")


def cmd_train(args):
    if args.variant not in VARIANTS:
        sys.exit(f"Unknown variant {args.variant!r}. Choose from: {', '.join(VARIANTS)}")
    flags, what = VARIANTS[args.variant]
    out = RUNS / f"{args.variant}_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    log = out / "log.txt"
    minutes = [m.strip() for m in args.stage_minutes.split(",")] if args.stage_minutes else []
    steps = [s.strip() for s in args.stage_steps.split(",")]
    if len(steps) != 3 or (minutes and len(minutes) != 3):
        sys.exit("--stage-steps and --stage-minutes need 3 comma-separated values (one per stage)")
    print(f"=== {args.variant} (seed {args.seed}): {what} -> {out} ===")
    policy = out / "drl_optical_policy.zip"

    val = {}
    for i, decisions in enumerate(STAGE_DECISIONS, start=1):
        stage_zip = out / f"stage{i}.zip"
        if not stage_zip.exists():
            print(f"\n=== Stage {i}: " + (f"{decisions} decisions per episode" if decisions else "full lifetimes")
                  + " ===")
            cmd = [sys.executable, DRL, "--out-dir", out, "--n-envs", str(args.n_envs),
                   "--timesteps", steps[i - 1], "--seed", str(args.seed)] + flags
            if decisions:
                cmd += ["--max-decisions", str(decisions)]
            if minutes and minutes[i - 1] not in ("", "0"):
                cmd += ["--time-limit-minutes", minutes[i - 1]]
            if i > 1 or policy.exists():      # stage 1 interrupted earlier -> continue it
                cmd += ["--resume", policy]
            run(cmd, log)
            shutil.copy2(policy, stage_zip)
            settings = policy.with_suffix(".settings.json")
            if settings.exists():
                shutil.copy2(settings, stage_zip.with_suffix(".settings.json"))
        val_json = out / f"val_stage{i}.json"
        if not val_json.exists():
            print(f"\n=== Stage {i}: validation (seeds {args.val_seeds}) ===")
            run([sys.executable, DRL, "--evaluate", stage_zip, "--val-seeds", args.val_seeds, "--out-dir", out,
                 "--eval-json", val_json] + flags + common_eval_args(args), log)
        rows = json.loads(val_json.read_text())["rows"]
        val[i] = {"days": mean_of(rows, "drl", "days_simulated"), "reward": mean_of(rows, "drl", "reward_return"),
                  "random_days": mean_of(rows, "random", "days_simulated"),
                  "random_reward": mean_of(rows, "random", "reward_return"),
                  "incremental_days": mean_of(rows, "incremental", "days_simulated")}
        v = val[i]
        print(f"Stage {i} on validation: DRL {v['days']:.0f} days (reward {v['reward']:.3f}); "
              f"random agent {v['random_days']:.0f}; incremental {v['incremental_days']:.0f}")
        # lifetime first; the reward breaks ties (e.g. when --eval-max-days cut every run at the same day)
        if (round(v["days"], 1), v["reward"]) <= (round(v["random_days"], 1), v["random_reward"]):
            print("  WARNING: not better than the random agent at this stage.")

    best = max(val, key=lambda i: (round(val[i]["days"], 1), val[i]["reward"]))
    print(f"\nBest stage on validation: stage {best}")
    test_json = out / "test.json"
    if not test_json.exists() or json.loads(test_json.read_text()).get("policy") != str(out / f"stage{best}.zip"):
        print(f"\n=== Final evaluation on the test seeds ({args.test_seeds}) ===")
        run([sys.executable, DRL, "--evaluate", out / f"stage{best}.zip", "--val-seeds", args.test_seeds,
             "--eval-heuristics", TEST_HEURISTICS, "--eval-gatekeeper-incremental", "--out-dir", out,
             "--eval-json", test_json] + flags + common_eval_args(args), log)
    test_rows = json.loads(test_json.read_text())["rows"]
    summary = {"variant": args.variant, "seed": args.seed, "description": what, "best_stage": best,
               "validation": val, "test_rows": test_rows}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nDone. Summary: {out / 'summary.json'}. Run 'python sim/ml/run_experiments.py summary' "
          f"for the table across runs.")


def cmd_timing(args):
    out = RUNS / "gatekeeper_timing"
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, DRL, "--time-gatekeeper", "--val-seeds", args.val_seeds, "--out-dir", out,
           "--eval-workers", str(args.eval_workers)]
    if args.eval_max_days:
        cmd += ["--eval-max-days", str(args.eval_max_days)]
    run(cmd, out / "log.txt")


def cmd_summary(args):
    runs = sorted(RUNS.glob("*/summary.json"))
    if not runs:
        sys.exit(f"No finished runs in {RUNS} yet.")
    by_variant, baseline_rows = {}, {}
    for p in runs:
        s = json.loads(p.read_text())
        drl = [r for r in s["test_rows"] if r["method"] == "drl"]
        by_variant.setdefault(s["variant"], []).append({
            "seed": s["seed"],
            "days": sum(float(r["days_simulated"]) for r in drl) / len(drl),
            "cost": sum(float(r["total_cost"]) for r in drl) / len(drl),
            "reward": sum(float(r["reward_return"]) for r in drl) / len(drl),
            "fallbacks": sum(float(r["fallbacks"]) for r in drl) / len(drl),
            "hit": sum(str(r["lifetime_reached_threshold"]) == "True" for r in drl), "n": len(drl)})
        for r in s["test_rows"]:
            if r["method"] != "drl" and (r["method"] != "random" or s["variant"] == "full"):
                baseline_rows[(r["method"], int(r["seed"]))] = r

    def sd(xs):
        return statistics.stdev(xs) if len(xs) > 1 else 0.0

    lines = [("method", "training seeds", "lifetime days (mean ± sd)", "total cost $M", "days per $1M",
              "reward", "fallbacks", "hit threshold")]
    for variant, rs in by_variant.items():
        d = [r["days"] for r in rs]
        c = [r["cost"] / 1e6 for r in rs]
        lines.append((f"DRL: {variant}", ",".join(str(r["seed"]) for r in rs),
                      f"{statistics.mean(d):.0f} ± {sd(d):.0f}", f"{statistics.mean(c):.1f}",
                      f"{statistics.mean(r['days'] / (r['cost'] / 1e6) for r in rs):.1f}",
                      f"{statistics.mean(r['reward'] for r in rs):.3f}",
                      f"{statistics.mean(r['fallbacks'] for r in rs):.1f}",
                      f"{sum(r['hit'] for r in rs)}/{sum(r['n'] for r in rs)}"))
    for method in sorted({m for m, _ in baseline_rows}):
        rs = [r for (m, _), r in baseline_rows.items() if m == method]
        d = [float(r["days_simulated"]) for r in rs]
        c = [float(r["total_cost"]) / 1e6 for r in rs]
        lines.append((method, "-", f"{statistics.mean(d):.0f}", f"{statistics.mean(c):.1f}",
                      f"{statistics.mean(dd / cc for dd, cc in zip(d, c)):.1f}",
                      f"{statistics.mean(float(r['reward_return']) for r in rs):.3f}",
                      f"{statistics.mean(float(r['fallbacks']) for r in rs):.1f}",
                      f"{sum(str(r['lifetime_reached_threshold']) == 'True' for r in rs)}/{len(rs)}"))
    widths = [max(len(str(l[i])) for l in lines) for i in range(len(lines[0]))]
    for k, l in enumerate(lines):
        print(" | ".join(str(x).ljust(w) for x, w in zip(l, widths)))
        if k == 0:
            print("-+-".join("-" * w for w in widths))
    import csv
    with open(RUNS / "summary.csv", "w", newline="") as f:
        csv.writer(f).writerows(lines)
    print(f"\nTest seeds {sorted({int(r['seed']) for r in baseline_rows.values()})}. Saved: {RUNS / 'summary.csv'}")
    print("Lifetime and days per $1M compare fairly only if every run hit the threshold (last column).")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    def eval_opts(p):
        p.add_argument("--eval-workers", type=int, default=os.cpu_count() or 1,
                       help="simulations to run in parallel during evaluation (default: all CPUs)")
        p.add_argument("--eval-max-days", type=int, default=None,
                       help="quick look only: stop every evaluation run at this day")
        p.add_argument("--val-seeds", default=VAL_SEEDS, help=f"validation seeds (default {VAL_SEEDS})")
        p.add_argument("--test-seeds", default=TEST_SEEDS, help=f"test seeds (default {TEST_SEEDS})")

    p = sub.add_parser("baselines", help="heuristics + gatekeeper/incremental on validation and test seeds")
    eval_opts(p)
    p = sub.add_parser("train", help="one full training run of a variant")
    p.add_argument("--variant", default="full", help=f"one of: {', '.join(VARIANTS)}")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--n-envs", type=int, required=True, help="parallel simulators = CPUs of the job")
    p.add_argument("--stage-steps", default="20000,20000,40000",
                   help="max training steps for stages 1,2,3 (default 20000,20000,40000)")
    p.add_argument("--stage-minutes", default=None,
                   help="time limit in minutes for stages 1,2,3, e.g. 480,480,960 (a stage ends at "
                        "whichever comes first)")
    eval_opts(p)
    p = sub.add_parser("gatekeeper-timing", help="gatekeeper vs incremental's own check: agreement and speed")
    eval_opts(p)
    sub.add_parser("summary", help="table of every finished run")
    p = sub.add_parser("list", help="show the variants")

    args = ap.parse_args()
    if args.command == "list":
        for k, (flags, what) in VARIANTS.items():
            print(f"{k:<18} {what}   {' '.join(flags)}")
        return
    {"baselines": cmd_baselines, "train": cmd_train, "gatekeeper-timing": cmd_timing,
     "summary": cmd_summary}[args.command](args)


if __name__ == "__main__":
    main()
