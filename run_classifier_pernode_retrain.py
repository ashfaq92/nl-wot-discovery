"""
Time the supervised classifier's full retrain on each federated node partition,
plus the full dataset, over five seeds -> mean +/- STD per scope.

Answers the "you never measured per-node retraining" objection to the
maintenance comparison: prior work's federated design trains one classifier
per node on the rows that node manages (Llopis et al. 2025), so the honest
per-node maintenance cost is a retrain on that node's own (smaller) trace
partition, not on the full trace. This driver partitions the undeduplicated
trace by destinationLocation using the same node layout as federated.py,
trains the identical classifier per node (same hyperparameters, same random
75/25 split), and records wall-clock training time.

Accuracy note: val_top1 is recorded for sanity only; per-node accuracy is NOT
a paper claim (the manuscript compares accuracy in the centralized setting,
the classifier's best case).

Run on a quiet machine, from code/:
  python run_classifier_pernode_retrain.py [csv] [--full-only]

Runs on CPU by default (the reported environment). Set FORCE_DEVICE=cuda for a
scratch GPU timing (used only for the manuscript's hardware footnote); its
output gets a _gpu suffix and its accuracy is never reported.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from classifier_baseline import train_classifier, RECORD_COLS  # noqa: E402
from federated import NODE_LOCATIONS  # noqa: E402

SEEDS = [42, 43, 44, 45, 46]


def msd(a):
    a = np.asarray(a, dtype=float)
    return a.mean(), (a.std(ddof=1) if len(a) > 1 else 0.0)


def time_scope(csv_path, scope, device):
    times, vals = [], []
    n_rows = n_classes = None
    for seed in SEEDS:
        model, tok, classes, info = train_classifier(
            csv_path, device=device, seed=seed, verbose=False)
        times.append(info["train_time_s"])
        vals.append(info["val_top1"])
        n_rows, n_classes = info["n_rows"], info["num_classes"]
        del model
    t_m, t_s = msd(times)
    v_m, _ = msd(vals)
    print(f"  {scope:>6}: rows={n_rows:>7}  classes={n_classes:>3}  "
          f"retrain={t_m:6.1f} +/- {t_s:4.1f} s  val_top1={v_m:.3f}")
    return {"scope": scope, "n_rows": n_rows, "n_classes": n_classes,
            "retrain_s_mean": round(t_m, 1), "retrain_s_std": round(t_s, 1),
            "val_top1_mean": round(v_m, 3), "n_seeds": len(SEEDS),
            "device": device}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    csv = args[0] if args else str(SCRIPT_DIR / "mainSimulationAccessTraces.csv")
    device = os.environ.get("FORCE_DEVICE") or "cpu"
    full_only = "--full-only" in sys.argv

    print(f"device = {device}   seeds = {SEEDS}   full_only = {full_only}")
    rows = [time_scope(csv, "full", device)]

    if not full_only:
        df = pd.read_csv(csv)[RECORD_COLS].dropna()
        with tempfile.TemporaryDirectory() as tmp:
            for node, locs in NODE_LOCATIONS.items():
                if not locs:  # coordinator, no devices -> nothing to train
                    continue
                sub = df[df["destinationLocation"].isin(locs)]
                node_csv = Path(tmp) / f"node{node}.csv"
                sub.to_csv(node_csv, index=False)
                rows.append(time_scope(str(node_csv), f"node{node}", device))

    out = SCRIPT_DIR / ("results_classifier_pernode_retrain.csv" if device == "cpu"
                        else "results_classifier_pernode_retrain_gpu.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    node_rows = [r for r in rows if r["scope"] != "full"]
    if node_rows:
        lo = min(r["retrain_s_mean"] for r in node_rows)
        hi = max(r["retrain_s_mean"] for r in node_rows)
        print(f"\nper-node retrain range ({device}): {lo:.1f}-{hi:.1f} s "
              f"(full: {rows[0]['retrain_s_mean']:.1f} s)")
    print(f"saved {out.name} in {SCRIPT_DIR}")


if __name__ == "__main__":
    main()
