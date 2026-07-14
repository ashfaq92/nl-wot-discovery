"""
In-distribution multi-seed evaluation of the Llopis-style supervised classifier.

Motivation
----------
The original `classifier_baseline.py` / `run_leakage_demo.py` report a single
fixed-seed (SEED=42) validation accuracy. This experiment re-runs the SAME
faithful classifier across several seeds and reports mean +/- STD for three
in-distribution metrics, on the held-out validation split only (no 73-query
benchmark here):

    accuracy            : plain top-1 accuracy
    macro_f1            : F1 averaged unweighted over classes (class-imbalance aware)
    balanced_accuracy   : mean per-class recall (chance = 1/num_classes)

Both in-distribution splits are evaluated:
    random         : IID row split (prior-work setup; duplicate sentences may
                     appear in both train and test -> the leaky inflated number)
    group_sentence : leakage-free split (no generated sentence in both sets)

The classifier training is delegated verbatim to
`classifier_baseline.train_classifier(..., seed=, split=)`, which now returns the
validation labels/predictions in its `info` dict, so this driver only varies the
seed and computes/aggregates metrics. Nothing here re-implements the model.

Run (from anywhere; paths are resolved relative to code/):
    python exploratory/classifier_indist_multiseed.py [csv]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score)

# --- make code/ importable and locate the dataset, regardless of CWD ----------
SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from classifier_baseline import train_classifier  # noqa: E402

SEEDS = [42, 43, 44, 45, 46]
SPLITS = ["random", "group_sentence"]
METRICS = ["accuracy", "macro_f1", "balanced_accuracy"]


def metrics_for(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }


def main():
    csv = sys.argv[1] if len(sys.argv) > 1 else str(CODE_DIR / "mainSimulationAccessTraces.csv")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dev_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    print(f"device = {device} ({dev_name})")
    print(f"seeds  = {SEEDS}")
    print(f"splits = {SPLITS}\n")

    per_seed = []
    for split in SPLITS:
        for seed in SEEDS:
            print(f"=== split={split:14s} seed={seed} ===")
            _, _, _, info = train_classifier(
                csv, device=device, split=split, seed=seed, verbose=False)
            m = metrics_for(info["val_true"], info["val_pred"])
            row = {"split": split, "seed": seed, "n_val": info["n_val"],
                   "num_classes": info["num_classes"],
                   "train_s": round(info["train_time_s"], 1), **m}
            per_seed.append(row)
            print(f"  n_val={info['n_val']:6d}  acc={m['accuracy']:.4f}  "
                  f"macroF1={m['macro_f1']:.4f}  balAcc={m['balanced_accuracy']:.4f}  "
                  f"({row['train_s']}s)")

    per_seed_df = pd.DataFrame(per_seed)
    per_seed_path = SCRIPT_DIR / "results_classifier_indist_perseed.csv"
    per_seed_df.to_csv(per_seed_path, index=False)

    # --- aggregate: mean +/- STD across seeds, per split -----------------------
    summary_rows = []
    for split in SPLITS:
        sub = per_seed_df[per_seed_df["split"] == split]
        # n_val is constant for the random (row) split but varies a lot for
        # group_sentence (25% of *sentences* held out; sentences carry very
        # different duplicate masses) -> report the range, not a point value.
        row = {"split": split, "n_seeds": len(sub),
               "n_val_min": int(sub["n_val"].min()),
               "n_val_max": int(sub["n_val"].max()),
               "num_classes": int(sub["num_classes"].iloc[0])}
        for met in METRICS:
            # population? no -> sample std (ddof=1) is the convention for n seeds
            row[f"{met}_mean"] = round(float(sub[met].mean()), 4)
            row[f"{met}_std"] = round(float(sub[met].std(ddof=1)), 4)
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_path = SCRIPT_DIR / "results_classifier_indist_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # --- human-readable report -------------------------------------------------
    print("\n" + "=" * 70)
    print(f"In-distribution classifier, mean +/- STD over {len(SEEDS)} seeds {SEEDS}")
    print("=" * 70)
    for r in summary_rows:
        nval = (f"{r['n_val_min']}" if r["n_val_min"] == r["n_val_max"]
                else f"{r['n_val_min']}-{r['n_val_max']}")
        print(f"\n[{r['split']}]  (n_val={nval}, classes={r['num_classes']})")
        for met in METRICS:
            print(f"  {met:18s} = {r[met + '_mean']:.4f} +/- {r[met + '_std']:.4f}")

    print(f"\nsaved {per_seed_path.name} and {summary_path.name} in {SCRIPT_DIR}")


if __name__ == "__main__":
    main()
