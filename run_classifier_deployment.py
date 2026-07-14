"""
Measure the supervised classifier's deployment/maintenance numbers on CPU, over
five seeds -> mean +/- STD, to match the retriever rows of the deployment table.
Reports: parameter count, model size, full-(re)train time, and per-query inference
latency (p50/p95). Adding a device forces a full retrain, so the train time IS the
per-device update cost for the classifier.

Run on a quiet machine, from code/:  python run_classifier_deployment.py [csv]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR
sys.path.insert(0, str(CODE_DIR))

from classifier_baseline import train_classifier, ClassifierRetriever  # noqa: E402
from eval_lib import Timer  # noqa: E402
from queries import QUERIES  # noqa: E402

DEVICE = "cpu"
SEEDS = [42, 43, 44, 45, 46]


def msd(a):
    a = np.asarray(a, dtype=float)
    return a.mean(), (a.std(ddof=1) if len(a) > 1 else 0.0)


def main():
    csv = sys.argv[1] if len(sys.argv) > 1 else str(CODE_DIR / "mainSimulationAccessTraces.csv")
    print(f"device = {DEVICE}   seeds = {SEEDS}  (each training is one full retrain)")
    retrains, p50s, p95s = [], [], []
    params = size_mib = None
    for seed in SEEDS:
        model, tok, classes, info = train_classifier(csv, device=DEVICE, seed=seed, verbose=False)
        if params is None:
            params = sum(p.numel() for p in model.parameters())
            size_mib = params * 4 / 1024 ** 2
        retrains.append(info["train_time_s"])
        clf = ClassifierRetriever(model, tok, classes, info["maxlen"], device=DEVICE)
        for q in QUERIES[:5]:
            clf.search(q["query"], 10)
        lat = []
        for _ in range(3):
            for q in QUERIES:
                with Timer() as t:
                    clf.search(q["query"], 10)
                lat.append(t.s)
        lat = np.array(lat) * 1000.0
        p50s.append(float(np.percentile(lat, 50)))
        p95s.append(float(np.percentile(lat, 95)))
        print(f"  seed {seed}: retrain={info['train_time_s']:.1f}s  "
              f"p50={p50s[-1]:.2f}ms p95={p95s[-1]:.2f}ms  val_top1={info['val_top1']:.4f}")

    rt_m, rt_s = msd(retrains); p50_m, p50_s = msd(p50s); p95_m, p95_s = msd(p95s)
    pd.DataFrame([{
        "params_M": round(params / 1e6, 2), "model_size_MiB": round(size_mib, 2),
        "retrain_s_mean": round(rt_m, 1), "retrain_s_std": round(rt_s, 1),
        "p50_ms_mean": round(p50_m, 2), "p50_ms_std": round(p50_s, 2),
        "p95_ms_mean": round(p95_m, 2), "p95_ms_std": round(p95_s, 2),
        "n_seeds": len(SEEDS), "device": DEVICE,
    }]).to_csv(SCRIPT_DIR / "results_classifier_deployment.csv", index=False)
    print("\n=== classifier deployment numbers (CPU, mean +/- STD over "
          f"{len(SEEDS)} runs) ===")
    print(f"  parameters      : {params/1e6:.2f} M   ({size_mib:.2f} MiB)")
    print(f"  full (re)train  : {rt_m:.1f} +/- {rt_s:.1f} s  (~{rt_m/60:.1f} min)")
    print(f"  query lat p50   : {p50_m:.2f} +/- {p50_s:.2f} ms")
    print(f"  query lat p95   : {p95_m:.2f} +/- {p95_s:.2f} ms")
    print(f"saved results_classifier_deployment.csv in {SCRIPT_DIR}")


if __name__ == "__main__":
    main()
