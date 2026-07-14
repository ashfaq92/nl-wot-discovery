"""
Leakage demonstration (2.5b).

The Dataset Audit argues that prior work's ~0.79 validation accuracy is inflated
by duplication: with a random row split, duplicates of a test sentence appear in
the training set. We test this directly by training the same classifier under
two splits and comparing validation top-1 accuracy:

  random         : IID row split (leaky)            -> reproduces ~0.79
  group_sentence : no sentence in both train+test   -> leakage-free

The gap quantifies the leakage. (We also report the 73-query benchmark accuracy
for reference, which is the out-of-distribution generalization the paper cares
about most.)

Run:  python run_leakage_demo.py [csv]
"""

import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pandas as pd

from classifier_baseline import train_classifier, ClassifierRetriever
from eval_lib import evaluate
from queries import QUERIES

CSV = sys.argv[1] if len(sys.argv) > 1 else "mainSimulationAccessTraces.csv"

rows = []
for split in ("random", "group_sentence"):
    print(f"\n=== training with split={split} ===")
    model, tok, classes, info = train_classifier(CSV, device="cpu", split=split)
    retr = ClassifierRetriever(model, tok, classes, info["maxlen"])
    res = evaluate(retr, QUERIES, ks=(1, 3))
    rows.append({
        "split": split,
        "val_top1": round(info["val_top1"], 3),
        "train_s": round(info["train_time_s"], 1),
        "bench_hit@1": round(res["overall"]["hit@1"], 3),
        "bench_hit@3": round(res["overall"]["hit@3"], 3),
    })
    print(f"  val_top1={info['val_top1']:.3f}  bench_hit@1={res['overall']['hit@1']:.3f}")

df = pd.DataFrame(rows)
df.to_csv("results_leakage_demo.csv", index=False)
print("\n" + df.to_string(index=False))
if len(df) == 2:
    drop = df.loc[0, "val_top1"] - df.loc[1, "val_top1"]
    print(f"\nLeakage = random - group_sentence validation accuracy = {drop:+.3f}")
print("saved results_leakage_demo.csv")
