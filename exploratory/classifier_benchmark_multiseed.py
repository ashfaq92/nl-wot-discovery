"""
Supervised classifier on the 73-query NL benchmark, over 5 seeds -> mean +/- STD.

This is the multi-seed version of the head-to-head classifier column (manuscript
Table "headtohead"). The classifier is trained (random split, matching prior work)
once per seed, wrapped behind the retriever .search() interface, and scored on the
SAME 73-query benchmark via eval_lib.evaluate. We report per-category Hit@1 and the
overall Hit@1/Hit@3/MRR plus the no-answer AUROC, each as mean +/- STD across seeds.

The frozen retriever it is compared against is deterministic, so only the classifier
needs multiple seeds.

Run (from anywhere):  python exploratory/classifier_benchmark_multiseed.py [csv]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from classifier_baseline import train_classifier, ClassifierRetriever  # noqa: E402
from eval_lib import evaluate  # noqa: E402
from queries import QUERIES  # noqa: E402

SEEDS = [42, 43, 44, 45, 46]
KS = (1, 3, 5, 10)
# answerable categories shown in the head-to-head table (no_answer excluded: no expected endpoint)
CATS = ["templated", "paraphrased", "synonym", "abstract",
        "ambiguous_location", "ambiguous_device"]


def msd(vals):
    a = np.array(vals, dtype=float)
    return round(float(a.mean()), 4), (round(float(a.std(ddof=1)), 4) if len(a) > 1 else 0.0)


def main():
    csv = sys.argv[1] if len(sys.argv) > 1 else str(CODE_DIR / "mainSimulationAccessTraces.csv")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}   seeds = {SEEDS}")

    per_seed = []
    for seed in SEEDS:
        model, tok, classes, info = train_classifier(
            csv, device=device, split="random", seed=seed, verbose=False)
        retr = ClassifierRetriever(model, tok, classes, info["maxlen"], device=device)
        res = evaluate(retr, QUERIES, ks=KS)
        o, na, bc = res["overall"], res["no_answer"], res["by_category"]
        row = {"seed": seed,
               "val_top1": round(info["val_top1"], 4),
               "hit@1": o["hit@1"], "hit@3": o["hit@3"], "hit@5": o["hit@5"],
               "hit@10": o["hit@10"], "mrr": o["mrr"], "auroc_na": na["auroc"]}
        for c in CATS:
            row[f"cat_{c}"] = bc[c]["hit@1"] if c in bc else float("nan")
        per_seed.append(row)
        print(f"  seed {seed}: val_top1={row['val_top1']:.3f}  "
              f"bench hit@1={o['hit@1']:.3f} hit@3={o['hit@3']:.3f} mrr={o['mrr']:.3f} "
              f"auroc_na={na['auroc']:.3f}")

    ps = pd.DataFrame(per_seed)
    ps.to_csv(SCRIPT_DIR / "results_classifier_benchmark_perseed.csv", index=False)

    metrics = ["val_top1", "hit@1", "hit@3", "hit@5", "hit@10", "mrr", "auroc_na"] + \
              [f"cat_{c}" for c in CATS]
    summ = {"n_seeds": len(SEEDS)}
    for m in metrics:
        summ[f"{m}_mean"], summ[f"{m}_std"] = msd(ps[m].tolist())
    pd.DataFrame([summ]).to_csv(SCRIPT_DIR / "results_classifier_benchmark_summary.csv", index=False)

    print("\n" + "=" * 60)
    print(f"Classifier on 73-query benchmark, mean +/- STD over {len(SEEDS)} seeds")
    print("=" * 60)
    print(f"  in-distribution val top-1 : {summ['val_top1_mean']:.3f} +/- {summ['val_top1_std']:.3f}")
    print("  benchmark overall:")
    for m in ["hit@1", "hit@3", "hit@5", "hit@10", "mrr", "auroc_na"]:
        print(f"    {m:9s} = {summ[m + '_mean']:.3f} +/- {summ[m + '_std']:.3f}")
    print("  benchmark per-category Hit@1:")
    for c in CATS:
        print(f"    {c:20s} = {summ['cat_' + c + '_mean']:.3f} +/- {summ['cat_' + c + '_std']:.3f}")
    print(f"\nsaved results_classifier_benchmark_perseed.csv and _summary.csv in {SCRIPT_DIR}")


if __name__ == "__main__":
    main()
