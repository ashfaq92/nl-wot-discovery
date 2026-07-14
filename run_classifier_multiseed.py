"""
Combined CPU classifier evaluation over 5 seeds -> mean +/- STD, single pass.

Per the paper's environment decision, ALL reported accuracy is on CPU. To avoid
retraining the classifier separately for each analysis, this trains it once per
seed (CPU) and derives, from the same model:
  (1) in-distribution validation top-1 accuracy,
  (2) the 73-query benchmark head-to-head (overall + per-category Hit@1, AUROC),
  (3) per-endpoint in-distribution recall@1/@3, binned by endpoint frequency,
      against the frozen retriever (built once on CPU).

5 trainings total (~6-8 min each on CPU). Run on a quiet machine.

Run (from code/):  python run_classifier_multiseed.py [csv]
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR
sys.path.insert(0, str(CODE_DIR))

from classifier_baseline import build_training_data, train_classifier, ClassifierRetriever  # noqa: E402
from eval_lib import load_records, build_record_text, evaluate, hit_at_k  # noqa: E402
from queries import QUERIES  # noqa: E402
from retrievers import BiEncoderRetriever  # noqa: E402

SEEDS = [42, 43, 44, 45, 46]
DEVICE = "cpu"
TEST_SIZE = 0.25
POOL = 10
KS = (1, 3, 5, 10)
BI_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
CATS = ["templated", "paraphrased", "synonym", "abstract",
        "ambiguous_location", "ambiguous_device"]
BINS = [0, 1.5, 15, 150, 1500, 1e9]
BIN_LABELS = ["~1 (singleton)", "2-15", "16-150", "151-1500", ">1500"]


def msd(vals, nd=4):
    a = np.asarray(vals, dtype=float)
    return round(float(a.mean()), nd), (round(float(a.std(ddof=1)), nd) if len(a) > 1 else 0.0)


def main():
    csv = sys.argv[1] if len(sys.argv) > 1 else str(CODE_DIR / "mainSimulationAccessTraces.csv")
    print(f"device = {DEVICE}   seeds = {SEEDS}")

    texts, labels = build_training_data(csv, preprocess=True)
    labels = np.array(labels)
    n = len(texts)
    classes = sorted(set(labels))
    sent_of_class = defaultdict(set)
    ep_per_sent = defaultdict(set)
    for t, l in zip(texts, labels):
        sent_of_class[l].add(t); ep_per_sent[t].add(l)

    print("building retriever index (CPU) ...")
    recs = build_record_text(load_records(csv), fmt="sentence")
    retr = BiEncoderRetriever(recs, BI_ENCODER, device=DEVICE)
    retr_cache = {}

    def retr_rank(s):
        if s not in retr_cache:
            retr_cache[s] = [e for e, _ in retr.search(s, POOL)]
        return retr_cache[s]

    bench_rows = []
    pc = defaultdict(lambda: {"clf1": [], "clf3": [], "retr1": [], "retr3": [], "n": []})

    for seed in SEEDS:
        print(f"\n=== seed {seed} (training on CPU) ===")
        model, tok, cls_c, info = train_classifier(
            csv, device=DEVICE, split="random", seed=seed, verbose=False)
        assert cls_c == classes
        clf = ClassifierRetriever(model, tok, classes, info["maxlen"], device=DEVICE)

        # (2) benchmark head-to-head
        res = evaluate(clf, QUERIES, ks=KS)
        o, na, bc = res["overall"], res["no_answer"], res["by_category"]
        brow = {"seed": seed, "val_top1": info["val_top1"],
                "hit@1": o["hit@1"], "hit@3": o["hit@3"], "hit@5": o["hit@5"],
                "hit@10": o["hit@10"], "mrr": o["mrr"], "auroc_na": na["auroc"]}
        for c in CATS:
            brow[f"cat_{c}"] = bc[c]["hit@1"] if c in bc else float("nan")
            brow[f"cat3_{c}"] = bc[c]["hit@3"] if c in bc else float("nan")
        bench_rows.append(brow)
        print(f"  val_top1={info['val_top1']:.4f}  bench hit@1={o['hit@1']:.3f} "
              f"hit@3={o['hit@3']:.3f} mrr={o['mrr']:.3f} auroc_na={na['auroc']:.3f}")

        # (3) per-class in-distribution recall
        idx_va = train_test_split(np.arange(n), test_size=TEST_SIZE, random_state=seed)[1]
        val_sent = [texts[i] for i in idx_va]
        val_true = labels[idx_va]
        clf_cache = {}
        by_cls = defaultdict(lambda: {"clf1": 0, "clf3": 0, "retr1": 0, "retr3": 0, "n": 0})
        for s, true_ep in zip(val_sent, val_true):
            if s not in clf_cache:
                clf_cache[s] = [e for e, _ in clf.search(s, POOL)]
            cr, rr, exp = clf_cache[s], retr_rank(s), [true_ep]
            b = by_cls[true_ep]; b["n"] += 1
            b["clf1"] += hit_at_k(cr, exp, 1); b["clf3"] += hit_at_k(cr, exp, 3)
            b["retr1"] += hit_at_k(rr, exp, 1); b["retr3"] += hit_at_k(rr, exp, 3)
        for c, b in by_cls.items():
            for m in ("clf1", "clf3", "retr1", "retr3"):
                pc[c][m].append(b[m] / b["n"])
            pc[c]["n"].append(b["n"])

    # ---- aggregate benchmark (head-to-head) ----------------------------------
    bdf = pd.DataFrame(bench_rows)
    bsum = {"n_seeds": len(SEEDS)}
    for m in (["val_top1", "hit@1", "hit@3", "hit@5", "hit@10", "mrr", "auroc_na"]
              + [f"cat_{c}" for c in CATS] + [f"cat3_{c}" for c in CATS]):
        bsum[f"{m}_mean"], bsum[f"{m}_std"] = msd(bdf[m].tolist())
    pd.DataFrame([bsum]).to_csv(SCRIPT_DIR / "results_classifier_multiseed.csv", index=False)

    # ---- per-class -> frequency bins -----------------------------------------
    rows = []
    for c, a in pc.items():
        if not a["n"]:
            continue
        d = {"endpoint": c, "shared": any(len(ep_per_sent[s]) > 1 for s in sent_of_class[c]),
             "mean_n_val": float(np.mean(a["n"]))}
        for m, lbl in [("clf1", "clf_r1"), ("retr1", "retr_r1"), ("clf3", "clf_r3"), ("retr3", "retr_r3")]:
            d[f"{lbl}_mean"] = float(np.mean(a[m]))
        rows.append(d)
    pcdf = pd.DataFrame(rows)
    pcdf.to_csv(SCRIPT_DIR / "results_classifier_perclass.csv", index=False)
    pcdf["bin"] = pd.cut(pcdf.mean_n_val, bins=BINS, labels=BIN_LABELS)
    g = pcdf.groupby("bin", observed=True).agg(
        n_classes=("endpoint", "size"),
        clf_r1=("clf_r1_mean", "mean"), retr_r1=("retr_r1_mean", "mean"),
        clf_r3=("clf_r3_mean", "mean"), retr_r3=("retr_r3_mean", "mean")).round(3)
    macro = {"n_classes": len(pcdf),
             "clf_r1": pcdf.clf_r1_mean.mean(), "retr_r1": pcdf.retr_r1_mean.mean(),
             "clf_r3": pcdf.clf_r3_mean.mean(), "retr_r3": pcdf.retr_r3_mean.mean()}
    g.to_csv(SCRIPT_DIR / "results_classifier_perclass_bins.csv")

    # ---- report --------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"CPU classifier, mean +/- STD over {len(SEEDS)} seeds")
    print("=" * 70)
    print(f"  in-distribution val top-1 : {bsum['val_top1_mean']:.4f} +/- {bsum['val_top1_std']:.4f}")
    print("  benchmark overall:")
    for m in ["hit@1", "hit@3", "hit@5", "hit@10", "mrr", "auroc_na"]:
        print(f"    {m:9s} = {bsum[m + '_mean']:.3f} +/- {bsum[m + '_std']:.3f}")
    print("  benchmark per-category Hit@1:")
    for c in CATS:
        print(f"    {c:20s} = {bsum['cat_' + c + '_mean']:.3f} +/- {bsum['cat_' + c + '_std']:.3f}")
    print("\n  per-class recall by endpoint frequency:")
    print(g.to_string())
    print(f"  Macro (all={macro['n_classes']}): clf@1={macro['clf_r1']:.3f} retr@1={macro['retr_r1']:.3f} "
          f"clf@3={macro['clf_r3']:.3f} retr@3={macro['retr_r3']:.3f}")
    print(f"\nsaved results_classifier_multiseed.csv and results_classifier_perclass*.csv in {SCRIPT_DIR}")


if __name__ == "__main__":
    main()
