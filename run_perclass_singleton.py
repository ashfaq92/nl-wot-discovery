"""
Per-endpoint in-distribution recall with singleton classes INCLUDED via copying.

The 57 endpoints that occur once in the
whole trace cannot normally be in both train and val. Here we *copy* each such
single row so it appears once in training AND once in validation (the two copies
are identical). Concretely the row's index is placed in both `tr_idx` and `va_idx`,
so the classifier trains on that exact (sentence -> endpoint) once and is then
evaluated on an identical copy. This tests: given exactly ONE training instance of
a device (the strongest case for memorization), can each method return it?

  - 1-row classes : index in BOTH train and val (1 train copy, 1 val copy)
  - >=2-row classes: >=1 row to val, the rest (>=1) to train (disjoint)

All 170 endpoints are scored. This backs the paper's claim that even copying a
device's lone record into the training set cannot lift the classifier on
singletons.

Runs on CPU by default (the reported environment). Set FORCE_DEVICE=cuda for a
scratch GPU run; its outputs get a `_gpu` suffix and are never reported.

Run (from code/):  python run_perclass_singleton.py [csv]
"""

from __future__ import annotations

import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR
sys.path.insert(0, str(CODE_DIR))

from classifier_baseline import build_training_data, train_classifier, ClassifierRetriever  # noqa: E402
from eval_lib import load_records, build_record_text, hit_at_k  # noqa: E402
from retrievers import BiEncoderRetriever  # noqa: E402

SEEDS = [42, 43, 44, 45, 46]
TEST_SIZE = 0.25
POOL = 10
BI_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
FREQ_BINS = [0.5, 1.5, 3, 15, 150, 1500, 1e9]
FREQ_LABELS = ["1 (singleton)", "2-3", "4-15", "16-150", "151-1500", ">1500"]


def split_with_singleton_copy(labels, seed, test_size=TEST_SIZE):
    """>=1 train & >=1 val per class; single-row classes are COPIED (their one row
    index goes to both train and val)."""
    rng = np.random.RandomState(seed)
    by_c = defaultdict(list)
    for i, l in enumerate(labels):
        by_c[l].append(i)
    tr, va = [], []
    for c, idx in by_c.items():
        idx = np.array(idx); rng.shuffle(idx)
        k = len(idx)
        if k == 1:
            tr.append(idx[0]); va.append(idx[0])      # copy: trains and validates
        else:
            n_val = max(1, int(round(test_size * k)))
            n_val = min(n_val, k - 1)
            va.extend(idx[:n_val]); tr.extend(idx[n_val:])
    return np.array(tr), np.array(va)


def main():
    csv = sys.argv[1] if len(sys.argv) > 1 else str(CODE_DIR / "mainSimulationAccessTraces.csv")
    device = os.environ.get("FORCE_DEVICE") or "cpu"
    tag = "" if device == "cpu" else "_gpu"
    print(f"device = {device}   seeds = {SEEDS}")

    texts, labels = build_training_data(csv, preprocess=True)
    texts = np.array(texts); labels = np.array(labels)
    freq = Counter(labels)
    n_single = sum(1 for c in freq if freq[c] == 1)
    print(f"endpoints: {len(freq)} total; {n_single} singletons (copied into both sets); all scored")

    print(f"building retriever index ({device}) ...")
    recs = build_record_text(load_records(csv), fmt="sentence")
    retr = BiEncoderRetriever(recs, BI_ENCODER, device=device)
    retr_cache = {}

    def retr_rank(s):
        if s not in retr_cache:
            retr_cache[s] = [e for e, _ in retr.search(s, POOL)]
        return retr_cache[s]

    pc = defaultdict(lambda: {"clf1": [], "clf3": [], "retr1": [], "retr3": [],
                              "ntrain": [], "nval": []})
    val_accs = []
    for seed in SEEDS:
        tr_idx, va_idx = split_with_singleton_copy(labels, seed)
        tr_counts = Counter(labels[tr_idx])
        model, tok, classes, info = train_classifier(
            csv, device=device, seed=seed, split_indices=(tr_idx, va_idx), verbose=False)
        clf = ClassifierRetriever(model, tok, classes, info["maxlen"], device=device)
        val_accs.append(info["val_top1"])

        val_sent, val_true = texts[va_idx], labels[va_idx]
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
            pc[c]["nval"].append(b["n"]); pc[c]["ntrain"].append(int(tr_counts[c]))
        print(f"  seed {seed}: val_top1={info['val_top1']:.4f}  scored classes={len(by_cls)}")

    rows = []
    for c, a in pc.items():
        rows.append({"endpoint": c, "total_rows": int(freq[c]),
                     "mean_train_rows": round(float(np.mean(a["ntrain"])), 2),
                     "clf_r1": round(float(np.mean(a["clf1"])), 4),
                     "retr_r1": round(float(np.mean(a["retr1"])), 4),
                     "clf_r3": round(float(np.mean(a["clf3"])), 4),
                     "retr_r3": round(float(np.mean(a["retr3"])), 4),
                     "n_seeds": len(a["clf1"])})
    df = pd.DataFrame(rows).sort_values("total_rows").reset_index(drop=True)
    df.to_csv(SCRIPT_DIR / f"results_perclass_singleton{tag}.csv", index=False)

    df["bin"] = pd.cut(df.total_rows, bins=FREQ_BINS, labels=FREQ_LABELS)
    g = df.groupby("bin", observed=True).agg(
        n_classes=("endpoint", "size"),
        clf_r1=("clf_r1", "mean"), retr_r1=("retr_r1", "mean"),
        clf_r3=("clf_r3", "mean"), retr_r3=("retr_r3", "mean")).round(3)
    g.to_csv(SCRIPT_DIR / f"results_perclass_singleton_bins{tag}.csv")

    print("\n" + "=" * 78)
    print(f"SINGLETONS COPIED into train+val (all 170 scored), {device}, mean over {len(SEEDS)} seeds")
    print("=" * 78)
    print(f"  in-distribution val top-1 : {np.mean(val_accs):.4f} +/- {np.std(val_accs, ddof=1):.4f}")
    print("\n  per-endpoint recall by total trace frequency:")
    print(g.to_string())
    print(f"\n  Macro over all {len(df)} classes: clf@1={df.clf_r1.mean():.3f} retr@1={df.retr_r1.mean():.3f} "
          f"clf@3={df.clf_r3.mean():.3f} retr@3={df.retr_r3.mean():.3f}")
    sing = df[df.total_rows == 1]
    print(f"  Singletons only (n={len(sing)}, 1 train copy each): "
          f"clf@1={sing.clf_r1.mean():.3f} retr@1={sing.retr_r1.mean():.3f} "
          f"clf@3={sing.clf_r3.mean():.3f} retr@3={sing.retr_r3.mean():.3f}")
    print(f"\nsaved results_perclass_singleton{tag}.csv and results_perclass_singleton_bins{tag}.csv in {SCRIPT_DIR}")


if __name__ == "__main__":
    main()
