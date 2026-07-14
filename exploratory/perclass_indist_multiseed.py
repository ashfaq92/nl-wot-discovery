"""
Per-class (per-endpoint) in-distribution comparison, multi-seed, with top-1 and
top-3 recall: supervised classifier vs. frozen retriever.

Extends perclass_indist_compare.py along two axes:
  1. top-1 AND top-3 recall for BOTH methods (scored uniformly through each
     method's .search() over the same validation sentences). top-3 tests whether
     the shared-sentence ambiguity that caps top-1 is resolved when the method may
     return more than one candidate.
  2. 5 seeds (42-46): per-class recall is aggregated as mean +/- STD.

For every seed we reconstruct the classifier's random-split validation set, score
both methods on its sentences, and compute per-endpoint recall@1 and recall@3
(recall_c = fraction of class-c validation rows whose true endpoint appears in the
de-duplicated top-k). The retriever index is seed-invariant, so its per-sentence
ranking is computed once and cached; only the classifier retrains per seed.

A class is `shared_sentence` if one of its generated sentences also maps to another
endpoint -- the collision a single top-1 answer cannot resolve.

Run (from anywhere):  python exploratory/perclass_indist_multiseed.py [csv]
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from classifier_baseline import build_training_data, train_classifier, ClassifierRetriever  # noqa: E402
from eval_lib import load_records, build_record_text, hit_at_k  # noqa: E402
from retrievers import BiEncoderRetriever  # noqa: E402

SEEDS = [42, 43, 44, 45, 46]
TEST_SIZE = 0.25
BI_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
POOL = 10            # retrieve enough candidates to read off 3 distinct endpoints


def macro(d):
    return float(np.mean(list(d.values()))) if d else float("nan")


def msd(vals):
    a = np.array(vals, dtype=float)
    return (round(float(a.mean()), 4),
            round(float(a.std(ddof=1)), 4) if len(a) > 1 else 0.0)


def main():
    csv = sys.argv[1] if len(sys.argv) > 1 else str(CODE_DIR / "mainSimulationAccessTraces.csv")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}   seeds = {SEEDS}")

    # --- seed-invariant pieces -------------------------------------------------
    texts, labels = build_training_data(csv, preprocess=True)
    labels = np.array(labels)
    n = len(texts)
    classes = sorted(set(labels))

    ep_per_sent = defaultdict(set)
    for t, l in zip(texts, labels):
        ep_per_sent[t].add(l)
    is_shared_class = {}            # endpoint -> bool (any of its sentences collides)
    sent_of_class = defaultdict(set)
    for t, l in zip(texts, labels):
        sent_of_class[l].add(t)
    for c in classes:
        is_shared_class[c] = any(len(ep_per_sent[s]) > 1 for s in sent_of_class[c])

    print("building retriever index (seed-invariant) ...")
    recs = build_record_text(load_records(csv), fmt="sentence")
    retr = BiEncoderRetriever(recs, BI_ENCODER, device=device)
    retr_rank_cache = {}            # sentence -> ranked endpoint list (top POOL)

    def retr_rank(s):
        if s not in retr_rank_cache:
            retr_rank_cache[s] = [e for e, _ in retr.search(s, POOL)]
        return retr_rank_cache[s]

    # --- per-seed: per-class recall@1 / recall@3 for both methods --------------
    # accumulate per class across seeds
    acc = defaultdict(lambda: {"clf1": [], "clf3": [], "retr1": [], "retr3": [], "n": []})
    per_seed_summary = []

    for seed in SEEDS:
        print(f"\n=== seed {seed} ===")
        idx_va = train_test_split(np.arange(n), test_size=TEST_SIZE, random_state=seed)[1]
        val_sent = [texts[i] for i in idx_va]
        val_true = labels[idx_va]

        model, tok, classes_c, info = train_classifier(
            csv, device=device, split="random", seed=seed, verbose=False)
        assert classes_c == classes
        clf = ClassifierRetriever(model, tok, classes, info["maxlen"], device=device)

        clf_rank_cache = {}
        def clf_rank(s):
            if s not in clf_rank_cache:
                clf_rank_cache[s] = [e for e, _ in clf.search(s, POOL)]
            return clf_rank_cache[s]

        # per-row hits, grouped by class
        by_cls = defaultdict(lambda: {"clf1": 0, "clf3": 0, "retr1": 0, "retr3": 0, "n": 0})
        for s, true_ep in zip(val_sent, val_true):
            exp = [true_ep]
            cr, rr = clf_rank(s), retr_rank(s)
            b = by_cls[true_ep]
            b["n"] += 1
            b["clf1"] += hit_at_k(cr, exp, 1); b["clf3"] += hit_at_k(cr, exp, 3)
            b["retr1"] += hit_at_k(rr, exp, 1); b["retr3"] += hit_at_k(rr, exp, 3)

        # per-class recall this seed -> accumulate
        rec = {m: {} for m in ("clf1", "clf3", "retr1", "retr3")}
        for c, b in by_cls.items():
            for m in ("clf1", "clf3", "retr1", "retr3"):
                rec[m][c] = b[m] / b["n"]
            a = acc[c]
            a["clf1"].append(rec["clf1"][c]); a["clf3"].append(rec["clf3"][c])
            a["retr1"].append(rec["retr1"][c]); a["retr3"].append(rec["retr3"][c])
            a["n"].append(b["n"])

        # per-seed summary (micro = row-weighted; macro = unweighted over classes)
        nrows = len(val_sent)
        present = list(by_cls)
        uniq_c = [c for c in present if not is_shared_class[c]]
        shar_c = [c for c in present if is_shared_class[c]]
        row = {"seed": seed, "n_val": nrows, "n_classes": len(present)}
        for m in ("clf1", "clf3", "retr1", "retr3"):
            row[f"micro_{m}"] = sum(by_cls[c][m] for c in present) / nrows
            row[f"macro_{m}"] = macro(rec[m])
            row[f"macro_{m}_unique"] = macro({c: rec[m][c] for c in uniq_c})
            row[f"macro_{m}_shared"] = macro({c: rec[m][c] for c in shar_c})
        per_seed_summary.append(row)
        print(f"  micro: clf@1={row['micro_clf1']:.4f} retr@1={row['micro_retr1']:.4f} "
              f"clf@3={row['micro_clf3']:.4f} retr@3={row['micro_retr3']:.4f}")
        print(f"  macro: clf@1={row['macro_clf1']:.4f} retr@1={row['macro_retr1']:.4f} "
              f"clf@3={row['macro_clf3']:.4f} retr@3={row['macro_retr3']:.4f}")

    # --- per-class aggregate CSV (mean +/- STD across seeds) -------------------
    rows = []
    for c in classes:
        a = acc[c]
        if not a["n"]:
            continue
        d = {"endpoint": c, "shared_sentence": is_shared_class[c],
             "n_seeds_present": len(a["n"]), "mean_n_val": round(float(np.mean(a["n"])), 1)}
        for m, lbl in [("clf1", "clf_r1"), ("retr1", "retr_r1"),
                       ("clf3", "clf_r3"), ("retr3", "retr_r3")]:
            d[f"{lbl}_mean"], d[f"{lbl}_std"] = msd(a[m])
        d["delta_r1_mean"] = round(d["retr_r1_mean"] - d["clf_r1_mean"], 4)
        rows.append(d)
    df = pd.DataFrame(rows).sort_values(
        ["delta_r1_mean", "mean_n_val"], ascending=[False, False]).reset_index(drop=True)
    perclass_path = SCRIPT_DIR / "results_perclass_indist_multiseed.csv"
    df.to_csv(perclass_path, index=False)

    # --- summary: mean +/- STD over seeds --------------------------------------
    ps = pd.DataFrame(per_seed_summary)
    summ_rows = []
    stat_cols = {"micro": "micro_{m}", "macro": "macro_{m}",
                 "macro_unique": "macro_{m}_unique", "macro_shared": "macro_{m}_shared"}
    for m, lbl in [("clf1", "classifier@1"), ("retr1", "retriever@1"),
                   ("clf3", "classifier@3"), ("retr3", "retriever@3")]:
        r = {"method": lbl}
        for stat, tmpl in stat_cols.items():
            r[f"{stat}_mean"], r[f"{stat}_std"] = msd(ps[tmpl.format(m=m)].tolist())
        summ_rows.append(r)
    summ = pd.DataFrame(summ_rows)
    summ_path = SCRIPT_DIR / "results_perclass_indist_multiseed_summary.csv"
    summ.to_csv(summ_path, index=False)

    print("\n" + "=" * 78)
    print(f"Per-class in-distribution comparison, mean +/- STD over {len(SEEDS)} seeds")
    print("=" * 78)
    hdr = f"{'method':14s} {'micro':>16s} {'macro(all)':>16s} {'macro(unique)':>16s} {'macro(shared)':>16s}"
    print(hdr)
    for r in summ_rows:
        print(f"{r['method']:14s} "
              f"{r['micro_mean']:.3f}±{r['micro_std']:.3f}   "
              f"{r['macro_mean']:.3f}±{r['macro_std']:.3f}   "
              f"{r['macro_unique_mean']:.3f}±{r['macro_unique_std']:.3f}   "
              f"{r['macro_shared_mean']:.3f}±{r['macro_shared_std']:.3f}")
    print(f"\nsaved {perclass_path.name} and {summ_path.name} in {SCRIPT_DIR}")


if __name__ == "__main__":
    main()
