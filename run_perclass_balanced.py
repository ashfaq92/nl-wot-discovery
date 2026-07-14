"""
Per-endpoint recall under (A) proportional-stratified and (B) class-balanced
training, both with singleton duplication.

Shared split per seed (stratified + singleton copy): every one of the 170
endpoints gets >=1 train and >=1 val row; single-row endpoints have their lone row
in both (one train copy, one val copy). On this split we train the classifier two
ways and score both against the (fixed, untrained) frozen retriever on the SAME
held-out validation rows:

  (A) stratified : train on the natural proportional train rows (frequent classes
                   dominate) -- the realistic imbalance.
  (B) balanced   : resample EACH class's training rows to TARGET (downsample the
                   majority, upsample the minority with replacement) so every class
                   contributes equally. Capped at TARGET to keep the set modest
                   (170 x TARGET rows, not the 170 x 19,032 a naive upsample gives).

Question: does removing the class imbalance let the classifier finally learn the
rare/singleton endpoints, or does it still fail? These numbers are the paper's
per-class table (tab:perclass-indist) and the re-balancing discussion.

Both trained models are also scored on the 73-query benchmark, and a summary CSV
(results_perclass_balanced_summary.csv) persists the validation accuracies and
benchmark numbers, so the manuscript's "balancing costs overall accuracy and does
not transfer to the benchmark" claims trace to a saved artifact.

Runs on CPU by default (the reported environment). Set FORCE_DEVICE=cuda for a
scratch GPU run; its outputs get a `_gpu` suffix and are never reported.

Run (from code/):  python run_perclass_balanced.py [csv]
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
from eval_lib import load_records, build_record_text, hit_at_k, evaluate  # noqa: E402
from queries import QUERIES  # noqa: E402
from retrievers import BiEncoderRetriever  # noqa: E402

SEEDS = [42, 43, 44, 45, 46]
TEST_SIZE = 0.25
TARGET = 500           # per-class training rows after balancing (threshold)
POOL = 10
BI_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
FREQ_BINS = [0.5, 1.5, 3, 15, 150, 1500, 1e9]
FREQ_LABELS = ["1 (singleton)", "2-3", "4-15", "16-150", "151-1500", ">1500"]


def stratified_singleton_copy(labels, seed, test_size=TEST_SIZE):
    """Proportional per-class split; single-row classes copied into both sets."""
    rng = np.random.RandomState(seed)
    by_c = defaultdict(list)
    for i, l in enumerate(labels):
        by_c[l].append(i)
    tr, va = [], []
    for c, idx in by_c.items():
        idx = np.array(idx); rng.shuffle(idx)
        k = len(idx)
        if k == 1:
            tr.append(idx[0]); va.append(idx[0])
        else:
            n_val = min(max(1, int(round(test_size * k))), k - 1)
            va.extend(idx[:n_val]); tr.extend(idx[n_val:])
    return np.array(tr), np.array(va)


def balance_train(tr_idx, labels, target, seed):
    """Resample each class's training rows to exactly `target` (down/upsample)."""
    rng = np.random.RandomState(seed + 1000)
    by_c = defaultdict(list)
    for i in tr_idx:
        by_c[labels[i]].append(i)
    out = []
    for c, idx in by_c.items():
        idx = np.array(idx)
        replace = len(idx) < target
        out.extend(rng.choice(idx, size=target, replace=replace).tolist())
    return np.array(out)


def main():
    csv = sys.argv[1] if len(sys.argv) > 1 else str(CODE_DIR / "mainSimulationAccessTraces.csv")
    device = os.environ.get("FORCE_DEVICE") or "cpu"
    tag = "" if device == "cpu" else "_gpu"
    print(f"device = {device}   seeds = {SEEDS}   balance target = {TARGET}/class")

    texts, labels = build_training_data(csv, preprocess=True)
    texts = np.array(texts); labels = np.array(labels)
    freq = Counter(labels)
    print(f"endpoints: {len(freq)}; balanced train set = {len(freq)*TARGET} rows "
          f"(vs natural ~{int(0.75*len(labels))})")

    print(f"building retriever index ({device}) ...")
    recs = build_record_text(load_records(csv), fmt="sentence")
    retr = BiEncoderRetriever(recs, BI_ENCODER, device=device)
    retr_cache = {}

    def retr_rank(s):
        if s not in retr_cache:
            retr_cache[s] = [e for e, _ in retr.search(s, POOL)]
        return retr_cache[s]

    pc = defaultdict(lambda: {"s1": [], "s3": [], "b1": [], "b3": [], "r1": [], "r3": []})
    acc_s, acc_b = [], []
    bench = defaultdict(list)   # benchmark transfer of both trained models
    BENCH_CATS = ("paraphrased", "synonym", "abstract")
    for seed in SEEDS:
        tr_idx, va_idx = stratified_singleton_copy(labels, seed)
        bal_idx = balance_train(tr_idx, labels, TARGET, seed)

        ms, toks, classes, infos = train_classifier(
            csv, device=device, seed=seed, split_indices=(tr_idx, va_idx), verbose=False)
        mb, tokb, _, infob = train_classifier(
            csv, device=device, seed=seed, split_indices=(bal_idx, va_idx), verbose=False)
        clf_s = ClassifierRetriever(ms, toks, classes, infos["maxlen"], device=device)
        clf_b = ClassifierRetriever(mb, tokb, classes, infob["maxlen"], device=device)
        acc_s.append(infos["val_top1"]); acc_b.append(infob["val_top1"])

        # does (re-)training transfer to the benchmark queries?
        for name, clf in (("strat", clf_s), ("bal", clf_b)):
            res = evaluate(clf, QUERIES, ks=(1, 3))
            bench[f"{name}_hit1"].append(res["overall"]["hit@1"])
            bench[f"{name}_hit3"].append(res["overall"]["hit@3"])
            for c in BENCH_CATS:
                if c in res["by_category"]:
                    bench[f"{name}_{c}_hit1"].append(res["by_category"][c]["hit@1"])

        val_sent, val_true = texts[va_idx], labels[va_idx]
        cs, cb = {}, {}
        by_cls = defaultdict(lambda: {"s1": 0, "s3": 0, "b1": 0, "b3": 0, "r1": 0, "r3": 0, "n": 0})
        for s, true_ep in zip(val_sent, val_true):
            if s not in cs:
                cs[s] = [e for e, _ in clf_s.search(s, POOL)]
                cb[s] = [e for e, _ in clf_b.search(s, POOL)]
            exp = [true_ep]; rr = retr_rank(s)
            b = by_cls[true_ep]; b["n"] += 1
            b["s1"] += hit_at_k(cs[s], exp, 1); b["s3"] += hit_at_k(cs[s], exp, 3)
            b["b1"] += hit_at_k(cb[s], exp, 1); b["b3"] += hit_at_k(cb[s], exp, 3)
            b["r1"] += hit_at_k(rr, exp, 1);    b["r3"] += hit_at_k(rr, exp, 3)
        for c, b in by_cls.items():
            for m in ("s1", "s3", "b1", "b3", "r1", "r3"):
                pc[c][m].append(b[m] / b["n"])
        print(f"  seed {seed}: val_top1 strat={infos['val_top1']:.3f} balanced={infob['val_top1']:.3f}")

    rows = []
    for c, a in pc.items():
        rows.append({"endpoint": c, "total_rows": int(freq[c]),
                     "retr_r1": round(np.mean(a["r1"]), 4), "retr_r3": round(np.mean(a["r3"]), 4),
                     "strat_clf_r1": round(np.mean(a["s1"]), 4), "strat_clf_r3": round(np.mean(a["s3"]), 4),
                     "bal_clf_r1": round(np.mean(a["b1"]), 4), "bal_clf_r3": round(np.mean(a["b3"]), 4)})
    df = pd.DataFrame(rows).sort_values("total_rows").reset_index(drop=True)
    df.to_csv(SCRIPT_DIR / f"results_perclass_balanced{tag}.csv", index=False)

    df["bin"] = pd.cut(df.total_rows, bins=FREQ_BINS, labels=FREQ_LABELS)
    g = df.groupby("bin", observed=True).agg(
        n=("endpoint", "size"),
        retr_r1=("retr_r1", "mean"), strat_clf_r1=("strat_clf_r1", "mean"), bal_clf_r1=("bal_clf_r1", "mean"),
        retr_r3=("retr_r3", "mean"), strat_clf_r3=("strat_clf_r3", "mean"), bal_clf_r3=("bal_clf_r3", "mean")
    ).round(3)
    g.to_csv(SCRIPT_DIR / f"results_perclass_balanced_bins{tag}.csv")

    summary = {"n_seeds": len(SEEDS), "target_per_class": TARGET,
               "val_top1_strat_mean": round(float(np.mean(acc_s)), 4),
               "val_top1_strat_std": round(float(np.std(acc_s, ddof=1)), 4),
               "val_top1_bal_mean": round(float(np.mean(acc_b)), 4),
               "val_top1_bal_std": round(float(np.std(acc_b, ddof=1)), 4),
               "macro_r1_strat": round(float(df.strat_clf_r1.mean()), 4),
               "macro_r1_bal": round(float(df.bal_clf_r1.mean()), 4),
               "macro_r1_retr": round(float(df.retr_r1.mean()), 4)}
    for k, v in bench.items():
        summary[f"bench_{k}_mean"] = round(float(np.mean(v)), 4)
        summary[f"bench_{k}_std"] = round(float(np.std(v, ddof=1)), 4) if len(v) > 1 else 0.0
    pd.DataFrame([summary]).to_csv(
        SCRIPT_DIR / f"results_perclass_balanced_summary{tag}.csv", index=False)

    print("\n" + "=" * 90)
    print(f"Stratified vs class-balanced (TARGET={TARGET}/class) classifier, {device}, mean over {len(SEEDS)} seeds")
    print("(retriever is the same fixed reference for both)")
    print("=" * 90)
    print(f"  val top-1 (natural val): stratified {np.mean(acc_s):.4f}  balanced {np.mean(acc_b):.4f}")
    print("\n  per-endpoint top-1 recall by frequency:    retr | strat-clf | bal-clf")
    print(g[["n", "retr_r1", "strat_clf_r1", "bal_clf_r1"]].to_string())
    print("\n  per-endpoint top-3 recall by frequency:    retr | strat-clf | bal-clf")
    print(g[["n", "retr_r3", "strat_clf_r3", "bal_clf_r3"]].to_string())
    print(f"\n  Macro (all {len(df)}): retr@1={df.retr_r1.mean():.3f}  "
          f"strat-clf@1={df.strat_clf_r1.mean():.3f}  bal-clf@1={df.bal_clf_r1.mean():.3f}")
    sing = df[df.total_rows == 1]
    print(f"  Singletons (n={len(sing)}): retr@1={sing.retr_r1.mean():.3f}  "
          f"strat-clf@1={sing.strat_clf_r1.mean():.3f}  bal-clf@1={sing.bal_clf_r1.mean():.3f}")
    print("\n  benchmark transfer (73-query benchmark, hit@1):")
    print(f"    stratified {np.mean(bench['strat_hit1']):.3f}  balanced {np.mean(bench['bal_hit1']):.3f}")
    for c in BENCH_CATS:
        print(f"    {c:12s} strat={np.mean(bench[f'strat_{c}_hit1']):.3f}  "
              f"bal={np.mean(bench[f'bal_{c}_hit1']):.3f}")
    print(f"\nsaved results_perclass_balanced{tag}.csv, results_perclass_balanced_bins{tag}.csv "
          f"and results_perclass_balanced_summary{tag}.csv in {SCRIPT_DIR}")


if __name__ == "__main__":
    main()
