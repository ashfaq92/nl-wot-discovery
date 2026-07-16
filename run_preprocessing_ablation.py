"""
Preprocessing ablation (appendix): does text preprocessing (camelCase split,
lowercasing, slash removal -- eval_lib.preprocess_text) drive the retrieval-
versus-classifier gap, or does the retrieval advantage hold without it?

Prior work's classifier reimplementation performs no such normalization; this
was confirmed directly with the original author (Llopis et al.), who reported
no camelCase splitting or lowercasing in their pipeline. Since our default
setup applies eval_lib.preprocess_text to both the retriever and the
classifier baseline (a fair, consistent choice, but a deviation from prior
work), a reviewer could reasonably ask whether that shared preprocessing step
is responsible for retrieval's advantage over the classifier.

This script reruns the deployed retriever (all-MiniLM-L6-v2, no reranker) and
the classifier baseline (5 seeds, matching run_classifier_multiseed.py's
protocol) with preprocessing OFF on both sides -- raw field-concatenated
sentences, original casing, unsplit camelCase -- and reports the same overall
Hit@1/Hit@3/MRR metrics as the preprocessed (default) numbers already in
results_centralized_sentence.csv / results_classifier_multiseed.csv, so the
two settings can be compared directly.

Run (from code/):  python run_preprocessing_ablation.py [csv]
"""

import os
import sys

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pandas as pd

from eval_lib import load_records, build_record_text, evaluate
from queries import QUERIES
from retrievers import BiEncoderRetriever
from classifier_baseline import train_classifier, ClassifierRetriever

CSV = sys.argv[1] if len(sys.argv) > 1 else "mainSimulationAccessTraces.csv"
BI_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
SEEDS = [42, 43, 44, 45, 46]
KS = (1, 3, 5, 10)


def summarize(res):
    o = res["overall"]
    return {f"hit@{k}": round(o[f"hit@{k}"], 3) for k in KS} | {"mrr": round(o["mrr"], 3)}


def msd(vals, nd=3):
    import numpy as np
    a = np.asarray(vals, dtype=float)
    return round(float(a.mean()), nd), (round(float(a.std(ddof=1)), nd) if len(a) > 1 else 0.0)


def main():
    raw_records = load_records(CSV)
    rows = []

    print("[retriever] all-MiniLM-L6-v2, no rerank")
    for preprocess in (True, False):
        records = build_record_text(raw_records, fmt="sentence", preprocess=preprocess)
        retr = BiEncoderRetriever(records, BI_ENCODER, use_reranker=False,
                                  device="cpu", preprocess_query=preprocess)
        res = evaluate(retr, QUERIES, ks=KS)
        row = {"method": "retriever", "preprocess": preprocess, **summarize(res)}
        rows.append(row)
        print(f"  preprocess={preprocess!s:5s} " +
              " ".join(f"{k}={row[k]:.3f}" for k in ["hit@1", "hit@3", "mrr"]))

    print("\n[classifier] 5 seeds, no preprocessing")
    seed_rows = []
    for seed in SEEDS:
        model, tok, classes, info = train_classifier(
            CSV, preprocess=False, device="cpu", split="random", seed=seed, verbose=False)
        clf = ClassifierRetriever(model, tok, classes, info["maxlen"],
                                  preprocess=False, device="cpu")
        res = evaluate(clf, QUERIES, ks=KS)
        srow = {"seed": seed, "val_top1": info["val_top1"], **summarize(res)}
        seed_rows.append(srow)
        print(f"  seed={seed} val_top1={srow['val_top1']:.4f} "
              f"hit@1={srow['hit@1']:.3f} hit@3={srow['hit@3']:.3f} mrr={srow['mrr']:.3f}")

    sdf = pd.DataFrame(seed_rows)
    crow = {"method": "classifier", "preprocess": False}
    for m in ["val_top1", "hit@1", "hit@3", "hit@5", "hit@10", "mrr"]:
        mean, std = msd(sdf[m].tolist())
        crow[f"{m}_mean"] = mean
        crow[f"{m}_std"] = std
    rows.append(crow)

    df = pd.DataFrame(rows)
    df.to_csv("results_preprocessing_ablation.csv", index=False)
    print("\n" + df.to_string(index=False))
    print("\nsaved results_preprocessing_ablation.csv")
    print("\nCompare against the preprocessed (default) numbers already in "
          "results_centralized_sentence.csv (retriever) and "
          "results_classifier_multiseed.csv (classifier).")


if __name__ == "__main__":
    main()
