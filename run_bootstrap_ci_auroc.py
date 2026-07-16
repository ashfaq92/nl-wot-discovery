"""Bootstrap 95% CIs for the no-answer AUROC and AUPRC (no-answer class)
comparisons in the centralized table.

For each method (five frozen bi-encoders, the best reranked configuration,
and the three training-free baselines), computes the per-query top-1 score
and answerable flag over all 73 benchmark queries, then bootstraps (B=10,000,
resampling query indices with replacement) a percentile 95% CI for:
  (1) AUROC (answerable-vs-no-answer separability of the top-1 score), and
  (2) AUPRC on the no-answer class (positive = no-answer, ranked by low score),
      the imbalance-aware metric given only 12/73 no-answer queries.
Resamples that draw zero queries of either class are skipped (undefined AUROC).

The pipeline is deterministic (frozen encoders, no training), so no seeds are
involved beyond the fixed bootstrap RNG (42).

Run (from code/):  python run_bootstrap_ci_auroc.py [csv]
"""

import os
import sys

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from eval_lib import load_records, build_record_text, evaluate
from queries import QUERIES
from retrievers import (BM25Retriever, BiEncoderRetriever,
                        ExactLookupRetriever, TfidfRetriever)

WINNER = "all-MiniLM-L6-v2"
BI = ["sentence-transformers/all-MiniLM-L6-v2",
      "sentence-transformers/all-MiniLM-L12-v2",
      "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
      "sentence-transformers/all-mpnet-base-v2",
      "sentence-transformers/multi-qa-mpnet-base-dot-v1"]
BEST_RERANK = ("sentence-transformers/all-mpnet-base-v2",
               "cross-encoder/ms-marco-TinyBERT-L-2-v2")
B = 10_000
RNG_SEED = 42
CSV = sys.argv[1] if len(sys.argv) > 1 else "mainSimulationAccessTraces.csv"


def per_query_score(retriever):
    res = evaluate(retriever, QUERIES, ks=(1, 3, 5))
    pq = res["per_query"].sort_values("id")
    return (pq["top_score"].to_numpy(dtype=float),
            pq["answerable"].to_numpy(dtype=bool),
            pq["id"].tolist())


def bootstrap_ci(y, s, idx):
    """idx: (B, n) resample indices. Returns (auroc_pt, auroc_lo, auroc_hi,
    auprc_pt, auprc_lo, auprc_hi) skipping resamples missing a class."""
    aurocs, auprcs = [], []
    for row in idx:
        yy, ss = y[row], s[row]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        aurocs.append(roc_auc_score(yy, ss))
        auprcs.append(average_precision_score(1 - yy, -ss))
    aurocs, auprcs = np.array(aurocs), np.array(auprcs)
    auroc_pt = roc_auc_score(y, s)
    auprc_pt = average_precision_score(1 - y, -s)
    return (round(float(auroc_pt), 3),
            round(float(np.percentile(aurocs, 2.5)), 3),
            round(float(np.percentile(aurocs, 97.5)), 3),
            round(float(auprc_pt), 3),
            round(float(np.percentile(auprcs, 2.5)), 3),
            round(float(np.percentile(auprcs, 97.5)), 3))


if __name__ == "__main__":
    records = build_record_text(load_records(CSV), fmt="sentence")

    methods = {}
    ids_ref = None
    for name in BI:
        short = name.split("/")[-1]
        s, y, ids = per_query_score(BiEncoderRetriever(records, name, device="cpu"))
        methods[short] = (y, s)
        ids_ref = ids_ref or ids
        assert ids == ids_ref
    bi_name, cross_name = BEST_RERANK
    rerank = BiEncoderRetriever(records, bi_name, cross_encoder_name=cross_name,
                                use_reranker=True, device="cpu")
    s, y, _ = per_query_score(rerank)
    methods[f"{bi_name.split('/')[-1]}+{cross_name.split('/')[-1]}"] = (y, s)
    for cls, key in [(ExactLookupRetriever, "exact"), (TfidfRetriever, "tfidf"),
                     (BM25Retriever, "bm25")]:
        s, y, _ = per_query_score(cls(records))
        methods[key] = (y, s)

    n = len(ids_ref)
    rng = np.random.default_rng(RNG_SEED)
    idx = rng.integers(0, n, size=(B, n))

    rows = []
    for name, (y, s) in methods.items():
        auroc_pt, auroc_lo, auroc_hi, auprc_pt, auprc_lo, auprc_hi = bootstrap_ci(y, s, idx)
        rows.append({"method": name, "n_queries": n, "n_no_answer": int((~y).sum()),
                     "auroc": auroc_pt, "auroc_ci_lo": auroc_lo, "auroc_ci_hi": auroc_hi,
                     "auprc_no_answer": auprc_pt,
                     "auprc_no_answer_ci_lo": auprc_lo, "auprc_no_answer_ci_hi": auprc_hi})

    df = pd.DataFrame(rows)
    df.to_csv("results_bootstrap_ci_auroc.csv", index=False)
    print(df.to_string(index=False))
    print("\nsaved results_bootstrap_ci_auroc.csv")
