"""Paired bootstrap 95% CIs for the centralized Hit@1 comparisons.

For each method in the centralized table (five frozen bi-encoders, the best
reranked configuration, and the three training-free baselines), computes the
per-query Hit@1 vector over the answerable queries, then:
  (1) a percentile bootstrap CI for each method's own Hit@1, and
  (2) a PAIRED bootstrap CI for the difference between the deployed winner
      (all-MiniLM-L6-v2) and every rival, resampling query indices with
      replacement so both methods are evaluated on the same resample.
The pipeline is deterministic, so no seeds are involved beyond the fixed
bootstrap RNG (42). B = 10,000 resamples.

Run (from code/):  python run_bootstrap_ci.py [csv]
"""

import os
import sys

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd

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


def per_query_hit1(retriever):
    res = evaluate(retriever, QUERIES, ks=(1, 3, 5))
    pq = res["per_query"]
    ans = pq[pq["answerable"]].sort_values("id")
    return ans["hit@1"].to_numpy(dtype=float), ans["id"].tolist()


if __name__ == "__main__":
    records = build_record_text(load_records(CSV), fmt="sentence")

    methods = {}
    ids_ref = None
    for name in BI:
        short = name.split("/")[-1]
        hits, ids = per_query_hit1(BiEncoderRetriever(records, name, device="cpu"))
        methods[short] = hits
        ids_ref = ids_ref or ids
        assert ids == ids_ref
    bi_name, cross_name = BEST_RERANK
    rerank = BiEncoderRetriever(records, bi_name, cross_encoder_name=cross_name,
                                use_reranker=True, device="cpu")
    methods[f"{bi_name.split('/')[-1]}+{cross_name.split('/')[-1]}"] = \
        per_query_hit1(rerank)[0]
    methods["exact"] = per_query_hit1(ExactLookupRetriever(records))[0]
    methods["tfidf"] = per_query_hit1(TfidfRetriever(records))[0]
    methods["bm25"] = per_query_hit1(BM25Retriever(records))[0]

    n = len(ids_ref)
    rng = np.random.default_rng(RNG_SEED)
    idx = rng.integers(0, n, size=(B, n))

    rows = []
    w = methods[WINNER]
    for name, h in methods.items():
        own = h[idx].mean(axis=1)
        row = {"method": name, "n_queries": n,
               "hit1": round(h.mean(), 3),
               "ci_lo": round(float(np.percentile(own, 2.5)), 3),
               "ci_hi": round(float(np.percentile(own, 97.5)), 3)}
        if name != WINNER:
            diff = (w - h)[idx].mean(axis=1)
            row.update({
                "diff_vs_winner": round(float((w - h).mean()), 3),
                "diff_ci_lo": round(float(np.percentile(diff, 2.5)), 3),
                "diff_ci_hi": round(float(np.percentile(diff, 97.5)), 3),
                "excludes_zero": bool(np.percentile(diff, 2.5) > 0
                                      or np.percentile(diff, 97.5) < 0)})
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv("results_bootstrap_ci.csv", index=False)
    print(df.to_string(index=False))
    print("\nsaved results_bootstrap_ci.csv")
