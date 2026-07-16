"""
Centralized retrieval experiment: full baseline ladder + paper bi-encoder matrix,
scored by eval_lib.evaluate over the 73-query benchmark. Saves
results_centralized_<format>.csv (overall) and
results_centralized_<format>_percat.csv (per-category breakdown behind tab:percat).

Run:  python run_centralized.py [path/to/csv] [record_format]
  record_format in {sentence, tuple, td_like}; default sentence.
"""

import os
import sys

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd

from eval_lib import load_records, build_record_text, evaluate, Timer, DEFAULT_RETRIEVE_K
from queries import QUERIES
from retrievers import (ExactLookupRetriever, TfidfRetriever, BM25Retriever,
                        BiEncoderRetriever)

KS = (1, 3, 5, 10)
CSV = sys.argv[1] if len(sys.argv) > 1 else "mainSimulationAccessTraces.csv"
FMT = sys.argv[2] if len(sys.argv) > 2 else "sentence"

LAT_RUNS = 5      # matches "every reported timing is the mean over five runs"
LAT_WARMUP = 5    # queries, un-timed, before the timed runs


def measure_latency_ms(retriever, retrieve_k=DEFAULT_RETRIEVE_K,
                       runs=LAT_RUNS, warmup=LAT_WARMUP):
    """p50/p95 query latency, each the mean over `runs` warmed-up passes over
    the full query set (mirrors run_deployment.py's protocol)."""
    for q in QUERIES[:warmup]:
        retriever.search(q["query"], retrieve_k)
    p50s, p95s = [], []
    for _ in range(runs):
        lat = []
        for q in QUERIES:
            with Timer() as t:
                retriever.search(q["query"], retrieve_k)
            lat.append(t.s)
        lat = np.array(lat) * 1000.0
        p50s.append(float(np.percentile(lat, 50)))
        p95s.append(float(np.percentile(lat, 95)))
    return float(np.mean(p50s)), float(np.mean(p95s))

# Frozen bi-encoders spanning small -> base. All are symmetric sentence
# encoders that need no query/passage prefix (so the comparison is fair without
# model-specific prompting). We report the best empirically -- there is no
# pre-committed lineup.
BI = ["sentence-transformers/all-MiniLM-L6-v2",
      "sentence-transformers/all-MiniLM-L12-v2",
      "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
      "sentence-transformers/all-mpnet-base-v2",
      "sentence-transformers/multi-qa-mpnet-base-dot-v1"]
CROSS = ["cross-encoder/ms-marco-TinyBERT-L-2-v2",
         "cross-encoder/ms-marco-MiniLM-L-6-v2",
         "cross-encoder/ms-marco-MiniLM-L-12-v2"]

records = build_record_text(load_records(CSV), fmt=FMT)
print(f"format={FMT}  records={len(records)}")

rows = []
cat_rows = []


def add(tag, res, load_s=None, index_s=None, retriever=None):
    o, na = res["overall"], res["no_answer"]
    row = {"retriever": tag, "format": FMT}
    for k in KS:
        row[f"hit@{k}"] = round(o[f"hit@{k}"], 3)
    row["mrr"] = round(o["mrr"], 3)
    row["recall@5"] = round(o["recall@5"], 3)
    row["na_auroc"] = round(na["auroc"], 3)
    row["na_auprc"] = round(na["auprc_no_answer"], 3)  # rare-class (abstain) AUPRC
    row["na_f1"] = round(na["best_f1"], 3)
    row["load_s"] = round(load_s, 2) if load_s is not None else ""
    row["index_s"] = round(index_s, 3) if index_s is not None else ""
    p50_ms, p95_ms = measure_latency_ms(retriever)
    row["p50_ms"] = round(p50_ms, 2)
    row["p95_ms"] = round(p95_ms, 2)
    rows.append(row)
    for cat, m in res["by_category"].items():
        cat_rows.append({"retriever": tag, "format": FMT, "category": cat,
                         "n": m["n"], "hit@1": round(m["hit@1"], 3),
                         "hit@3": round(m["hit@3"], 3),
                         "mrr": round(m["mrr"], 3)})
    print(f"  {tag:42s} hit@1={row['hit@1']:.3f} hit@3={row['hit@3']:.3f} "
          f"mrr={row['mrr']:.3f} na_auroc={row['na_auroc']:.3f} p95={row['p95_ms']}ms")


print("\n[lexical baselines]")
for R, tag in [(ExactLookupRetriever, "exact"), (TfidfRetriever, "tfidf"),
               (BM25Retriever, "bm25")]:
    r = R(records)
    add(tag, evaluate(r, QUERIES, ks=KS), retriever=r)

print("\n[bi-encoders, no rerank]")
for bi in BI:
    r = BiEncoderRetriever(records, bi, use_reranker=False, device="cpu")
    add(f"bi:{r.name}", evaluate(r, QUERIES, ks=KS), r.load_time_s, r.index_build_time_s,
        retriever=r)

print("\n[bi-encoder + cross-encoder rerank]")
for bi in BI:
    for cr in CROSS:
        r = BiEncoderRetriever(records, bi, cr, use_reranker=True, device="cpu")
        add(f"bi:{r.name}+{r.cross_name}", evaluate(r, QUERIES, ks=KS),
            r.load_time_s, r.index_build_time_s, retriever=r)

df = pd.DataFrame(rows)
out = f"results_centralized_{FMT}.csv"
df.to_csv(out, index=False)
cat_out = f"results_centralized_{FMT}_percat.csv"
pd.DataFrame(cat_rows).to_csv(cat_out, index=False)
print("\n" + df.to_string(index=False))
print(f"\nsaved {out} and {cat_out}")
