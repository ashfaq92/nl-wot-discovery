"""
Record-format ablation (2.6): does the retrieval advantage come from semantic
matching, or from the specific sentence template? Compares the format-sensitive
retrievers across the three record serializations, no reranking.

  sentence : "I need to {op} {service} in {location}"
  tuple    : "operation: {op}; service: {service}; location: {location}"
  td_like  : "Thing: {service} in {location}. Affordance: {op} {service}. ..."

Exact lookup is format-invariant (it matches structured field values, not the
serialized text), so it is reported once as a reference.

Run:  python run_format_ablation.py [csv]
"""

import os
import sys

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pandas as pd

from eval_lib import (DEFAULT_RETRIEVE_K, load_records, build_record_text,
                      evaluate)
from queries import QUERIES
from retrievers import ExactLookupRetriever, TfidfRetriever, BM25Retriever, BiEncoderRetriever

KS = (1, 3, 5, 10)
RETRIEVE_K = DEFAULT_RETRIEVE_K
CSV = sys.argv[1] if len(sys.argv) > 1 else "mainSimulationAccessTraces.csv"
FORMATS = ("sentence", "tuple", "td_like")
BI = "sentence-transformers/all-MiniLM-L6-v2"

base = load_records(CSV)
rows = []


def add(fmt, name, res):
    o, na = res["overall"], res["no_answer"]
    rows.append({
        "format": fmt, "retriever": name,
        "hit@1": round(o["hit@1"], 3), "hit@3": round(o["hit@3"], 3),
        "hit@5": round(o["hit@5"], 3), "mrr": round(o["mrr"], 3),
        "na_auroc": round(na["auroc"], 3),
        "na_auprc": round(na["auprc_no_answer"], 3),
    })


# exact lookup: format-invariant, run once
add("(invariant)", "exact",
    evaluate(ExactLookupRetriever(base), QUERIES, ks=KS,
             retrieve_k=RETRIEVE_K))

for fmt in FORMATS:
    recs = build_record_text(base, fmt=fmt)
    add(fmt, "tfidf", evaluate(TfidfRetriever(recs), QUERIES, ks=KS,
                                retrieve_k=RETRIEVE_K))
    add(fmt, "bm25", evaluate(BM25Retriever(recs), QUERIES, ks=KS,
                               retrieve_k=RETRIEVE_K))
    bi = BiEncoderRetriever(recs, BI, use_reranker=False, device="cpu")
    add(fmt, "MiniLM-L6", evaluate(bi, QUERIES, ks=KS,
                                    retrieve_k=RETRIEVE_K))

df = pd.DataFrame(rows)
df.to_csv("results_format_ablation.csv", index=False)
print(df.to_string(index=False))
print("\nsaved results_format_ablation.csv")
