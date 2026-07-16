"""
Shared evaluation library for WoT discovery retrieval experiments.

This module is deliberately retriever-agnostic. Every baseline (exact lookup,
BM25/TF-IDF, frozen bi-encoder, bi-encoder + cross-encoder, the Transformer
recommender) implements one small interface and is then scored by the same
`evaluate()` driver, so all numbers in the paper are comparable.

Contents
--------
1. Data + record building   : load_records, build_record_text, preprocess_text
2. Ranking metrics          : hit_at_k, recall_at_k, reciprocal_rank, precision_at_k
3. No-answer handling        : no_answer_scores (threshold accept/abstain, AUROC, best-F1)
4. Timing / memory           : Timer, embedding_memory_mb, process_memory_mb
5. Retriever interface + driver : Retriever (protocol), evaluate

A *retriever* is any object exposing:

    search(query: str, top_k: int) -> list[tuple[str, float]]

returning (endpoint, score) pairs ranked by descending score. Endpoints may
repeat (the same device endpoint can come from several operation records); the
driver de-duplicates by first (best) occurrence before scoring.
"""

from __future__ import annotations

import re
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

RECORD_COLS = ["operation", "destinationServiceType",
               "destinationLocation", "accessedNodeAddress"]

RECORD_FORMATS = ("sentence", "tuple", "td_like")
DEFAULT_RETRIEVE_K = 50


# ---------------------------------------------------------------------------
# 1. Data + record building
# ---------------------------------------------------------------------------

def preprocess_text(text: str) -> str:
    """Normalize record/query text: split camelCase, split '...room' suffixes,
    drop slashes, collapse whitespace, lowercase.

    Mirrors the preprocessing used in the existing notebooks so results are
    comparable across experiments.
    """
    text = str(text)
    text = re.sub(r"(?<!^)(?=[A-Z])", " ", text)          # camelCase -> camel Case
    text = re.sub(r"(\b\w+)(room)(\w*\b)", r"\1 room\3", text, flags=re.IGNORECASE)
    text = text.replace("/", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def load_records(csv_path: str) -> pd.DataFrame:
    """Load the trace CSV and return the unique service records (286 rows).

    Deduplicates over (operation, service, location, endpoint) -- the exact
    deduplication reported in the Dataset Audit section.
    """
    df = pd.read_csv(csv_path)
    records = (df[RECORD_COLS]
               .dropna()
               .drop_duplicates()
               .reset_index(drop=True)
               .copy())
    return records


def build_record_text(records: pd.DataFrame, fmt: str = "sentence",
                      include_endpoint: bool = False, preprocess: bool = True) -> pd.DataFrame:
    """Add a 'record_text' column rendering each record in the chosen format.

    fmt:
      - "sentence" : "I need to {op} {service} in {location}"  (Llopis-style)
      - "tuple"    : "operation: {op}; service: {service}; location: {location}"
      - "td_like"  : richer TD-style serialization

    include_endpoint: if True, append the endpoint href to the text (only
      meaningful for "td_like"; off by default since queries never contain
      endpoint tokens, so it cannot help matching and only adds noise).

    preprocess: if False, skip preprocess_text() (no camelCase split, no
      lowercasing) so the raw field-concatenated sentence is left untouched --
      matching prior work's classifier reimplementation, which performs no
      such normalization (see run_preprocessing_ablation.py).
    """
    if fmt not in RECORD_FORMATS:
        raise ValueError(f"unknown fmt {fmt!r}; choose from {RECORD_FORMATS}")

    out = records.copy()
    op = out["operation"].astype(str)
    svc = out["destinationServiceType"].astype(str).str.lstrip("/")
    loc = out["destinationLocation"].astype(str)
    ep = out["accessedNodeAddress"].astype(str)

    if fmt == "sentence":
        raw = "I need to " + op + " " + svc + " in " + loc
    elif fmt == "tuple":
        raw = ("operation: " + op + "; service: " + svc + "; location: " + loc)
    else:  # td_like
        raw = ("Thing: " + svc + " in " + loc +
               ". Affordance: " + op + " " + svc +
               ". Location: " + loc)
        if include_endpoint:
            raw = raw + ". Endpoint: " + ep

    out["record_text"] = raw.map(preprocess_text) if preprocess else raw
    return out


# ---------------------------------------------------------------------------
# 2. Ranking metrics
# ---------------------------------------------------------------------------

def _dedup_keep_order(endpoints):
    """De-duplicate endpoints keeping first (best-ranked) occurrence."""
    seen = set()
    out = []
    for e in endpoints:
        if e not in seen:
            seen.add(e)
            out.append(e)
    return out


def _dedup_scored(results):
    """De-duplicate scored (endpoint, score) results, keeping the first
    (best-ranked) occurrence of each endpoint."""
    seen = set()
    out = []
    for ep, sc in results:
        if ep not in seen:
            seen.add(ep)
            out.append((ep, float(sc)))
    return out


def guaranteed_rank(results, expected):
    """Worst-case rank of the first acceptable endpoint under adversarial ties.

    Many DS2OS service records serialize to identical text: they share the same
    operation/service/location while exposing different endpoint addresses (e.g.
    a sensor's ``.../movement`` reading and its ``.../lastChange`` timestamp), so
    a text-only retriever scores them identically and their relative order is
    arbitrary (decided by trace row order alone). We therefore credit a method
    only for a position it is *guaranteed* whatever order exact score ties break
    in. For the best-scoring acceptable endpoint e, that position is

        (# endpoints scoring strictly higher) + (# non-acceptable endpoints tied
        with e) + 1,

    because an adversary can place every strictly-higher record and every
    non-acceptable tied record ahead of e, but cannot push an acceptable record
    past its own tie group. Consequences: a rank-1 tie against any non-acceptable
    sibling is never banked as a hit@1 (the win would be a coin flip), whereas a
    tie whose members are all acceptable stays a guaranteed hit.

    ``results`` is the retriever's scored output ``[(endpoint, score), ...]``.
    Returns the guaranteed rank (>= 1), or None if no acceptable endpoint was
    retrieved.
    """
    ded = _dedup_scored(results)
    best = None
    for ep, sc in ded:
        if ep in expected:
            higher = sum(1 for _, s in ded if s > sc)
            nonacc_tied = sum(1 for e2, s in ded if s == sc and e2 not in expected)
            r = higher + nonacc_tied + 1
            best = r if best is None else min(best, r)
    return best


def hit_at_k(ranked_endpoints, expected, k) -> float:
    """1.0 if any acceptable endpoint appears in the top-k (de-duplicated).

    Endpoint-only variant, blind to score ties; used by the in-distribution
    per-class analysis. Benchmark scoring (see ``evaluate``) uses the tie-aware
    ``hit_at_k_scored`` instead.
    """
    topk = _dedup_keep_order(ranked_endpoints)[:k]
    return 1.0 if any(e in expected for e in topk) else 0.0


def hit_at_k_scored(results, expected, k) -> float:
    """1.0 if an acceptable endpoint is *guaranteed* within the top-k under any
    resolution of exact score ties (see ``guaranteed_rank``)."""
    r = guaranteed_rank(results, expected)
    return 1.0 if (r is not None and r <= k) else 0.0


def reciprocal_rank_scored(results, expected) -> float:
    """1 / guaranteed rank of the first acceptable endpoint, else 0. Ties are
    resolved pessimistically (bottom of any non-acceptable tie block), matching
    ``hit_at_k_scored`` so both metrics use one tie convention."""
    r = guaranteed_rank(results, expected)
    return (1.0 / r) if r else 0.0


def recall_at_k(ranked_endpoints, expected, k) -> float:
    """Fraction of acceptable endpoints retrieved in top-k.

    For unambiguous queries (|expected| == 1) this equals hit@k. For ambiguous
    queries it measures how much of the acceptable set is surfaced.
    """
    if not expected:
        return float("nan")
    topk = set(_dedup_keep_order(ranked_endpoints)[:k])
    return len(topk & set(expected)) / len(set(expected))


def precision_at_k(ranked_endpoints, expected, k) -> float:
    """Fraction of the top-k that are acceptable."""
    topk = _dedup_keep_order(ranked_endpoints)[:k]
    if not topk:
        return 0.0
    return sum(1 for e in topk if e in expected) / len(topk)


def reciprocal_rank(ranked_endpoints, expected) -> float:
    """1/rank of the first acceptable endpoint (de-duplicated), else 0."""
    for i, e in enumerate(_dedup_keep_order(ranked_endpoints), start=1):
        if e in expected:
            return 1.0 / i
    return 0.0


# ---------------------------------------------------------------------------
# 3. No-answer handling
# ---------------------------------------------------------------------------

def no_answer_scores(top_scores, is_answerable, threshold=None):
    """Evaluate the system's ability to abstain on no-answer queries.

    Args:
      top_scores : per-query top-1 retrieval score (higher = more confident
                   an answer exists).
      is_answerable : per-query bool (True if the query has a real target).
      threshold : optional fixed decision threshold; if given, report accuracy
                  of accept(>=thr)/abstain(<thr).

    Returns a dict with AUROC (answerable vs no-answer separability), AUPRC for
    both the answerable and the (rarer, more informative) no-answer class, the
    best achievable F1 over thresholds and its threshold, and -- if a fixed
    threshold is provided -- accept/abstain accuracy at that threshold.

    AUPRC for the no-answer class (positive = no-answer, scored by low
    confidence) is the imbalance-aware metric for "can the system abstain?":
    no-answer queries are the minority, so AUROC can be optimistic while
    average precision on that rare class is more honest.

    Degrades gracefully if all queries share one class (metrics undefined).
    """
    from sklearn.metrics import (roc_auc_score, precision_recall_curve,
                                 average_precision_score)

    y = np.asarray(is_answerable, dtype=int)
    s = np.asarray(top_scores, dtype=float)
    out = {"n": int(len(y)), "n_answerable": int(y.sum()),
           "n_no_answer": int((1 - y).sum())}

    if out["n_answerable"] == 0 or out["n_no_answer"] == 0:
        out["auroc"] = float("nan")
        out["auprc_answerable"] = float("nan")
        out["auprc_no_answer"] = float("nan")
        out["best_f1"] = float("nan")
        out["best_f1_threshold"] = float("nan")
    else:
        out["auroc"] = float(roc_auc_score(y, s))
        # positive = answerable, ranked by score
        out["auprc_answerable"] = float(average_precision_score(y, s))
        # positive = no-answer (rare), ranked by LOW score (-s)
        out["auprc_no_answer"] = float(average_precision_score(1 - y, -s))
        prec, rec, thr = precision_recall_curve(y, s)
        f1 = np.divide(2 * prec * rec, prec + rec,
                       out=np.zeros_like(prec), where=(prec + rec) > 0)
        best = int(np.argmax(f1))
        out["best_f1"] = float(f1[best])
        # precision_recall_curve returns thresholds of length len(prec)-1
        out["best_f1_threshold"] = float(thr[best]) if best < len(thr) else float("nan")

    if threshold is not None:
        pred = (s >= threshold).astype(int)
        out["threshold"] = float(threshold)
        out["accept_abstain_accuracy"] = float((pred == y).mean())
        # answerable wrongly abstained / no-answer wrongly accepted
        ans = y == 1
        noa = y == 0
        out["false_abstain_rate"] = float((pred[ans] == 0).mean()) if ans.any() else float("nan")
        out["false_accept_rate"] = float((pred[noa] == 1).mean()) if noa.any() else float("nan")
    return out


# ---------------------------------------------------------------------------
# 4. Timing / memory
# ---------------------------------------------------------------------------

class Timer:
    """Context manager measuring wall-clock seconds: `with Timer() as t: ...; t.s`."""

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.s = time.perf_counter() - self._t0
        return False


def embedding_memory_mb(array) -> float:
    """MiB used by a numpy embedding matrix."""
    return float(np.asarray(array).nbytes) / (1024 * 1024)


def process_memory_mb():
    """Resident set size in MiB, or NaN if psutil is unavailable."""
    try:
        import os
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# 5. Retriever interface + evaluation driver
# ---------------------------------------------------------------------------

def evaluate(retriever, queries, ks=(1, 3, 5, 10),
             no_answer_threshold=None, retrieve_k=None):
    """Score a retriever over the query benchmark.

    Args:
      retriever : object with .search(query, top_k) -> list[(endpoint, score)].
      queries   : list of query dicts (see queries.py).
      ks        : k values for Hit@k / Recall@k / Precision@k.
      no_answer_threshold : optional fixed accept/abstain threshold.
      retrieve_k : raw top_k requested from the retriever. Defaults to
                   DEFAULT_RETRIEVE_K so endpoint de-duplication cannot
                   silently truncate Hit@k.

    Hit@k and MRR are tie-aware (see ``guaranteed_rank``): identical-text records
    tie exactly, and a method is credited only for a rank it is guaranteed under
    any resolution of those ties, so the reported numbers do not depend on the
    trace's row order.

    Returns dict with:
      per_query : pandas DataFrame (one row per query, metrics + latency + score)
      overall   : aggregate metrics over answerable queries
      by_category : aggregate metrics per category
      no_answer : output of no_answer_scores over all queries
      latency   : {mean, p50, p95} seconds
    """
    retrieve_k = max(retrieve_k or DEFAULT_RETRIEVE_K, max(ks))
    rows = []

    for q in queries:
        expected = set(q["expected"])
        answerable = len(expected) > 0
        with Timer() as t:
            results = retriever.search(q["query"], retrieve_k)
        ranked_eps = [ep for ep, _ in results]
        top_score = results[0][1] if results else float("-inf")

        row = {
            "id": q["id"],
            "category": q["category"],
            "ambiguous": q.get("ambiguous", False),
            "provenance": q.get("provenance", ""),
            "answerable": answerable,
            "latency_s": t.s,
            "top_score": top_score,
            "top1_endpoint": ranked_eps[0] if ranked_eps else None,
        }
        if answerable:
            for k in ks:
                row[f"hit@{k}"] = hit_at_k_scored(results, expected, k)
                row[f"recall@{k}"] = recall_at_k(ranked_eps, expected, k)
                row[f"precision@{k}"] = precision_at_k(ranked_eps, expected, k)
            row["mrr"] = reciprocal_rank_scored(results, expected)
        rows.append(row)

    per_query = pd.DataFrame(rows)

    metric_cols = ([f"hit@{k}" for k in ks] + [f"recall@{k}" for k in ks]
                   + [f"precision@{k}" for k in ks] + ["mrr"])
    ans = per_query[per_query["answerable"]]

    def _agg(frame):
        return {m: float(frame[m].mean()) for m in metric_cols if m in frame}

    overall = _agg(ans)
    overall["n"] = int(len(ans))
    by_category = {cat: {**_agg(g), "n": int(len(g))}
                   for cat, g in ans.groupby("category")}

    na = no_answer_scores(per_query["top_score"].tolist(),
                          per_query["answerable"].tolist(),
                          threshold=no_answer_threshold)

    lat = per_query["latency_s"].to_numpy()
    latency = {
        "mean": float(lat.mean()),
        "p50": float(np.percentile(lat, 50)),
        "p95": float(np.percentile(lat, 95)),
    }

    return {
        "per_query": per_query,
        "overall": overall,
        "by_category": by_category,
        "no_answer": na,
        "latency": latency,
    }


# ---------------------------------------------------------------------------
# Smoke test: a trivial token-overlap retriever, just to exercise the pipeline
# end-to-end (no models needed). Run:  python eval_lib.py [path/to/csv]
# ---------------------------------------------------------------------------

class _TokenOverlapRetriever:
    """Tiny lexical retriever for self-testing only (not a paper baseline)."""

    def __init__(self, records):
        self.endpoints = records["accessedNodeAddress"].tolist()
        self.tokens = [set(t.split()) for t in records["record_text"]]

    def search(self, query, top_k):
        q = set(preprocess_text(query).split())
        scored = []
        for ep, toks in zip(self.endpoints, self.tokens):
            inter = len(q & toks)
            denom = (len(q) + len(toks)) or 1
            scored.append((ep, inter / denom))
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:top_k]


if __name__ == "__main__":
    import sys
    from queries import QUERIES, validate, category_counts

    csv = sys.argv[1] if len(sys.argv) > 1 else "mainSimulationAccessTraces.csv"
    validate()
    print("query categories:", category_counts())

    records = build_record_text(load_records(csv), fmt="sentence")
    print(f"indexed records: {len(records)}  | example: {records['record_text'].iloc[0]!r}")

    retr = _TokenOverlapRetriever(records)
    res = evaluate(retr, QUERIES, ks=(1, 3, 5))

    o = res["overall"]
    print(f"\n[token-overlap smoke test]  n_answerable={o['n']}")
    print(f"  Hit@1={o['hit@1']:.3f}  Hit@3={o['hit@3']:.3f}  Hit@5={o['hit@5']:.3f}  MRR={o['mrr']:.3f}")
    print(f"  no-answer AUROC={res['no_answer']['auroc']:.3f}  best-F1={res['no_answer']['best_f1']:.3f}")
    print(f"  latency mean={res['latency']['mean']*1000:.2f}ms p95={res['latency']['p95']*1000:.2f}ms")
    print("\nby category (Hit@3):")
    for cat, m in res["by_category"].items():
        print(f"  {cat:20s} n={m['n']:2d}  Hit@3={m['hit@3']:.3f}")
