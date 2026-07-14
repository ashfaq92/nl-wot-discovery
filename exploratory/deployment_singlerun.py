"""
Deployment / resource-footprint study (Phase 4).

Two questions:
  4.1 Footprint: model size, index memory, index-build time, isolated query
      latency (p50/p95), and accuracy (Hit@1/Hit@4) for a small vs a base
      encoder -- the edge-suitability evidence.
  4.2 Maintainability: the per-device cost of keeping the index current
      (embed + append/replace/remove one record) versus retraining the
      supervised classifier when a device (label) is added.

Latency is measured in a focused loop (warmup first) to avoid the noisy
numbers seen when sweeping many models under background load.

Run:  python run_deployment.py [csv]
"""

import os
import sys

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd

from eval_lib import load_records, build_record_text, evaluate, Timer
from queries import QUERIES
from retrievers import BiEncoderRetriever

CSV = sys.argv[1] if len(sys.argv) > 1 else "mainSimulationAccessTraces.csv"
MODELS = ["sentence-transformers/all-MiniLM-L6-v2",
          "sentence-transformers/all-mpnet-base-v2"]
LATENCY_REPEATS = 3      # x73 queries
INDEX_OP_TRIALS = 30     # per index op
CLASSIFIER_RETRAIN_S = 435.0  # measured in classifier_baseline.py (full retrain)

records = build_record_text(load_records(CSV), fmt="sentence")
rows = []


def n_params(model):
    return sum(p.numel() for p in model.parameters())


for name in MODELS:
    r = BiEncoderRetriever(records, name, use_reranker=False, device="cpu")
    short = r.name

    # --- accuracy (no rerank) ---
    res = evaluate(r, QUERIES, ks=(1, 4))
    hit1, hit4 = res["overall"]["hit@1"], res["overall"]["hit@4"]

    # --- isolated query latency (warmup then timed loop) ---
    for q in QUERIES[:5]:
        r.search(q["query"], 10)
    lat = []
    for _ in range(LATENCY_REPEATS):
        for q in QUERIES:
            with Timer() as t:
                r.search(q["query"], 10)
            lat.append(t.s)
    lat = np.array(lat) * 1000.0  # ms

    # --- 4.2 index maintenance: per-device add / update / delete ---
    emb = r.emb
    sample_text = records["record_text"].iloc[0]
    add_t, upd_t, del_t = [], [], []
    for _ in range(INDEX_OP_TRIALS):
        with Timer() as t:                       # add: embed + append
            e = r.bi.encode([sample_text], normalize_embeddings=True,
                            show_progress_bar=False)
            _ = np.vstack([emb, e])
        add_t.append(t.s)
        with Timer() as t:                       # update: embed + replace row
            e = r.bi.encode([sample_text], normalize_embeddings=True,
                            show_progress_bar=False)
            tmp = emb.copy(); tmp[0] = e[0]
        upd_t.append(t.s)
        with Timer() as t:                       # delete: drop a row
            _ = np.delete(emb, 0, axis=0)
        del_t.append(t.s)

    rows.append({
        "model": short,
        "params_M": round(n_params(r.bi) / 1e6, 1),
        "index_mem_MiB": round(r.embedding_mb, 3),
        "load_s": round(r.load_time_s, 2),
        "index_build_s": round(r.index_build_time_s, 3),
        "lat_p50_ms": round(float(np.percentile(lat, 50)), 2),
        "lat_p95_ms": round(float(np.percentile(lat, 95)), 2),
        "hit@1": round(hit1, 3),
        "hit@4": round(hit4, 3),
        "idx_add_ms": round(float(np.median(add_t)) * 1000, 2),
        "idx_update_ms": round(float(np.median(upd_t)) * 1000, 2),
        "idx_delete_ms": round(float(np.median(del_t)) * 1000, 3),
    })
    print(f"done {short}")

df = pd.DataFrame(rows)
df.to_csv("results_deployment.csv", index=False)
print("\n" + df.to_string(index=False))
print(f"\nMaintainability contrast: adding one device updates the index in"
      f" ~{df['idx_add_ms'].min():.0f}-{df['idx_add_ms'].max():.0f} ms, versus"
      f" ~{CLASSIFIER_RETRAIN_S:.0f} s to retrain the supervised classifier"
      f" (new label) -- about {CLASSIFIER_RETRAIN_S*1000/df['idx_add_ms'].max():.0f}x.")
print("saved results_deployment.csv")
