"""
Deployment latency / footprint with mean +/- STD, measured on CPU.

Per the paper's environment decision, ALL timing is reported on the commodity CPU
(no GPU at inference). Repeats the isolated measurement K times to attach
mean +/- STD to index-build time, p50/p95 query latency, and per-device index
add/delete cost, for the small (MiniLM-L6) and base (mpnet) encoders. These are
the retriever rows of the paper's deployment table; the classifier row comes
from run_classifier_deployment.py.

Run on a quiet machine (no other jobs), from code/:  python run_deployment.py [csv]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR
sys.path.insert(0, str(CODE_DIR))

from eval_lib import load_records, build_record_text, evaluate, Timer  # noqa: E402
from queries import QUERIES  # noqa: E402
from retrievers import BiEncoderRetriever  # noqa: E402

MODELS = ["sentence-transformers/all-MiniLM-L6-v2",
          "sentence-transformers/all-mpnet-base-v2"]
K_RUNS = 5            # repeated builds+latency measurements -> mean+/-STD
LAT_REPEATS = 3       # x73 queries per run
DEVICE = "cpu"        # reported timing environment


def msd(a):
    a = np.asarray(a, dtype=float)
    return round(float(a.mean()), 2), (round(float(a.std(ddof=1)), 2) if len(a) > 1 else 0.0)


def main():
    csv = sys.argv[1] if len(sys.argv) > 1 else str(CODE_DIR / "mainSimulationAccessTraces.csv")
    records = build_record_text(load_records(csv), fmt="sentence")
    rows = []
    for name in MODELS:
        p50s, p95s, builds, adds, dels = [], [], [], [], []
        params_m = index_mb = hit1 = hit4 = None
        for run in range(K_RUNS):
            r = BiEncoderRetriever(records, name, use_reranker=False, device=DEVICE)
            builds.append(r.index_build_time_s * 1000.0)   # ms
            if params_m is None:
                params_m = round(sum(p.numel() for p in r.bi.parameters()) / 1e6, 1)
                index_mb = round(r.embedding_mb, 3)
                res = evaluate(r, QUERIES, ks=(1, 4))
                hit1, hit4 = round(res["overall"]["hit@1"], 3), round(res["overall"]["hit@4"], 3)
            for q in QUERIES[:5]:                           # warmup
                r.search(q["query"], 10)
            lat = []
            for _ in range(LAT_REPEATS):
                for q in QUERIES:
                    with Timer() as t:
                        r.search(q["query"], 10)
                    lat.append(t.s)
            lat = np.array(lat) * 1000.0
            p50s.append(float(np.percentile(lat, 50)))
            p95s.append(float(np.percentile(lat, 95)))
            # per-device index ops: add (embed one record + append), delete (drop a row)
            emb, sample = r.emb, records["record_text"].iloc[0]
            at, dt = [], []
            for _ in range(30):
                with Timer() as t:
                    e = r.bi.encode([sample], normalize_embeddings=True, show_progress_bar=False)
                    _ = np.vstack([emb, e])
                at.append(t.s)
                with Timer() as t:
                    _ = np.delete(emb, 0, axis=0)
                dt.append(t.s)
            adds.append(float(np.median(at)) * 1000.0)
            dels.append(float(np.median(dt)) * 1000.0)
        p50_m, p50_s = msd(p50s); p95_m, p95_s = msd(p95s); b_m, b_s = msd(builds)
        add_m, add_s = msd(adds); del_m, del_s = msd(dels)
        rows.append({"model": name.split("/")[-1], "params_M": params_m,
                     "index_mem_MiB": index_mb, "hit@1": hit1, "hit@4": hit4,
                     "build_ms_mean": b_m, "build_ms_std": b_s,
                     "p50_ms_mean": p50_m, "p50_ms_std": p50_s,
                     "p95_ms_mean": p95_m, "p95_ms_std": p95_s,
                     "add_ms_mean": round(add_m, 2), "add_ms_std": round(add_s, 2),
                     "del_ms_mean": round(del_m, 3), "del_ms_std": round(del_s, 3)})
        print(f"done {rows[-1]['model']}: p50={p50_m}+/-{p50_s}ms  p95={p95_m}+/-{p95_s}ms  "
              f"build={b_m}+/-{b_s}ms  add={add_m:.2f}+/-{add_s:.2f}ms  del={del_m:.3f}ms")

    df = pd.DataFrame(rows)
    df.to_csv(SCRIPT_DIR / "results_deployment.csv", index=False)
    print("\n" + df.to_string(index=False))
    print(f"\n(device={DEVICE}, {K_RUNS} runs x {LAT_REPEATS}x73 queries)")
    print(f"saved results_deployment.csv in {SCRIPT_DIR}")


if __name__ == "__main__":
    main()
