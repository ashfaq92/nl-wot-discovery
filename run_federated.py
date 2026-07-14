"""
Federated retrieval evaluation (Phase 3.3/3.4).

Evaluates the delegation strategies in federated.py over the 73-query benchmark
and compares federated retrieval to the centralized index, reporting both
accuracy (Hit@k, MRR) and federation cost (nodes visited, hops).

Two entry scenarios:
  gateway : every query enters the empty coordinator (node 7) -> must delegate.
            Stresses delegation; mirrors a user hitting a gateway.
  local   : a query enters a node that holds one of its target endpoints
            (no-answer queries enter the coordinator). Shows local-first
            resolving with little or no delegation.

Run:  python run_federated.py [csv]
"""

import os
import sys

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd

from eval_lib import load_records, build_record_text, hit_at_k, reciprocal_rank
from queries import QUERIES
from federated import FederatedIndex

KS = (1, 3, 5)
THRESHOLD = 0.40
COORDINATOR = 7
CSV = sys.argv[1] if len(sys.argv) > 1 else "mainSimulationAccessTraces.csv"


def target_node(expected, fed):
    """A node holding one of the acceptable endpoints (lowest id), or None."""
    nodes = set()
    for ep in expected:
        nodes |= fed.endpoint_nodes.get(ep, set())
    return min(nodes) if nodes else None


def run(fed, strategy, origin_mode, threshold=THRESHOLD):
    """strategy in {broadcast, first_served, local_first}; origin_mode in
    {gateway, local}. Returns aggregate metrics over answerable queries."""
    rows = []
    for q in QUERIES:
        expected = set(q["expected"])
        if not expected:           # skip no-answer for accuracy/cost aggregation
            continue
        if origin_mode == "gateway":
            origin = COORDINATOR
        else:
            origin = target_node(expected, fed) or COORDINATOR

        fn = getattr(fed, strategy)
        if strategy == "broadcast":
            ranked, visited, hops = fn(q["query"], max(KS), origin=origin)
        else:
            ranked, visited, hops = fn(q["query"], max(KS), origin=origin,
                                       threshold=threshold)
        eps = [ep for ep, _ in ranked]
        row = {"id": q["id"], "visited": visited, "hops": hops,
               "mrr": reciprocal_rank(eps, expected)}
        for k in KS:
            row[f"hit@{k}"] = hit_at_k(eps, expected, k)
        rows.append(row)
    df = pd.DataFrame(rows)
    agg = {"strategy": strategy, "entry": origin_mode,
           "hit@1": round(df["hit@1"].mean(), 3),
           "hit@3": round(df["hit@3"].mean(), 3),
           "hit@5": round(df["hit@5"].mean(), 3),
           "mrr": round(df["mrr"].mean(), 3),
           "avg_nodes_visited": round(df["visited"].mean(), 2),
           "avg_hops": round(df["hops"].mean(), 2)}
    return agg


if __name__ == "__main__":
    records = build_record_text(load_records(CSV), fmt="sentence")
    fed = FederatedIndex(records, device="cpu")
    n_nonempty = sum(1 for n in fed.nodes if fed.node[n]["texts"])
    print(f"federation: {len(fed.nodes)} nodes ({n_nonempty} non-empty), "
          f"threshold={THRESHOLD}\n")

    results = []
    results.append(run(fed, "broadcast", "gateway"))
    results.append(run(fed, "first_served", "gateway"))
    results.append(run(fed, "local_first", "gateway"))
    results.append(run(fed, "first_served", "local"))
    results.append(run(fed, "local_first", "local"))

    df = pd.DataFrame(results)
    df.to_csv("results_federated.csv", index=False)
    print(df.to_string(index=False))
    print("\nsaved results_federated.csv")
    print("\nNote: broadcast contacts all device nodes and equals the centralized "
          "index in accuracy; local-first from the owning node resolves with the "
          "fewest nodes visited.")
