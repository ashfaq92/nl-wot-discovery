"""Sensitivity of the federated delegation threshold tau.

Sweeps tau over {0.30, 0.40, 0.50, 0.60, 0.70} for the two delegating
strategies (broadcast ignores tau) under both entry modes, reporting
accuracy (Hit@k, MRR) and federation cost (nodes visited, hops).
Backs the manuscript claim that tau = 0.40 is not a tuned choice.
The upper values bracket the 0.60 softmax-confidence threshold prior work
uses for federated delegation (different signal, similar operating point).

Run (from code/):  python run_federated_tau_sweep.py [csv]
"""

import os
import sys

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pandas as pd

from eval_lib import load_records, build_record_text
from federated import FederatedIndex
from run_federated import run

TAUS = [0.30, 0.40, 0.50, 0.60, 0.70]
CSV = sys.argv[1] if len(sys.argv) > 1 else "mainSimulationAccessTraces.csv"

if __name__ == "__main__":
    records = build_record_text(load_records(CSV), fmt="sentence")
    fed = FederatedIndex(records, device="cpu")

    rows = []
    for tau in TAUS:
        for strategy in ("first_served", "local_first"):
            for entry in ("gateway", "local"):
                agg = run(fed, strategy, entry, threshold=tau)
                agg["tau"] = tau
                rows.append(agg)

    cols = ["tau", "strategy", "entry", "hit@1", "hit@3", "hit@5", "mrr",
            "avg_nodes_visited", "avg_hops"]
    df = pd.DataFrame(rows)[cols]
    df.to_csv("results_federated_tau_sweep.csv", index=False)
    print(df.to_string(index=False))
    print("\nsaved results_federated_tau_sweep.csv")
