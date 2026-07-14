"""
Per-class (per-endpoint) in-distribution comparison: supervised classifier vs.
frozen retriever, on the SAME held-out validation rows.

Both methods are scored on the classifier's random-split validation set (seed 42):
each validation row is a generated sentence ("I need to {op} {service} in
{location}") whose true label is one of the 170 endpoint classes. The retriever's
record serialization is built identically, so feeding it the validation sentences
is a fair head-to-head.

For every endpoint class present in the validation set we report top-1 recall:
    recall_c = (# val rows of class c whose top-1 prediction == c) / (# val rows of c)

- Classifier: argmax of the 170-way softmax (from classifier_baseline).
- Retriever : top-1 endpoint from BiEncoderRetriever(all-MiniLM-L6-v2) over the
              286 deduplicated records.

A class is flagged `shared_sentence` when at least one of its generated sentences
also maps to a different endpoint -- the structural collision where a single-label
decision (softmax OR top-1 retrieval) cannot separate the devices.

Run (from anywhere):  python exploratory/perclass_indist_compare.py [csv]
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from classifier_baseline import build_training_data, train_classifier  # noqa: E402
from eval_lib import load_records, build_record_text  # noqa: E402
from retrievers import BiEncoderRetriever  # noqa: E402

SEED = 42
TEST_SIZE = 0.25
BI_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    csv = sys.argv[1] if len(sys.argv) > 1 else str(CODE_DIR / "mainSimulationAccessTraces.csv")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    # --- in-distribution data + the exact seed-42 random validation split ------
    texts, labels = build_training_data(csv, preprocess=True)
    labels = np.array(labels)
    n = len(texts)
    classes = sorted(set(labels))
    cls2idx = {c: i for i, c in enumerate(classes)}

    idx_tr, idx_va = train_test_split(np.arange(n), test_size=TEST_SIZE, random_state=SEED)
    val_sent = [texts[i] for i in idx_va]
    val_true_ep = labels[idx_va]

    # sentence -> set of endpoints (collision detection over the FULL data)
    ep_per_sent = defaultdict(set)
    for t, l in zip(texts, labels):
        ep_per_sent[t].add(l)

    # --- classifier: train on the same split, get val predictions --------------
    print("training classifier (random split, seed 42) ...")
    _, _, classes_c, info = train_classifier(
        csv, device=device, split="random", seed=SEED, verbose=False)
    assert classes_c == classes
    # train_test_split(X, y, random_state) partitions identically to the arange
    # split above, so info['val_pred'] aligns row-for-row with idx_va / val_sent.
    assert np.array_equal(info["val_true"], np.array([cls2idx[e] for e in val_true_ep])), \
        "val split mismatch -- ordering assumption broken"
    clf_pred_ep = np.array([classes[i] for i in info["val_pred"]])
    print(f"  classifier val accuracy = {(clf_pred_ep == val_true_ep).mean():.4f}")

    # --- retriever: top-1 endpoint for each unique validation sentence ---------
    print("building retriever index ...")
    recs = build_record_text(load_records(csv), fmt="sentence")
    retr = BiEncoderRetriever(recs, BI_ENCODER, device=device)
    uniq = sorted(set(val_sent))
    print(f"  scoring {len(uniq)} unique validation sentences ...")
    sent2pred = {s: retr.search(s, top_k=1)[0][0] for s in uniq}
    retr_pred_ep = np.array([sent2pred[s] for s in val_sent])
    print(f"  retriever val accuracy  = {(retr_pred_ep == val_true_ep).mean():.4f}")

    # --- per-class top-1 recall -----------------------------------------------
    rows = []
    for c in classes:
        mask = val_true_ep == c
        n_c = int(mask.sum())
        if n_c == 0:
            continue                       # class absent from this val split
        clf_rec = float((clf_pred_ep[mask] == c).mean())
        retr_rec = float((retr_pred_ep[mask] == c).mean())
        shared = any(len(ep_per_sent[s]) > 1 for s in np.array(val_sent)[mask])
        # a readable example sentence for this class
        example = np.array(val_sent)[mask][0]
        rows.append({"endpoint": c, "n_val_rows": n_c,
                     "shared_sentence": shared,
                     "clf_recall": round(clf_rec, 4),
                     "retr_recall": round(retr_rec, 4),
                     "delta_retr_minus_clf": round(retr_rec - clf_rec, 4),
                     "example": example})

    df = pd.DataFrame(rows).sort_values(
        ["delta_retr_minus_clf", "n_val_rows"], ascending=[True, False]).reset_index(drop=True)
    out_path = SCRIPT_DIR / "results_perclass_indist.csv"
    df.to_csv(out_path, index=False)

    # --- summary ---------------------------------------------------------------
    n_cls = len(df)
    clf_macro, retr_macro = df["clf_recall"].mean(), df["retr_recall"].mean()
    clf_micro = float((clf_pred_ep == val_true_ep).mean())
    retr_micro = float((retr_pred_ep == val_true_ep).mean())
    retr_wins = int((df["delta_retr_minus_clf"] > 0).sum())
    clf_wins = int((df["delta_retr_minus_clf"] < 0).sum())
    ties = int((df["delta_retr_minus_clf"] == 0).sum())
    shared = df[df["shared_sentence"]]
    unique = df[~df["shared_sentence"]]

    print("\n" + "=" * 72)
    print(f"Per-class in-distribution comparison over {n_cls} endpoint classes "
          f"(val split, seed {SEED})")
    print("=" * 72)
    print(f"  micro (overall val accuracy):  classifier {clf_micro:.4f}   retriever {retr_micro:.4f}")
    print(f"  macro (mean per-class recall): classifier {clf_macro:.4f}   retriever {retr_macro:.4f}")
    print(f"  classes where retriever > classifier : {retr_wins}")
    print(f"  classes where classifier > retriever : {clf_wins}")
    print(f"  classes tied                         : {ties}")
    print(f"\n  unique-sentence classes (n={len(unique)}): "
          f"clf macro {unique['clf_recall'].mean():.4f}  retr macro {unique['retr_recall'].mean():.4f}")
    print(f"  shared-sentence classes (n={len(shared)}): "
          f"clf macro {shared['clf_recall'].mean():.4f}  retr macro {shared['retr_recall'].mean():.4f}")

    print("\n  10 classes where the retriever most outperforms the classifier:")
    better = df[df["delta_retr_minus_clf"] > 0].sort_values(
        "delta_retr_minus_clf", ascending=False)
    for _, r in better.head(10).iterrows():
        print(f"    {r['endpoint']:34s} n={r['n_val_rows']:5d} shared={int(r['shared_sentence'])} "
              f"clf={r['clf_recall']:.2f} retr={r['retr_recall']:.2f}  | {r['example']}")

    worse = df[df["delta_retr_minus_clf"] < 0]
    if len(worse):
        print(f"\n  classes where the classifier beats the retriever (n={len(worse)}):")
        for _, r in worse.iterrows():
            print(f"    {r['endpoint']:34s} n={r['n_val_rows']:5d} shared={int(r['shared_sentence'])} "
                  f"clf={r['clf_recall']:.2f} retr={r['retr_recall']:.2f}  | {r['example']}")

    print(f"\nsaved {out_path.name} in {SCRIPT_DIR}")


if __name__ == "__main__":
    main()
