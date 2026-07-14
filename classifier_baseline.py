"""
Supervised endpoint-classifier baseline (faithful PyTorch port of the
Llopis et al. 2023 architecture) for the retrieval-vs-classification head-to-head.

This is NOT an attempt to reproduce Table 6/7 of the original paper exactly
(those illustrative confidences proved fragile to reproduce even for the
original authors). It is a faithful re-implementation of their *architecture*
and *training setup* whose aggregate accuracy matches the reported order
(~0.79 top-1 on a random split). We additionally cite the authors' published
numbers in the paper.

Architecture (from Llopis et al. 2023 and the author's own model definition):
    Input -> TokenAndPositionEmbedding(embed=128)
          -> TransformerBlock(heads=2, ff=128, dropout=0.1)
          -> GlobalAveragePooling1D -> Dropout(0.1)
          -> Dense(512, relu) -> Dropout(0.5)
          -> Dense(256, relu) -> Dropout(0.5)
          -> Dense(num_classes, softmax)
    Adam, sparse categorical cross-entropy, 10 epochs, batch 5000.

The trained model is wrapped as `ClassifierRetriever` exposing the standard
.search(query, top_k) -> [(endpoint, prob), ...] interface, so it is scored on
the SAME 73-query benchmark as every retriever, via eval_lib.evaluate.

Run:  python classifier_baseline.py [csv]
"""

from __future__ import annotations

import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from eval_lib import preprocess_text, Timer

SEED = 42
RECORD_COLS = ["operation", "destinationServiceType",
               "destinationLocation", "accessedNodeAddress"]


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data: full trace -> generated sentences + endpoint labels
# ---------------------------------------------------------------------------

def build_training_data(csv_path, preprocess=True):
    """Generated sentence + endpoint label for every row of the FULL dataset
    (faithful to prior work: training uses all rows, not the deduplicated set).
    """
    df = pd.read_csv(csv_path)[RECORD_COLS].dropna()
    sent = ("I need to " + df["operation"].astype(str) + " "
            + df["destinationServiceType"].astype(str).str.lstrip("/") + " in "
            + df["destinationLocation"].astype(str))
    if preprocess:
        sent = sent.map(preprocess_text)
    return sent.tolist(), df["accessedNodeAddress"].astype(str).tolist()


class WordTokenizer:
    """Deterministic word->index tokenizer (0=pad, 1=oov). Avoids the
    nondeterminism of hash encoding that complicated prior reproduction."""

    def __init__(self):
        self.w2i = {"<pad>": 0, "<oov>": 1}

    def fit(self, texts):
        for t in texts:
            for w in t.split():
                if w not in self.w2i:
                    self.w2i[w] = len(self.w2i)
        return self

    @property
    def vocab_size(self):
        return len(self.w2i)

    def encode(self, text, maxlen):
        ids = [self.w2i.get(w, 1) for w in text.split()][:maxlen]
        return ids + [0] * (maxlen - len(ids))


# ---------------------------------------------------------------------------
# Model (faithful port)
# ---------------------------------------------------------------------------

class LlopisClassifier(nn.Module):
    def __init__(self, vocab_size, maxlen, num_classes,
                 embed_dim=128, num_heads=2, ff_dim=128):
        super().__init__()
        self.maxlen = maxlen
        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(maxlen, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=0.1, batch_first=True)
        self.drop1 = nn.Dropout(0.1)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, ff_dim), nn.ReLU(),
                                 nn.Linear(ff_dim, embed_dim))
        self.drop2 = nn.Dropout(0.1)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.pool_drop = nn.Dropout(0.1)
        self.fc1 = nn.Linear(embed_dim, 512)
        self.d1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.d2 = nn.Dropout(0.5)
        self.out = nn.Linear(256, num_classes)

    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)
        a, _ = self.attn(h, h, h)
        h = self.ln1(h + self.drop1(a))
        f = self.ffn(h)
        h = self.ln2(h + self.drop2(f))
        h = h.mean(dim=1)                 # GlobalAveragePooling1D
        h = self.pool_drop(h)
        h = torch.relu(self.fc1(h)); h = self.d1(h)
        h = torch.relu(self.fc2(h)); h = self.d2(h)
        return self.out(h)                # logits


def train_classifier(csv_path, preprocess=True, epochs=10, batch_size=5000,
                     test_size=0.25, maxlen=None, device="cpu", verbose=True,
                     split="random", seed=SEED, split_indices=None):
    """Train on the full data and report validation top-1 accuracy.

    split:
      "random"         : IID row split (the prior-work setup). Duplicated rows
                         of a sentence can land in both train and test -> the
                         leaky split that inflates accuracy.
      "group_sentence" : leakage-free split where no generated sentence appears
                         in both train and test (GroupShuffleSplit by sentence).

    Returns (model, tokenizer, classes, info).
    """
    from sklearn.model_selection import train_test_split, GroupShuffleSplit

    set_seed(seed)
    texts, labels = build_training_data(csv_path, preprocess=preprocess)

    classes = sorted(set(labels))
    cls2idx = {c: i for i, c in enumerate(classes)}
    y = np.array([cls2idx[l] for l in labels], dtype=np.int64)

    tok = WordTokenizer().fit(texts)
    if maxlen is None:
        maxlen = max(len(t.split()) for t in texts)
    X = np.array([tok.encode(t, maxlen) for t in texts], dtype=np.int64)

    if split_indices is not None:
        tr_idx, va_idx = split_indices
        Xtr, Xva, ytr, yva = X[tr_idx], X[va_idx], y[tr_idx], y[va_idx]
    elif split == "group_sentence":
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        tr_idx, va_idx = next(gss.split(X, y, groups=texts))
        Xtr, Xva, ytr, yva = X[tr_idx], X[va_idx], y[tr_idx], y[va_idx]
    else:
        Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=test_size,
                                              random_state=seed)
    Xtr_t = torch.tensor(Xtr); ytr_t = torch.tensor(ytr)

    model = LlopisClassifier(tok.vocab_size, maxlen, len(classes)).to(device)
    opt = torch.optim.Adam(model.parameters())
    lossf = nn.CrossEntropyLoss()

    with Timer() as t:
        model.train()
        n = len(Xtr_t)
        for ep in range(epochs):
            perm = torch.randperm(n)
            tot = 0.0
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                xb = Xtr_t[idx].to(device); yb = ytr_t[idx].to(device)
                opt.zero_grad()
                logits = model(xb)
                loss = lossf(logits, yb)
                loss.backward(); opt.step()
                tot += loss.item() * len(idx)
            if verbose:
                print(f"  epoch {ep + 1}/{epochs}  loss={tot / n:.4f}")
    train_time = t.s

    # validation top-1
    model.eval()
    with torch.no_grad():
        va_logits = model(torch.tensor(Xva).to(device))
        va_pred = va_logits.argmax(1).cpu().numpy()
    val_acc = float((va_pred == yva).mean())

    info = {"val_top1": val_acc, "train_time_s": train_time, "maxlen": maxlen,
            "vocab_size": tok.vocab_size, "num_classes": len(classes),
            "n_rows": len(texts), "epochs": epochs, "seed": seed, "split": split,
            "val_true": yva, "val_pred": va_pred, "n_val": int(len(yva))}
    if verbose:
        print(f"  val top-1 = {val_acc:.3f}  (train {train_time:.1f}s, "
              f"{len(classes)} classes, vocab {tok.vocab_size})")
    return model, tok, classes, info


class ClassifierRetriever:
    """Wraps the trained classifier behind the standard retriever interface.
    search() returns the top-k endpoint labels by softmax probability."""

    def __init__(self, model, tokenizer, classes, maxlen,
                 preprocess=True, device="cpu"):
        self.name = "llopis-classifier"
        self.model = model.eval()
        self.tok = tokenizer
        self.classes = classes
        self.maxlen = maxlen
        self.preprocess = preprocess
        self.device = device

    def search(self, query, top_k):
        q = preprocess_text(query) if self.preprocess else query
        ids = torch.tensor([self.tok.encode(q, self.maxlen)], device=self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(ids)[0], dim=-1)
        k = min(top_k, len(self.classes))
        top = torch.topk(probs, k)
        return [(self.classes[i], float(p))
                for p, i in zip(top.values.tolist(), top.indices.tolist())]

    def describe(self):
        return {"retriever": "llopis-classifier", "n_classes": len(self.classes)}


if __name__ == "__main__":
    from eval_lib import evaluate
    from queries import QUERIES

    csv = sys.argv[1] if len(sys.argv) > 1 else "mainSimulationAccessTraces.csv"
    model, tok, classes, info = train_classifier(csv, device="cpu")
    print("\ntrain info:", info)

    retr = ClassifierRetriever(model, tok, classes, info["maxlen"])
    res = evaluate(retr, QUERIES, ks=(1, 3, 5, 10))
    o, na = res["overall"], res["no_answer"]
    print(f"\n[classifier on 73-query benchmark]")
    print(f"  overall: hit@1={o['hit@1']:.3f} hit@3={o['hit@3']:.3f} "
          f"hit@5={o['hit@5']:.3f} hit@10={o['hit@10']:.3f} mrr={o['mrr']:.3f}")
    print(f"  no-answer AUROC={na['auroc']:.3f}")
    print("  by category (hit@1 / hit@3):")
    for cat, m in res["by_category"].items():
        print(f"    {cat:20s} n={m['n']:2d}  {m['hit@1']:.3f} / {m['hit@3']:.3f}")
