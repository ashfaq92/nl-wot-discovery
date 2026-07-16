"""
Retriever implementations for the WoT discovery benchmark.

Every retriever exposes the interface that eval_lib.evaluate expects:

    search(query: str, top_k: int) -> list[tuple[str, float]]

returning (endpoint, score) pairs ranked by descending score.

This file currently holds the frozen bi-encoder retriever (with optional
cross-encoder reranking). The lexical baselines (exact lookup, BM25/TF-IDF)
and the Transformer recommender are added in the Phase 2 steps that follow.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from eval_lib import preprocess_text, Timer, embedding_memory_mb


class BiEncoderRetriever:
    """Frozen sentence-transformer bi-encoder, with optional cross-encoder rerank.

    The index (record embeddings) is built once at construction. At query time
    the query is embedded with the same frozen encoder, candidates are scored by
    dot product over L2-normalized embeddings, and -- if a cross-encoder is
    supplied -- the top `rerank_pool` candidates are reranked.

    Timing/footprint attributes set at construction:
        load_time_s, index_build_time_s, embedding_mb
    """

    def __init__(self, records, bi_encoder_name,
                 cross_encoder_name=None, use_reranker=False,
                 rerank_pool=50, device="cpu",
                 text_col="record_text", endpoint_col="accessedNodeAddress",
                 preprocess_query=True):
        from sentence_transformers import SentenceTransformer, CrossEncoder

        self.name = bi_encoder_name.split("/")[-1]
        self.endpoints = records[endpoint_col].tolist()
        self.texts = records[text_col].tolist()
        self.rerank_pool = rerank_pool
        self.preprocess_query = preprocess_query
        self.use_reranker = use_reranker and cross_encoder_name is not None
        self.cross_name = cross_encoder_name.split("/")[-1] if self.use_reranker else None

        with Timer() as t:
            self.bi = SentenceTransformer(bi_encoder_name, device=device)
        self.load_time_s = t.s

        with Timer() as t:
            self.emb = self.bi.encode(self.texts, normalize_embeddings=True,
                                      batch_size=32, show_progress_bar=False)
        self.index_build_time_s = t.s
        self.embedding_mb = embedding_memory_mb(self.emb)

        self.cross = None
        if self.use_reranker:
            with Timer() as t:
                self.cross = CrossEncoder(cross_encoder_name, device=device)
            self.cross_load_time_s = t.s

    def search(self, query, top_k):
        q = preprocess_text(query) if self.preprocess_query else query
        q_emb = self.bi.encode([q], normalize_embeddings=True,
                               show_progress_bar=False)[0]
        sims = self.emb @ q_emb

        pool = max(top_k, self.rerank_pool if self.use_reranker else top_k)
        pool = min(pool, len(self.endpoints))
        top_idx = np.argpartition(-sims, pool - 1)[:pool]
        top_idx = top_idx[np.argsort(-sims[top_idx])]

        if self.use_reranker:
            pairs = [(q, self.texts[i]) for i in top_idx]
            scores = np.asarray(self.cross.predict(pairs))
            order = np.argsort(-scores)
            ranked = [(self.endpoints[top_idx[j]], float(scores[j])) for j in order]
        else:
            ranked = [(self.endpoints[i], float(sims[i])) for i in top_idx]

        return ranked[:top_k]

    def describe(self):
        d = {"retriever": "bi-encoder", "bi_encoder": self.name,
             "reranker": self.cross_name or "none",
             "load_time_s": round(self.load_time_s, 3),
             "index_build_time_s": round(self.index_build_time_s, 3),
             "embedding_mb": round(self.embedding_mb, 3),
             "n_records": len(self.endpoints)}
        return d


class ExactLookupRetriever:
    """Non-semantic structured lookup baseline (the brittleness reference point).

    A record is scored by how many of its *own* structured field values appear,
    as substrings, in the preprocessed query: service and location count double,
    operation counts single (max raw score 5, normalized to [0, 1]). This is a
    proxy for JSONPath/SPARQL-style exact matching: it only fires when the user
    happens to use the directory's own vocabulary. No hand-built synonyms are
    used -- the match vocabulary is exactly the dataset's field values -- so the
    baseline is fully reproducible. It should excel on templated queries and on
    no-answer queries (score 0 -> abstain), and degrade on paraphrases.

    Ties are broken by original record order (stable), so ranking among equally
    scored candidates is deterministic but uninformed -- by design.
    """

    W_SERVICE, W_LOCATION, W_OPERATION = 2, 2, 1
    _MAX = W_SERVICE + W_LOCATION + W_OPERATION

    def __init__(self, records,
                 service_col="destinationServiceType",
                 location_col="destinationLocation",
                 operation_col="operation",
                 endpoint_col="accessedNodeAddress"):
        self.name = "exact-lookup"
        self.endpoints = records[endpoint_col].tolist()
        self.svc = [preprocess_text(str(s).lstrip("/")) for s in records[service_col]]
        self.loc = [preprocess_text(s) for s in records[location_col]]
        self.op = [preprocess_text(s) for s in records[operation_col]]
        self.load_time_s = 0.0
        with Timer() as t:
            # "index build" here is just caching the normalized field strings
            self._n = len(self.endpoints)
        self.index_build_time_s = t.s
        self.embedding_mb = 0.0

    def search(self, query, top_k):
        q = preprocess_text(query)
        scored = []
        for i in range(self._n):
            score = 0
            if self.svc[i] and self.svc[i] in q:
                score += self.W_SERVICE
            if self.loc[i] and self.loc[i] in q:
                score += self.W_LOCATION
            if self.op[i] and self.op[i] in q:
                score += self.W_OPERATION
            scored.append((self.endpoints[i], score / self._MAX))
        scored.sort(key=lambda x: x[1], reverse=True)  # stable: ties keep order
        return scored[:top_k]

    def describe(self):
        return {"retriever": "exact-lookup", "n_records": self._n,
                "weights": {"service": self.W_SERVICE,
                            "location": self.W_LOCATION,
                            "operation": self.W_OPERATION}}


class TfidfRetriever:
    """Lexical baseline: TF-IDF vectors over record_text with cosine similarity.

    Uses scikit-learn's TfidfVectorizer (1-2 grams). L2-normalized vectors mean
    the linear kernel is cosine similarity. Stronger than exact lookup because
    partial token overlap is rewarded, but still purely lexical (no synonymy).
    """

    def __init__(self, records, text_col="record_text",
                 endpoint_col="accessedNodeAddress", ngram_range=(1, 2)):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.name = "tfidf"
        self.endpoints = records[endpoint_col].tolist()
        texts = records[text_col].tolist()
        self.vectorizer = TfidfVectorizer(ngram_range=ngram_range)
        with Timer() as t:
            self.matrix = self.vectorizer.fit_transform(texts)
        self.index_build_time_s = t.s
        self.load_time_s = 0.0
        self.embedding_mb = self.matrix.data.nbytes / (1024 * 1024)

    def search(self, query, top_k):
        from sklearn.metrics.pairwise import linear_kernel
        q = self.vectorizer.transform([preprocess_text(query)])
        sims = linear_kernel(q, self.matrix).ravel()
        pool = min(top_k, len(self.endpoints))
        idx = np.argpartition(-sims, pool - 1)[:pool]
        idx = idx[np.argsort(-sims[idx])]
        return [(self.endpoints[i], float(sims[i])) for i in idx]

    def describe(self):
        return {"retriever": "tfidf", "n_records": len(self.endpoints),
                "vocab": len(self.vectorizer.vocabulary_),
                "index_build_time_s": round(self.index_build_time_s, 3)}


class BM25Retriever:
    """Lexical baseline: Okapi BM25 over whitespace-tokenized record_text.

    Self-contained implementation (no rank_bm25 dependency). Standard params
    k1=1.5, b=0.75; non-negative idf. The canonical sparse retrieval baseline
    the dense bi-encoder must beat to justify embeddings.
    """

    def __init__(self, records, text_col="record_text",
                 endpoint_col="accessedNodeAddress", k1=1.5, b=0.75):
        self.name = "bm25"
        self.k1, self.b = k1, b
        self.endpoints = records[endpoint_col].tolist()
        with Timer() as t:
            self.docs = [t_.split() for t_ in records[text_col]]
            self.doc_len = np.array([len(d) for d in self.docs], dtype=float)
            self.avgdl = float(self.doc_len.mean()) if len(self.doc_len) else 0.0
            self.tf = [Counter(d) for d in self.docs]
            df = Counter()
            for d in self.docs:
                df.update(set(d))
            n = len(self.docs)
            # BM25 idf (non-negative variant)
            self.idf = {term: np.log(1 + (n - f + 0.5) / (f + 0.5))
                        for term, f in df.items()}
        self.index_build_time_s = t.s
        self.load_time_s = 0.0
        self.embedding_mb = 0.0

    def search(self, query, top_k):
        q_terms = preprocess_text(query).split()
        scores = np.zeros(len(self.docs))
        for i, tf in enumerate(self.tf):
            dl = self.doc_len[i]
            s = 0.0
            for term in q_terms:
                f = tf.get(term, 0)
                if f == 0:
                    continue
                idf = self.idf.get(term, 0.0)
                denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                s += idf * (f * (self.k1 + 1)) / denom
            scores[i] = s
        pool = min(top_k, len(self.endpoints))
        idx = np.argpartition(-scores, pool - 1)[:pool]
        idx = idx[np.argsort(-scores[idx])]
        return [(self.endpoints[i], float(scores[i])) for i in idx]

    def describe(self):
        return {"retriever": "bm25", "n_records": len(self.endpoints),
                "k1": self.k1, "b": self.b,
                "index_build_time_s": round(self.index_build_time_s, 3)}
