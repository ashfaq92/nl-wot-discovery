"""
Federated frozen retrieval (Phase 3).

The 286 service records are partitioned across discovery nodes by location,
mirroring the 7-node smart-home layout of Llopis et al. (2025, Table 5) so the
comparison to their distributed-AI federation is apples-to-apples. Each node
holds its own local retrieval index (per-node embeddings); the bi-encoder is
frozen and shared, so unlike the prior work there is no per-node training.

This module (3.1) builds the partition and the per-node indexes, and exposes
single-node search. Delegation strategies (local-first, broadcast-merge,
first-served) and the federated evaluation are added in 3.2.

Node -> locations mapping mirrors Llopis et al. (2025) Table 5:
  1: Bedroom, room_10
  2: Dinningroom, Livingroom, Entrance
  3: Garage, Watterroom, room_1
  4: BedroomParents, BedroomChildren, Bathroom, Showerroom
  5: Kitchen, room_2, room_3
  6: room_4 .. room_9
  7: (coordinator; holds no devices)
"""

from __future__ import annotations

import numpy as np

from eval_lib import load_records, build_record_text, Timer, preprocess_text

NODE_LOCATIONS = {
    1: ["Bedroom", "room_10"],
    2: ["Dinningroom", "Livingroom", "Entrance"],
    3: ["Garage", "Watterroom", "room_1"],
    4: ["BedroomParents", "BedroomChildren", "Bathroom", "Showerroom"],
    5: ["Kitchen", "room_2", "room_3"],
    6: ["room_4", "room_5", "room_6", "room_7", "room_8", "room_9"],
    7: [],  # coordinator / entry node, no local devices
}

# location -> node (inverse map)
LOCATION_NODE = {loc: n for n, locs in NODE_LOCATIONS.items() for loc in locs}

# Federation topology (undirected adjacency), a connected graph over the 7
# nodes with the coordinator (7) centrally connected. Used for delegation.
FEDERATION_GRAPH = {
    1: [2, 5],
    2: [1, 3, 7],
    3: [2, 4, 7],
    4: [3, 5],
    5: [1, 4, 6],
    6: [5, 7],
    7: [2, 3, 6],
}


def node_of_endpoint_location(location):
    return LOCATION_NODE.get(location)


class FederatedIndex:
    """Per-node frozen retrieval indexes over a location-partitioned corpus.

    One shared frozen bi-encoder is loaded once; each node gets its own
    embedding matrix over the records whose location maps to that node.
    """

    def __init__(self, records, bi_encoder_name="sentence-transformers/all-MiniLM-L6-v2",
                 node_locations=NODE_LOCATIONS, device="cpu",
                 text_col="record_text", endpoint_col="accessedNodeAddress",
                 location_col="destinationLocation"):
        from sentence_transformers import SentenceTransformer

        self.name = bi_encoder_name.split("/")[-1]
        with Timer() as t:
            self.encoder = SentenceTransformer(bi_encoder_name, device=device)
        self.load_time_s = t.s

        loc_to_node = {loc: n for n, locs in node_locations.items() for loc in locs}
        self.nodes = sorted(node_locations.keys())
        self.node = {}            # node -> dict(endpoints, texts, emb, build_s)
        unassigned = set()

        for n in self.nodes:
            mask = records[location_col].map(lambda l: loc_to_node.get(l) == n)
            sub = records[mask]
            endpoints = sub[endpoint_col].tolist()
            texts = sub[text_col].tolist()
            if texts:
                with Timer() as t:
                    emb = self.encoder.encode(texts, normalize_embeddings=True,
                                              batch_size=32, show_progress_bar=False)
                build_s = t.s
            else:
                emb = np.zeros((0, self.encoder.get_sentence_embedding_dimension()),
                               dtype=np.float32)
                build_s = 0.0
            self.node[n] = {"endpoints": endpoints, "texts": texts,
                            "emb": emb, "build_s": build_s}

        # endpoint -> set of nodes that hold it (an endpoint can appear in
        # more than one location in the trace data)
        self.endpoint_nodes = {}
        for ep, loc in zip(records[endpoint_col], records[location_col]):
            n = loc_to_node.get(loc)
            if n is not None:
                self.endpoint_nodes.setdefault(str(ep), set()).add(n)

        # sanity: every record assigned to exactly one node
        for loc in records[location_col].unique():
            if loc not in loc_to_node:
                unassigned.add(loc)
        self.unassigned_locations = unassigned

    def _embed(self, query):
        return self.encoder.encode([preprocess_text(query)],
                                   normalize_embeddings=True,
                                   show_progress_bar=False)[0]

    def _search_emb(self, node, q_emb, top_k):
        d = self.node[node]
        if not d["texts"]:
            return []
        sims = d["emb"] @ q_emb
        k = min(top_k, len(d["endpoints"]))
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [(d["endpoints"][i], float(sims[i])) for i in idx]

    def node_search(self, node, query, top_k):
        """Local retrieval within a single node. Returns [(endpoint, score), ...]."""
        return self._search_emb(node, self._embed(query), top_k)

    # ----- federation topology helpers -----

    def _bfs_levels(self, origin, graph=FEDERATION_GRAPH):
        """Yield (node, hop_depth) in BFS order from origin over the graph."""
        seen = {origin}
        frontier = [origin]
        depth = 0
        while frontier:
            for n in frontier:
                yield n, depth
            nxt = []
            for n in frontier:
                for m in graph.get(n, []):
                    if m not in seen:
                        seen.add(m)
                        nxt.append(m)
            frontier = nxt
            depth += 1

    @staticmethod
    def _merge(cands, top_k):
        """Merge (endpoint, score) candidates: best score per endpoint, sorted."""
        best = {}
        for ep, sc in cands:
            if ep not in best or sc > best[ep]:
                best[ep] = sc
        ranked = sorted(best.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    # ----- delegation strategies -----
    # Each returns (ranked_results, nodes_visited, hops), where nodes_visited
    # counts nodes whose index was actually searched and hops is the BFS depth
    # reached.

    def broadcast(self, query, top_k, origin=7, graph=FEDERATION_GRAPH):
        """Contact every reachable node, merge all candidates. Accuracy-preserving;
        equivalent to a single centralized index over the union of records."""
        q = self._embed(query)
        cands, visited, max_hop = [], 0, 0
        for node, depth in self._bfs_levels(origin, graph):
            if self.node[node]["texts"]:
                cands.extend(self._search_emb(node, q, top_k))
                visited += 1
            max_hop = max(max_hop, depth)
        return self._merge(cands, top_k), visited, max_hop

    def first_served(self, query, top_k, origin=7, threshold=0.4,
                     graph=FEDERATION_GRAPH):
        """Follow BFS order; stop at the first node whose top score >= threshold
        and return that node's results. If none qualify, return the best seen."""
        q = self._embed(query)
        visited = 0
        best_node_results, best_score, best_hop = [], -1.0, 0
        for node, depth in self._bfs_levels(origin, graph):
            if not self.node[node]["texts"]:
                continue
            res = self._search_emb(node, q, top_k)
            visited += 1
            if res and res[0][1] > best_score:
                best_node_results, best_score, best_hop = res, res[0][1], depth
            if res and res[0][1] >= threshold:
                return res, visited, depth
        return best_node_results, visited, best_hop

    def local_first(self, query, top_k, origin=7, threshold=0.4,
                    graph=FEDERATION_GRAPH):
        """Search the origin first; delegate over BFS only while the best score
        so far is below threshold. Merge candidates across visited nodes."""
        q = self._embed(query)
        cands, visited, reached_hop = [], 0, 0
        for node, depth in self._bfs_levels(origin, graph):
            if not self.node[node]["texts"]:
                continue
            res = self._search_emb(node, q, top_k)
            visited += 1
            reached_hop = depth
            cands.extend(res)
            cur_best = max((s for _, s in cands), default=-1.0)
            if cur_best >= threshold:
                break
        return self._merge(cands, top_k), visited, reached_hop

    def stats(self):
        return {
            "nodes": len(self.nodes),
            "per_node_records": {n: len(self.node[n]["endpoints"]) for n in self.nodes},
            "per_node_build_s": {n: round(self.node[n]["build_s"], 3) for n in self.nodes},
            "load_time_s": round(self.load_time_s, 2),
            "unassigned_locations": sorted(self.unassigned_locations),
        }


if __name__ == "__main__":
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else "mainSimulationAccessTraces.csv"
    records = build_record_text(load_records(csv), fmt="sentence")
    fed = FederatedIndex(records)
    st = fed.stats()
    print("federation stats:")
    print("  nodes:", st["nodes"], " encoder load:", st["load_time_s"], "s")
    print("  unassigned locations:", st["unassigned_locations"])
    total = 0
    for n in fed.nodes:
        c = st["per_node_records"][n]
        total += c
        print(f"  node {n}: {c:3d} records  (build {st['per_node_build_s'][n]}s)  "
              f"locations={NODE_LOCATIONS[n]}")
    print(f"  total assigned records: {total} (expected 286)")

    # quick local-search sanity on the node that should hold the kitchen light
    print("\nlocal search at node 5 (Kitchen) for 'please switch on the kitchen lights':")
    for ep, sc in fed.node_search(5, "please switch on the kitchen lights", 3):
        print(f"  {sc:.3f}  {ep}")
