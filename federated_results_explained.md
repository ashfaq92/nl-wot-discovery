# Federated results, explained with simple arithmetic

This note explains every number in the manuscript's federated table
(`results_federated.csv`, Table `tab:federated`) so it can be verified by hand.
Source of truth: `federated.py` (partition, graph, strategies) and
`run_federated.py` (the evaluation loop).

## 1. The partition (why 21–62 records per node)

The 286 records are assigned to nodes by their `destinationLocation`
(`NODE_LOCATIONS` in `federated.py`, mirroring Llopis et al. 2025, Table 5):

| Node | Locations | Role |
|---|---|---|
| 1 | Bedroom, room_10 | devices |
| 2 | Dinningroom, Livingroom, Entrance | devices |
| 3 | Garage, Watterroom, room_1 | devices |
| 4 | BedroomParents, BedroomChildren, Bathroom, Showerroom | devices |
| 5 | Kitchen, room_2, room_3 | devices |
| 6 | room_4 … room_9 | devices |
| 7 | (none) | coordinator / gateway |

Each location belongs to exactly one node, so the six device-node record counts
sum to 286. Run `python federated.py` to print the exact per-node counts.

## 2. The topology (why "6 nodes, 2 hops" for broadcast)

`FEDERATION_GRAPH` (undirected):

```
1: 2,5    2: 1,3,7    3: 2,4,7    4: 3,5
5: 1,4,6  6: 5,7      7: 2,3,6
```

Breadth-first search (BFS) from the gateway (node 7):

- depth 0: {7}
- depth 1: {2, 3, 6}   (7's neighbors)
- depth 2: {1, 4, 5}   (2→1, 3→4, 6→5)

**Broadcast** searches every node that holds records. Node 7 holds none, so it
is traversed but not counted as "visited". Hence for *every* query:
nodes visited = 6, hops = max BFS depth = 2. That is exactly the table row
`broadcast, gateway: Nodes 6.0, Hops 2.0` (no averaging needed; it is constant).
Broadcast then merges all candidates by best-score-per-endpoint, which is
mathematically the same ranking a single centralized index would produce, so
Hit@1/Hit@3/MRR (0.79 / 0.92 / 0.85) match the centralized row by construction.

## 3. What "nodes" and "hops" count (from the code)

- `nodes visited` = number of nodes whose **index was actually searched**
  (empty node 7 never counts).
- `hops` differs slightly per strategy, matching each function's return value:
  - `broadcast`: max BFS depth reached (always 2 from any entry).
  - `first_served`: BFS depth of the node whose result was **returned**.
  - `local_first`: BFS depth **reached when the loop stopped**.

This is why first-served (local) shows 0.1 hops but local-first (local) shows
0.54 hops even though both visit the same 2.33 nodes on average: when the
answer is found at depth 0 or 1, first-served reports the depth of the
*answering* node while local-first reports the depth of the *last-visited*
node.

## 4. The delegation rule (threshold τ = 0.40)

Both delegating strategies visit device nodes in BFS order from the entry
point and use the cosine-similarity of the local top-1 result:

- **first-served**: stop at the first node whose top score ≥ 0.40 and return
  *that node's* list (no merging). If no node qualifies, return the best seen.
- **local-first**: keep visiting and *merge* everything seen so far; stop as
  soon as the best merged score ≥ 0.40.

## 5. Why the gateway rows collapse to Hit@1 ≈ 0.54

From node 7, BFS visits device nodes in the order 2, 3, 6, then 1, 4, 5.
The entry node contributes no candidates, so both strategies simply walk this
fixed order and stop at the first node with a confident (≥ 0.40) local answer.
A semantically similar but *wrong-room* device (e.g., a light in another room)
often clears 0.40 before the true node is reached, so the walk stops early:
on average after ~2.97 nodes at depth ~1.3. Both strategies degrade
identically on Hit@1 (0.54) because they apply the same stopping rule to the
same node order; they differ only in the hop bookkeeping above.

Sanity check on the averages: with six device nodes and stops mostly at the
first or second visited node, an average of 2.97 nodes means roughly "stops by
the third node on a typical query", consistent with hops ≈ 1.25–1.39 (the
first three visited nodes sit at depth 1).

## 6. Why the local rows do better with fewer nodes

`local` entry means the query starts at a node that owns one of its target
endpoints. The correct record is then in the *first* index searched, and its
score usually clears 0.40 immediately: hit at depth 0, 1 node visited. The
average of 2.33 nodes (not 1.0) comes from the minority of queries whose
local top score stays below 0.40 (paraphrases/abstract phrasings), which
trigger further delegation. Hit@3 (0.95) even exceeds broadcast (0.92)
because searching a smaller, mostly-correct index first removes distractor
devices from other rooms that broadcast merges in.

## 7. Reproducing the table

```bash
cd code
python run_federated.py          # writes results_federated.csv
python federated.py              # prints partition + per-node counts
```

Rounding: the CSV stores e.g. 0.787 → the paper reports 0.79; 2.97 → "about
three nodes"; 61 answerable queries are the denominator for all accuracy
columns (12 no-answer queries are excluded).
