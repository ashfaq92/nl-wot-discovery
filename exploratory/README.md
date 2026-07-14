# exploratory/

Side investigations, superseded drafts, and scratch notebooks that informed the
paper but are **not reported in it**. Nothing in the manuscript depends on this
folder; every reported number is produced by the `run_*.py` drivers one level
up in `code/`. Kept for provenance.

Scripts here import the shared core from `code/` (paths resolve relative to
this file), so they still run: `python exploratory/<script>.py` from `code/`.

## Promoted to code/ (reported in the paper)

These started here and were moved up (and renamed) once the manuscript adopted
their numbers:

| Was | Now |
|---|---|
| `cpu_all_classifier.py` | `../run_classifier_multiseed.py` |
| `deployment_latency_multirun_cpu.py` | `../run_deployment.py` |
| `measure_classifier_deploy.py` | `../run_classifier_deployment.py` |
| `perclass_stratified_vs_balanced_gpu.py` | `../run_perclass_balanced.py` |
| `perclass_singleton_copy_gpu.py` | `../run_perclass_singleton.py` |

## Superseded drafts

- **`deployment_singlerun.py`** (was `code/run_deployment.py`) and
  `results_deployment_singlerun.csv`: the original single-run deployment
  measurement. Superseded by the 5-run mean +/- STD version now at
  `../run_deployment.py`.
- **`classifier_benchmark_multiseed.py`**: classifier on the query benchmark,
  mean +/- STD (GPU if available). Subsumed by `../run_classifier_multiseed.py`,
  which derives the same numbers on CPU from a single pass.
- **`perclass_indist_compare.py`** / **`perclass_indist_multiseed.py`**:
  per-endpoint in-distribution recall, classifier vs frozen retriever
  (single-seed, then 5-seed). Early versions of the frequency-bin analysis now
  produced by `../run_classifier_multiseed.py` and `../run_perclass_balanced.py`.
- **`perclass_balanced_split_gpu.py`**: guarantees every evaluated endpoint has
  >=1 train and >=1 val row (singletons train-only, excluded from scoring), to
  separate "no training data" from "trained but could not learn". Folded into
  the singleton-copy and balanced-training designs now in `code/`.
- `results_strat_vs_balanced_perclass.csv` / `results_strat_vs_balanced_bins.csv`:
  GPU-produced runs of the balanced-training experiment. The authoritative CPU
  outputs moved up as `../results_perclass_balanced*.csv`.

## Removed from the paper

- **`classifier_indist_multiseed.py`**: multi-seed in-distribution evaluation
  under a `random` (IID, prior-work) vs `group_sentence` (leakage-free) split.
  The leakage-free split collapses accuracy to 0.52 +/- 0.13, but the number is
  fragile across seeds (the val-set size swings 52k-170k rows because sentences
  carry very different duplicate masses) and prior work used a random split, so
  the disjoint experiment never matched the baseline it critiqued. Removed; the
  why-the-classifier-fails argument in the paper is carried by the per-class
  frequency-bin analysis instead.
- **`run_leakage_demo.py`** (+ `results_leakage_demo.csv`):
  the earlier two-training random vs sentence-disjoint demo of the same
  hypothesis (~15 min). Removed from the paper for the same reason.

Scratch notebooks from early development (including the original TensorFlow
implementation of the Llopis et al. classifier, `audacity_baseline.ipynb`) and
the orphaned `results_embedding_classifier*.csv` were deleted in the July 2026
cleanup; they remain in git history.
