# GraphRec — Product Requirements Document

## 1. Summary

GraphRec is a heterogeneous graph-based movie recommender. It builds a knowledge
graph from the MovieLens dataset (users, movies, and side-feature nodes), generates
recommendations with Personalized PageRank (random-walk-with-restart), and exposes
configurable edge weights as a product lever for steering *what* gets recommended.
It ships with an honest offline evaluation against standard baselines, multi-hop
explanations, and an interactive graph visualization. It runs fully locally with no
GPU.

## 2. Goals

- Demonstrate an end-to-end, production-minded recommendation pipeline: data →
  graph construction → retrieval → evaluation → serving story.
- Show *why* a graph formulation is useful (relational, multi-hop, explainable)
  and measure it honestly against non-graph baselines.
- Make recommendation behavior **steerable** via edge weights, not just accurate.
- Keep it complete, polished, and runnable end-to-end on a MacBook.

## 3. Non-goals / guardrails

- No LLM in the critical path; no agent frameworks.
- Not overbuilt — scope is bounded by the milestones in §14.
- No GPU; everything runs locally and deterministically (fixed seeds).

## 4. Dataset

**MovieLens 100k.** Public, tiny, instant load.

- Item features: genres (19 flags), release year.
- User features: age, gender, occupation.

Features on both sides allow a genuinely heterogeneous graph.

## 5. Graph schema

**Node types:** `user`, `movie`, `genre`, `occupation`, `age_bucket`, `year_bucket`

**Edge types:**

| Edge | Meaning | Weight |
| --- | --- | --- |
| `user –rated– movie` | interaction | scaled by rating |
| `movie –has_genre– genre` | content | configurable |
| `user –has_occupation– occupation` | demographic | configurable |
| `user –in– age_bucket` | demographic | configurable |
| `movie –in– year_bucket` | temporal | configurable |

All edge-type weights are configurable (see §7).

## 6. Recommender

Personalized PageRank seeded on a user's liked movies → rank all `movie` nodes →
drop already-seen → return top-K.

- **Cold-start:** popularity fallback for unseen / cold users.
- Deterministic given a fixed seed and config.

## 7. Configurable edge weights (the product lever)

Recommendation behavior is steered by per-edge-type weights, not just by the data.

- **Demo objective:** recency boost — up-weight `year_bucket` / newer-film edges so
  fresher titles surface.
- **Generalization (README):** the same mechanism prioritizes any objective —
  promoting a genre, increasing diversity, or leaning on demographic affinity —
  by changing weights, with no code changes.

## 8. Explainability

Multi-hop path extraction, e.g. *"recommended because you liked Fargo → Crime →
L.A. Confidential."* Surfaced as text in the CLI; the same paths are reused to
highlight edges in the visualization.

## 9. Baselines

- Popularity (global top items)
- Item-item co-occurrence (cosine)
- Matrix factorization (TruncatedSVD)

Fair, cheap, and standard — enough to show where the graph wins and where it doesn't.

## 10. Evaluation

- **Split:** temporal (global timestamp cutoff — train on past, test on future).
- **Metrics:** Recall@10, NDCG@10, catalog coverage.
- **Outputs:**
  - Comparison table: PPR vs. the three baselines.
  - Ablation: effect of the recency weight on the metrics.

## 11. Visualization

Interactive pyvis HTML of a queried user's neighborhood plus the recommendation /
explanation subgraph. Always a local subgraph — the full 100k-edge graph is too
large to render meaningfully.

## 12. Interface (CLI, `typer`)

```
graphrec recommend --user 42 --k 10 [--recency-weight W ...]
graphrec eval                       # metrics + comparison table
graphrec visualize --user 42        # writes interactive HTML
```

## 13. Repo structure

```
graphrec/
  data/            # loader + MovieLens fetch
  graph/           # schema + construction (bipartite -> +features)
  recommend/       # PPR, weights, cold-start, explanations
  baselines/       # popularity, item-CF, SVD
  eval/            # temporal split, metrics, comparison
  viz/             # pyvis subgraph
  cli.py
tests/
README.md
```

### Tooling & setup

- **Python 3.12**, managed with **uv**.
- Runtime deps: `networkx`, `pandas`, `numpy`, `scipy`, `scikit-learn`, `pyvis`, `typer`.
- Dev deps: `pytest`, `ruff`.

```bash
uv python pin 3.12
uv add networkx pandas numpy scipy scikit-learn pyvis typer
uv add --dev pytest ruff
```

## 14. Milestones

Phased so an early stop still yields a complete, defensible project. Feature nodes
are purely additive — a bare bipartite graph is a strict subset of the full schema,
so there is no throwaway work.

- **M1 (must):** bipartite user–movie graph + PPR + CLI `recommend`
- **M2 (must):** temporal split + metrics + 3 baselines + comparison table
- **M3 (must):** feature nodes (full schema from §5)
- **M4 (should):** configurable weights + recency demo + ablation
- **M5 (should):** explanation paths
- **M6 (should):** pyvis visualization
- **M7 (must):** README + production story
- Tests written alongside throughout.

## 15. Production story (README)

- **Neo4j** as the production graph store (NetworkX is the local/demo choice).
- **Candidate generation → ranking** split; PPR as candidate generation.
- Offline PPR **precompute** + served recommendations; **batch vs. real-time**.
- **Cold-start** handling.
- **Monitoring & eval-in-prod** (Braintrust).
- `implicit` ALS as the production MF choice (SVD is the local default).

## 16. Future work — graph evolution

Today's graph is built from *existing structured* features. The natural evolution is
to add **LLM-extracted concepts/themes** from movie text (plot, reviews) as new node
types, creating richer relations between movies and improving recommendations over
time. The graph — and the recommendations it produces — get better as the graph
grows, without changing the retrieval algorithm.

## 17. Risks & mitigations

| Risk | Mitigation |
| --- | --- |
| PPR hub bias (high-degree feature nodes pull toward popular items) | configurable edge weights + degree normalization |
| Temporal-split sparsity for cold users | popularity fallback |
| Visualization at scale | always render a local subgraph |
| MF baseline install friction on macOS | TruncatedSVD by default; ALS noted as prod choice |

## 18. Quality roadmap (prioritized next steps)

Concrete, measurable improvements that target the gaps the evaluation surfaced
(accuracy at parity with baselines; coverage stuck ~0.05). Ordered by ROI.

1. **Degree discounting (RP3β-style).** Divide PPR scores by `degree(movie)^β` to
   penalize popular hubs. Directly attacks the low-coverage / popularity-bias
   problem; ~minutes of work and runs through the existing eval harness.
2. **Denoise the graph.** Build `rated` edges only for ratings ≥ 4 ("liked") and
   drop over-generic mega-hubs (e.g. `Drama`). Cheap accuracy and less diffusion.
3. **Validation-set hyperparameter search.** Carve a validation slice from the
   train period and search α, `FeatureWeights`, `recency_weight`, and β against it,
   so the weighting claims are empirical rather than hand-set.
4. **Production scaling (discuss, not necessarily build).** Two-stage
   candidate-generation → learned ranker (PPR → LightGBM / LambdaMART), and
   LightGCN as the learned-propagation upgrade — the "how I'd push quality at
   scale" answer.

## 19. Observability — experiment tracking (MLflow)

Eval and ablation runs are logged to **MLflow** so results are recorded,
comparable, and reproducible rather than living in terminal scrollback.

- **Optional extra** (`uv sync --extra tracking`); lazy-imported so the core
  install stays light. Opt in per run with `--mlflow`.
- **Local file backend** (`./mlruns`), no server; browse with `uv run mlflow ui`.
- **One run per config:** each method (eval) or `recency_weight` (ablation) is a
  run, with params (k, test_frac, n_users, seed, alpha) and metrics
  (recall_at_10, ndcg_at_10, coverage, avg_year) — so MLflow's compare/plot views
  work across methods and sweeps.
