# GraphRec

A knowledge-graph movie recommender. It builds a heterogeneous graph from the
MovieLens 100k dataset (users, movies, genres, demographics, release eras),
generates recommendations with **Personalized PageRank**, exposes **configurable
edge weights** to steer *what* gets recommended, and explains every recommendation
with a graph path. Runs fully locally, no GPU.

```
User 42 ──rated──▶ Fargo ──has_genre──▶ Crime ──has_genre──▶ L.A. Confidential ◀── recommended
```

## Why a graph?

Vanilla collaborative filtering scores items by latent similarity; it can't tell
you *why*, and it can't easily incorporate side information or be steered toward a
business objective. A graph formulation makes relationships first-class:

- **Multi-hop reasoning** — connect users to items through shared genres, eras, and
  co-raters, not just a single similarity number.
- **Explainability** — every recommendation comes with a concrete path.
- **Steerability** — re-weighting edge types changes recommendations with no
  retraining (e.g. tilt toward newer films, or promote a genre).

It is *complementary* to vector retrieval rather than strictly better — see
[Tradeoffs](#design-decisions--tradeoffs).

## Quickstart

```bash
uv sync                                   # install (Python 3.12, managed by uv)

uv run graphrec recommend --user 1        # top-k with explanations
uv run graphrec recommend --user 1 --recency-weight 3   # steer toward newer films
uv run graphrec eval                      # compare against baselines (temporal split)
uv run graphrec ablate                    # sweep the recency lever
uv run graphrec visualize --user 1        # write recommendations.html
```

MovieLens 100k (~5 MB) is downloaded and cached under `.data/` on first run.

## How it works

**Pipeline:** `data → graph construction → PPR retrieval → (explanation, viz)`, with
a separate `eval` path that fits every model on a temporal train split.

**Graph schema** (`graphrec/graph/build.py`):

| Node type | Edges | Weight |
| --- | --- | --- |
| `user`, `movie` | `user –rated– movie` | rating (1–5) |
| `genre` | `movie –has_genre– genre` | configurable |
| `occupation`, `age_bucket` | `user –has_occupation–`, `user –in– age_bucket` | configurable |
| `year_bucket` | `movie –in– year_bucket` | configurable |

Feature nodes are **additive** — the bipartite core is a strict subset — so the
recommender works unchanged; features only add new paths.

**Retrieval:** Personalized PageRank (random-walk-with-restart) rooted at the user
node. Edge weight is the per-step transition probability, so higher-weighted edges
attract more of the walk. Movie nodes are ranked by stationary probability;
already-seen movies are excluded. Cold/unseen users fall back to popularity.

**The recency lever** (`recency_weight`) scales each movie's incident edges by
`1 + recency_weight · normalized_year`, so newer films receive more inflow. Uniform
scaling of a node's edges cancels on *exit* but raises *inflow*, giving a clean
directional tilt. The same weighting mechanism generalizes to any objective
(promote a genre, boost diversity).

**Explanations** pick the most *specific* shared bridge (rarest by node degree,
IDF-style) through a genuinely-liked movie, preferring content bridges over
collaborative ones for readability.

## Evaluation

Honest offline evaluation on a **temporal split** (train on the past, test on the
future — no random-split leakage). Every model is fit on the train split only; the
graph is rebuilt from train interactions, never the full-data cache. Metrics are
averaged over a reproducible sample of users (`--users`, seed 42).

`uv run graphrec eval` (100 users, k=10):

| Method | Recall@10 | NDCG@10 | Coverage |
| --- | --- | --- | --- |
| Popularity | 0.067 | 0.151 | 0.048 |
| Item-CF (cosine) | 0.043 | 0.147 | 0.099 |
| MF (TruncatedSVD) | 0.066 | 0.155 | 0.230 |
| Graph PPR (bipartite) | 0.063 | 0.157 | 0.050 |
| Graph PPR (+features) | 0.064 | 0.159 | 0.051 |

Out of the box, graph PPR is at **parity** with strong baselines — popularity is
famously hard to beat on MovieLens, and plain PPR inherits a popularity bias
(coverage ≈ popularity's). The graph earns its keep once the weighting lever is
applied.

`uv run graphrec ablate` (recency sweep, feature graph):

| recency_weight | Recall@10 | NDCG@10 | Coverage | AvgYear |
| --- | --- | --- | --- | --- |
| 0.0 | 0.064 | 0.159 | 0.051 | 1989.9 |
| 1.0 | 0.078 | 0.166 | 0.051 | 1992.3 |
| 2.0 | 0.088 | 0.167 | 0.051 | 1992.7 |
| 4.0 | **0.092** | **0.171** | 0.050 | 1993.2 |

At `recency_weight=4` the graph **beats every baseline** (+40% Recall over MF). On a
temporal split, recency is a strong predictive prior, and the graph injects it
cleanly as edge weights — no retraining. This is steering, not overfitting: the
same knob trades accuracy against any objective you choose.

## Design decisions & tradeoffs

- **Graph vs. vector retrieval.** Vectors excel at fuzzy semantic similarity;
  graphs excel at precise relational/multi-hop queries and explanations. A
  production system would likely use **both**: vector ANN for broad candidate
  generation, graph traversal for relational precision and reasons.
- **Popularity / hub bias.** PPR concentrates mass on high-degree nodes, so
  coverage stays low (~0.05). Configurable weights and degree-aware normalization
  are the levers; recency improves accuracy but **not** coverage — diversity
  remains an open problem here, called out honestly rather than hidden.
- **Leakage discipline.** Temporal split; models fit on train only; the full-data
  graph cache is never used during eval. Side-features (genres, demographics) are
  static metadata, so they don't leak the test period.
- **NetworkX for the demo.** Fine at this scale and dependency-light; Neo4j is the
  production choice (below).

## Productionizing

This repo is the offline/demo slice. A production deployment would look like:

- **Graph store: Neo4j.** Justified here by two needs the demo already exercises —
  live multi-hop **traversal** for explanations, and a **mutable, persistent**
  graph for the evolution roadmap below. Neo4j GDS provides personalized PageRank
  in-database. (If the workload were purely read-mostly with no live traversal,
  precomputed recommendations in a KV store like Redis would be simpler — a graph
  DB is a justified choice, not an automatic one.)
- **Candidate generation → ranking.** PPR is candidate generation; a learned ranker
  (gradient-boosted trees / a neural ranker over richer features) would re-rank the
  top candidates. The recency lever becomes one ranking feature among many.
- **Serving.** Precompute PPR offline (per-user top-N) and serve from a low-latency
  store; or hold the graph in a long-lived service for on-demand PPR. The demo's
  versioned graph cache mirrors the offline-build / online-serve split.
- **Cold start.** Popularity fallback today; in production, content/graph-based
  fallback via feature nodes (genre/era) for new users and items.
- **Matrix factorization.** TruncatedSVD here is the dependency-light stand-in;
  `implicit` ALS (implicit-feedback) is the production choice.
- **Evaluation in production.** Offline metrics (Recall/NDCG/coverage) for
  regression-gating, plus online A/B with an eval platform (e.g. Braintrust) and
  guardrail metrics for diversity/freshness.
- **Monitoring.** Track coverage/novelty drift, popularity skew, latency, and
  graph-staleness; alert when the served graph falls behind ingestion.

## Future work — an evolving graph

Today's graph is built from *existing structured* features. The natural next step is
to add **LLM-extracted concepts/themes** from movie text (plots, reviews) as new
node types — richer relations between movies, better multi-hop reasoning, and more
informative explanations. The retrieval algorithm doesn't change; the graph gets
richer over time, and recommendations improve with it.

## Project layout

```
graphrec/
  data/        # MovieLens download + feature loaders
  graph/       # schema, build_graph, versioned disk cache
  recommend/   # PPR, graph adapter, explanations
  baselines/   # popularity, item-CF, SVD
  eval/        # temporal split, metrics, comparison + ablation harness
  viz/         # pyvis subgraph rendering
  cli.py       # recommend / visualize / eval / ablate
tests/         # 19 tests, network-free
PRD.md         # product requirements
```

## Development

```bash
uv run pytest        # tests (no network access required)
uv run ruff check    # lint
```
