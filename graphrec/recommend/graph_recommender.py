"""Adapter that exposes the Graph PPR method via the evaluation interface.

Crucially, ``fit`` rebuilds the graph from the *training* interactions only — the
disk cache (full data) is never used during evaluation, so there is no future
leakage. Movie/user side-features are static metadata (not interactions), so
including them does not leak the test period.
"""

from __future__ import annotations

import pandas as pd

from graphrec.graph.build import build_bipartite_graph, build_graph
from graphrec.recommend import ppr


class GraphPPRRecommender:
    """Personalized PageRank over a graph built from training interactions.

    With ``movie_features``/``user_features`` it builds the full heterogeneous
    graph; without them it builds the bipartite core (the M2 baseline).
    """

    def __init__(
        self,
        alpha: float = 0.85,
        movie_features: pd.DataFrame | None = None,
        user_features: pd.DataFrame | None = None,
        recency_weight: float = 0.0,
    ) -> None:
        self.alpha = alpha
        self.movie_features = movie_features
        self.user_features = user_features
        self.recency_weight = recency_weight
        self._with_features = (
            movie_features is not None or user_features is not None
        )
        base = "Graph PPR (+features)" if self._with_features else "Graph PPR"
        self.name = f"{base} r={recency_weight:g}" if recency_weight else base

    def fit(self, train: pd.DataFrame) -> None:
        if self._with_features:
            self.graph = build_graph(
                train,
                self.movie_features,
                self.user_features,
                recency_weight=self.recency_weight,
            )
        else:
            self.graph = build_bipartite_graph(train)

    def recommend(self, user_id: int, k: int, exclude: set[int]) -> list[int]:
        try:
            recs = ppr.recommend(self.graph, int(user_id), k, alpha=self.alpha)
        except KeyError:  # user not present in the training graph
            return []
        return [rec.movie_id for rec in recs]
