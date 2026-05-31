"""Tests for the bipartite PPR recommender (no network access)."""

from __future__ import annotations

import pandas as pd
import pytest

from graphrec.graph.build import build_bipartite_graph
from graphrec.recommend import ppr


def _toy_ratings() -> pd.DataFrame:
    """Two users with overlapping taste, each with one distinct movie."""
    return pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 2],
            "movie_id": [10, 20, 10, 20, 30],
            "rating": [5, 4, 5, 4, 5],
            "timestamp": [0, 0, 0, 0, 0],
        }
    )


def test_recommend_excludes_seen_and_surfaces_neighbor_item() -> None:
    graph = build_bipartite_graph(_toy_ratings())
    recs = ppr.recommend(graph, user_id=1, k=5)
    rec_ids = {rec.movie_id for rec in recs}

    # User 1 rated 10 and 20; those must not be recommended back.
    assert 10 not in rec_ids
    assert 20 not in rec_ids
    # Movie 30, liked by the taste-aligned user 2, should surface.
    assert 30 in rec_ids


def test_recommend_respects_k() -> None:
    graph = build_bipartite_graph(_toy_ratings())
    assert len(ppr.recommend(graph, user_id=2, k=1)) <= 1


def test_unknown_user_raises() -> None:
    graph = build_bipartite_graph(_toy_ratings())
    with pytest.raises(KeyError):
        ppr.recommend(graph, user_id=999)
