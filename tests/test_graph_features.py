"""Tests for the heterogeneous feature graph (no network access)."""

from __future__ import annotations

import pandas as pd

from graphrec.graph.build import (
    age_bucket_label,
    build_graph,
    year_bucket_label,
)
from graphrec.recommend import ppr


def _ratings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2],
            "movie_id": [10, 20, 10, 30],
            "rating": [5, 4, 5, 5],
            "timestamp": [0, 0, 0, 0],
        }
    )


def _movie_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "movie_id": [10, 20, 30],
            "genres": [["Comedy"], ["Drama"], ["Comedy", "Crime"]],
            "year": [1995, 1980, 1975],
        }
    )


def _user_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [1, 2],
            "age": [30, 20],
            "occupation": ["engineer", "student"],
        }
    )


def test_age_and_year_buckets() -> None:
    assert age_bucket_label(17) == "<18"
    assert age_bucket_label(30) == "25-34"
    assert age_bucket_label(70) == "56+"
    assert year_bucket_label(1995) == "1990s"
    assert year_bucket_label(1975) == "1970s"


def test_feature_nodes_and_edges_present() -> None:
    graph = build_graph(_ratings(), _movie_features(), _user_features())

    assert graph.nodes["genre:Comedy"]["kind"] == "genre"
    assert graph.nodes["occupation:engineer"]["kind"] == "occupation"
    assert graph.nodes["age_bucket:25-34"]["kind"] == "age_bucket"
    assert graph.nodes["year_bucket:1990s"]["kind"] == "year_bucket"

    assert graph.edges["movie:10", "genre:Comedy"]["relation"] == "has_genre"
    assert graph.edges["user:1", "occupation:engineer"]["relation"] == "has_occupation"
    # the bipartite core is preserved
    assert graph.edges["user:1", "movie:10"]["relation"] == "rated"


def test_recommend_returns_only_movies_on_feature_graph() -> None:
    graph = build_graph(_ratings(), _movie_features(), _user_features())
    recs = ppr.recommend(graph, user_id=1, k=10)
    rec_ids = {rec.movie_id for rec in recs}

    # Never recommends feature nodes or already-seen movies...
    assert all(isinstance(rec.movie_id, int) for rec in recs)
    assert 10 not in rec_ids and 20 not in rec_ids
    # ...and surfaces movie 30, reachable via user 2 and the shared Comedy genre.
    assert 30 in rec_ids
