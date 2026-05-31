"""Tests for configurable edge weights and the recency lever (no network)."""

from __future__ import annotations

import pandas as pd

from graphrec.graph.build import FeatureWeights, build_graph


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
            "year": [1995, 1980, 1975],  # 10 newest, 30 oldest
        }
    )


def test_feature_weights_applied_to_edges() -> None:
    graph = build_graph(
        _ratings(), _movie_features(), weights=FeatureWeights(genre=3.0)
    )
    assert graph.edges["movie:10", "genre:Comedy"]["weight"] == 3.0


def test_recency_boosts_newer_movie_edges_more() -> None:
    base = build_graph(_ratings(), _movie_features(), recency_weight=0.0)
    boosted = build_graph(_ratings(), _movie_features(), recency_weight=2.0)

    # Newest film (1995) is boosted; oldest (1975, the min) is unchanged.
    newest_ratio = (
        boosted.edges["user:1", "movie:10"]["weight"]
        / base.edges["user:1", "movie:10"]["weight"]
    )
    oldest_ratio = (
        boosted.edges["user:2", "movie:30"]["weight"]
        / base.edges["user:2", "movie:30"]["weight"]
    )
    assert oldest_ratio == 1.0
    assert newest_ratio > oldest_ratio
    # span 1975..1995, weight=2 -> newest factor = 1 + 2*1 = 3
    assert newest_ratio == 3.0


def test_recency_off_is_a_no_op() -> None:
    base = build_graph(_ratings(), _movie_features(), recency_weight=0.0)
    again = build_graph(_ratings(), _movie_features())
    assert (
        base.edges["user:1", "movie:10"]["weight"]
        == again.edges["user:1", "movie:10"]["weight"]
    )
