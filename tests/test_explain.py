"""Tests for path-based explanations (no network access)."""

from __future__ import annotations

import pandas as pd

from graphrec.graph.build import build_graph
from graphrec.recommend.explain import explain, format_explanation


def _ratings() -> pd.DataFrame:
    # user 1 likes 10 (Comedy) and 20 (Drama); user 2 likes 10 and 30.
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


def test_explanation_uses_a_connected_liked_movie() -> None:
    graph = build_graph(_ratings(), _movie_features())
    # Movie 30 connects to the user only through movie 10 (shared Comedy genre,
    # and co-rater user 2) — never through movie 20 (no shared bridge).
    exp = explain(graph, user_id=1, movie_id=30)

    assert exp is not None
    assert exp.liked_movie_id == 10
    assert exp.bridge_kind in {"genre", "user"}


def test_format_explanation_reads_naturally() -> None:
    graph = build_graph(_ratings(), _movie_features())
    exp = explain(graph, user_id=1, movie_id=30)
    titles = {10: "Toy Movie (1995)", 30: "Other Movie (1975)"}

    text = format_explanation(exp, titles, "Other Movie (1975)")
    assert text.startswith("because you liked Toy Movie (1995)")
    assert "Other Movie (1975)" in text


def test_no_explanation_for_unknown_user() -> None:
    graph = build_graph(_ratings(), _movie_features())
    assert explain(graph, user_id=999, movie_id=30) is None
