"""Tests for the recommendation visualization (no network access)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from graphrec.graph.build import build_graph
from graphrec.recommend import ppr
from graphrec.viz.render import render_recommendations


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
        {"user_id": [1, 2], "age": [30, 20], "occupation": ["engineer", "student"]}
    )


def test_render_writes_standalone_html(tmp_path: Path) -> None:
    graph = build_graph(_ratings(), _movie_features(), _user_features())
    titles = {10: "A (1995)", 20: "B (1980)", 30: "C (1975)"}
    recs = ppr.recommend(graph, user_id=1, k=2, titles=titles)

    out = render_recommendations(graph, 1, recs, titles, tmp_path / "viz.html")

    assert out.exists()
    html = out.read_text()
    assert "User 1" in html  # the queried user node is rendered
    assert "C (1975)" in html  # the recommended movie is rendered
    # assets are embedded (works offline), not pulled from a CDN
    assert "vis-network" in html
