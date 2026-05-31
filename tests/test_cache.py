"""Tests for the graph disk cache (no network access)."""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from graphrec.graph.build import build_bipartite_graph


def _toy_ratings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [1, 1, 2],
            "movie_id": [10, 20, 10],
            "rating": [5, 4, 5],
            "timestamp": [0, 0, 0],
        }
    )


def test_graph_pickle_roundtrip_preserves_structure(tmp_path: Path) -> None:
    """The cache relies on pickle preserving nodes, edges, weights, and kinds."""
    graph = build_bipartite_graph(_toy_ratings())

    path = tmp_path / "graph.pkl"
    with path.open("wb") as handle:
        pickle.dump(graph, handle)
    with path.open("rb") as handle:
        restored = pickle.load(handle)

    assert set(restored.nodes) == set(graph.nodes)
    assert set(restored.edges) == set(graph.edges)
    assert restored.nodes["user:1"]["kind"] == "user"
    assert restored.nodes["movie:10"]["kind"] == "movie"
    assert restored.edges["user:1", "movie:10"]["weight"] == 5.0
