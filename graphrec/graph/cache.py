"""Disk cache for the constructed graph.

Building the graph from the raw interaction files takes ~0.5s. Since each CLI
invocation is a fresh process, nothing persists in memory between calls, so we
cache the built graph to disk and reload it on subsequent calls.

This mirrors the production offline/serving split: the graph is built once
offline; serving reuses it. Note this caches the *graph structure* only — PageRank
still runs per request, rooted at the queried user (see PRD §15 for the heavier
"precompute candidate generation" optimization we deliberately skip here).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import networkx as nx

from graphrec.data.loader import (
    DEFAULT_CACHE,
    load_movie_features,
    load_ratings,
    load_user_features,
)
from graphrec.graph.build import build_graph

# Bump when graph construction changes so stale caches are ignored.
GRAPH_VERSION = "feature-v1"


def _cache_path(cache_dir: Path) -> Path:
    return cache_dir / f"graph_{GRAPH_VERSION}.pkl"


def get_graph(*, cache_dir: Path | None = None, rebuild: bool = False) -> nx.Graph:
    """Return the bipartite graph, loading from disk cache when available.

    Pass ``rebuild=True`` to force reconstruction (e.g. after the data changes).
    """
    cache_dir = cache_dir or DEFAULT_CACHE
    path = _cache_path(cache_dir)

    if path.exists() and not rebuild:
        with path.open("rb") as handle:
            return pickle.load(handle)  # noqa: S301 - trusted local cache

    graph = build_graph(
        load_ratings(cache_dir),
        load_movie_features(cache_dir),
        load_user_features(cache_dir),
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(graph, handle)
    return graph
