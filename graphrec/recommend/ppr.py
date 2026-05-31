"""Personalized PageRank recommender."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from graphrec.graph.build import user_node


@dataclass
class Recommendation:
    """A single scored recommendation."""

    movie_id: int
    score: float
    title: str | None = None


def recommend(
    graph: nx.Graph,
    user_id: int,
    k: int = 10,
    *,
    alpha: float = 0.85,
    titles: dict[int, str] | None = None,
) -> list[Recommendation]:
    """Recommend top-k movies for ``user_id`` via Personalized PageRank.

    The personalization vector restarts the walk at the user's node, so the walk
    explores ``user -> rated movies -> similar users -> their movies``. Movies the
    user has already rated are excluded.
    """
    source = user_node(user_id)
    if source not in graph:
        raise KeyError(f"unknown user_id: {user_id}")

    scores = nx.pagerank(
        graph, alpha=alpha, personalization={source: 1.0}, weight="weight"
    )

    seen = set(graph.neighbors(source))
    ranked = sorted(
        (
            (node, score)
            for node, score in scores.items()
            if graph.nodes[node].get("kind") == "movie" and node not in seen
        ),
        key=lambda item: item[1],
        reverse=True,
    )

    results: list[Recommendation] = []
    for node, score in ranked[:k]:
        movie_id = int(node.split(":", 1)[1])
        title = titles.get(movie_id) if titles else None
        results.append(Recommendation(movie_id=movie_id, score=score, title=title))
    return results
