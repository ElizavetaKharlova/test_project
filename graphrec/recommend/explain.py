"""Path-based explanations for recommendations.

A recommended movie R and one of the user's liked movies L are connected through
a common neighbor — which on this graph can only be a shared genre, a shared
release-era bucket, or another user who rated both. That 3-hop path
(user -> L -> bridge -> R) is a human-readable reason.

We pick the *most specific* such path: a bridge that connects few things (a niche
genre, a close-taste co-rater) is far more informative than a giant hub like
"Drama". Specificity is 1/degree of the bridge node (an IDF-style signal). We also
prefer content bridges over collaborative ones for readability, and only explain
via movies the user actually rated highly — so "because you liked X" is honest.
Scoring by raw edge-weight product would instead always pick collaborative paths,
since rating edges dominate feature edges numerically.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from graphrec.graph.build import movie_node, user_node

_BRIDGE_KINDS = ("genre", "year_bucket", "user")
_LIKE_THRESHOLD = 4.0


@dataclass
class Explanation:
    """The most specific explanatory path for a single recommendation."""

    liked_movie_id: int
    bridge_kind: str  # "genre" | "year_bucket" | "user"
    bridge_label: str  # e.g. "Crime", "1990s"; empty for a user bridge
    specificity: float  # 1 / degree(bridge); higher = rarer, more informative


def explain(graph: nx.Graph, user_id: int, movie_id: int) -> Explanation | None:
    """Return the most specific user -> liked -> bridge -> movie path, or None."""
    user = user_node(user_id)
    target = movie_node(movie_id)
    if user not in graph or target not in graph:
        return None

    target_neighbors = set(graph.neighbors(target))
    rated = sorted(
        (n for n in graph.neighbors(user) if graph.nodes[n].get("kind") == "movie"),
        key=lambda n: (-graph.edges[user, n]["weight"], n),
    )
    # Explain through genuinely-liked movies when any exist; else any rated movie.
    liked = [n for n in rated if graph.edges[user, n]["weight"] >= _LIKE_THRESHOLD]
    liked = liked or rated

    best: Explanation | None = None
    best_key: tuple[bool, float] | None = None
    for movie in liked:
        for bridge in sorted(set(graph.neighbors(movie)) & target_neighbors):
            kind = graph.nodes[bridge].get("kind")
            if kind not in _BRIDGE_KINDS:
                continue
            is_content = kind != "user"
            specificity = 1.0 / graph.degree(bridge)
            key = (is_content, specificity)
            if best_key is None or key > best_key:
                best_key = key
                label = bridge.split(":", 1)[1] if is_content else ""
                best = Explanation(
                    liked_movie_id=int(movie.split(":", 1)[1]),
                    bridge_kind=kind,
                    bridge_label=label,
                    specificity=specificity,
                )
    return best


def format_explanation(
    explanation: Explanation, titles: dict[int, str], target_title: str
) -> str:
    """Render an Explanation as a one-line reason string."""
    liked = titles.get(
        explanation.liked_movie_id, f"movie {explanation.liked_movie_id}"
    )
    if explanation.bridge_kind == "genre":
        bridge = explanation.bridge_label
    elif explanation.bridge_kind == "year_bucket":
        bridge = f"{explanation.bridge_label} films"
    else:
        bridge = "similar viewers"
    return f"because you liked {liked} → {bridge} → {target_title}"
