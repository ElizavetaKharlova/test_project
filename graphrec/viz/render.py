"""Render the recommendation subgraph as a standalone interactive HTML file.

The full 100k-edge graph is far too large to draw, so we render only the story
behind a single user's recommendations: the queried user, the recommended movies,
and the liked-movie -> bridge -> recommendation explanation paths. The HTML embeds
its assets inline so it is portable and works offline.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import networkx as nx
from pyvis.network import Network

from graphrec.graph.build import movie_node, user_node
from graphrec.recommend.explain import explain
from graphrec.recommend.ppr import Recommendation

# role -> (color, shape, size)
_STYLE = {
    "query_user": ("#f4b400", "star", 30),
    "recommended": ("#0f9d58", "dot", 22),
    "liked": ("#4285f4", "dot", 18),
    "genre": ("#ff6d00", "box", 16),
    "year_bucket": ("#9c27b0", "box", 16),
    "user": ("#00acc1", "triangle", 16),
}


def render_recommendations(
    graph: nx.Graph,
    user_id: int,
    recs: Sequence[Recommendation],
    titles: dict[int, str],
    output: str | Path,
) -> Path:
    """Write an interactive HTML visualization of a user's recommendations."""
    net = Network(
        height="760px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#222222",
        directed=False,
        cdn_resources="in_line",
    )
    net.barnes_hut(spring_length=160)

    nodes: set[str] = set()
    edges: set[frozenset[str]] = set()

    def add_node(node_id: str, role: str, label: str) -> None:
        if node_id in nodes:
            return
        color, shape, size = _STYLE[role]
        net.add_node(
            node_id, label=label, title=role, color=color, shape=shape, size=size
        )
        nodes.add(node_id)

    def add_edge(a: str, b: str, label: str, dashed: bool = False) -> None:
        key = frozenset((a, b))
        if key in edges:
            return
        net.add_edge(a, b, title=label, dashes=dashed)
        edges.add(key)

    user = user_node(user_id)
    add_node(user, "query_user", f"User {user_id}")

    for rec in recs:
        target = movie_node(rec.movie_id)
        add_node(target, "recommended", rec.title or f"movie {rec.movie_id}")

        reason = explain(graph, user_id, rec.movie_id)
        if reason is None:
            add_edge(user, target, "recommended", dashed=True)
            continue

        liked = movie_node(reason.liked_movie_id)
        bridge = reason.bridge_node
        bridge_label = (
            f"User {bridge.split(':', 1)[1]}"
            if reason.bridge_kind == "user"
            else reason.bridge_label
        )
        add_node(liked, "liked", titles.get(reason.liked_movie_id, liked))
        add_node(bridge, reason.bridge_kind, bridge_label)

        add_edge(user, liked, graph.edges[user, liked]["relation"])
        add_edge(liked, bridge, graph.edges[liked, bridge]["relation"])
        add_edge(bridge, target, graph.edges[bridge, target]["relation"])

    output = Path(output)
    net.save_graph(str(output))
    return output
