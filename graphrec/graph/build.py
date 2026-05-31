"""Graph construction from MovieLens interactions.

Nodes are typed via string-prefixed ids (``"user:42"``, ``"movie:7"``) and a
``kind`` attribute. Using prefixes keeps node namespaces disjoint and makes it
trivial to add feature node types (genre, occupation, ...) in later milestones
without colliding ids.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import pandas as pd


def user_node(user_id: int) -> str:
    """Return the graph node id for a user."""
    return f"user:{user_id}"


def movie_node(movie_id: int) -> str:
    """Return the graph node id for a movie."""
    return f"movie:{movie_id}"


def genre_node(name: str) -> str:
    """Return the graph node id for a genre."""
    return f"genre:{name}"


def occupation_node(name: str) -> str:
    """Return the graph node id for an occupation."""
    return f"occupation:{name}"


def age_bucket_node(label: str) -> str:
    """Return the graph node id for an age bucket."""
    return f"age_bucket:{label}"


def year_bucket_node(label: str) -> str:
    """Return the graph node id for a release-year bucket."""
    return f"year_bucket:{label}"


@dataclass(frozen=True)
class FeatureWeights:
    """Per-edge-type weights for feature edges (rating edges keep their rating).

    Defaults are uniform in M3; M4 exposes these as the steering "product lever".
    """

    genre: float = 1.0
    occupation: float = 1.0
    age: float = 1.0
    year: float = 1.0


def age_bucket_label(age: int) -> str:
    """Map a raw age to a coarse bucket label."""
    for upper, label in [
        (18, "<18"), (25, "18-24"), (35, "25-34"),
        (45, "35-44"), (50, "45-49"), (56, "50-55"),
    ]:  # fmt: skip
        if age < upper:
            return label
    return "56+"


def year_bucket_label(year: int) -> str:
    """Map a release year to its decade label, e.g. 1994 -> '1990s'."""
    return f"{int(year) // 10 * 10}s"


def build_bipartite_graph(ratings: pd.DataFrame) -> nx.Graph:
    """Build a weighted bipartite user-movie graph.

    Edge weight is the rating, so stronger preferences carry more random-walk
    probability during Personalized PageRank.
    """
    edges = pd.DataFrame(
        {
            "source": "user:" + ratings["user_id"].astype(str),
            "target": "movie:" + ratings["movie_id"].astype(str),
            "weight": ratings["rating"].astype(float),
        }
    )
    graph = nx.from_pandas_edgelist(edges, "source", "target", edge_attr="weight")

    kinds = {
        node: ("user" if node.startswith("user:") else "movie") for node in graph
    }
    nx.set_node_attributes(graph, kinds, "kind")
    nx.set_edge_attributes(graph, "rated", "relation")
    return graph


def build_graph(
    ratings: pd.DataFrame,
    movie_features: pd.DataFrame | None = None,
    user_features: pd.DataFrame | None = None,
    weights: FeatureWeights | None = None,
) -> nx.Graph:
    """Build the full heterogeneous graph: bipartite core plus feature nodes.

    Feature edges are additive — the bipartite graph is a strict subset — so the
    PPR recommender works unchanged; the feature nodes only add new paths.
    """
    weights = weights or FeatureWeights()
    graph = build_bipartite_graph(ratings)
    if movie_features is not None:
        _add_movie_feature_nodes(graph, movie_features, weights)
    if user_features is not None:
        _add_user_feature_nodes(graph, user_features, weights)
    return graph


def _add_movie_feature_nodes(
    graph: nx.Graph, movie_features: pd.DataFrame, weights: FeatureWeights
) -> None:
    for row in movie_features.itertuples(index=False):
        movie = movie_node(row.movie_id)
        if movie not in graph:  # only annotate movies that were actually rated
            continue
        for genre in row.genres:
            node = genre_node(genre)
            graph.add_node(node, kind="genre")
            graph.add_edge(movie, node, weight=weights.genre, relation="has_genre")
        if pd.notna(row.year):
            node = year_bucket_node(year_bucket_label(row.year))
            graph.add_node(node, kind="year_bucket")
            graph.add_edge(movie, node, weight=weights.year, relation="in_year_bucket")


def _add_user_feature_nodes(
    graph: nx.Graph, user_features: pd.DataFrame, weights: FeatureWeights
) -> None:
    for row in user_features.itertuples(index=False):
        user = user_node(row.user_id)
        if user not in graph:
            continue
        occupation = occupation_node(row.occupation)
        graph.add_node(occupation, kind="occupation")
        graph.add_edge(
            user, occupation, weight=weights.occupation, relation="has_occupation"
        )
        age = age_bucket_node(age_bucket_label(row.age))
        graph.add_node(age, kind="age_bucket")
        graph.add_edge(user, age, weight=weights.age, relation="in_age_bucket")
