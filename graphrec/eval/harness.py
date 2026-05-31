"""Comparison harness: fit every method on the train split and score it on test.

Produces a tidy DataFrame with one row per method and columns for Recall@K,
NDCG@K, catalog coverage, and the average release year of recommendations (a
read-out for the recency lever). PageRank is the per-user bottleneck, so
evaluation runs on a reproducible sample of users by default (``n_users``); pass
0 for all.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from graphrec.baselines.item_cf import ItemCFRecommender
from graphrec.baselines.mf import SVDRecommender
from graphrec.baselines.popularity import PopularityRecommender
from graphrec.eval.metrics import coverage, ndcg_at_k, recall_at_k
from graphrec.eval.split import temporal_split
from graphrec.recommend.graph_recommender import GraphPPRRecommender


def _items_by_user(frame: pd.DataFrame) -> dict[int, set[int]]:
    grouped = frame.groupby("user_id")["movie_id"]
    return {int(user): {int(m) for m in movies} for user, movies in grouped}


def _sample_eval_users(
    train_items: dict[int, set[int]],
    test_items: dict[int, set[int]],
    n_users: int,
    seed: int,
) -> list[int]:
    # Only users with a training profile AND held-out positives are evaluable,
    # and the same user set is scored for every method (fair comparison).
    eligible = sorted(user for user in test_items if user in train_items)
    rng = np.random.default_rng(seed)
    if n_users and len(eligible) > n_users:
        return [int(u) for u in rng.choice(eligible, size=n_users, replace=False)]
    return eligible


def _year_map(movie_features: pd.DataFrame | None) -> dict[int, float] | None:
    if movie_features is None:
        return None
    return {
        int(m): (np.nan if pd.isna(y) else float(y))
        for m, y in zip(movie_features["movie_id"], movie_features["year"])
    }


def _score(
    rec,
    users: Sequence[int],
    train_items: dict[int, set[int]],
    test_items: dict[int, set[int]],
    k: int,
    n_catalog: int,
    year_map: dict[int, float] | None,
) -> dict[str, float]:
    recalls, ndcgs, recommended_union, years = [], [], set(), []
    for user in users:
        relevant = test_items[user]
        topk = rec.recommend(user, k, train_items.get(user, set()))
        recalls.append(recall_at_k(topk, relevant, k))
        ndcgs.append(ndcg_at_k(topk, relevant, k))
        recommended_union.update(topk[:k])
        if year_map is not None:
            years.extend(
                year_map[m] for m in topk[:k] if not np.isnan(year_map.get(m, np.nan))
            )
    metrics = {
        f"Recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"NDCG@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "Coverage": coverage(recommended_union, n_catalog),
    }
    if year_map is not None:
        metrics["AvgYear"] = float(np.mean(years)) if years else 0.0
    return metrics


def evaluate(
    ratings: pd.DataFrame,
    *,
    movie_features: pd.DataFrame | None = None,
    user_features: pd.DataFrame | None = None,
    k: int = 10,
    test_frac: float = 0.2,
    n_users: int = 100,
    seed: int = 42,
    alpha: float = 0.85,
) -> pd.DataFrame:
    """Evaluate Graph PPR against baselines on a temporal split.

    When ``movie_features``/``user_features`` are provided, a second graph row
    using the full feature schema is added, and an AvgYear column is reported.
    """
    train, test = temporal_split(ratings, test_frac=test_frac)
    train_items = _items_by_user(train)
    test_items = _items_by_user(test)
    users = _sample_eval_users(train_items, test_items, n_users, seed)
    n_catalog = train["movie_id"].nunique()
    year_map = _year_map(movie_features)

    recommenders = [
        PopularityRecommender(),
        ItemCFRecommender(),
        SVDRecommender(seed=seed),
        GraphPPRRecommender(alpha=alpha),
    ]
    if movie_features is not None or user_features is not None:
        recommenders.append(
            GraphPPRRecommender(
                alpha=alpha,
                movie_features=movie_features,
                user_features=user_features,
            )
        )

    rows = []
    for rec in recommenders:
        rec.fit(train)
        metrics = _score(rec, users, train_items, test_items, k, n_catalog, year_map)
        rows.append({"Method": rec.name, **metrics})
    return pd.DataFrame(rows)


def recency_ablation(
    ratings: pd.DataFrame,
    movie_features: pd.DataFrame,
    user_features: pd.DataFrame,
    *,
    recency_weights: Sequence[float],
    k: int = 10,
    test_frac: float = 0.2,
    n_users: int = 100,
    seed: int = 42,
    alpha: float = 0.85,
) -> pd.DataFrame:
    """Sweep the recency weight on the feature-graph PPR and report the tradeoff.

    Shows how steering toward newer films (rising AvgYear) trades off against the
    accuracy/coverage metrics.
    """
    train, test = temporal_split(ratings, test_frac=test_frac)
    train_items = _items_by_user(train)
    test_items = _items_by_user(test)
    users = _sample_eval_users(train_items, test_items, n_users, seed)
    n_catalog = train["movie_id"].nunique()
    year_map = _year_map(movie_features)

    rows = []
    for recency_weight in recency_weights:
        rec = GraphPPRRecommender(
            alpha=alpha,
            movie_features=movie_features,
            user_features=user_features,
            recency_weight=recency_weight,
        )
        rec.fit(train)
        metrics = _score(rec, users, train_items, test_items, k, n_catalog, year_map)
        rows.append({"recency_weight": float(recency_weight), **metrics})
    return pd.DataFrame(rows)
