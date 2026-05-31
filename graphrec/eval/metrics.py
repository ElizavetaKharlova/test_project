"""Top-K ranking metrics.

All functions take a ranked list of recommended movie ids and the set of relevant
(held-out) movie ids for a single user. Aggregate by averaging over users.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np


def recall_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    """Fraction of a user's relevant items that appear in the top-k."""
    if not relevant:
        return 0.0
    hits = sum(1 for movie in recommended[:k] if movie in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    """Normalized discounted cumulative gain with binary relevance."""
    if not relevant:
        return 0.0
    dcg = sum(
        1.0 / np.log2(rank + 2)
        for rank, movie in enumerate(recommended[:k])
        if movie in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def coverage(recommended_items: Iterable[int], n_catalog_items: int) -> float:
    """Catalog coverage: share of the catalog that appears in any user's top-k.

    A proxy for aggregate diversity — low coverage means the system funnels every
    user toward the same few popular items.
    """
    if n_catalog_items == 0:
        return 0.0
    return len(set(recommended_items)) / n_catalog_items
