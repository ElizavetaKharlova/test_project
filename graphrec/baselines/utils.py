"""Shared helpers for the matrix-based baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp


def build_user_item_matrix(
    train: pd.DataFrame,
) -> tuple[sp.csr_matrix, dict[int, int], np.ndarray]:
    """Build a sparse user-item rating matrix from training interactions.

    Returns ``(matrix, user_index, item_ids)`` where ``user_index`` maps a user id
    to its row, and ``item_ids[j]`` is the movie id of column ``j``.
    """
    users = train["user_id"].unique()
    items = train["movie_id"].unique()
    user_index = {int(u): i for i, u in enumerate(users)}
    item_index = {int(m): j for j, m in enumerate(items)}

    rows = train["user_id"].map(user_index).to_numpy()
    cols = train["movie_id"].map(item_index).to_numpy()
    vals = train["rating"].to_numpy(dtype=float)
    matrix = sp.csr_matrix((vals, (rows, cols)), shape=(len(users), len(items)))

    item_ids = np.array([int(m) for m in items])
    return matrix, user_index, item_ids


def topk_from_scores(
    scores: np.ndarray, item_ids: np.ndarray, exclude: set[int], k: int
) -> list[int]:
    """Return the top-k movie ids by score, skipping excluded (already-seen) ids."""
    out: list[int] = []
    for idx in np.argsort(scores)[::-1]:
        movie_id = int(item_ids[idx])
        if movie_id in exclude:
            continue
        out.append(movie_id)
        if len(out) >= k:
            break
    return out
