"""Matrix factorization baseline via truncated SVD.

TruncatedSVD on the sparse rating matrix is the dependency-light, CPU-only stand-in
for a production implicit-feedback model (e.g. ``implicit`` ALS). See PRD §15.
"""

from __future__ import annotations

import pandas as pd
from sklearn.decomposition import TruncatedSVD

from graphrec.baselines.utils import build_user_item_matrix, topk_from_scores


class SVDRecommender:
    """Low-rank reconstruction of the user-item matrix; score = user·item factors."""

    name = "MF (SVD)"

    def __init__(self, n_components: int = 50, seed: int = 42) -> None:
        self.n_components = n_components
        self.seed = seed

    def fit(self, train: pd.DataFrame) -> None:
        self.matrix, self.user_index, self.item_ids = build_user_item_matrix(train)
        n_comp = min(self.n_components, min(self.matrix.shape) - 1)
        self.svd = TruncatedSVD(n_components=n_comp, random_state=self.seed)
        self.user_factors = self.svd.fit_transform(self.matrix)
        self.item_factors = self.svd.components_.T

    def recommend(self, user_id: int, k: int, exclude: set[int]) -> list[int]:
        row = self.user_index.get(int(user_id))
        if row is None:
            return []
        scores = self.user_factors[row] @ self.item_factors.T
        return topk_from_scores(scores, self.item_ids, exclude, k)
