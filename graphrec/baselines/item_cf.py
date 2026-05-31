"""Item-item collaborative filtering with cosine similarity."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from graphrec.baselines.utils import build_user_item_matrix, topk_from_scores


class ItemCFRecommender:
    """Score candidates by cosine similarity to the items a user already rated."""

    name = "Item-CF"

    def fit(self, train: pd.DataFrame) -> None:
        self.matrix, self.user_index, self.item_ids = build_user_item_matrix(train)
        # items x items cosine similarity over the user-rating vectors
        self.similarity = cosine_similarity(self.matrix.T, dense_output=True)

    def recommend(self, user_id: int, k: int, exclude: set[int]) -> list[int]:
        row = self.user_index.get(int(user_id))
        if row is None:
            return []
        user_vector = self.matrix[row].toarray().ravel()
        scores = self.similarity @ user_vector
        return topk_from_scores(scores, self.item_ids, exclude, k)
