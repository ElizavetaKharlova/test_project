"""Most-popular baseline: recommend globally popular items, minus what's seen."""

from __future__ import annotations

import pandas as pd


class PopularityRecommender:
    """Rank by training interaction count; identical for every user."""

    name = "Popularity"

    def fit(self, train: pd.DataFrame) -> None:
        counts = train.groupby("movie_id").size().sort_values(ascending=False)
        self._ranked = [int(movie) for movie in counts.index]

    def recommend(self, user_id: int, k: int, exclude: set[int]) -> list[int]:
        out: list[int] = []
        for movie in self._ranked:
            if movie in exclude:
                continue
            out.append(movie)
            if len(out) >= k:
                break
        return out
