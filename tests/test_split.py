"""Tests for the temporal split."""

from __future__ import annotations

import pandas as pd

from graphrec.eval.split import temporal_split


def _ratings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "movie_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "rating": [5] * 10,
            "timestamp": list(range(10)),
        }
    )


def test_temporal_split_has_no_future_leak() -> None:
    train, test = temporal_split(_ratings(), test_frac=0.3)

    assert len(test) >= 1
    assert len(train) + len(test) == 10
    # every training interaction precedes every test interaction
    assert train["timestamp"].max() <= test["timestamp"].min()


def test_temporal_split_is_disjoint() -> None:
    train, test = temporal_split(_ratings(), test_frac=0.3)
    assert set(train.index).isdisjoint(test.index)
