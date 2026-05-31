"""Temporal train/test split.

A global timestamp cutoff: the earliest ``1 - test_frac`` of interactions (by time)
are training, the most recent ``test_frac`` are test. This mirrors production —
we only ever train on the past and predict the future — and avoids the optimistic
leakage of a random split.
"""

from __future__ import annotations

import pandas as pd


def temporal_split(
    ratings: pd.DataFrame, *, test_frac: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split interactions by a global timestamp quantile.

    Returns ``(train, test)`` where every train timestamp precedes every test
    timestamp.
    """
    cutoff = ratings["timestamp"].quantile(1.0 - test_frac)
    train = ratings[ratings["timestamp"] <= cutoff].copy()
    test = ratings[ratings["timestamp"] > cutoff].copy()
    return train, test
