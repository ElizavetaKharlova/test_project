"""Tests for ranking metrics."""

from __future__ import annotations

import math

from graphrec.eval.metrics import coverage, ndcg_at_k, recall_at_k


def test_recall_at_k() -> None:
    assert recall_at_k([1, 2, 3], {2, 5}, 3) == 0.5
    assert recall_at_k([1, 2, 3], {1, 2, 3}, 3) == 1.0
    assert recall_at_k([4, 5], {1}, 2) == 0.0
    assert recall_at_k([1], set(), 1) == 0.0  # no relevant -> 0, not a crash
    # only the first k entries count
    assert recall_at_k([9, 9, 2], {2}, 2) == 0.0


def test_ndcg_at_k() -> None:
    assert ndcg_at_k([1, 2, 3], {1, 2, 3}, 3) == 1.0
    # single relevant item at rank 2 (index 1): gain 1/log2(3), ideal 1/log2(2)=1
    assert math.isclose(ndcg_at_k([1, 2, 3], {2}, 3), 1.0 / math.log2(3), rel_tol=1e-9)
    assert ndcg_at_k([1, 2], set(), 2) == 0.0


def test_coverage() -> None:
    assert coverage([1, 1, 2], 4) == 0.5
    assert coverage([], 10) == 0.0
    assert coverage([1, 2, 3], 0) == 0.0
