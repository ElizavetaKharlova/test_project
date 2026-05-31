"""Evaluation: temporal split, metrics, and the comparison harness."""

from graphrec.eval.harness import evaluate
from graphrec.eval.metrics import coverage, ndcg_at_k, recall_at_k
from graphrec.eval.split import temporal_split

__all__ = ["coverage", "evaluate", "ndcg_at_k", "recall_at_k", "temporal_split"]
