"""Tests for optional MLflow tracking (skipped if mlflow isn't installed)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


def test_log_dataframe_records_one_run_per_row(tmp_path: Path) -> None:
    mlflow = pytest.importorskip("mlflow")
    from graphrec.eval import tracking

    table = pd.DataFrame(
        [
            {"Method": "A", "Recall@10": 0.1, "NDCG@10": 0.2, "Coverage": 0.3},
            {"Method": "B", "Recall@10": 0.4, "NDCG@10": 0.5, "Coverage": 0.6},
        ]
    )
    uri = f"sqlite:///{tmp_path}/mlflow.db"

    tracking.log_dataframe(
        "test-exp", table, {"k": 10, "seed": 42}, label_col="Method", tracking_uri=uri
    )

    mlflow.set_tracking_uri(uri)
    runs = mlflow.search_runs(experiment_names=["test-exp"])
    assert len(runs) == 2
    assert set(runs["tags.Method"]) == {"A", "B"}
    # '@' is sanitized out of metric keys
    assert "metrics.recall_at_10" in runs.columns
