"""Optional MLflow experiment tracking for eval and ablation results.

MLflow is an optional dependency (the 'tracking' extra), imported lazily so the
core package stays dependency-light. Install with `uv sync --extra tracking`.

Each row of an eval/ablation table becomes one MLflow run (one config = one run),
which makes MLflow's compare/plot views useful across methods or recency sweeps.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd

# SQLite backend; the legacy ./mlruns file store is deprecated as of Feb 2026.
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"


def _import_mlflow():
    try:
        import mlflow
    except ImportError as exc:  # pragma: no cover - only without the extra
        raise RuntimeError(
            "MLflow is not installed. Install the tracking extra: "
            "uv sync --extra tracking"
        ) from exc
    return mlflow


def _metric_key(column: str) -> str:
    """Sanitize a column name into a valid MLflow metric key (no '@')."""
    return column.replace("@", "_at_").replace(" ", "_").lower()


def log_dataframe(
    experiment: str,
    table: pd.DataFrame,
    base_params: Mapping[str, Any],
    *,
    label_col: str,
    param_cols: Iterable[str] = (),
    tracking_uri: str | None = None,
) -> str:
    """Log each row of a results table as a separate MLflow run.

    ``label_col`` distinguishes runs (e.g. "Method" or "recency_weight").
    ``param_cols`` columns are logged as params rather than metrics. Returns the
    active tracking URI (e.g. the local ``./mlruns`` store).
    """
    mlflow = _import_mlflow()
    mlflow.set_tracking_uri(tracking_uri or DEFAULT_TRACKING_URI)
    mlflow.set_experiment(experiment)

    param_columns = set(param_cols)
    for row in table.to_dict(orient="records"):
        label = row[label_col]
        run_name = str(label) if label_col == "Method" else f"{label_col}={label}"
        with mlflow.start_run(run_name=run_name):
            params = dict(base_params)
            params.update({col: row[col] for col in param_columns})
            mlflow.log_params(params)
            mlflow.set_tag(label_col, label)
            metrics = {
                _metric_key(col): float(val)
                for col, val in row.items()
                if col != label_col
                and col not in param_columns
                and isinstance(val, (int, float))
                and not isinstance(val, bool)
            }
            mlflow.log_metrics(metrics)
    return mlflow.get_tracking_uri()
