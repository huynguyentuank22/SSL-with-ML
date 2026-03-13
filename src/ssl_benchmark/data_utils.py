"""
data_utils.py
=============
Data loading and preprocessing utilities for the SSL benchmark.

Loads train/test parquet files, infers feature columns,
and infers task types (regression vs classification).
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_TARGET_COLS = ["duration", "avgpcon", "ec", "pclass"]

# Tasks that are classification (low cardinality integers)
CLASSIFICATION_TARGETS = {"pclass"}

# Extra columns to drop even if not in target list
DEFAULT_EXCLUDE_COLS: List[str] = ["month"]

REGRESSION_METRICS  = ["mae", "rmse", "mse", "r2"]
CLASSIFICATION_METRICS = ["accuracy", "f1_macro", "precision_macro", "recall_macro"]


# ---------------------------------------------------------------------------
# Task type inference
# ---------------------------------------------------------------------------
def infer_task_type(series: pd.Series, name: str) -> str:
    """Return 'classification' or 'regression' for a target column."""
    if name in CLASSIFICATION_TARGETS:
        return "classification"
    n_unique = series.nunique()
    if pd.api.types.is_integer_dtype(series) and n_unique <= 20:
        log.info(
            f"Task '{name}' has {n_unique} unique integer values — "
            "treating as classification."
        )
        return "classification"
    return "regression"


# ---------------------------------------------------------------------------
# Feature column detection
# ---------------------------------------------------------------------------
def get_feature_cols(
    df: pd.DataFrame,
    target_cols: List[str] = DEFAULT_TARGET_COLS,
    exclude_cols: Optional[List[str]] = None,
) -> List[str]:
    """Return feature column list, excluding targets and any extra cols."""
    drop = set(target_cols)
    if exclude_cols:
        drop.update(exclude_cols)
    for col in DEFAULT_EXCLUDE_COLS:
        drop.add(col)
    feat_cols = [c for c in df.columns if c not in drop]
    return feat_cols


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------
def load_data(
    data_dir: Path,
    target_cols: List[str] = DEFAULT_TARGET_COLS,
    exclude_cols: Optional[List[str]] = None,
    scale: bool = True,
    scaler_path: Optional[Path] = None,
) -> Dict:
    """
    Load train/test parquet files, return a structured dict with:
      - X_train, X_test  (numpy float32, optionally scaled)
      - y_train, y_test  per task
      - feature_cols
      - task_types
      - scaler
    """
    train_path = data_dir / "train.parquet"
    test_path  = data_dir / "test.parquet"

    log.info(f"Loading train data from: {train_path}")
    log.info(f"Loading test  data from: {test_path}")

    df_train = pd.read_parquet(train_path)
    df_test  = pd.read_parquet(test_path)

    feat_cols = get_feature_cols(df_train, target_cols, exclude_cols)
    log.info(f"Feature columns ({len(feat_cols)}): {feat_cols}")
    log.info(f"Train size: {len(df_train)} rows")
    log.info(f"Test  size: {len(df_test)} rows")

    X_train = df_train[feat_cols].values.astype(np.float32)
    X_test  = df_test[feat_cols].values.astype(np.float32)

    # Optional scaling (stored so embeddings use same stats)
    scaler = None
    if scale:
        if scaler_path and Path(scaler_path).exists():
            import joblib
            scaler = joblib.load(scaler_path)
            log.info(f"Loaded existing scaler from {scaler_path}")
        else:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train).astype(np.float32)
            log.info("Fitted new StandardScaler on training data.")
        X_test = scaler.transform(X_test).astype(np.float32)

    # Build per-task targets
    task_types: Dict[str, str] = {}
    y_train_dict: Dict[str, np.ndarray] = {}
    y_test_dict:  Dict[str, np.ndarray] = {}

    available_tasks = [t for t in target_cols if t in df_train.columns]
    for task in available_tasks:
        task_types[task] = infer_task_type(df_train[task], task)
        y_train_dict[task] = df_train[task].values
        y_test_dict[task]  = df_test[task].values
        log.info(
            f"  Task '{task}': type={task_types[task]}, "
            f"train={len(y_train_dict[task])}, test={len(y_test_dict[task])}"
        )

    return {
        "X_train":     X_train,
        "X_test":      X_test,
        "y_train":     y_train_dict,
        "y_test":      y_test_dict,
        "feature_cols": feat_cols,
        "task_types":  task_types,
        "scaler":      scaler,
        "df_train":    df_train,
        "df_test":     df_test,
    }
