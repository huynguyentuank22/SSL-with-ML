"""
downstream.py
=============
Downstream probing: train XGBoost / LightGBM / CatBoost / cuML KNN / cuML RF
on frozen embeddings and evaluate on test embeddings.

Supports:
  - Regression tasks:    duration, avgpcon, ec
  - Classification task: pclass
"""

import logging
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try importing the three boosting libraries (graceful degradation)
# ---------------------------------------------------------------------------
_HAS_XGB = False
_HAS_LGB = False
_HAS_CAT = False

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    log.warning("xgboost not installed — XGBoost downstream models will be skipped.")

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    log.warning("lightgbm not installed — LightGBM downstream models will be skipped.")

try:
    import catboost as cat
    _HAS_CAT = True
except ImportError:
    log.warning("catboost not installed — CatBoost downstream models will be skipped.")

# cuML (RAPIDS) — GPU-accelerated KNN and Random Forest
# Falls back gracefully if not available (e.g. no RAPIDS install)
_HAS_CUML = False
try:
    import cuml  # noqa: F401 — existence check
    _HAS_CUML = True
    log.info("cuML detected — KNN and RandomForest will run on GPU.")
except ImportError:
    log.warning("cuml not installed — KNN and RandomForest (cuML) will be skipped.")


# ===========================================================================
# Model factory
# ===========================================================================

def _has_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_downstream_models(task_type: str) -> Dict[str, Any]:
    """
    Return dict of { model_name: estimator_instance } for the given task_type.
    GPU is used when available.
    """
    use_gpu = _has_gpu()
    models: Dict[str, Any] = {}

    if _HAS_XGB:
        if task_type == "regression":
            xgb_params = dict(random_state=42, verbosity=0, n_estimators=300)
            if use_gpu:
                xgb_params["device"] = "cuda"
            models["xgboost"] = xgb.XGBRegressor(**xgb_params)
        else:
            xgb_params = dict(random_state=42, verbosity=0, n_estimators=300,
                              use_label_encoder=False, eval_metric="mlogloss")
            if use_gpu:
                xgb_params["device"] = "cuda"
            models["xgboost"] = xgb.XGBClassifier(**xgb_params)

    if _HAS_LGB:
        if task_type == "regression":
            lgb_params = dict(random_state=42, verbose=-1, n_estimators=300)
            if use_gpu:
                lgb_params["device"] = "gpu"
            models["lightgbm"] = lgb.LGBMRegressor(**lgb_params)
        else:
            lgb_params = dict(random_state=42, verbose=-1, n_estimators=300)
            if use_gpu:
                lgb_params["device"] = "gpu"
            models["lightgbm"] = lgb.LGBMClassifier(**lgb_params)

    if _HAS_CAT:
        if task_type == "regression":
            cat_params = dict(random_seed=42, verbose=0, iterations=300)
            if use_gpu:
                cat_params["task_type"] = "GPU"
            models["catboost"] = cat.CatBoostRegressor(**cat_params)
        else:
            cat_params = dict(random_seed=42, verbose=0, iterations=300)
            if use_gpu:
                cat_params["task_type"] = "GPU"
            models["catboost"] = cat.CatBoostClassifier(**cat_params)

    if _HAS_CUML:
        # cuML models always run on GPU; no device flag needed
        if task_type == "regression":
            from cuml.neighbors import KNeighborsRegressor
            from cuml.ensemble import RandomForestRegressor as cuRFRegressor
            models["cuml_knn"] = KNeighborsRegressor(n_neighbors=5)
            models["cuml_rf"]  = cuRFRegressor(n_estimators=300, random_state=42)
        else:
            from cuml.neighbors import KNeighborsClassifier
            from cuml.ensemble import RandomForestClassifier as cuRFClassifier
            models["cuml_knn"] = KNeighborsClassifier(n_neighbors=5)
            models["cuml_rf"]  = cuRFClassifier(n_estimators=300, random_state=42)

    return models


# ===========================================================================
# Metric helpers
# ===========================================================================

def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "mse":  float(mse),
        "r2":   float(r2_score(y_true, y_pred)),
    }


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    return {
        "accuracy":        float(accuracy_score(y_true, y_pred)),
        "f1_macro":        float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro":    float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


# ===========================================================================
# Main probing function
# ===========================================================================

def run_downstream_probe(
    *,
    emb_train: np.ndarray,
    emb_test:  np.ndarray,
    y_train:   np.ndarray,
    y_test:    np.ndarray,
    task_name:       str,
    task_type:       str,
    ssl_model_name:  str,
    arch_name:       str,
    layer_name:      str,
    output_dir:      Path,
) -> List[Dict]:
    """
    Train each downstream boosting model on (emb_train, y_train),
    predict on emb_test, compute metrics, save predictions.

    Returns a list of result dicts (one per downstream model).
    """
    models = get_downstream_models(task_type)
    if not models:
        log.warning("No downstream models available — skipping probe.")
        return []

    # LightGBM needs integer labels for classification (-1 not allowed)
    if task_type == "classification":
        uniq = np.unique(y_train)
        label_map  = {v: i for i, v in enumerate(uniq)}
        inv_map    = {i: v for v, i in label_map.items()}
        y_train_ml = np.array([label_map[v] for v in y_train])
        y_test_ml  = np.array([label_map[v] for v in y_test])
    else:
        y_train_ml = y_train.astype(np.float32)
        y_test_ml  = y_test.astype(np.float32)
        label_map  = None
        inv_map    = None

    results = []

    for model_name, estimator in models.items():
        try:
            log.info(
                f"    Fitting {model_name} | task={task_name} | "
                f"{ssl_model_name}/{arch_name}/{layer_name}"
            )
            estimator.fit(emb_train, y_train_ml)
            y_pred_ml = estimator.predict(emb_test)
            # Ensure y_pred_ml is a flat 1-D array of scalars (CatBoost can
            # return object arrays or 2-D arrays in rare cases)
            y_pred_ml = np.asarray(y_pred_ml).flatten()

            if task_type == "classification":
                # Cast each label to a plain Python int so it can be used as a
                # dict key — CatBoost may return numpy int arrays
                y_pred_orig = np.array([inv_map[int(v)] for v in y_pred_ml])
                y_true_orig = y_test     # original labels
                metrics = compute_classification_metrics(y_true_orig, y_pred_orig)

                # Try to get class probabilities
                try:
                    y_proba = estimator.predict_proba(emb_test)
                except Exception:
                    y_proba = None
            else:
                y_pred_orig = y_pred_ml
                y_true_orig = y_test
                metrics = compute_regression_metrics(y_true_orig, y_pred_orig)
                y_proba = None

            # ── Save predictions ──────────────────────────────────────────
            pred_dir = (output_dir / "predictions" /
                        task_name / ssl_model_name / arch_name /
                        layer_name)
            pred_dir.mkdir(parents=True, exist_ok=True)
            pred_path = pred_dir / f"{model_name}_predictions.csv"

            pred_df = pd.DataFrame({
                "sample_index":       np.arange(len(y_true_orig)),
                "task_name":          task_name,
                "ssl_model_name":     ssl_model_name,
                "architecture_name":  arch_name,
                "layer_name":         layer_name,
                "downstream_model":   model_name,
                "y_true":             y_true_orig,
                "y_pred":             y_pred_orig,
            })
            if y_proba is not None and y_proba.ndim == 2:
                for c in range(y_proba.shape[1]):
                    pred_df[f"y_proba_class_{c}"] = y_proba[:, c]
            pred_df.to_csv(pred_path, index=False)

            # ── Compile result row ────────────────────────────────────────
            row = {
                "task_name":           task_name,
                "task_type":           task_type,
                "ssl_model_name":      ssl_model_name,
                "architecture_name":   arch_name,
                "layer_name":          layer_name,
                "downstream_model":    model_name,
                "train_size":          len(y_train),
                "test_size":           len(y_test),
                "embedding_dim":       emb_train.shape[1],
            }
            row.update(metrics)
            results.append(row)

            metric_str = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            log.info(f"      → {metric_str}")

        except Exception as e:
            log.error(
                f"    ERROR in {model_name} | {task_name}/{ssl_model_name}/"
                f"{arch_name}/{layer_name}: {e}\n{traceback.format_exc()}"
            )

    return results


# ===========================================================================
# Summarise results
# ===========================================================================

def build_summary(all_results: List[Dict], output_dir: Path) -> pd.DataFrame:
    """Save full summary CSV and subset CSVs (best by task / SSL / layer)."""
    if not all_results:
        log.warning("No results to summarise.")
        return pd.DataFrame()

    df = pd.DataFrame(all_results)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "summary_results.csv", index=False)
    log.info(f"Summary saved → {output_dir / 'summary_results.csv'}")

    # ── Best by task ──────────────────────────────────────────────────────
    reg_mask  = df["task_type"] == "regression"
    cls_mask  = df["task_type"] == "classification"

    best_rows = []
    if reg_mask.any():
        best_rows.append(
            df[reg_mask].sort_values("r2", ascending=False).groupby("task_name").first()
        )
    if cls_mask.any():
        best_rows.append(
            df[cls_mask].sort_values("f1_macro", ascending=False).groupby("task_name").first()
        )
    if best_rows:
        best_task = pd.concat(best_rows)
        best_task.to_csv(output_dir / "best_by_task.csv")
        log.info(f"Best-by-task saved → {output_dir / 'best_by_task.csv'}")

    # ── Best by SSL model ─────────────────────────────────────────────────
    rows_ssl = []
    for ttype, metric in [("regression", "r2"), ("classification", "f1_macro")]:
        mask = df["task_type"] == ttype
        if mask.any():
            rows_ssl.append(
                df[mask].sort_values(metric, ascending=False)
                .groupby(["task_name", "ssl_model_name"]).first()
            )
    if rows_ssl:
        best_ssl = pd.concat(rows_ssl)
        best_ssl.to_csv(output_dir / "best_by_ssl_model.csv")
        log.info(f"Best-by-SSL saved → {output_dir / 'best_by_ssl_model.csv'}")

    # ── Best by layer ─────────────────────────────────────────────────────
    rows_layer = []
    for ttype, metric in [("regression", "r2"), ("classification", "f1_macro")]:
        mask = df["task_type"] == ttype
        if mask.any():
            rows_layer.append(
                df[mask].sort_values(metric, ascending=False)
                .groupby(["task_name", "ssl_model_name", "architecture_name", "layer_name"])
                .first()
            )
    if rows_layer:
        best_layer = pd.concat(rows_layer)
        best_layer.to_csv(output_dir / "best_by_layer.csv")
        log.info(f"Best-by-layer saved → {output_dir / 'best_by_layer.csv'}")

    return df
