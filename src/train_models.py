"""
train_models.py
===============
Train RandomForest, XGBoost, LightGBM, CatBoost, cuML KNN, cuML RF
với default hyperparams.
GPU được bật tự động nếu model hỗ trợ; fallback về CPU nếu không có CUDA.

Targets : duration, avgpcon, pclass, ec
Input   : data/train.parquet, data/test.parquet
Output  : data/results/
            metrics.csv            — MAE, RMSE, MSE, R2 cho mỗi (model, target)
            predictions.parquet    — y_test + y_pred của tất cả (model, target)
            test_with_preds.parquet — full test df + tất cả cột dự đoán
"""

import os
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR   = Path(r"/kaggle/input/datasets/huy281204/f-data-cleaned")
OUTPUT_DIR = Path("/kaggle/working/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLS = ["duration", "avgpcon", "pclass", "ec"]
TASK_TYPES = {
    "duration": "regression",
    "avgpcon": "regression",
    "pclass": "classification",
    "ec": "classification"
}
RANDOM_SEED = 42


# ── Load data ────────────────────────────────────────────────────────────────
print("=" * 60)
print("Loading data ...")
df_train = pd.read_parquet(DATA_DIR / "train.parquet")
df_test  = pd.read_parquet(DATA_DIR / "test.parquet")

DROP_COLS = TARGET_COLS + ["month"]
feat_cols = [c for c in df_train.columns if c not in DROP_COLS]

X_train = df_train[feat_cols].values.astype(np.float32)
X_test  = df_test[feat_cols].values.astype(np.float32)

print(f"X_train : {X_train.shape}")
print(f"X_test  : {X_test.shape}")
print(f"Features: {len(feat_cols)} cols  |  {feat_cols[:5]} ...")
print("=" * 60)


# ── Helpers ──────────────────────────────────────────────────────────────────
def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                model_name: str, target: str) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "model" : model_name,
        "target": target,
        "type"  : "regression",
        "MAE"   : mean_absolute_error(y_true, y_pred),
        "RMSE"  : float(np.sqrt(mse)),
        "MSE"   : float(mse),
        "R2"    : r2_score(y_true, y_pred),
        "accuracy": None,
        "f1": None,
    }


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                   model_name: str, target: str) -> dict:
    return {
        "model" : model_name,
        "target": target,
        "type"  : "classification",
        "MAE"   : None,
        "RMSE"  : None,
        "MSE"   : None,
        "R2"    : None,
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average='binary'),
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    model_name: str, target: str) -> dict:
    task_type = TASK_TYPES[target]
    if task_type == "regression":
        return compute_regression_metrics(y_true, y_pred, model_name, target)
    else:
        return compute_classification_metrics(y_true, y_pred, model_name, target)


def make_xgb(gpu: bool, task_type: str = "regression"):
    import xgboost as xgb
    if task_type == "classification":
        if gpu:
            return xgb.XGBClassifier(
                tree_method="hist", device="cuda",
                random_state=RANDOM_SEED, n_jobs=-1
            )
        return xgb.XGBClassifier(
            tree_method="hist",
            random_state=RANDOM_SEED, n_jobs=-1
        )
    else:
        if gpu:
            return xgb.XGBRegressor(
                tree_method="hist", device="cuda",
                random_state=RANDOM_SEED, n_jobs=-1
            )
        return xgb.XGBRegressor(
            tree_method="hist",
            random_state=RANDOM_SEED, n_jobs=-1
        )


def make_lgb(gpu: bool, task_type: str = "regression"):
    import lightgbm as lgb
    if task_type == "classification":
        if gpu:
            return lgb.LGBMClassifier(
                device="gpu", random_state=RANDOM_SEED,
                n_jobs=-1, verbose=-1
            )
        return lgb.LGBMClassifier(
            random_state=RANDOM_SEED, n_jobs=-1, verbose=-1
        )
    else:
        if gpu:
            return lgb.LGBMRegressor(
                device="gpu", random_state=RANDOM_SEED,
                n_jobs=-1, verbose=-1
            )
        return lgb.LGBMRegressor(
            random_state=RANDOM_SEED, n_jobs=-1, verbose=-1
        )


def make_cat(gpu: bool, task_type: str = "regression"):
    from catboost import CatBoostRegressor, CatBoostClassifier
    if task_type == "classification":
        if gpu:
            return CatBoostClassifier(
                task_type="GPU", random_seed=RANDOM_SEED, verbose=0
            )
        return CatBoostClassifier(
            random_seed=RANDOM_SEED, verbose=0
        )
    else:
        if gpu:
            return CatBoostRegressor(
                task_type="GPU", random_seed=RANDOM_SEED, verbose=0
            )
        return CatBoostRegressor(
            random_seed=RANDOM_SEED, verbose=0
        )


def make_cuml_knn(gpu: bool, task_type: str = "regression"):
    """cuML KNN — always runs on GPU; `gpu` param kept for interface consistency."""
    if task_type == "classification":
        from cuml.neighbors import KNeighborsClassifier
        return KNeighborsClassifier(n_neighbors=5)
    else:
        from cuml.neighbors import KNeighborsRegressor
        return KNeighborsRegressor(n_neighbors=5)


def make_cuml_rf(gpu: bool, task_type: str = "regression"):
    """cuML Random Forest — always runs on GPU; `gpu` param kept for interface consistency."""
    if task_type == "classification":
        from cuml.ensemble import RandomForestClassifier as cuRFC
        return cuRFC(n_estimators=300, random_state=RANDOM_SEED)
    else:
        from cuml.ensemble import RandomForestRegressor as cuRFR
        return cuRFR(n_estimators=300, random_state=RANDOM_SEED)


def fit_with_gpu_fallback(name: str, make_fn, X_tr, y_tr, task_type: str):
    """Thử tạo model với GPU; nếu lỗi thì fallback CPU.
    cuML models không có CPU fallback — nếu lỗi thì raise ngay.
    """
    is_cuml = name.startswith("cuML")
    candidates = [True] if is_cuml else [True, False]
    for use_gpu in candidates:
        try:
            m = make_fn(use_gpu, task_type)
            m.fit(X_tr, y_tr)
            device = "GPU" if (use_gpu or is_cuml) else "CPU"
            return m, device
        except Exception as e:
            if use_gpu and not is_cuml:
                print(f"    [GPU fallback] {name}: {e}")
            else:
                raise
    return None, None   # unreachable


def build_model_specs():
    """Trả về list (name, make_fn)."""
    specs = []
    try:
        import xgboost  # noqa: F401
        specs.append(("XGBoost", make_xgb))
    except ImportError:
        print("[SKIP] xgboost not installed.")
    try:
        import lightgbm  # noqa: F401
        specs.append(("LightGBM", make_lgb))
    except ImportError:
        print("[SKIP] lightgbm not installed.")
    try:
        import catboost  # noqa: F401
        specs.append(("CatBoost", make_cat))
    except ImportError:
        print("[SKIP] catboost not installed.")
    try:
        import cuml  # noqa: F401
        specs.append(("cuML_KNN", make_cuml_knn))
        specs.append(("cuML_RF",  make_cuml_rf))
    except ImportError:
        print("[SKIP] cuml not installed — KNN and RandomForest (GPU) will be skipped.")
    return specs


# ── Train & evaluate ─────────────────────────────────────────────────────────
model_specs = build_model_specs()

all_metrics: list[dict] = []
all_preds:   list[dict] = []   # long-format: index, target, model, y_true, y_pred
pred_wide = df_test[TARGET_COLS].copy().reset_index(drop=True)  # wide: 1 col per (target, model)

for target in TARGET_COLS:
    task_type = TASK_TYPES[target]
    
    # For classification, use int dtype; for regression, use float32
    if task_type == "classification":
        y_train = df_train[target].values.astype(np.int32)
        y_test  = df_test[target].values.astype(np.int32)
    else:
        y_train = df_train[target].values.astype(np.float32)
        y_test  = df_test[target].values.astype(np.float32)

    print(f"\n{'─'*60}")
    print(f"Target: {target} ({task_type})")

    for name, make_fn in model_specs:
        print(f"  [{name}]", end=" ", flush=True)
        try:
            model, device = fit_with_gpu_fallback(name, make_fn, X_train, y_train, task_type)

            y_pred = model.predict(X_test)
            if task_type == "classification":
                y_pred = y_pred.astype(np.int32)
            else:
                y_pred = y_pred.astype(np.float32)
                
            m = compute_metrics(y_test, y_pred, name, target)
            m["device"] = device
            all_metrics.append(m)

            # Long-format predictions
            for i, (yt, yp) in enumerate(zip(y_test, y_pred)):
                all_preds.append({"idx": i, "target": target,
                                  "model": name, "y_true": float(yt), "y_pred": float(yp)})

            # Wide-format column for test_with_preds
            pred_wide[f"{target}_{name}"] = y_pred

            # Print metrics based on task type
            if task_type == "regression":
                print(
                    f"({device})  "
                    f"MAE={m['MAE']:.4f}  "
                    f"RMSE={m['RMSE']:.4f}  "
                    f"MSE={m['MSE']:.4f}  "
                    f"R2={m['R2']:.4f}"
                )
            else:  # classification
                print(
                    f"({device})  "
                    f"Accuracy={m['accuracy']:.4f}  "
                    f"F1={m['f1']:.4f}"
                )
        except Exception as exc:
            print(f"FAILED — {exc}")


# ── Save results ──────────────────────────────────────────────────────────────
print(f"\n{'='*60}")

metrics_df = pd.DataFrame(all_metrics)
metrics_path = OUTPUT_DIR / "metrics.csv"
metrics_df.to_csv(metrics_path, index=False)
print(f"Saved metrics          -> {metrics_path}")

# predictions.parquet — long format: mỗi hàng = 1 (sample, target, model)
# Dễ dùng cho plotting, groupby, so sánh model
pred_long = pd.DataFrame(all_preds)
pred_path = OUTPUT_DIR / "predictions.parquet"
pred_long.to_parquet(pred_path, index=False)
print(f"Saved predictions      -> {pred_path}  (long format: idx/target/model/y_true/y_pred)")

# test_with_preds.parquet — wide format: full test features + all prediction columns
# Dễ dùng để inspect từng sample, debug, feature analysis
full_test_df = pd.concat(
    [df_test.reset_index(drop=True),
     pred_wide.drop(columns=TARGET_COLS)],   # bỏ y_true vì đã có trong df_test
    axis=1
)
full_path = OUTPUT_DIR / "test_with_preds.parquet"
full_test_df.to_parquet(full_path, index=False)
print(f"Saved test_with_preds  -> {full_path}  (wide format: features + pred cols)")

print(f"\n{'='*60}")
print("=== METRICS SUMMARY ===")
print(
    metrics_df
    .sort_values(["target", "MAE"])
    .to_string(index=False)
)
