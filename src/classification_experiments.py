import glob
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("/kaggle/working")
TRAIN_FILENAME = "jobs_labeled_train_scaled.csv"
TEST_FILENAME = "jobs_labeled_test_scaled.csv"
TASKS = ["model", "model_group", "base_model"]
TARGET_COLUMNS = set(TASKS)
MODEL_NAMES = ["xgboost", "catboost", "knn_cuml", "random_forest_cuml"]

MANUAL_CATEGORY_COLUMNS = {
    "partition_db",
    "partition_gaia",
    "partition_normal",
    "partition_test",
    "partition_xeon-p8",
    "constraint_\\n",
    "constraint_opteron",
    "constraint_opteron&6274",
    "constraint_xeon-e5",
    "constraint_xeon-g6",
    "constraint_xeon-g6&6248",
    "constraint_xeon-p8",
    "flag_12",
    "flag_2",
    "flag_4",
    "flag_8",
    "job_type_llmapreduce:map",
    "job_type_llsub:batch",
    "job_type_llsub:interactive",
    "job_type_other",
}

try:
    from xgboost import XGBClassifier
except Exception as e:
    XGBClassifier = None
    XGB_IMPORT_ERROR = str(e)
else:
    XGB_IMPORT_ERROR = ""

try:
    from catboost import CatBoostClassifier
except Exception as e:
    CatBoostClassifier = None
    CATBOOST_IMPORT_ERROR = str(e)
else:
    CATBOOST_IMPORT_ERROR = ""

try:
    from cuml.neighbors import KNeighborsClassifier as cuKNeighborsClassifier
    from cuml.ensemble import RandomForestClassifier as cuRandomForestClassifier
except Exception as e:
    cuKNeighborsClassifier = None
    cuRandomForestClassifier = None
    CUML_IMPORT_ERROR = str(e)
else:
    CUML_IMPORT_ERROR = ""


def log(message: str) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def resolve_input_paths(
    train_path: str = "", test_path: str = ""
) -> Tuple[str, str]:
    if train_path and test_path:
        if Path(train_path).exists() and Path(test_path).exists():
            return train_path, test_path

    train_candidates = glob.glob(f"/kaggle/input/**/{TRAIN_FILENAME}", recursive=True)
    test_candidates = glob.glob(f"/kaggle/input/**/{TEST_FILENAME}", recursive=True)

    if not train_candidates:
        raise FileNotFoundError(f"Khong tim thay {TRAIN_FILENAME} trong /kaggle/input")
    if not test_candidates:
        raise FileNotFoundError(f"Khong tim thay {TEST_FILENAME} trong /kaggle/input")

    train_resolved = sorted(train_candidates)[0]
    test_resolved = sorted(test_candidates)[0]
    return train_resolved, test_resolved


def drop_noise_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    to_drop = []
    for col in df.columns:
        col_l = str(col).strip().lower()
        if col_l.startswith("unnamed"):
            to_drop.append(col)
        elif col_l in {"index", "level_0"}:
            to_drop.append(col)

    cleaned = df.drop(columns=to_drop, errors="ignore")
    return cleaned, to_drop


def load_and_clean_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    log(f"Load train: {train_path}")
    train_df_raw = pd.read_csv(train_path)
    log(f"Load test: {test_path}")
    test_df_raw = pd.read_csv(test_path)

    log(f"Shape train raw: {train_df_raw.shape}")
    log(f"Shape test raw: {test_df_raw.shape}")

    train_df, dropped_train = drop_noise_columns(train_df_raw)
    test_df, dropped_test = drop_noise_columns(test_df_raw)

    log(f"Da drop {len(dropped_train)} cot noise train: {dropped_train}")
    log(f"Da drop {len(dropped_test)} cot noise test: {dropped_test}")
    log(f"Shape train sau clean: {train_df.shape}")
    log(f"Shape test sau clean: {test_df.shape}")

    cleanup_log = {
        "train_raw_shape": list(train_df_raw.shape),
        "test_raw_shape": list(test_df_raw.shape),
        "train_clean_shape": list(train_df.shape),
        "test_clean_shape": list(test_df.shape),
        "dropped_columns_train": dropped_train,
        "dropped_columns_test": dropped_test,
    }
    return train_df, test_df, cleanup_log


def _is_binary_dummy_column(train_s: pd.Series, test_s: pd.Series, name: str) -> bool:
    lname = str(name).lower()
    if any(token in lname for token in ["dummy", "onehot", "__"]):
        return True

    combined = pd.concat([train_s, test_s], axis=0)
    vals = pd.Series(combined.dropna().unique())
    if vals.empty:
        return False

    numeric_vals = pd.to_numeric(vals, errors="coerce")
    if numeric_vals.isna().any():
        return False

    unique_vals = set(numeric_vals.astype(float).tolist())
    return unique_vals.issubset({0.0, 1.0})


def build_feature_sets(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[Dict[str, List[str]], Dict[str, Any]]:
    target_cols = TARGET_COLUMNS

    base_train_features = [c for c in train_df.columns if c not in target_cols]
    base_test_features = [c for c in test_df.columns if c not in target_cols]

    log(f"So feature ban dau train (tru target): {len(base_train_features)}")
    log(f"So feature ban dau test (tru target): {len(base_test_features)}")

    full_features = sorted([c for c in base_train_features if c in set(base_test_features)])

    numeric_train = [c for c in train_df.select_dtypes(include=[np.number]).columns if c not in target_cols]
    numeric_test = [c for c in test_df.select_dtypes(include=[np.number]).columns if c not in target_cols]
    numeric_common = sorted([c for c in numeric_train if c in set(numeric_test)])

    manual_category_present = [c for c in numeric_common if c in MANUAL_CATEGORY_COLUMNS]
    manual_category_missing = sorted(list(MANUAL_CATEGORY_COLUMNS - set(numeric_common)))

    removed_dummy_cols = []
    removed_manual_category_cols = []
    numeric_only_features = []
    for col in numeric_common:
        if col in MANUAL_CATEGORY_COLUMNS:
            removed_manual_category_cols.append(col)
            continue

        is_dummy = _is_binary_dummy_column(train_df[col], test_df[col], col)
        if is_dummy:
            removed_dummy_cols.append(col)
        else:
            numeric_only_features.append(col)

    log(f"So feature sau align full: {len(full_features)}")
    log(f"So feature numeric-only (sau filter dummy/onehot): {len(numeric_only_features)}")
    log(f"So cot category loai theo danh sach co san: {len(removed_manual_category_cols)}")
    if manual_category_missing:
        log(f"Cac cot category khong thay trong tap numeric_common: {manual_category_missing}")

    feature_sets = {
        "full": full_features,
        "numeric": numeric_only_features,
    }

    feature_log = {
        "base_feature_count_train": len(base_train_features),
        "base_feature_count_test": len(base_test_features),
        "full_feature_count": len(full_features),
        "numeric_feature_count": len(numeric_only_features),
        "manual_category_columns_count": len(MANUAL_CATEGORY_COLUMNS),
        "manual_category_columns": sorted(list(MANUAL_CATEGORY_COLUMNS)),
        "removed_manual_category_columns_count": len(removed_manual_category_cols),
        "removed_manual_category_columns": sorted(removed_manual_category_cols),
        "manual_category_columns_present_in_numeric_common": sorted(manual_category_present),
        "manual_category_columns_missing_in_numeric_common": manual_category_missing,
        "removed_dummy_columns_count": len(removed_dummy_cols),
        "removed_dummy_columns": removed_dummy_cols,
        "full_features": full_features,
        "numeric_features": numeric_only_features,
    }

    return feature_sets, feature_log


def prepare_task_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: List[str],
    task: str,
) -> Dict[str, Any]:
    if task not in train_df.columns or task not in test_df.columns:
        raise KeyError(f"Khong tim thay cot target {task}")

    train_target = train_df[task]
    test_target = test_df[task]

    train_valid_mask = train_target.notna()
    test_valid_mask = test_target.notna()

    dropped_train_nan = int((~train_valid_mask).sum())
    dropped_test_nan = int((~test_valid_mask).sum())

    if dropped_train_nan > 0 or dropped_test_nan > 0:
        log(f"Task={task}: drop NaN target train={dropped_train_nan}, test={dropped_test_nan}")

    train_filtered = train_df.loc[train_valid_mask].copy()
    test_filtered = test_df.loc[test_valid_mask].copy()

    y_train_raw = train_filtered[task].astype(str).values
    y_test_raw = test_filtered[task].astype(str).values

    train_classes = set(pd.unique(y_train_raw).tolist())
    test_classes = set(pd.unique(y_test_raw).tolist())

    unseen_in_test = sorted(list(test_classes - train_classes))
    if unseen_in_test:
        raise ValueError(
            f"Task={task}: Test co class khong ton tai trong train: {unseen_in_test}"
        )

    le = LabelEncoder()
    le.fit(y_train_raw)

    y_train = le.transform(y_train_raw).astype(np.int32)
    y_test = le.transform(y_test_raw).astype(np.int32)

    x_train = train_filtered[features].to_numpy(dtype=np.float32)
    x_test = test_filtered[features].to_numpy(dtype=np.float32)

    return {
        "X_train": x_train,
        "X_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_test_raw": y_test_raw,
        "label_encoder": le,
        "test_index": test_filtered.index.to_numpy(),
        "dropped_train_nan_target": dropped_train_nan,
        "dropped_test_nan_target": dropped_test_nan,
    }


def compute_class_weights_for_task(
    y_train: np.ndarray, le: LabelEncoder
) -> Tuple[Dict[int, float], Dict[str, float], np.ndarray]:
    classes_int = np.unique(y_train)
    cw = compute_class_weight(class_weight="balanced", classes=classes_int, y=y_train)

    class_weight_int = {int(c): float(w) for c, w in zip(classes_int, cw)}
    class_weight_label = {
        str(le.inverse_transform([int(c)])[0]): float(w)
        for c, w in zip(classes_int, cw)
    }
    sample_weight = np.array([class_weight_int[int(v)] for v in y_train], dtype=np.float32)

    return class_weight_int, class_weight_label, sample_weight


def _build_model(
    model_name: str,
    class_weight_int: Dict[int, float],
) -> Tuple[Any, str, bool]:
    if model_name == "xgboost":
        if XGBClassifier is None:
            raise ImportError(f"Khong import duoc xgboost: {XGB_IMPORT_ERROR}")
        model = XGBClassifier(
            device="cuda",
            eval_metric="mlogloss",
            random_state=42,
        )
        return model, "sample_weight", True

    if model_name == "catboost":
        if CatBoostClassifier is None:
            raise ImportError(f"Khong import duoc catboost: {CATBOOST_IMPORT_ERROR}")
        sorted_keys = sorted(class_weight_int.keys())
        class_weights_list = [class_weight_int[k] for k in sorted_keys]
        model = CatBoostClassifier(
            task_type="GPU",
            verbose=0,
            random_seed=42,
            class_weights=class_weights_list,
        )
        return model, "class_weights", True

    if model_name == "knn_cuml":
        if cuKNeighborsClassifier is None:
            raise ImportError(f"Khong import duoc cuML KNN: {CUML_IMPORT_ERROR}")
        model = cuKNeighborsClassifier(n_neighbors=7, weights='distance')
        return model, "not_supported", False

    if model_name == "random_forest_cuml":
        if cuRandomForestClassifier is None:
            raise ImportError(f"Khong import duoc cuML RandomForest: {CUML_IMPORT_ERROR}")
        model = cuRandomForestClassifier(random_state=42)
        return model, "sample_weight_if_supported", True

    raise ValueError(f"Model khong hop le: {model_name}")


def train_and_evaluate_model(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    y_test_raw: np.ndarray,
    sample_weight_train: np.ndarray,
    class_weight_int: Dict[int, float],
    label_encoder: LabelEncoder,
) -> Dict[str, Any]:
    result = {
        "metrics": None,
        "y_pred_raw": None,
        "error_message": "",
        "train_time_sec": None,
        "predict_time_sec": None,
        "weighting_status": "",
    }

    try:
        model, weighting_mode, expects_weighting = _build_model(model_name, class_weight_int)
        log(f"Model={model_name}: weighting_mode={weighting_mode}")

        train_start = time.perf_counter()

        if model_name == "xgboost":
            model.fit(x_train, y_train, sample_weight=sample_weight_train)
            result["weighting_status"] = "used sample_weight"

        elif model_name == "catboost":
            model.fit(x_train, y_train)
            result["weighting_status"] = "used class_weights"

        elif model_name == "knn_cuml":
            model.fit(x_train, y_train)
            result["weighting_status"] = "not supported by KNN"

        elif model_name == "random_forest_cuml":
            try:
                model.fit(x_train, y_train, sample_weight=sample_weight_train)
                result["weighting_status"] = "used sample_weight"
            except TypeError:
                model.fit(x_train, y_train)
                result["weighting_status"] = "sample_weight not supported (TypeError)"
            except Exception as e:
                if "sample_weight" in str(e).lower():
                    model.fit(x_train, y_train)
                    result["weighting_status"] = "sample_weight not supported (runtime)"
                else:
                    raise

        train_end = time.perf_counter()
        result["train_time_sec"] = float(train_end - train_start)

        pred_start = time.perf_counter()
        y_pred = model.predict(x_test)
        pred_end = time.perf_counter()
        result["predict_time_sec"] = float(pred_end - pred_start)

        if hasattr(y_pred, "to_pandas"):
            y_pred = y_pred.to_pandas().values
        elif hasattr(y_pred, "get"):
            y_pred = y_pred.get()

        y_pred = np.asarray(y_pred).reshape(-1)

        if y_pred.dtype.kind not in {"i", "u"}:
            y_pred = np.rint(y_pred).astype(np.int32)
        else:
            y_pred = y_pred.astype(np.int32)

        if y_pred.min() < 0 or y_pred.max() >= len(label_encoder.classes_):
            raise ValueError(
                f"Gia tri y_pred ngoai range class: min={y_pred.min()}, max={y_pred.max()}, n_class={len(label_encoder.classes_)}"
            )

        y_test = np.asarray(y_test).astype(np.int32)

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1_micro": float(f1_score(y_test, y_pred, average="micro", zero_division=0)),
            "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
            "precision_micro": float(precision_score(y_test, y_pred, average="micro", zero_division=0)),
            "precision_weighted": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
            "recall_micro": float(recall_score(y_test, y_pred, average="micro", zero_division=0)),
            "recall_weighted": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        }

        y_pred_raw = label_encoder.inverse_transform(y_pred)

        result["metrics"] = metrics
        result["y_pred_raw"] = y_pred_raw

        log(
            f"Model={model_name}: train_time={result['train_time_sec']:.4f}s, "
            f"predict_time={result['predict_time_sec']:.4f}s"
        )

        if not expects_weighting:
            log(f"Model={model_name}: khong ho tro class/sample weighting")
        else:
            log(f"Model={model_name}: {result['weighting_status']}")

    except Exception as e:
        result["error_message"] = str(e)
        log(f"Model={model_name} loi: {e}")

    return result


def save_outputs(
    metrics_rows: List[Dict[str, Any]],
    prediction_rows: List[Dict[str, Any]],
    run_log: Dict[str, Any],
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics_cols = [
        "feature_mode",
        "task",
        "model_name",
        "accuracy",
        "f1_micro",
        "f1_weighted",
        "precision_micro",
        "precision_weighted",
        "recall_micro",
        "recall_weighted",
        "error_message",
    ]
    metrics_df = pd.DataFrame(metrics_rows)
    if metrics_df.empty:
        metrics_df = pd.DataFrame(columns=metrics_cols)
    else:
        metrics_df = metrics_df.reindex(columns=metrics_cols)

    predictions_cols = [
        "feature_mode",
        "task",
        "model_name",
        "index",
        "y_true",
        "y_pred",
    ]
    pred_df = pd.DataFrame(prediction_rows)
    if pred_df.empty:
        pred_df = pd.DataFrame(columns=predictions_cols)
    else:
        pred_df = pred_df.reindex(columns=predictions_cols)

    metrics_path = OUTPUT_DIR / "classification_metrics_summary.csv"
    pred_path = OUTPUT_DIR / "classification_predictions_detailed.csv"
    log_path = OUTPUT_DIR / "classification_run_log.json"

    metrics_df.to_csv(metrics_path, index=False)
    pred_df.to_csv(pred_path, index=False)

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(run_log, f, ensure_ascii=False, indent=2)

    log(f"Saved: {metrics_path}")
    log(f"Saved: {pred_path}")
    log(f"Saved: {log_path}")


def main() -> None:
    overall_start = time.perf_counter()

    run_log: Dict[str, Any] = {
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "inputs": {},
        "data_cleanup": {},
        "feature_modes": {},
        "tasks": {},
        "experiments": [],
        "import_status": {
            "xgboost_available": XGBClassifier is not None,
            "catboost_available": CatBoostClassifier is not None,
            "cuml_available": (cuKNeighborsClassifier is not None and cuRandomForestClassifier is not None),
            "xgboost_import_error": XGB_IMPORT_ERROR,
            "catboost_import_error": CATBOOST_IMPORT_ERROR,
            "cuml_import_error": CUML_IMPORT_ERROR,
        },
    }

    metrics_rows: List[Dict[str, Any]] = []
    prediction_rows: List[Dict[str, Any]] = []

    try:
        train_path, test_path = resolve_input_paths()
        run_log["inputs"]["train_path"] = train_path
        run_log["inputs"]["test_path"] = test_path

        train_df, test_df, cleanup_log = load_and_clean_data(train_path, test_path)
        run_log["data_cleanup"] = cleanup_log

        feature_sets, feature_log = build_feature_sets(train_df, test_df)
        run_log["feature_modes"] = {
            "full": {
                "feature_count": feature_log["full_feature_count"],
                "features": feature_log["full_features"],
            },
            "numeric": {
                "feature_count": feature_log["numeric_feature_count"],
                "features": feature_log["numeric_features"],
                "removed_dummy_columns_count": feature_log["removed_dummy_columns_count"],
                "removed_dummy_columns": feature_log["removed_dummy_columns"],
            },
        }

        for feature_mode, feature_cols in feature_sets.items():
            log("=" * 70)
            log(f"Feature mode={feature_mode}, so feature={len(feature_cols)}")

            if len(feature_cols) == 0:
                err = f"Feature mode={feature_mode} khong co feature hop le"
                log(err)
                for task in TASKS:
                    for model_name in MODEL_NAMES:
                        metrics_rows.append(
                            {
                                "feature_mode": feature_mode,
                                "task": task,
                                "model_name": model_name,
                                "accuracy": np.nan,
                                "f1_micro": np.nan,
                                "f1_weighted": np.nan,
                                "precision_micro": np.nan,
                                "precision_weighted": np.nan,
                                "recall_micro": np.nan,
                                "recall_weighted": np.nan,
                                "error_message": err,
                            }
                        )
                continue

            for task in TASKS:
                log("-" * 70)
                log(f"Task={task}")

                try:
                    task_data = prepare_task_data(train_df, test_df, feature_cols, task)

                    x_train = task_data["X_train"]
                    x_test = task_data["X_test"]
                    y_train = task_data["y_train"]
                    y_test = task_data["y_test"]
                    y_test_raw = task_data["y_test_raw"]
                    test_index = task_data["test_index"]
                    le = task_data["label_encoder"]

                    class_weight_int, class_weight_label, sample_weight_train = compute_class_weights_for_task(y_train, le)

                    train_dist = pd.Series(y_train).value_counts().sort_index().to_dict()
                    train_dist_label = {
                        str(le.inverse_transform([int(k)])[0]): int(v)
                        for k, v in train_dist.items()
                    }

                    test_dist = pd.Series(y_test).value_counts().sort_index().to_dict()
                    test_dist_label = {
                        str(le.inverse_transform([int(k)])[0]): int(v)
                        for k, v in test_dist.items()
                    }

                    log(f"Task={task}: so class train={len(train_dist_label)}")
                    log(f"Task={task}: distribution train={train_dist_label}")
                    log(f"Task={task}: distribution test={test_dist_label}")
                    log(f"Task={task}: class weights={class_weight_label}")

                    if task not in run_log["tasks"]:
                        run_log["tasks"][task] = {}

                    run_log["tasks"][task][feature_mode] = {
                        "num_classes_train": len(train_dist_label),
                        "class_distribution_train": train_dist_label,
                        "class_distribution_test": test_dist_label,
                        "class_weights": class_weight_label,
                        "dropped_nan_train_target": task_data["dropped_train_nan_target"],
                        "dropped_nan_test_target": task_data["dropped_test_nan_target"],
                    }

                except Exception as prep_err:
                    prep_err_msg = f"Prepare task data failed: {prep_err}"
                    log(prep_err_msg)
                    for model_name in MODEL_NAMES:
                        metrics_rows.append(
                            {
                                "feature_mode": feature_mode,
                                "task": task,
                                "model_name": model_name,
                                "accuracy": np.nan,
                                "f1_micro": np.nan,
                                "f1_weighted": np.nan,
                                "precision_micro": np.nan,
                                "precision_weighted": np.nan,
                                "recall_micro": np.nan,
                                "recall_weighted": np.nan,
                                "error_message": prep_err_msg,
                            }
                        )
                        run_log["experiments"].append(
                            {
                                "feature_mode": feature_mode,
                                "task": task,
                                "model_name": model_name,
                                "error_message": prep_err_msg,
                            }
                        )
                    continue

                for model_name in MODEL_NAMES:
                    log(f"Run experiment: mode={feature_mode}, task={task}, model={model_name}")
                    exp_start = time.perf_counter()

                    result = train_and_evaluate_model(
                        model_name=model_name,
                        x_train=x_train,
                        y_train=y_train,
                        x_test=x_test,
                        y_test=y_test,
                        y_test_raw=y_test_raw,
                        sample_weight_train=sample_weight_train,
                        class_weight_int=class_weight_int,
                        label_encoder=le,
                    )

                    exp_end = time.perf_counter()
                    exp_total = float(exp_end - exp_start)

                    if result["error_message"]:
                        metrics_rows.append(
                            {
                                "feature_mode": feature_mode,
                                "task": task,
                                "model_name": model_name,
                                "accuracy": np.nan,
                                "f1_micro": np.nan,
                                "f1_weighted": np.nan,
                                "precision_micro": np.nan,
                                "precision_weighted": np.nan,
                                "recall_micro": np.nan,
                                "recall_weighted": np.nan,
                                "error_message": result["error_message"],
                            }
                        )
                    else:
                        m = result["metrics"]
                        metrics_rows.append(
                            {
                                "feature_mode": feature_mode,
                                "task": task,
                                "model_name": model_name,
                                "accuracy": m["accuracy"],
                                "f1_micro": m["f1_micro"],
                                "f1_weighted": m["f1_weighted"],
                                "precision_micro": m["precision_micro"],
                                "precision_weighted": m["precision_weighted"],
                                "recall_micro": m["recall_micro"],
                                "recall_weighted": m["recall_weighted"],
                                "error_message": "",
                            }
                        )

                        y_pred_raw = result["y_pred_raw"]
                        for idx_val, y_true_i, y_pred_i in zip(test_index, y_test_raw, y_pred_raw):
                            prediction_rows.append(
                                {
                                    "feature_mode": feature_mode,
                                    "task": task,
                                    "model_name": model_name,
                                    "index": int(idx_val) if str(idx_val).isdigit() else idx_val,
                                    "y_true": y_true_i,
                                    "y_pred": y_pred_i,
                                }
                            )

                    run_log["experiments"].append(
                        {
                            "feature_mode": feature_mode,
                            "task": task,
                            "model_name": model_name,
                            "train_time_sec": result["train_time_sec"],
                            "predict_time_sec": result["predict_time_sec"],
                            "total_time_sec": exp_total,
                            "weighting_status": result["weighting_status"],
                            "error_message": result["error_message"],
                        }
                    )

        overall_end = time.perf_counter()
        run_log["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        run_log["total_runtime_sec"] = float(overall_end - overall_start)

        save_outputs(metrics_rows, prediction_rows, run_log)

    except Exception as e:
        log(f"Pipeline loi nghiem trong: {e}")
        run_log["fatal_error"] = str(e)
        run_log["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        run_log["total_runtime_sec"] = float(time.perf_counter() - overall_start)

        save_outputs(metrics_rows, prediction_rows, run_log)


if __name__ == "__main__":
    main()
