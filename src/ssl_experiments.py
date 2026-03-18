import glob
import json
import random
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("/kaggle/working")
UNLABELED_FILENAME = "jobs_unlabeled_scaled.csv"
TRAIN_FILENAME = "jobs_labeled_train_scaled.csv"
TEST_FILENAME = "jobs_labeled_test_scaled.csv"

TASKS = ["model", "model_group", "base_model"]
TARGET_COLUMNS = set(TASKS)

# Keep exactly as provided in the requirement.
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

VIME_CONFIG = {
    "p_m": 0.3,
    "alpha": 2.0,
    "K": 3,
    "beta": 1.0,
    "ssl_epochs": 30,
    "batch_size": 256,
    "learning_rate": 1e-3,
    "random_state": 42,
}

PROBE_CONFIG = {
    "max_iter": 3000,
    "random_state": 42,
}

ENCODER_CONFIGS = {
    "vime_original": {
        "hidden_dims": [256, 128],
        "activation": "relu",
        "dropout": 0.0,
        "norm": None,
        "residual": False,
    },
    "mlp_variant_1": {
        "hidden_dims": [256, 192, 128],
        "activation": "relu",
        "dropout": 0.1,
        "norm": None,
        "residual": False,
    },
    "mlp_variant_2": {
        "hidden_dims": [256, 192, 128],
        "activation": "relu",
        "dropout": 0.1,
        "norm": "batchnorm",
        "residual": False,
    },
    "mlp_variant_3": {
        "hidden_dims": [256, 256, 128],
        "activation": "gelu",
        "dropout": 0.1,
        "norm": "layernorm",
        "residual": True,
    },
    "mlp_variant_4": {
        "hidden_dims": [512, 256, 128],
        "activation": "silu",
        "dropout": 0.15,
        "norm": "layernorm",
        "residual": False,
    },
}


def log(message: str) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_input_paths(
    unlabeled_path: str = "",
    train_path: str = "",
    test_path: str = "",
) -> Tuple[str, str, str]:
    if unlabeled_path and train_path and test_path:
        if Path(unlabeled_path).exists() and Path(train_path).exists() and Path(test_path).exists():
            return unlabeled_path, train_path, test_path

    unlabeled_candidates = glob.glob(f"/kaggle/input/**/{UNLABELED_FILENAME}", recursive=True)
    train_candidates = glob.glob(f"/kaggle/input/**/{TRAIN_FILENAME}", recursive=True)
    test_candidates = glob.glob(f"/kaggle/input/**/{TEST_FILENAME}", recursive=True)

    if not unlabeled_candidates:
        raise FileNotFoundError(f"Khong tim thay {UNLABELED_FILENAME} trong /kaggle/input")
    if not train_candidates:
        raise FileNotFoundError(f"Khong tim thay {TRAIN_FILENAME} trong /kaggle/input")
    if not test_candidates:
        raise FileNotFoundError(f"Khong tim thay {TEST_FILENAME} trong /kaggle/input")

    unlabeled_resolved = sorted(unlabeled_candidates)[0]
    train_resolved = sorted(train_candidates)[0]
    test_resolved = sorted(test_candidates)[0]
    return unlabeled_resolved, train_resolved, test_resolved


def _drop_noise_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    cols_to_drop = []
    for col in df.columns:
        col_l = str(col).strip().lower()
        if col_l.startswith("unnamed"):
            cols_to_drop.append(col)
        elif col_l in {"index", "level_0"}:
            cols_to_drop.append(col)

    cleaned = df.drop(columns=cols_to_drop, errors="ignore")
    return cleaned, cols_to_drop


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def load_and_clean_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    unlabeled_path, train_path, test_path = resolve_input_paths()

    log(f"Load unlabeled: {unlabeled_path}")
    unlabeled_raw = pd.read_csv(unlabeled_path)
    log(f"Load labeled train: {train_path}")
    train_raw = pd.read_csv(train_path)
    log(f"Load labeled test: {test_path}")
    test_raw = pd.read_csv(test_path)

    log(f"unlabeled raw shape: {unlabeled_raw.shape}")
    log(f"labeled_train raw shape: {train_raw.shape}")
    log(f"labeled_test raw shape: {test_raw.shape}")

    unlabeled_clean = _clean_columns(unlabeled_raw)
    train_clean = _clean_columns(train_raw)
    test_clean = _clean_columns(test_raw)

    unlabeled_df, dropped_unlabeled = _drop_noise_columns(unlabeled_clean)
    train_df, dropped_train = _drop_noise_columns(train_clean)
    test_df, dropped_test = _drop_noise_columns(test_clean)

    for target in TASKS:
        if target not in train_df.columns:
            raise KeyError(f"Khong tim thay target '{target}' trong labeled train")
        if target not in test_df.columns:
            raise KeyError(f"Khong tim thay target '{target}' trong labeled test")

    log(f"unlabeled clean shape: {unlabeled_df.shape}")
    log(f"labeled_train clean shape: {train_df.shape}")
    log(f"labeled_test clean shape: {test_df.shape}")

    run_info = {
        "input_paths": {
            "unlabeled": unlabeled_path,
            "labeled_train": train_path,
            "labeled_test": test_path,
        },
        "data_shapes": {
            "unlabeled_raw": list(unlabeled_raw.shape),
            "labeled_train_raw": list(train_raw.shape),
            "labeled_test_raw": list(test_raw.shape),
            "unlabeled_clean": list(unlabeled_df.shape),
            "labeled_train_clean": list(train_df.shape),
            "labeled_test_clean": list(test_df.shape),
        },
        "dropped_noise_columns": {
            "unlabeled": dropped_unlabeled,
            "labeled_train": dropped_train,
            "labeled_test": dropped_test,
        },
    }
    return unlabeled_df, train_df, test_df, run_info


def build_numeric_feature_set(
    unlabeled_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[List[str], Dict[str, Any]]:
    manual_category_norm = {str(c).strip() for c in MANUAL_CATEGORY_COLUMNS}

    unlabeled_numeric = [c for c in unlabeled_df.select_dtypes(include=[np.number]).columns if c not in TARGET_COLUMNS]
    train_numeric = [c for c in train_df.select_dtypes(include=[np.number]).columns if c not in TARGET_COLUMNS]
    test_numeric = [c for c in test_df.select_dtypes(include=[np.number]).columns if c not in TARGET_COLUMNS]

    log(f"numeric columns unlabeled ban dau: {len(unlabeled_numeric)}")
    log(f"numeric columns labeled_train ban dau: {len(train_numeric)}")
    log(f"numeric columns labeled_test ban dau: {len(test_numeric)}")

    common_numeric = sorted(list(set(unlabeled_numeric) & set(train_numeric) & set(test_numeric)))

    removed_manual_category_cols = [c for c in common_numeric if str(c).strip() in manual_category_norm]
    final_features = [c for c in common_numeric if str(c).strip() not in manual_category_norm]

    manual_present = sorted([c for c in common_numeric if str(c).strip() in manual_category_norm])
    manual_missing = sorted(list(manual_category_norm - {str(c).strip() for c in common_numeric}))

    log(f"so cot bi loai do manual category: {len(removed_manual_category_cols)}")
    if manual_missing:
        log(f"manual category khong ton tai trong common numeric: {manual_missing}")

    log(f"so feature numeric cuoi cung: {len(final_features)}")
    log(f"danh sach feature numeric cuoi cung: {final_features}")

    if len(final_features) == 0:
        raise ValueError("Khong con numeric feature nao sau khi filter")

    feature_log = {
        "numeric_count_initial": {
            "unlabeled": len(unlabeled_numeric),
            "labeled_train": len(train_numeric),
            "labeled_test": len(test_numeric),
        },
        "numeric_common_count": len(common_numeric),
        "removed_manual_category_count": len(removed_manual_category_cols),
        "removed_manual_category_columns": sorted(removed_manual_category_cols),
        "manual_category_columns": sorted(list(MANUAL_CATEGORY_COLUMNS)),
        "manual_category_present_in_common": manual_present,
        "manual_category_missing_in_common": manual_missing,
        "final_numeric_feature_count": len(final_features),
        "final_numeric_features": final_features,
    }
    return final_features, feature_log


def _make_norm(norm_name: str, dim: int) -> nn.Module:
    if norm_name == "batchnorm":
        return nn.BatchNorm1d(dim)
    if norm_name == "layernorm":
        return nn.LayerNorm(dim)
    return nn.Identity()


def _make_act(name: str) -> nn.Module:
    name = (name or "relu").lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    return nn.ReLU()


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, activation: str, dropout: float, norm: str):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.norm1 = _make_norm(norm, dim)
        self.act = _make_act(activation)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim, dim)
        self.norm2 = _make_norm(norm, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.norm2(out)
        out = out + x
        out = self.act(out)
        return out


class TabularEncoder(nn.Module):
    def __init__(self, input_dim: int, config: Dict[str, Any]):
        super().__init__()
        hidden_dims = config["hidden_dims"]
        activation = config.get("activation", "relu")
        dropout = float(config.get("dropout", 0.0))
        norm = config.get("norm", None)
        residual = bool(config.get("residual", False))

        layers: List[nn.Module] = []
        prev_dim = input_dim
        for i, h in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, h))
            if norm is not None:
                layers.append(_make_norm(norm, h))
            layers.append(_make_act(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            if residual and i > 0 and prev_dim == h:
                layers.append(ResidualBlock(h, activation=activation, dropout=dropout, norm=(norm or "layernorm")))
            prev_dim = h

        self.net = nn.Sequential(*layers)
        self.embedding_dim = hidden_dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_encoder(config: Dict[str, Any], input_dim: int) -> nn.Module:
    return TabularEncoder(input_dim=input_dim, config=config)


class VIMESelfModel(nn.Module):
    def __init__(self, encoder: nn.Module, input_dim: int):
        super().__init__()
        self.encoder = encoder
        emb_dim = encoder.embedding_dim
        self.mask_head = nn.Linear(emb_dim, input_dim)
        self.recon_head = nn.Linear(emb_dim, input_dim)

    def forward(self, x_tilde: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x_tilde)
        m_logit = self.mask_head(z)
        x_hat = self.recon_head(z)
        return m_logit, x_hat


def _generate_corrupted_batch(x: torch.Tensor, p_m: float) -> Tuple[torch.Tensor, torch.Tensor]:
    bsz, dim = x.shape
    mask = (torch.rand((bsz, dim), device=x.device) < p_m).float()

    x_tilde = x.clone()
    for j in range(dim):
        perm = torch.randperm(bsz, device=x.device)
        shuffled = x[perm, j]
        x_tilde[:, j] = mask[:, j] * shuffled + (1.0 - mask[:, j]) * x[:, j]

    return x_tilde, mask


def vime_self_train(
    X_unlabeled: np.ndarray,
    encoder_config: Dict[str, Any],
    vime_config: Dict[str, Any],
    device: str,
) -> Tuple[nn.Module, Dict[str, Any]]:
    input_dim = X_unlabeled.shape[1]
    encoder = build_encoder(encoder_config, input_dim=input_dim)
    model = VIMESelfModel(encoder=encoder, input_dim=input_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(vime_config["learning_rate"]))
    bce = nn.BCEWithLogitsLoss()

    epochs = int(vime_config["ssl_epochs"])
    batch_size = int(vime_config["batch_size"])
    p_m = float(vime_config["p_m"])
    alpha = float(vime_config["alpha"])
    beta = float(vime_config["beta"])

    x_tensor = torch.from_numpy(X_unlabeled.astype(np.float32))
    n = x_tensor.shape[0]

    history = []
    start_time = time.perf_counter()

    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n)
        epoch_losses = []

        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            x_batch = x_tensor[idx].to(device)

            x_tilde, mask = _generate_corrupted_batch(x_batch, p_m=p_m)
            m_logit, x_hat = model(x_tilde)

            mask_loss = bce(m_logit, mask)
            masked_count = int(mask.sum().item())
            if masked_count > 0:
                recon_loss = ((x_hat - x_batch) ** 2 * mask).sum() / (mask.sum() + 1e-8)
            else:
                recon_loss = torch.tensor(0.0, device=device)

            loss = beta * mask_loss + alpha * recon_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(float(loss.detach().cpu().item()))

        epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else np.nan
        history.append(epoch_loss)
        log(f"SSL epoch {epoch + 1}/{epochs} - loss={epoch_loss:.6f}")

    end_time = time.perf_counter()

    ssl_info = {
        "ssl_train_time_sec": float(end_time - start_time),
        "final_ssl_loss": float(history[-1]) if history else np.nan,
        "ssl_loss_history": history,
    }
    return model.encoder, ssl_info


def extract_embeddings(encoder: nn.Module, X: np.ndarray, batch_size: int, device: str) -> np.ndarray:
    encoder = encoder.to(device)
    encoder.eval()

    x_tensor = torch.from_numpy(X.astype(np.float32))
    out = []

    with torch.no_grad():
        for i in range(0, x_tensor.shape[0], batch_size):
            xb = x_tensor[i : i + batch_size].to(device)
            zb = encoder(xb)
            out.append(zb.detach().cpu().numpy())

    z = np.concatenate(out, axis=0) if out else np.zeros((0, encoder.embedding_dim), dtype=np.float32)
    return z


def run_linear_probe(
    Z_train: np.ndarray,
    y_train_raw: np.ndarray,
    Z_test: np.ndarray,
    y_test_raw: np.ndarray,
) -> Dict[str, Any]:
    train_classes = set(pd.unique(y_train_raw).tolist())
    test_classes = set(pd.unique(y_test_raw).tolist())
    unseen = sorted(list(test_classes - train_classes))
    if unseen:
        raise ValueError(f"Test co class khong ton tai trong train: {unseen}")

    le = LabelEncoder()
    le.fit(y_train_raw)

    y_train = le.transform(y_train_raw)
    y_test = le.transform(y_test_raw)

    probe = LogisticRegression(
        class_weight="balanced",
        max_iter=int(PROBE_CONFIG["max_iter"]),
        random_state=int(PROBE_CONFIG["random_state"]),
        n_jobs=-1,
    )

    t0 = time.perf_counter()
    probe.fit(Z_train, y_train)
    t1 = time.perf_counter()

    y_pred = probe.predict(Z_test)
    t2 = time.perf_counter()

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_micro": float(f1_score(y_test, y_pred, average="micro", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "precision_micro": float(precision_score(y_test, y_pred, average="micro", zero_division=0)),
        "precision_weighted": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall_micro": float(recall_score(y_test, y_pred, average="micro", zero_division=0)),
        "recall_weighted": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
    }

    y_pred_raw = le.inverse_transform(y_pred)
    return {
        "metrics": metrics,
        "probe_train_time_sec": float(t1 - t0),
        "probe_predict_time_sec": float(t2 - t1),
        "y_pred_raw": y_pred_raw,
    }


def run_ssl_once_per_encoder(
    X_unlabeled: np.ndarray,
    encoder_configs: Dict[str, Dict[str, Any]],
    vime_config: Dict[str, Any],
    device: str,
) -> Dict[str, Dict[str, Any]]:
    ssl_results: Dict[str, Dict[str, Any]] = {}

    for encoder_name, encoder_config in encoder_configs.items():
        log("=" * 80)
        log(f"SSL pretraining encoder={encoder_name} bat dau")

        record = {
            "encoder": None,
            "ssl_train_time_sec": None,
            "final_ssl_loss": np.nan,
            "ssl_loss_history": [],
            "error_message": "",
            "embedding_dim": None,
            "n_features": int(X_unlabeled.shape[1]),
        }

        try:
            encoder, info = vime_self_train(
                X_unlabeled=X_unlabeled,
                encoder_config=encoder_config,
                vime_config=vime_config,
                device=device,
            )
            encoder = encoder.cpu()
            record["encoder"] = encoder
            record["ssl_train_time_sec"] = info["ssl_train_time_sec"]
            record["final_ssl_loss"] = info["final_ssl_loss"]
            record["ssl_loss_history"] = info["ssl_loss_history"]
            record["embedding_dim"] = int(encoder.embedding_dim)
            log(f"SSL pretraining encoder={encoder_name} xong, time={record['ssl_train_time_sec']:.4f}s")

        except Exception as e:
            record["error_message"] = str(e)
            log(f"SSL pretraining encoder={encoder_name} loi: {e}")

        ssl_results[encoder_name] = record

    return ssl_results


def _prepare_task_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    task: str,
) -> Dict[str, Any]:
    train_target = train_df[task]
    test_target = test_df[task]

    train_mask = train_target.notna()
    test_mask = test_target.notna()

    dropped_train_nan = int((~train_mask).sum())
    dropped_test_nan = int((~test_mask).sum())

    if dropped_train_nan > 0 or dropped_test_nan > 0:
        log(f"Task={task}: drop NaN target train={dropped_train_nan}, test={dropped_test_nan}")

    train_sub = train_df.loc[train_mask].copy()
    test_sub = test_df.loc[test_mask].copy()

    X_train = train_sub[feature_cols].to_numpy(dtype=np.float32)
    X_test = test_sub[feature_cols].to_numpy(dtype=np.float32)

    y_train_raw = train_sub[task].astype(str).values
    y_test_raw = test_sub[task].astype(str).values

    train_dist = train_sub[task].astype(str).value_counts(dropna=False).to_dict()
    test_dist = test_sub[task].astype(str).value_counts(dropna=False).to_dict()

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train_raw": y_train_raw,
        "y_test_raw": y_test_raw,
        "test_index": test_sub.index.to_numpy(),
        "train_dist": {str(k): int(v) for k, v in train_dist.items()},
        "test_dist": {str(k): int(v) for k, v in test_dist.items()},
        "dropped_train_nan": dropped_train_nan,
        "dropped_test_nan": dropped_test_nan,
    }


def run_downstream_task(
    task: str,
    encoder_name: str,
    ssl_record: Dict[str, Any],
    task_data: Dict[str, Any],
    batch_size: int,
    device: str,
    n_features: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    metrics_row = {
        "task": task,
        "encoder_name": encoder_name,
        "n_features": n_features,
        "embedding_dim": ssl_record.get("embedding_dim"),
        "ssl_train_time_sec": ssl_record.get("ssl_train_time_sec"),
        "probe_train_time_sec": np.nan,
        "probe_predict_time_sec": np.nan,
        "accuracy": np.nan,
        "f1_micro": np.nan,
        "f1_weighted": np.nan,
        "precision_micro": np.nan,
        "precision_weighted": np.nan,
        "recall_micro": np.nan,
        "recall_weighted": np.nan,
        "error_message": "",
    }

    pred_rows: List[Dict[str, Any]] = []
    exp_log: Dict[str, Any] = {
        "task": task,
        "encoder_name": encoder_name,
        "ssl_train_time_sec": ssl_record.get("ssl_train_time_sec"),
        "probe_train_time_sec": None,
        "probe_predict_time_sec": None,
        "error_message": "",
    }

    if ssl_record.get("error_message"):
        err = f"SSL encoder unavailable: {ssl_record['error_message']}"
        metrics_row["error_message"] = err
        exp_log["error_message"] = err
        return metrics_row, pred_rows, exp_log

    try:
        encoder = ssl_record["encoder"]
        if encoder is None:
            raise ValueError("Encoder is None")

        X_train = task_data["X_train"]
        X_test = task_data["X_test"]
        y_train_raw = task_data["y_train_raw"]
        y_test_raw = task_data["y_test_raw"]
        test_index = task_data["test_index"]

        log(f"Task={task}, encoder={encoder_name}: extract embeddings train/test")
        Z_train = extract_embeddings(encoder, X_train, batch_size=batch_size, device=device)
        Z_test = extract_embeddings(encoder, X_test, batch_size=batch_size, device=device)

        if Z_train.ndim != 2 or Z_test.ndim != 2:
            raise ValueError(f"Embedding shape khong hop le: Z_train={Z_train.shape}, Z_test={Z_test.shape}")
        if Z_train.shape[1] != Z_test.shape[1]:
            raise ValueError(f"Embedding dim mismatch: {Z_train.shape[1]} vs {Z_test.shape[1]}")

        probe_out = run_linear_probe(
            Z_train=Z_train,
            y_train_raw=y_train_raw,
            Z_test=Z_test,
            y_test_raw=y_test_raw,
        )

        metrics = probe_out["metrics"]
        metrics_row.update(metrics)
        metrics_row["probe_train_time_sec"] = probe_out["probe_train_time_sec"]
        metrics_row["probe_predict_time_sec"] = probe_out["probe_predict_time_sec"]
        metrics_row["embedding_dim"] = int(Z_train.shape[1])

        exp_log["probe_train_time_sec"] = probe_out["probe_train_time_sec"]
        exp_log["probe_predict_time_sec"] = probe_out["probe_predict_time_sec"]

        log(
            f"Task={task}, encoder={encoder_name}: "
            f"acc={metrics['accuracy']:.6f}, f1_weighted={metrics['f1_weighted']:.6f}, "
            f"probe_train={probe_out['probe_train_time_sec']:.4f}s, "
            f"probe_predict={probe_out['probe_predict_time_sec']:.4f}s"
        )

        y_pred_raw = probe_out["y_pred_raw"]
        for idx_val, y_true, y_pred in zip(test_index, y_test_raw, y_pred_raw):
            pred_rows.append(
                {
                    "task": task,
                    "encoder_name": encoder_name,
                    "index": int(idx_val) if str(idx_val).isdigit() else idx_val,
                    "y_true": y_true,
                    "y_pred": y_pred,
                }
            )

    except Exception as e:
        metrics_row["error_message"] = str(e)
        exp_log["error_message"] = str(e)
        log(f"Task={task}, encoder={encoder_name} loi: {e}")

    return metrics_row, pred_rows, exp_log


def save_outputs(
    metrics_rows: List[Dict[str, Any]],
    prediction_rows: List[Dict[str, Any]],
    ssl_rows: List[Dict[str, Any]],
    run_log: Dict[str, Any],
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics_cols = [
        "task",
        "encoder_name",
        "n_features",
        "embedding_dim",
        "ssl_train_time_sec",
        "probe_train_time_sec",
        "probe_predict_time_sec",
        "accuracy",
        "f1_micro",
        "f1_weighted",
        "precision_micro",
        "precision_weighted",
        "recall_micro",
        "recall_weighted",
        "error_message",
    ]
    pred_cols = ["task", "encoder_name", "index", "y_true", "y_pred"]
    ssl_cols = ["encoder_name", "n_features", "ssl_train_time_sec", "final_ssl_loss", "error_message"]

    metrics_df = pd.DataFrame(metrics_rows)
    pred_df = pd.DataFrame(prediction_rows)
    ssl_df = pd.DataFrame(ssl_rows)

    if metrics_df.empty:
        metrics_df = pd.DataFrame(columns=metrics_cols)
    else:
        metrics_df = metrics_df.reindex(columns=metrics_cols)

    if pred_df.empty:
        pred_df = pd.DataFrame(columns=pred_cols)
    else:
        pred_df = pred_df.reindex(columns=pred_cols)

    if ssl_df.empty:
        ssl_df = pd.DataFrame(columns=ssl_cols)
    else:
        ssl_df = ssl_df.reindex(columns=ssl_cols)

    metrics_path = OUTPUT_DIR / "vime_linear_probe_metrics_summary.csv"
    pred_path = OUTPUT_DIR / "vime_linear_probe_predictions_detailed.csv"
    ssl_path = OUTPUT_DIR / "vime_ssl_training_summary.csv"
    log_path = OUTPUT_DIR / "vime_linear_probe_run_log.json"

    metrics_df.to_csv(metrics_path, index=False)
    pred_df.to_csv(pred_path, index=False)
    ssl_df.to_csv(ssl_path, index=False)

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(run_log, f, ensure_ascii=False, indent=2)

    log(f"Saved: {metrics_path}")
    log(f"Saved: {pred_path}")
    log(f"Saved: {ssl_path}")
    log(f"Saved: {log_path}")


def main() -> None:
    set_seed(int(VIME_CONFIG["random_state"]))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Torch device: {device}")

    run_log: Dict[str, Any] = {
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_file_paths": {},
        "final_numeric_feature_columns": [],
        "MANUAL_CATEGORY_COLUMNS": sorted(list(MANUAL_CATEGORY_COLUMNS)),
        "encoder_configs": ENCODER_CONFIGS,
        "vime_hyperparameters": VIME_CONFIG,
        "probe_hyperparameters": PROBE_CONFIG,
        "data": {},
        "class_distributions": {},
        "ssl_runs": [],
        "downstream_runs": [],
    }

    metrics_rows: List[Dict[str, Any]] = []
    prediction_rows: List[Dict[str, Any]] = []
    ssl_rows: List[Dict[str, Any]] = []

    overall_start = time.perf_counter()

    try:
        unlabeled_df, train_df, test_df, data_info = load_and_clean_data()
        run_log["input_file_paths"] = data_info["input_paths"]
        run_log["data"] = {
            "shapes": data_info["data_shapes"],
            "dropped_noise_columns": data_info["dropped_noise_columns"],
        }

        feature_cols, feature_log = build_numeric_feature_set(unlabeled_df, train_df, test_df)
        run_log["final_numeric_feature_columns"] = feature_cols
        run_log["feature_selection"] = feature_log

        X_unlabeled = unlabeled_df[feature_cols].to_numpy(dtype=np.float32)
        n_features = int(X_unlabeled.shape[1])

        if X_unlabeled.ndim != 2:
            raise ValueError(f"X_unlabeled shape khong hop le: {X_unlabeled.shape}")

        log(f"X_unlabeled shape: {X_unlabeled.shape}")

        ssl_results = run_ssl_once_per_encoder(
            X_unlabeled=X_unlabeled,
            encoder_configs=ENCODER_CONFIGS,
            vime_config=VIME_CONFIG,
            device=device,
        )

        for encoder_name, record in ssl_results.items():
            ssl_rows.append(
                {
                    "encoder_name": encoder_name,
                    "n_features": n_features,
                    "ssl_train_time_sec": record.get("ssl_train_time_sec"),
                    "final_ssl_loss": record.get("final_ssl_loss"),
                    "error_message": record.get("error_message", ""),
                }
            )
            run_log["ssl_runs"].append(
                {
                    "encoder_name": encoder_name,
                    "n_features": n_features,
                    "embedding_dim": record.get("embedding_dim"),
                    "ssl_train_time_sec": record.get("ssl_train_time_sec"),
                    "final_ssl_loss": record.get("final_ssl_loss"),
                    "error_message": record.get("error_message", ""),
                }
            )

        task_data_map: Dict[str, Dict[str, Any]] = {}
        for task in TASKS:
            log("-" * 80)
            log(f"Prepare downstream task: {task}")
            try:
                task_data = _prepare_task_data(train_df, test_df, feature_cols=feature_cols, task=task)
                task_data_map[task] = task_data
                run_log["class_distributions"][task] = {
                    "train": task_data["train_dist"],
                    "test": task_data["test_dist"],
                    "dropped_nan_train": task_data["dropped_train_nan"],
                    "dropped_nan_test": task_data["dropped_test_nan"],
                }

                log(f"Task={task} class distribution train: {task_data['train_dist']}")
                log(f"Task={task} class distribution test : {task_data['test_dist']}")

            except Exception as e:
                err = f"Prepare task failed: {e}"
                run_log["class_distributions"][task] = {"error_message": err}
                task_data_map[task] = {"error_message": err}
                log(f"Task={task} loi prepare: {e}")

        for encoder_name, ssl_record in ssl_results.items():
            log("=" * 80)
            log(f"Downstream encoder: {encoder_name}")

            for task in TASKS:
                log(f"Run downstream task={task}, encoder={encoder_name}")

                if "error_message" in task_data_map[task]:
                    err_msg = task_data_map[task]["error_message"]
                    row = {
                        "task": task,
                        "encoder_name": encoder_name,
                        "n_features": n_features,
                        "embedding_dim": ssl_record.get("embedding_dim"),
                        "ssl_train_time_sec": ssl_record.get("ssl_train_time_sec"),
                        "probe_train_time_sec": np.nan,
                        "probe_predict_time_sec": np.nan,
                        "accuracy": np.nan,
                        "f1_micro": np.nan,
                        "f1_weighted": np.nan,
                        "precision_micro": np.nan,
                        "precision_weighted": np.nan,
                        "recall_micro": np.nan,
                        "recall_weighted": np.nan,
                        "error_message": err_msg,
                    }
                    metrics_rows.append(row)
                    run_log["downstream_runs"].append(
                        {
                            "task": task,
                            "encoder_name": encoder_name,
                            "error_message": err_msg,
                        }
                    )
                    continue

                metrics_row, pred_rows, exp_log = run_downstream_task(
                    task=task,
                    encoder_name=encoder_name,
                    ssl_record=ssl_record,
                    task_data=task_data_map[task],
                    batch_size=int(VIME_CONFIG["batch_size"]),
                    device=device,
                    n_features=n_features,
                )
                metrics_rows.append(metrics_row)
                prediction_rows.extend(pred_rows)
                run_log["downstream_runs"].append(exp_log)

    except Exception as e:
        log(f"Pipeline loi nghiem trong: {e}")
        run_log["fatal_error"] = str(e)

    overall_end = time.perf_counter()
    run_log["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    run_log["total_runtime_sec"] = float(overall_end - overall_start)

    save_outputs(
        metrics_rows=metrics_rows,
        prediction_rows=prediction_rows,
        ssl_rows=ssl_rows,
        run_log=run_log,
    )


if __name__ == "__main__":
    main()
