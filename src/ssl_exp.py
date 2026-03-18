import os
import re
import json
import glob
import time
import copy
import random
import argparse
import traceback
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

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

TARGET_COLUMNS = ["model", "model_group", "base_model"]


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def to_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, set):
        return sorted([to_serializable(v) for v in obj], key=lambda x: str(x))
    if isinstance(obj, Counter):
        return {str(k): int(v) for k, v in obj.items()}
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Index):
        return [to_serializable(v) for v in obj.tolist()]
    if isinstance(obj, pd.Series):
        return [to_serializable(v) for v in obj.tolist()]
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    return obj


def resolve_input_path(explicit_path: str, filename: str) -> str:
    if explicit_path:
        if os.path.exists(explicit_path):
            return explicit_path
        raise FileNotFoundError(f"Provided path does not exist for {filename}: {explicit_path}")

    pattern = os.path.join("/kaggle/input", "**", filename)
    matches = sorted(glob.glob(pattern, recursive=True))
    if not matches:
        raise FileNotFoundError(f"Could not find {filename} under /kaggle/input/**")
    if len(matches) > 1:
        log(f"Found multiple matches for {filename}; using the first one: {matches[0]}")
    return matches[0]


def maybe_promote_unnamed_index(df: pd.DataFrame) -> pd.DataFrame:
    unnamed_candidates = [c for c in df.columns if re.match(r"(?i)^Unnamed:\s*0$", str(c).strip())]
    if unnamed_candidates:
        col = unnamed_candidates[0]
        if df[col].is_unique:
            df = df.set_index(col, drop=True)
            df.index.name = "index"
    return df


def make_columns_unique(columns):
    counts = {}
    output = []
    for c in columns:
        if c not in counts:
            counts[c] = 0
            output.append(c)
        else:
            counts[c] += 1
            output.append(f"{c}__dup{counts[c]}")
    return output


def normalize_column_name(col: str) -> str:
    col = str(col)
    col = col.replace("\ufeff", "")
    col = col.replace("\r", "")
    col = col.replace("\n", "\\n")
    col = re.sub(r"\s+", " ", col).strip()
    return col


def clean_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = maybe_promote_unnamed_index(df)
    cleaned = [normalize_column_name(c) for c in df.columns]
    df = df.copy()
    df.columns = make_columns_unique(cleaned)
    return df


def load_and_clean_data(unlabeled_path: str, labeled_train_path: str, labeled_test_path: str):
    log(f"Loading unlabeled data from: {unlabeled_path}")
    df_unlabeled = pd.read_csv(unlabeled_path, low_memory=False)
    log(f"Loading labeled train data from: {labeled_train_path}")
    df_train = pd.read_csv(labeled_train_path, low_memory=False)
    log(f"Loading labeled test data from: {labeled_test_path}")
    df_test = pd.read_csv(labeled_test_path, low_memory=False)

    df_unlabeled = clean_dataframe_columns(df_unlabeled)
    df_train = clean_dataframe_columns(df_train)
    df_test = clean_dataframe_columns(df_test)

    log(f"unlabeled shape: {df_unlabeled.shape}")
    log(f"labeled_train shape: {df_train.shape}")
    log(f"labeled_test shape: {df_test.shape}")

    for target in TARGET_COLUMNS:
        if target not in df_train.columns:
            raise KeyError(f"Missing target column '{target}' in labeled train data.")
        if target not in df_test.columns:
            raise KeyError(f"Missing target column '{target}' in labeled test data.")

    return df_unlabeled, df_train, df_test


def is_noise_column(col: str) -> bool:
    col_lower = col.lower()
    return (
        bool(re.match(r"(?i)^Unnamed:\s*\d+$", col))
        or col_lower in {"index", "level_0", "__index_level_0__"}
    )


def build_numeric_feature_set(
    df_unlabeled: pd.DataFrame,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
):
    def candidate_numeric_columns(df: pd.DataFrame):
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        removed_targets = [c for c in numeric_cols if c in TARGET_COLUMNS]
        removed_manual = [c for c in numeric_cols if c in MANUAL_CATEGORY_COLUMNS]
        removed_noise = [c for c in numeric_cols if is_noise_column(c)]

        keep_cols = []
        for c in numeric_cols:
            if c in TARGET_COLUMNS:
                continue
            if c in MANUAL_CATEGORY_COLUMNS:
                continue
            if is_noise_column(c):
                continue
            keep_cols.append(c)

        return {
            "numeric_initial": numeric_cols,
            "removed_targets": removed_targets,
            "removed_manual": removed_manual,
            "removed_noise": removed_noise,
            "kept_candidates": keep_cols,
        }

    info_unlabeled = candidate_numeric_columns(df_unlabeled)
    info_train = candidate_numeric_columns(df_train)
    info_test = candidate_numeric_columns(df_test)

    common_cols = [
        c for c in info_unlabeled["kept_candidates"]
        if c in set(info_train["kept_candidates"]) and c in set(info_test["kept_candidates"])
    ]

    if not common_cols:
        raise ValueError("No common numeric feature columns found across unlabeled/train/test after cleaning.")

    feature_info = {
        "unlabeled_numeric_initial_count": len(info_unlabeled["numeric_initial"]),
        "train_numeric_initial_count": len(info_train["numeric_initial"]),
        "test_numeric_initial_count": len(info_test["numeric_initial"]),
        "unlabeled_removed_manual_count": len(info_unlabeled["removed_manual"]),
        "train_removed_manual_count": len(info_train["removed_manual"]),
        "test_removed_manual_count": len(info_test["removed_manual"]),
        "unlabeled_removed_manual_columns": info_unlabeled["removed_manual"],
        "train_removed_manual_columns": info_train["removed_manual"],
        "test_removed_manual_columns": info_test["removed_manual"],
        "unlabeled_removed_noise_columns": info_unlabeled["removed_noise"],
        "train_removed_noise_columns": info_train["removed_noise"],
        "test_removed_noise_columns": info_test["removed_noise"],
        "unlabeled_removed_target_columns": info_unlabeled["removed_targets"],
        "train_removed_target_columns": info_train["removed_targets"],
        "test_removed_target_columns": info_test["removed_targets"],
        "final_numeric_feature_columns": common_cols,
        "n_features": len(common_cols),
        "manual_category_columns_missing_in_unlabeled": sorted([c for c in MANUAL_CATEGORY_COLUMNS if c not in df_unlabeled.columns]),
        "manual_category_columns_missing_in_train": sorted([c for c in MANUAL_CATEGORY_COLUMNS if c not in df_train.columns]),
        "manual_category_columns_missing_in_test": sorted([c for c in MANUAL_CATEGORY_COLUMNS if c not in df_test.columns]),
    }

    log(
        "Initial numeric columns - unlabeled/train/test: "
        f"{feature_info['unlabeled_numeric_initial_count']} / "
        f"{feature_info['train_numeric_initial_count']} / "
        f"{feature_info['test_numeric_initial_count']}"
    )
    log(
        "Removed manual category columns - unlabeled/train/test: "
        f"{feature_info['unlabeled_removed_manual_count']} / "
        f"{feature_info['train_removed_manual_count']} / "
        f"{feature_info['test_removed_manual_count']}"
    )
    log(f"Final numeric feature count: {feature_info['n_features']}")
    log(f"Final numeric feature columns: {feature_info['final_numeric_feature_columns']}")

    return common_cols, feature_info


def prepare_feature_matrix(df: pd.DataFrame, feature_cols, dataset_name: str) -> np.ndarray:
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{dataset_name}: missing feature columns after alignment: {missing_cols}")

    X = df.loc[:, feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    missing_count = int(X.isna().sum().sum())
    if missing_count > 0:
        log(f"{dataset_name}: found {missing_count} missing numeric feature values; filling with 0.0")
        X = X.fillna(0.0)

    X_np = X.to_numpy(dtype=np.float32)
    if X_np.ndim != 2 or X_np.shape[1] != len(feature_cols):
        raise ValueError(
            f"{dataset_name}: unexpected feature matrix shape {X_np.shape}; expected (?, {len(feature_cols)})"
        )
    return X_np


def get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")


def make_norm(norm_name: str, dim: int):
    if norm_name is None:
        return None
    norm_name = norm_name.lower()
    if norm_name == "batchnorm":
        return nn.BatchNorm1d(dim)
    if norm_name == "layernorm":
        return nn.LayerNorm(dim)
    raise ValueError(f"Unsupported norm type: {norm_name}")


class StandardMLPEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims,
        embedding_dim: int,
        activation: str = "relu",
        dropout: float = 0.0,
        norm: str = None,
    ):
        super().__init__()
        dims = list(hidden_dims) + [embedding_dim]
        layers = []
        prev = input_dim
        for i, out_dim in enumerate(dims):
            layers.append(nn.Linear(prev, out_dim))
            if norm is not None:
                layers.append(make_norm(norm, out_dim))
            layers.append(get_activation(activation))
            if dropout > 0 and i < len(dims) - 1:
                layers.append(nn.Dropout(dropout))
            prev = out_dim
        self.net = nn.Sequential(*layers)
        self.output_dim = embedding_dim

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int = None,
        activation: str = "relu",
        dropout: float = 0.1,
        norm: str = "layernorm",
    ):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.norm1 = make_norm(norm, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.norm2 = make_norm(norm, dim)
        self.act = get_activation(activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        if self.norm1 is not None:
            out = self.norm1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.fc2(out)
        if self.norm2 is not None:
            out = self.norm2(out)
        out = self.dropout(out)
        out = out + residual
        out = self.act(out)
        return out


class ResidualMLPEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_blocks: int,
        embedding_dim: int,
        activation: str = "gelu",
        dropout: float = 0.1,
        norm: str = "layernorm",
    ):
        super().__init__()
        in_layers = [nn.Linear(input_dim, hidden_dim)]
        if norm is not None:
            in_layers.append(make_norm(norm, hidden_dim))
        in_layers.append(get_activation(activation))
        if dropout > 0:
            in_layers.append(nn.Dropout(dropout))
        self.input_proj = nn.Sequential(*in_layers)
        self.blocks = nn.Sequential(
            *[
                ResidualBlock(
                    hidden_dim,
                    hidden_dim,
                    activation=activation,
                    dropout=dropout,
                    norm=norm,
                )
                for _ in range(num_blocks)
            ]
        )
        out_layers = [nn.Linear(hidden_dim, embedding_dim)]
        if norm is not None:
            out_layers.append(make_norm(norm, embedding_dim))
        out_layers.append(get_activation(activation))
        self.output_proj = nn.Sequential(*out_layers)
        self.output_dim = embedding_dim

    def forward(self, x):
        x = self.input_proj(x)
        x = self.blocks(x)
        x = self.output_proj(x)
        return x


class VIMESelfModel(nn.Module):
    def __init__(self, encoder: nn.Module, input_dim: int, feature_head_activation: str = "linear"):
        super().__init__()
        self.encoder = encoder
        self.mask_head = nn.Linear(encoder.output_dim, input_dim)
        self.feature_head = nn.Linear(encoder.output_dim, input_dim)
        self.feature_head_activation = feature_head_activation

    def forward(self, x):
        z = self.encoder(x)
        mask_logits = self.mask_head(z)
        feature_out = self.feature_head(z)
        if self.feature_head_activation == "sigmoid":
            feature_out = torch.sigmoid(feature_out)
        return z, mask_logits, feature_out


def scaled_dim(input_dim: int, multiplier: float, min_dim: int, max_dim: int) -> int:
    return int(max(min_dim, min(max_dim, round(input_dim * multiplier))))


def get_encoder_configs(input_dim: int):
    configs = {}

    configs["vime_original"] = {
        "encoder_name": "vime_original",
        "type": "standard_mlp",
        "hidden_dims": [
            scaled_dim(input_dim, 1.0, 64, 512),
            scaled_dim(input_dim, 0.75, 64, 384),
        ],
        "embedding_dim": scaled_dim(input_dim, 0.5, 32, 256),
        "activation": "relu",
        "dropout": 0.0,
        "norm": None,
    }

    configs["mlp_variant_1"] = {
        "encoder_name": "mlp_variant_1",
        "type": "standard_mlp",
        "hidden_dims": [
            scaled_dim(input_dim, 1.5, 128, 512),
            scaled_dim(input_dim, 1.0, 96, 384),
            scaled_dim(input_dim, 0.75, 64, 256),
        ],
        "embedding_dim": scaled_dim(input_dim, 0.5, 32, 192),
        "activation": "relu",
        "dropout": 0.10,
        "norm": None,
    }

    configs["mlp_variant_2"] = {
        "encoder_name": "mlp_variant_2",
        "type": "standard_mlp",
        "hidden_dims": [
            scaled_dim(input_dim, 1.5, 128, 512),
            scaled_dim(input_dim, 1.0, 96, 384),
            scaled_dim(input_dim, 0.75, 64, 256),
        ],
        "embedding_dim": scaled_dim(input_dim, 0.5, 32, 192),
        "activation": "relu",
        "dropout": 0.10,
        "norm": "batchnorm",
    }

    configs["mlp_variant_3"] = {
        "encoder_name": "mlp_variant_3",
        "type": "residual_mlp",
        "hidden_dim": scaled_dim(input_dim, 1.0, 128, 512),
        "num_blocks": 3,
        "embedding_dim": scaled_dim(input_dim, 0.75, 64, 256),
        "activation": "gelu",
        "dropout": 0.10,
        "norm": "layernorm",
    }

    configs["mlp_variant_4"] = {
        "encoder_name": "mlp_variant_4",
        "type": "standard_mlp",
        "hidden_dims": [
            scaled_dim(input_dim, 2.0, 256, 1024),
            scaled_dim(input_dim, 1.5, 192, 768),
            scaled_dim(input_dim, 1.0, 128, 512),
        ],
        "embedding_dim": scaled_dim(input_dim, 1.0, 64, 256),
        "activation": "silu",
        "dropout": 0.05,
        "norm": "layernorm",
    }

    return configs


def build_encoder(config, input_dim: int) -> nn.Module:
    if config["type"] == "standard_mlp":
        encoder = StandardMLPEncoder(
            input_dim=input_dim,
            hidden_dims=config["hidden_dims"],
            embedding_dim=config["embedding_dim"],
            activation=config["activation"],
            dropout=config["dropout"],
            norm=config["norm"],
        )
        return encoder

    if config["type"] == "residual_mlp":
        encoder = ResidualMLPEncoder(
            input_dim=input_dim,
            hidden_dim=config["hidden_dim"],
            num_blocks=config["num_blocks"],
            embedding_dim=config["embedding_dim"],
            activation=config["activation"],
            dropout=config["dropout"],
            norm=config["norm"],
        )
        return encoder

    raise ValueError(f"Unsupported encoder type: {config['type']}")


def mask_generator(p_m: float, x: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    return rng.binomial(1, p_m, size=x.shape).astype(np.float32)


def pretext_generator(m: np.ndarray, x: np.ndarray, rng: np.random.RandomState):
    no, dim = x.shape
    x_bar = np.empty_like(x, dtype=np.float32)
    for i in range(dim):
        idx = rng.permutation(no)
        x_bar[:, i] = x[idx, i]
    x_tilde = x * (1.0 - m) + x_bar * m
    m_new = (x != x_tilde).astype(np.float32)
    return m_new, x_tilde.astype(np.float32)


def infer_feature_head_activation(x_unlabeled: np.ndarray) -> str:
    x_min = float(np.nanmin(x_unlabeled))
    x_max = float(np.nanmax(x_unlabeled))
    return "sigmoid" if (x_min >= 0.0 and x_max <= 1.0) else "linear"


def vime_self_train(X_unlabeled: np.ndarray, encoder_config: dict, vime_config: dict):
    start_time = time.time()
    seed = int(vime_config["random_state"])
    set_seed(seed)
    rng = np.random.RandomState(seed)

    input_dim = X_unlabeled.shape[1]
    encoder = build_encoder(encoder_config, input_dim=input_dim)
    feature_head_activation = infer_feature_head_activation(X_unlabeled)

    device = vime_config["device"]
    model = VIMESelfModel(encoder, input_dim=input_dim, feature_head_activation=feature_head_activation).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(vime_config["learning_rate"]),
        weight_decay=float(vime_config.get("weight_decay", 0.0)),
    )
    bce_loss_fn = nn.BCEWithLogitsLoss()
    mse_loss_fn = nn.MSELoss()

    epochs = int(vime_config["ssl_epochs"])
    batch_size = int(vime_config["batch_size"])
    alpha = float(vime_config["alpha"])
    p_m = float(vime_config["p_m"])
    patience = int(vime_config.get("ssl_patience", 0))

    history = []
    best_loss = float("inf")
    best_state = None
    patience_counter = 0

    num_samples = X_unlabeled.shape[0]

    log(
        f"SSL start | encoder={encoder_config['encoder_name']} | "
        f"input_dim={input_dim} | embedding_dim={encoder.output_dim} | "
        f"feature_head_activation={feature_head_activation}"
    )

    model.train()
    for epoch in range(1, epochs + 1):
        perm = rng.permutation(num_samples)
        epoch_total_loss = 0.0
        epoch_mask_loss = 0.0
        epoch_feature_loss = 0.0
        epoch_batches = 0

        for start_idx in range(0, num_samples, batch_size):
            batch_idx = perm[start_idx:start_idx + batch_size]
            x_batch = X_unlabeled[batch_idx].astype(np.float32)

            m_batch = mask_generator(p_m, x_batch, rng)
            m_label, x_tilde = pretext_generator(m_batch, x_batch, rng)

            x_batch_t = torch.from_numpy(x_batch).to(device)
            x_tilde_t = torch.from_numpy(x_tilde).to(device)
            m_label_t = torch.from_numpy(m_label).to(device)

            optimizer.zero_grad()
            _, mask_logits, feature_out = model(x_tilde_t)

            mask_loss = bce_loss_fn(mask_logits, m_label_t)
            feature_loss = mse_loss_fn(feature_out, x_batch_t)
            total_loss = mask_loss + alpha * feature_loss

            total_loss.backward()
            optimizer.step()

            epoch_total_loss += float(total_loss.item())
            epoch_mask_loss += float(mask_loss.item())
            epoch_feature_loss += float(feature_loss.item())
            epoch_batches += 1

        avg_total = epoch_total_loss / max(epoch_batches, 1)
        avg_mask = epoch_mask_loss / max(epoch_batches, 1)
        avg_feature = epoch_feature_loss / max(epoch_batches, 1)

        history.append(
            {
                "epoch": epoch,
                "ssl_loss": avg_total,
                "mask_loss": avg_mask,
                "feature_loss": avg_feature,
            }
        )

        if epoch == 1 or epoch % max(1, min(10, epochs)) == 0 or epoch == epochs:
            log(
                f"SSL epoch {epoch}/{epochs} | encoder={encoder_config['encoder_name']} | "
                f"total_loss={avg_total:.6f} | mask_loss={avg_mask:.6f} | feature_loss={avg_feature:.6f}"
            )

        if avg_total < best_loss:
            best_loss = avg_total
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience > 0 and patience_counter >= patience:
            log(
                f"SSL early stopping triggered | encoder={encoder_config['encoder_name']} | "
                f"best_loss={best_loss:.6f} | stopped_epoch={epoch}"
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_time_sec = time.time() - start_time
    log(
        f"SSL end | encoder={encoder_config['encoder_name']} | "
        f"train_time_sec={train_time_sec:.3f} | final_best_ssl_loss={best_loss:.6f}"
    )

    model = model.to("cpu")
    model.eval()

    result = {
        "encoder": model.encoder,
        "embedding_dim": model.encoder.output_dim,
        "ssl_train_time_sec": train_time_sec,
        "final_ssl_loss": best_loss,
        "ssl_history": history,
        "feature_head_activation": feature_head_activation,
        "encoder_config": encoder_config,
    }
    return result


def extract_embeddings(encoder: nn.Module, X: np.ndarray, batch_size: int = 2048, device: str = "cpu") -> np.ndarray:
    encoder = copy.deepcopy(encoder).to(device)
    encoder.eval()

    dataset = TensorDataset(torch.from_numpy(X.astype(np.float32)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    outputs = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            z = encoder(xb)
            outputs.append(z.detach().cpu().numpy())

    Z = np.concatenate(outputs, axis=0) if outputs else np.zeros((0, encoder.output_dim), dtype=np.float32)

    if Z.ndim != 2:
        raise ValueError(f"Encoder output must be 2D, but got shape {Z.shape}")
    if Z.shape[0] != X.shape[0]:
        raise ValueError(f"Embedding row mismatch: got {Z.shape[0]} rows for input with {X.shape[0]} rows")

    return Z.astype(np.float32)


def compute_class_distribution(y):
    return {str(k): int(v) for k, v in Counter(pd.Series(y).astype(str).tolist()).items()}


def prepare_task_artifact(df_train: pd.DataFrame, df_test: pd.DataFrame, task: str):
    train_missing = int(df_train[task].isna().sum())
    test_missing = int(df_test[task].isna().sum())

    if train_missing > 0 or test_missing > 0:
        log(f"{task}: dropping missing targets | train_missing={train_missing} | test_missing={test_missing}")

    train_mask = df_train[task].notna().to_numpy()
    test_mask = df_test[task].notna().to_numpy()

    y_train = df_train.loc[train_mask, task].to_numpy()
    y_test = df_test.loc[test_mask, task].to_numpy()
    test_index = df_test.loc[test_mask].index.to_numpy()

    if len(np.unique(y_train)) < 2:
        raise ValueError(f"{task}: labeled train has fewer than 2 classes after dropping missing targets.")

    unseen_test_classes = sorted(set(pd.Series(y_test).astype(str).unique()) - set(pd.Series(y_train).astype(str).unique()))
    if unseen_test_classes:
        raise ValueError(f"{task}: test contains classes not present in train: {unseen_test_classes}")

    train_dist = compute_class_distribution(y_train)
    test_dist = compute_class_distribution(y_test)

    log(f"{task}: train class distribution = {train_dist}")
    log(f"{task}: test class distribution = {test_dist}")

    return {
        "task": task,
        "train_mask": train_mask,
        "test_mask": test_mask,
        "y_train": y_train,
        "y_test": y_test,
        "test_index": test_index,
        "train_class_distribution": train_dist,
        "test_class_distribution": test_dist,
        "train_missing_targets": train_missing,
        "test_missing_targets": test_missing,
    }


def run_linear_probe(Z_train, y_train, Z_test, y_test, random_state: int = 42, max_iter: int = 5000):
    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=max_iter,
        random_state=random_state,
        solver="lbfgs",
        multi_class="auto",
    )

    t0 = time.time()
    clf.fit(Z_train, y_train)
    train_time = time.time() - t0

    t1 = time.time()
    y_pred = clf.predict(Z_test)
    predict_time = time.time() - t1

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_micro": float(f1_score(y_test, y_pred, average="micro", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "precision_micro": float(precision_score(y_test, y_pred, average="micro", zero_division=0)),
        "precision_weighted": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall_micro": float(recall_score(y_test, y_pred, average="micro", zero_division=0)),
        "recall_weighted": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
    }

    return clf, y_pred, metrics, train_time, predict_time


def run_ssl_once_per_encoder(X_unlabeled: np.ndarray, encoder_configs: dict, vime_config: dict):
    pretrained_encoders = {}
    ssl_rows = []
    ssl_logs = {}

    for encoder_name, encoder_config in encoder_configs.items():
        row = {
            "encoder_name": encoder_name,
            "n_features": int(X_unlabeled.shape[1]),
            "ssl_train_time_sec": np.nan,
            "final_ssl_loss": np.nan,
            "error_message": "",
        }

        try:
            log(f"========== SSL encoder: {encoder_name} ==========")
            ssl_result = vime_self_train(X_unlabeled, encoder_config, vime_config)
            pretrained_encoders[encoder_name] = ssl_result

            row["ssl_train_time_sec"] = float(ssl_result["ssl_train_time_sec"])
            row["final_ssl_loss"] = float(ssl_result["final_ssl_loss"])
            ssl_logs[encoder_name] = {
                "status": "success",
                "embedding_dim": int(ssl_result["embedding_dim"]),
                "ssl_train_time_sec": float(ssl_result["ssl_train_time_sec"]),
                "final_ssl_loss": float(ssl_result["final_ssl_loss"]),
                "feature_head_activation": ssl_result["feature_head_activation"],
                "ssl_history": ssl_result["ssl_history"],
            }

        except Exception as e:
            error_message = str(e)
            log(f"SSL failed for encoder={encoder_name}: {error_message}")
            log(traceback.format_exc())

            pretrained_encoders[encoder_name] = None
            row["error_message"] = error_message
            ssl_logs[encoder_name] = {
                "status": "failed",
                "error_message": error_message,
                "traceback": traceback.format_exc(),
            }

        ssl_rows.append(row)

    return pretrained_encoders, ssl_rows, ssl_logs


def run_downstream_task(
    task: str,
    encoder_name: str,
    encoder_bundle: dict,
    task_artifact: dict,
    Z_train_full: np.ndarray,
    Z_test_full: np.ndarray,
    n_features: int,
    probe_max_iter: int,
    random_state: int,
):
    row = {
        "task": task,
        "encoder_name": encoder_name,
        "n_features": int(n_features),
        "embedding_dim": np.nan if encoder_bundle is None else int(encoder_bundle["embedding_dim"]),
        "ssl_train_time_sec": np.nan if encoder_bundle is None else float(encoder_bundle["ssl_train_time_sec"]),
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
    detailed_rows = []

    try:
        if encoder_bundle is None:
            raise ValueError(f"SSL encoder not available for {encoder_name}")

        if task_artifact.get("error_message"):
            raise ValueError(task_artifact["error_message"])

        train_mask = task_artifact["train_mask"]
        test_mask = task_artifact["test_mask"]
        y_train = task_artifact["y_train"]
        y_test = task_artifact["y_test"]
        test_index = task_artifact["test_index"]

        Z_train = Z_train_full[train_mask]
        Z_test = Z_test_full[test_mask]

        if Z_train.shape[0] != len(y_train):
            raise ValueError(f"{task}/{encoder_name}: Z_train rows {Z_train.shape[0]} != y_train length {len(y_train)}")
        if Z_test.shape[0] != len(y_test):
            raise ValueError(f"{task}/{encoder_name}: Z_test rows {Z_test.shape[0]} != y_test length {len(y_test)}")
        if Z_train.ndim != 2 or Z_test.ndim != 2:
            raise ValueError(f"{task}/{encoder_name}: embeddings must be 2D, got {Z_train.shape} and {Z_test.shape}")

        log(
            f"Downstream task start | task={task} | encoder={encoder_name} | "
            f"train_samples={Z_train.shape[0]} | test_samples={Z_test.shape[0]} | embedding_dim={Z_train.shape[1]}"
        )

        _, y_pred, metrics, probe_train_time, probe_predict_time = run_linear_probe(
            Z_train=Z_train,
            y_train=y_train,
            Z_test=Z_test,
            y_test=y_test,
            random_state=random_state,
            max_iter=probe_max_iter,
        )

        row["probe_train_time_sec"] = float(probe_train_time)
        row["probe_predict_time_sec"] = float(probe_predict_time)
        for k, v in metrics.items():
            row[k] = float(v)

        log(
            f"Downstream task end | task={task} | encoder={encoder_name} | "
            f"accuracy={row['accuracy']:.6f} | f1_weighted={row['f1_weighted']:.6f} | "
            f"probe_train_time_sec={row['probe_train_time_sec']:.3f} | probe_predict_time_sec={row['probe_predict_time_sec']:.3f}"
        )

        for idx_value, yt, yp in zip(test_index, y_test, y_pred):
            detailed_rows.append(
                {
                    "task": task,
                    "encoder_name": encoder_name,
                    "index": idx_value,
                    "y_true": yt,
                    "y_pred": yp,
                }
            )

    except Exception as e:
        row["error_message"] = str(e)
        log(f"Downstream failed | task={task} | encoder={encoder_name} | error={e}")
        log(traceback.format_exc())

    return row, detailed_rows


def save_outputs(
    metrics_rows,
    detailed_prediction_rows,
    ssl_rows,
    run_log,
    output_dir: str,
):
    os.makedirs(output_dir, exist_ok=True)

    metrics_path = os.path.join(output_dir, "vime_linear_probe_metrics_summary.csv")
    preds_path = os.path.join(output_dir, "vime_linear_probe_predictions_detailed.csv")
    ssl_path = os.path.join(output_dir, "vime_ssl_training_summary.csv")
    run_log_path = os.path.join(output_dir, "vime_linear_probe_run_log.json")

    metrics_columns = [
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
    preds_columns = ["task", "encoder_name", "index", "y_true", "y_pred"]
    ssl_columns = ["encoder_name", "n_features", "ssl_train_time_sec", "final_ssl_loss", "error_message"]

    metrics_df = pd.DataFrame(metrics_rows, columns=metrics_columns)
    ssl_df = pd.DataFrame(ssl_rows, columns=ssl_columns)
    preds_df = pd.DataFrame(detailed_prediction_rows, columns=preds_columns)

    metrics_df.to_csv(metrics_path, index=False)
    ssl_df.to_csv(ssl_path, index=False)
    preds_df.to_csv(preds_path, index=False)

    with open(run_log_path, "w", encoding="utf-8") as f:
        json.dump(to_serializable(run_log), f, ensure_ascii=False, indent=2)

    log(f"Saved metrics summary to: {metrics_path}")
    log(f"Saved detailed predictions to: {preds_path}")
    log(f"Saved SSL training summary to: {ssl_path}")
    log(f"Saved run log to: {run_log_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="VIME linear-probe experiment runner for Kaggle tabular data.")
    parser.add_argument("--unlabeled_path", type=str, default=None, help="Path to jobs_unlabeled_scaled.csv")
    parser.add_argument("--labeled_train_path", type=str, default=None, help="Path to jobs_labeled_train_scaled.csv")
    parser.add_argument("--labeled_test_path", type=str, default=None, help="Path to jobs_labeled_test_scaled.csv")
    parser.add_argument("--output_dir", type=str, default="/kaggle/working")
    parser.add_argument("--ssl_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--ssl_patience", type=int, default=10)
    parser.add_argument("--embedding_batch_size", type=int, default=2048)
    parser.add_argument("--probe_max_iter", type=int, default=5000)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.random_state)

    unresolved_input_paths = {
        "unlabeled_path": args.unlabeled_path,
        "labeled_train_path": args.labeled_train_path,
        "labeled_test_path": args.labeled_test_path,
    }

    input_paths = {
        "unlabeled_path": resolve_input_path(args.unlabeled_path, "jobs_unlabeled_scaled.csv"),
        "labeled_train_path": resolve_input_path(args.labeled_train_path, "jobs_labeled_train_scaled.csv"),
        "labeled_test_path": resolve_input_path(args.labeled_test_path, "jobs_labeled_test_scaled.csv"),
    }

    run_log = {
        "input_paths": input_paths,
        "provided_input_paths": unresolved_input_paths,
        "manual_category_columns": sorted(MANUAL_CATEGORY_COLUMNS),
        "target_columns": TARGET_COLUMNS,
        "feature_mode": "numeric",
        "vime_hyperparameters": {
            "p_m": 0.3,
            "alpha": 2.0,
            "K": 3,
            "beta": 1.0,
        },
        "train_hyperparameters": {
            "ssl_epochs": args.ssl_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "ssl_patience": args.ssl_patience,
            "embedding_batch_size": args.embedding_batch_size,
            "probe_max_iter": args.probe_max_iter,
            "random_state": args.random_state,
            "device": args.device,
        },
        "feature_info": {},
        "encoder_configs": {},
        "task_artifacts": {},
        "ssl_logs": {},
        "downstream_logs": [],
        "pipeline_error": "",
    }

    metrics_rows = []
    detailed_prediction_rows = []
    ssl_rows = []

    try:
        df_unlabeled, df_train, df_test = load_and_clean_data(
            unlabeled_path=input_paths["unlabeled_path"],
            labeled_train_path=input_paths["labeled_train_path"],
            labeled_test_path=input_paths["labeled_test_path"],
        )

        feature_cols, feature_info = build_numeric_feature_set(df_unlabeled, df_train, df_test)
        run_log["feature_info"] = feature_info

        X_unlabeled = prepare_feature_matrix(df_unlabeled, feature_cols, "unlabeled")
        X_train_full = prepare_feature_matrix(df_train, feature_cols, "labeled_train")
        X_test_full = prepare_feature_matrix(df_test, feature_cols, "labeled_test")

        log(
            f"Aligned feature matrix shapes | "
            f"unlabeled={X_unlabeled.shape} | train={X_train_full.shape} | test={X_test_full.shape}"
        )

        encoder_configs = get_encoder_configs(input_dim=X_unlabeled.shape[1])
        run_log["encoder_configs"] = encoder_configs

        task_artifacts = {}
        for task in TARGET_COLUMNS:
            try:
                task_artifacts[task] = prepare_task_artifact(df_train, df_test, task)
            except Exception as e:
                task_artifacts[task] = {
                    "error_message": str(e),
                    "traceback": traceback.format_exc(),
                }
                log(f"Task preparation failed for {task}: {e}")
                log(traceback.format_exc())
        run_log["task_artifacts"] = task_artifacts

        # K và beta được giữ lại trong config/log để bám sát khung VIME gốc,
        # nhưng downstream ở script này là linear probe nên không dùng vime_semi.
        vime_config = {
            "p_m": 0.3,
            "alpha": 2.0,
            "K": 3,
            "beta": 1.0,
            "ssl_epochs": args.ssl_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "ssl_patience": args.ssl_patience,
            "random_state": args.random_state,
            "device": args.device,
        }

        pretrained_encoders, ssl_rows, ssl_logs = run_ssl_once_per_encoder(
            X_unlabeled=X_unlabeled,
            encoder_configs=encoder_configs,
            vime_config=vime_config,
        )
        run_log["ssl_logs"] = ssl_logs

        for encoder_name, encoder_bundle in pretrained_encoders.items():
            if encoder_bundle is None:
                for task in TARGET_COLUMNS:
                    row, preds = run_downstream_task(
                        task=task,
                        encoder_name=encoder_name,
                        encoder_bundle=None,
                        task_artifact=task_artifacts.get(task, {"error_message": f"Task artifact missing for {task}"}),
                        Z_train_full=np.zeros((len(df_train), 1), dtype=np.float32),
                        Z_test_full=np.zeros((len(df_test), 1), dtype=np.float32),
                        n_features=len(feature_cols),
                        probe_max_iter=args.probe_max_iter,
                        random_state=args.random_state,
                    )
                    metrics_rows.append(row)
                    detailed_prediction_rows.extend(preds)
                    run_log["downstream_logs"].append(row)
                continue

            try:
                log(f"Extracting embeddings | encoder={encoder_name} | split=train")
                Z_train_full = extract_embeddings(
                    encoder=encoder_bundle["encoder"],
                    X=X_train_full,
                    batch_size=args.embedding_batch_size,
                    device=args.device,
                )

                log(f"Extracting embeddings | encoder={encoder_name} | split=test")
                Z_test_full = extract_embeddings(
                    encoder=encoder_bundle["encoder"],
                    X=X_test_full,
                    batch_size=args.embedding_batch_size,
                    device=args.device,
                )

                if Z_train_full.shape[1] != encoder_bundle["embedding_dim"]:
                    raise ValueError(
                        f"Encoder output shape mismatch for {encoder_name}: "
                        f"expected embedding_dim={encoder_bundle['embedding_dim']}, got {Z_train_full.shape[1]}"
                    )

            except Exception as e:
                embed_error = f"Embedding extraction failed for {encoder_name}: {e}"
                log(embed_error)
                log(traceback.format_exc())

                for task in TARGET_COLUMNS:
                    row = {
                        "task": task,
                        "encoder_name": encoder_name,
                        "n_features": int(len(feature_cols)),
                        "embedding_dim": int(encoder_bundle["embedding_dim"]),
                        "ssl_train_time_sec": float(encoder_bundle["ssl_train_time_sec"]),
                        "probe_train_time_sec": np.nan,
                        "probe_predict_time_sec": np.nan,
                        "accuracy": np.nan,
                        "f1_micro": np.nan,
                        "f1_weighted": np.nan,
                        "precision_micro": np.nan,
                        "precision_weighted": np.nan,
                        "recall_micro": np.nan,
                        "recall_weighted": np.nan,
                        "error_message": embed_error,
                    }
                    metrics_rows.append(row)
                    run_log["downstream_logs"].append(row)
                continue

            for task in TARGET_COLUMNS:
                row, preds = run_downstream_task(
                    task=task,
                    encoder_name=encoder_name,
                    encoder_bundle=encoder_bundle,
                    task_artifact=task_artifacts.get(task, {"error_message": f"Task artifact missing for {task}"}),
                    Z_train_full=Z_train_full,
                    Z_test_full=Z_test_full,
                    n_features=len(feature_cols),
                    probe_max_iter=args.probe_max_iter,
                    random_state=args.random_state,
                )
                metrics_rows.append(row)
                detailed_prediction_rows.extend(preds)
                run_log["downstream_logs"].append(row)

    except Exception as e:
        run_log["pipeline_error"] = str(e)
        log(f"Pipeline-level error: {e}")
        log(traceback.format_exc())

    finally:
        save_outputs(
            metrics_rows=metrics_rows,
            detailed_prediction_rows=detailed_prediction_rows,
            ssl_rows=ssl_rows,
            run_log=run_log,
            output_dir=args.output_dir,
        )

        if metrics_rows:
            metrics_df = pd.DataFrame(metrics_rows)
            ok_df = metrics_df[metrics_df["error_message"].fillna("") == ""].copy()
            if not ok_df.empty:
                best_df = (
                    ok_df.sort_values(["task", "f1_weighted", "accuracy"], ascending=[True, False, False])
                    .groupby("task", as_index=False)
                    .head(1)
                )
                log("Best encoder per task by f1_weighted then accuracy:")
                log(best_df[["task", "encoder_name", "f1_weighted", "accuracy"]].to_string(index=False))
            else:
                log("No successful downstream runs to summarize.")


if __name__ == "__main__":
    main()