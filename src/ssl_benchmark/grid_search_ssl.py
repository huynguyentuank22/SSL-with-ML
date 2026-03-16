"""
grid_search_ssl.py
==================
Grid-search SSL pretraining hyperparameters for ONE family at a time
(DAE, SCARF, or VIME), and select the best configuration by validation loss.

Notes:
- This script runs one family per execution.

Usage example:
    python -m ssl_benchmark.grid_search_ssl --family dae
"""

import argparse
import gc
import itertools
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch

from .data_utils import DEFAULT_TARGET_COLS, load_data
from .ssl_models import (
    DAEModel,
    SCARFModel,
    VIMEModel,
    make_loader,
    split_ssl_train_val,
)

log = logging.getLogger(__name__)


def _parse_int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_float_list(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_str_list(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _trial_space_dae(common: Dict[str, List], dae_arch: List[str]) -> Iterable[Dict]:
    for arch_name, epochs, batch_size, lr, es_pat, es_delta in itertools.product(
        dae_arch,
        common["epochs"],
        common["batch_size"],
        common["lr"],
        common["early_stopping_patience"],
        common["early_stopping_min_delta"],
    ):
        yield {
            "model_family": "dae",
            "arch_name": arch_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "early_stopping_patience": es_pat,
            "early_stopping_min_delta": es_delta,
        }


def _trial_space_vime(
    common: Dict[str, List],
    vime_arch: List[str],
    p_mask_grid: List[float],
    alpha_grid: List[float],
) -> Iterable[Dict]:
    for arch_name, epochs, batch_size, lr, es_pat, es_delta, p_mask, alpha in itertools.product(
        vime_arch,
        common["epochs"],
        common["batch_size"],
        common["lr"],
        common["early_stopping_patience"],
        common["early_stopping_min_delta"],
        p_mask_grid,
        alpha_grid,
    ):
        yield {
            "model_family": "vime",
            "arch_name": arch_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "early_stopping_patience": es_pat,
            "early_stopping_min_delta": es_delta,
            "p_mask": p_mask,
            "alpha": alpha,
        }


def _trial_space_scarf(
    common: Dict[str, List],
    scarf_arch: List[str],
    corruption_rate_grid: List[float],
    temperature_grid: List[float],
) -> Iterable[Dict]:
    for arch_name, epochs, batch_size, lr, es_pat, es_delta, corruption_rate, temperature in itertools.product(
        scarf_arch,
        common["epochs"],
        common["batch_size"],
        common["lr"],
        common["early_stopping_patience"],
        common["early_stopping_min_delta"],
        corruption_rate_grid,
        temperature_grid,
    ):
        yield {
            "model_family": "scarf",
            "arch_name": arch_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "early_stopping_patience": es_pat,
            "early_stopping_min_delta": es_delta,
            "corruption_rate": corruption_rate,
            "temperature": temperature,
        }


def _build_model(input_dim: int, seed: int, params: Dict):
    family = params["model_family"]
    common_kwargs = dict(
        input_dim=input_dim,
        arch_name=params["arch_name"],
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        lr=params["lr"],
        seed=seed,
        early_stopping_patience=params["early_stopping_patience"],
        early_stopping_min_delta=params["early_stopping_min_delta"],
        val_split=0.0,
    )

    if family == "dae":
        return DAEModel(**common_kwargs)
    if family == "vime":
        return VIMEModel(
            **common_kwargs,
            p_mask=params["p_mask"],
            alpha=params["alpha"],
        )
    if family == "scarf":
        return SCARFModel(
            **common_kwargs,
            corruption_rate=params["corruption_rate"],
            temperature=params["temperature"],
        )
    raise ValueError(f"Unsupported model family: {family}")


def _cleanup_model(model) -> None:
    if model is not None:
        try:
            for attr in vars(model).values():
                if isinstance(attr, torch.nn.Module):
                    attr.cpu()
                elif isinstance(attr, torch.nn.ModuleList):
                    attr.cpu()
        except Exception:
            pass
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _evaluate_valid_loss(model, X_val: np.ndarray, batch_size: int) -> float:
    if X_val is None or len(X_val) == 0:
        raise ValueError("Validation set is empty. Increase train size or val_split.")
    if not hasattr(model, "_evaluate_loss"):
        raise AttributeError("Model does not implement _evaluate_loss(loader).")

    val_loader = make_loader(X_val, batch_size=batch_size, shuffle=False)
    return float(model._evaluate_loss(val_loader))


def run_grid_search(
    *,
    X_train: np.ndarray,
    val_split: float,
    seed: int,
    trials: List[Dict],
    max_trials: int,
) -> pd.DataFrame:
    X_fit, X_val = split_ssl_train_val(X_train, val_split, seed)
    if X_val is None:
        raise ValueError("No validation split created. Set val_split > 0 and ensure enough rows.")

    if max_trials > 0:
        trials = trials[:max_trials]

    records: List[Dict] = []
    input_dim = X_fit.shape[1]

    log.info("=" * 80)
    log.info("Grid search started")
    log.info(f"Train rows: {len(X_fit)} | Valid rows: {len(X_val)} | Trials: {len(trials)}")
    log.info("=" * 80)

    for i, params in enumerate(trials, start=1):
        model = None
        try:
            model = _build_model(input_dim=input_dim, seed=seed, params=params)
            model.fit_ssl(X_fit)
            val_loss = _evaluate_valid_loss(model, X_val, batch_size=params["batch_size"])

            row = {
                "trial_id": i,
                "model_family": params["model_family"],
                "arch_name": params["arch_name"],
                "val_loss": val_loss,
                "params_json": json.dumps(params, sort_keys=True),
            }
            row.update(params)
            records.append(row)

            log.info(
                f"[{i:04d}/{len(trials):04d}] "
                f"{params['model_family']}/{params['arch_name']} "
                f"val_loss={val_loss:.6f}"
            )
        except Exception as exc:
            err = {
                "trial_id": i,
                "model_family": params.get("model_family", "unknown"),
                "arch_name": params.get("arch_name", "unknown"),
                "val_loss": np.nan,
                "params_json": json.dumps(params, sort_keys=True),
                "error": str(exc),
            }
            err.update(params)
            records.append(err)
            log.exception(f"Trial failed: {params}")
        finally:
            _cleanup_model(model)

    df = pd.DataFrame(records)
    return df


def _build_trials_from_args(args) -> List[Dict]:
    common = {
        "epochs": _parse_int_list(args.epochs_grid),
        "batch_size": _parse_int_list(args.batch_size_grid),
        "lr": _parse_float_list(args.lr_grid),
        "early_stopping_patience": _parse_int_list(args.early_stopping_patience_grid),
        "early_stopping_min_delta": _parse_float_list(args.early_stopping_min_delta_grid),
    }

    family = args.family

    if family == "dae":
        return list(
            _trial_space_dae(
                common=common,
                dae_arch=_parse_str_list(args.dae_arch_grid),
            )
        )

    if family == "vime":
        return list(
            _trial_space_vime(
                common=common,
                vime_arch=_parse_str_list(args.vime_arch_grid),
                p_mask_grid=_parse_float_list(args.vime_p_mask_grid),
                alpha_grid=_parse_float_list(args.vime_alpha_grid),
            )
        )

    if family == "scarf":
        return list(
            _trial_space_scarf(
                common=common,
                scarf_arch=_parse_str_list(args.scarf_arch_grid),
                corruption_rate_grid=_parse_float_list(args.scarf_corruption_rate_grid),
                temperature_grid=_parse_float_list(args.scarf_temperature_grid),
            )
        )

    raise ValueError(f"Unsupported family: {family}")


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]

    p = argparse.ArgumentParser(
        description="Grid search SSL hyperparameters for one family by validation loss."
    )
    p.add_argument("--family", type=str, required=True, choices=["dae", "scarf", "vime"])

    p.add_argument("--data_dir", type=Path, default=project_root / "data")
    p.add_argument("--output_dir", type=Path, default=project_root / "results" / "grid_search_ssl")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--max_trials", type=int, default=0, help="0 means run all generated trials.")
    p.add_argument("--no_scale", action="store_true")

    p.add_argument("--epochs_grid", type=str, default="50")
    p.add_argument("--batch_size_grid", type=str, default="256")
    p.add_argument("--lr_grid", type=str, default="0.001,0.0003,0.0001,0.00003")
    p.add_argument("--early_stopping_patience_grid", type=str, default="10")
    p.add_argument("--early_stopping_min_delta_grid", type=str, default="0.0001")

    p.add_argument("--dae_arch_grid", type=str, default="dae_large")

    p.add_argument("--vime_arch_grid", type=str, default="vime_large")
    p.add_argument("--vime_p_mask_grid", type=str, default="0.2,0.3")
    p.add_argument("--vime_alpha_grid", type=str, default="1.0,2.0")

    p.add_argument("--scarf_arch_grid", type=str, default="scarf_large")
    p.add_argument("--scarf_corruption_rate_grid", type=str, default="0.4,0.6")
    p.add_argument("--scarf_temperature_grid", type=str, default="0.07,0.1")

    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()
    family_output_dir = args.output_dir / args.family
    family_output_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(
        data_dir=args.data_dir,
        target_cols=DEFAULT_TARGET_COLS,
        scale=not args.no_scale,
    )
    X_train = data["X_train"]

    trials = _build_trials_from_args(args)
    if not trials:
        raise ValueError("No trials generated. Check --models and grid arguments.")

    df = run_grid_search(
        X_train=X_train,
        val_split=args.val_split,
        seed=args.seed,
        trials=trials,
        max_trials=args.max_trials,
    )

    all_csv = family_output_dir / "grid_search_all_trials.csv"
    df.to_csv(all_csv, index=False)

    ok_df = df[df["val_loss"].notna()].copy()
    if ok_df.empty:
        raise RuntimeError("All trials failed. Check logs and hyperparameter ranges.")

    best_trial = ok_df.sort_values("val_loss", ascending=True).head(1)
    best_trial_csv = family_output_dir / "grid_search_best_trial.csv"
    best_trial.to_csv(best_trial_csv, index=False)

    top10_csv = family_output_dir / "grid_search_top10.csv"
    ok_df.sort_values("val_loss", ascending=True).head(10).to_csv(top10_csv, index=False)

    log.info("=" * 80)
    log.info(f"Family                 -> {args.family}")
    log.info(f"Saved all trials        -> {all_csv}")
    log.info(f"Saved best trial        -> {best_trial_csv}")
    log.info(f"Saved top-10 leaderboard-> {top10_csv}")

    top = best_trial.iloc[0]
    log.info(
        "Best trial by valid loss: "
        f"{top['model_family']}/{top['arch_name']} val_loss={top['val_loss']:.6f}"
    )
    log.info("=" * 80)


if __name__ == "__main__":
    main()
