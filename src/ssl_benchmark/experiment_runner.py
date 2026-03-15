"""
experiment_runner.py
====================
Orchestrates the 3-tier SSL benchmark pipeline:
  Tier 1: SSL pretraining
  Tier 2: Layer-wise embedding extraction + saving
  Tier 3: Downstream probing (boosting models)

Entry-points used by run_ssl_benchmark.py:
  run_ssl_architecture(...)    — single SSL family × architecture
  run_full_benchmark(...)      — all families × all architectures
"""

import gc
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
# import torch

from .ssl_models import build_ssl_model, get_all_architectures, BaseSSLModel
from .downstream import run_downstream_probe, build_summary


def _free_memory(model: "BaseSSLModel" = None) -> None:
    """Release GPU memory held by a model and run garbage collection."""
    if model is not None:
        # Move all sub-modules to CPU so CUDA memory is freed immediately
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
        mem = torch.cuda.memory_reserved() / 1024 ** 2
        log.info(f"  [mem] GPU reserved after cleanup: {mem:.0f} MB")

log = logging.getLogger(__name__)

ALL_SSL_FAMILIES = ["vime", "scarf", "saint", "subtab", "dae", "tabnet"]


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def _emb_path(
    output_dir: Path,
    ssl_model_name: str,
    arch_name: str,
    layer_name: str,
    split: str,
) -> Path:
    return (
        output_dir / "embeddings" /
        ssl_model_name / arch_name / layer_name /
        f"{split}.npy"
    )


def save_embeddings(
    model: BaseSSLModel,
    X_train: np.ndarray,
    X_test: np.ndarray,
    arch_name: str,
    output_dir: Path,
) -> List[str]:
    """
    Extract embeddings layer-by-layer and save each to disk immediately.
    Arrays are NOT kept in memory after saving — returns list of layer names
    whose embeddings were saved successfully.
    """
    layers = model.list_available_layers()
    saved_layers: List[str] = []

    for layer in layers:
        try:
            log.info(f"  Extracting layer '{layer}' …")
            emb_train = model.extract_embeddings(X_train, layer)
            emb_test  = model.extract_embeddings(X_test,  layer)

            # Save to disk immediately, then release the arrays
            for split, arr in [("train", emb_train), ("test", emb_test)]:
                p = _emb_path(output_dir, model.name, arch_name, layer, split)
                p.parent.mkdir(parents=True, exist_ok=True)
                np.save(p, arr)
                del arr

            del emb_train, emb_test
            gc.collect()

            saved_layers.append(layer)
            log.info(
                f"    saved → "
                f"{_emb_path(output_dir, model.name, arch_name, layer, 'train').parent}"
            )
        except Exception as e:
            log.error(f"  Failed to extract layer '{layer}': {e}\n{traceback.format_exc()}")

    return saved_layers


def load_embeddings_from_disk(
    output_dir: Path,
    ssl_model_name: str,
    arch_name: str,
    layer_name: str,
) -> Optional[Dict[str, np.ndarray]]:
    """Load train/test embeddings from previously saved .npy files."""
    result = {}
    for split in ["train", "test"]:
        p = _emb_path(output_dir, ssl_model_name, arch_name, layer_name, split)
        if not p.exists():
            log.warning(f"Embedding file not found: {p}")
            return None
        result[split] = np.load(p)
    return result


# ---------------------------------------------------------------------------
# Single architecture run
# ---------------------------------------------------------------------------

def run_ssl_architecture(
    *,
    ssl_family:  str,
    arch_name:   str,
    X_train:     np.ndarray,
    X_test:      np.ndarray,
    y_train_dict: Dict[str, np.ndarray],
    y_test_dict:  Dict[str, np.ndarray],
    task_types:  Dict[str, str],
    output_dir:  Path,
    embeddings_dir: Optional[Path] = None,
    epochs:      int  = 50,
    batch_size:  int  = 256,
    lr:          float = 1e-3,
    seed:        int   = 42,
    early_stopping_patience: int = 10,
    early_stopping_min_delta: float = 1e-4,
    val_split: float = 0.1,
    skip_pretrain: bool = False,
    skip_extract:  bool = False,
    skip_probe:    bool = False,
) -> List[Dict]:
    """
    Full pipeline for one (ssl_family, arch_name) combination.

    Returns a list of metric dicts from downstream probing.
    """
    log.info("=" * 70)
    log.info(f"SSL family: {ssl_family}  |  Architecture: {arch_name}")
    log.info("=" * 70)

    embeddings_root = embeddings_dir if embeddings_dir is not None else output_dir

    ckpt_path = (output_dir / "checkpoints" /
                 ssl_family / f"{arch_name}.pt")

    # ── Tier 1: Pretrain ──────────────────────────────────────────────────
    try:
        model = build_ssl_model(
            ssl_family, arch_name,
            input_dim=X_train.shape[1],
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            seed=seed,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
            val_split=val_split,
        )
    except Exception as e:
        log.error(f"Could not build model {ssl_family}/{arch_name}: {e}")
        return []

    if not skip_pretrain:
        if ckpt_path.exists():
            log.info(f"Checkpoint found, loading → {ckpt_path}")
            try:
                model.load_checkpoint(ckpt_path)
            except Exception as e:
                log.warning(f"Failed to load checkpoint ({e}), retraining …")
                _do_pretrain(model, X_train, ckpt_path)
        else:
            _do_pretrain(model, X_train, ckpt_path)
    else:
        if ckpt_path.exists():
            model.load_checkpoint(ckpt_path)
        else:
            log.warning(
                f"--skip_pretrain set but no checkpoint at {ckpt_path}. "
                "Skipping this architecture."
            )
            return []

    # ── Tier 2: Extract embeddings ────────────────────────────────────────
    available_layers = model.list_available_layers()

    if not skip_extract:
        saved_layers = save_embeddings(model, X_train, X_test, arch_name, embeddings_root)
    else:
        # Determine which layers are already on disk
        saved_layers = [
            layer for layer in available_layers
            if _emb_path(embeddings_root, ssl_family, arch_name, layer, "train").exists()
            and _emb_path(embeddings_root, ssl_family, arch_name, layer, "test").exists()
        ]
        missing = set(available_layers) - set(saved_layers)
        if missing:
            log.warning(f"Embeddings not found on disk for layers: {missing}")

    # ── Free model from GPU memory before probing ─────────────────────────
    _free_memory(model)
    model = None  # ensure Python GC can collect

    if not saved_layers:
        log.error("No embeddings available — aborting probe stage.")
        return []

    # ── Tier 3: Downstream probing (stream one layer at a time) ───────────
    all_results: List[Dict] = []

    if not skip_probe:
        for layer_name in saved_layers:
            # Load this layer's embeddings from disk, probe, then release
            embs = load_embeddings_from_disk(embeddings_root, ssl_family, arch_name, layer_name)
            if embs is None:
                log.warning(f"  Skipping layer '{layer_name}' — embeddings missing.")
                continue

            emb_train = embs["train"]
            emb_test  = embs["test"]
            del embs

            for task_name, task_type in task_types.items():
                if task_name not in y_train_dict:
                    continue
                results = run_downstream_probe(
                    emb_train=emb_train,
                    emb_test=emb_test,
                    y_train=y_train_dict[task_name],
                    y_test=y_test_dict[task_name],
                    task_name=task_name,
                    task_type=task_type,
                    ssl_model_name=ssl_family,
                    arch_name=arch_name,
                    layer_name=layer_name,
                    output_dir=output_dir,
                )
                all_results.extend(results)

            # Release embedding arrays for this layer before loading the next
            del emb_train, emb_test
            gc.collect()

    return all_results


def _do_pretrain(model: BaseSSLModel, X_train: np.ndarray, ckpt_path: Path):
    try:
        model.fit_ssl(X_train)
        model.save_checkpoint(ckpt_path)
    except Exception as e:
        log.error(
            f"Pretraining failed for {model.name}/{model.arch_name}: "
            f"{e}\n{traceback.format_exc()}"
        )
    finally:
        # Always flush GPU after training regardless of success/failure
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Full benchmark
# ---------------------------------------------------------------------------

def run_full_benchmark(
    *,
    ssl_families:  List[str],
    X_train:       np.ndarray,
    X_test:        np.ndarray,
    y_train_dict:  Dict[str, np.ndarray],
    y_test_dict:   Dict[str, np.ndarray],
    task_types:    Dict[str, str],
    output_dir:    Path,
    embeddings_dir: Optional[Path] = None,
    epochs:        int   = 50,
    batch_size:    int   = 256,
    lr:            float = 1e-3,
    seed:          int   = 42,
    early_stopping_patience: int = 10,
    early_stopping_min_delta: float = 1e-4,
    val_split: float = 0.1,
    skip_pretrain: bool  = False,
    skip_extract:  bool  = False,
    skip_probe:    bool  = False,
    tasks_filter:  Optional[List[str]] = None,
) -> None:
    """
    Run the full SSL benchmark across all requested families and architectures.
    Results are accumulated and a final summary CSV is written.
    """
    if tasks_filter:
        task_types = {k: v for k, v in task_types.items() if k in tasks_filter}
        y_train_dict = {k: v for k, v in y_train_dict.items() if k in tasks_filter}
        y_test_dict  = {k: v for k, v in y_test_dict.items()  if k in tasks_filter}

    all_results: List[Dict] = []

    for family in ssl_families:
        archs = get_all_architectures(family)
        for arch in archs:
            try:
                results = run_ssl_architecture(
                    ssl_family=family,
                    arch_name=arch,
                    X_train=X_train,
                    X_test=X_test,
                    y_train_dict=y_train_dict,
                    y_test_dict=y_test_dict,
                    task_types=task_types,
                    output_dir=output_dir,
                    embeddings_dir=embeddings_dir,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    seed=seed,
                    early_stopping_patience=early_stopping_patience,
                    early_stopping_min_delta=early_stopping_min_delta,
                    val_split=val_split,
                    skip_pretrain=skip_pretrain,
                    skip_extract=skip_extract,
                    skip_probe=skip_probe,
                )
                all_results.extend(results)
            except Exception as e:
                log.error(
                    f"Unhandled error for {family}/{arch}: "
                    f"{e}\n{traceback.format_exc()}"
                )

    # Save per-run intermediate CSV to avoid losing data on crash
    if all_results:
        import pandas as pd
        pd.DataFrame(all_results).to_csv(
            output_dir / "summary_results.csv", index=False
        )
        log.info(f"Intermediate summary saved → {output_dir / 'summary_results.csv'}")

    # Final polished summary with best-by tables
    build_summary(all_results, output_dir)

    log.info("=" * 70)
    log.info("Benchmark complete.")
    log.info(f"Total result rows: {len(all_results)}")
    log.info(f"Output directory:  {output_dir}")
    log.info("=" * 70)
