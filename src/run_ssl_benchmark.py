#!/usr/bin/env python
"""
run_ssl_benchmark.py
====================
CLI entry-point for the SSL tabular benchmark framework.

Usage examples
--------------
# Pretrain + extract + probe ALL models on ALL tasks
  python src/run_ssl_benchmark.py

# Only run one SSL family
  python src/run_ssl_benchmark.py --ssl_model vime

# Multiple families
  python src/run_ssl_benchmark.py --ssl_model vime scarf dae

# Only one task
  python src/run_ssl_benchmark.py --task duration

# Multiple tasks
  python src/run_ssl_benchmark.py --task duration pclass

# Stage-level control
  python src/run_ssl_benchmark.py --extract_only        # assume checkpoints exist, skip probe
  python src/run_ssl_benchmark.py --probe_only          # assume embeddings exist
  python src/run_ssl_benchmark.py --ssl_model saint --task pclass --probe_only

Training hyper-parameter overrides
-----------------------------------
  python src/run_ssl_benchmark.py --epochs 100 --batch_size 512 --lr 0.0005

Output
------
  ssl_benchmark_outputs/
    checkpoints/         pretrained model checkpoints
    embeddings/          .npy embeddings per layer
    predictions/         prediction CSVs
    summary_results.csv  full metric table
    best_by_task.csv
    best_by_ssl_model.csv
    best_by_layer.csv
    logs/                rotating log file
"""

import argparse
import logging
import sys
from pathlib import Path

# Project root inferred from this file location: <repo>/src/run_ssl_benchmark.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "ssl_benchmark_outputs"

# Setup logging to both stdout and a file
LOG_DIR = OUTPUT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "benchmark.log", mode="a", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# Make sure <repo>/src is importable when running this file directly
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ssl_benchmark.data_utils import load_data, DEFAULT_TARGET_COLS
from ssl_benchmark.experiment_runner import (
    run_full_benchmark,
    run_ssl_architecture,
    ALL_SSL_FAMILIES,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="SSL Tabular Benchmark for HPC data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model selection
    p.add_argument(
        "--ssl_model", nargs="+",
        choices=ALL_SSL_FAMILIES + ["all"],
        default=["all"],
        help="Which SSL family/families to run. Default: all.",
    )

    # Task selection
    p.add_argument(
        "--task", nargs="+",
        default=None,
        help="Which prediction tasks to evaluate. Default: all 4 tasks.",
    )

    # Architecture selection (optional; if not given, all are run)
    p.add_argument(
        "--arch", nargs="+", default=None,
        help=(
            "Specific architecture names to run (e.g. vime_small saint_large). "
            "If omitted, all architectures for each selected ssl_model are run."
        ),
    )

    # Stage control
    p.add_argument(
        "--extract_only", action="store_true",
        help="Pretrain (or load ckpt) + extract embeddings only; skip probing.",
    )
    p.add_argument(
        "--probe_only", action="store_true",
        help="Load existing embeddings and run downstream probing only.",
    )

    # Training hyper-params
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--early_stopping_patience", type=int, default=10)
    p.add_argument("--early_stopping_min_delta", type=float, default=1e-4)
    p.add_argument("--val_split", type=float, default=0.1)

    # Data options
    p.add_argument(
        "--data_dir", type=Path, default=DATA_DIR,
        help="Directory containing train.parquet and test.parquet.",
    )
    p.add_argument(
        "--output_dir", type=Path, default=OUTPUT_DIR,
        help="Root output directory.",
    )
    p.add_argument(
        "--exclude_cols", nargs="*", default=None,
        help="Extra columns to exclude from features.",
    )
    p.add_argument(
        "--no_scale", action="store_true",
        help="Disable StandardScaler preprocessing.",
    )

    return p.parse_args()


def main():
    args = parse_args()

    log.info("=" * 70)
    log.info("SSL TABULAR BENCHMARK starting ...")
    log.info(f"  ssl_model   : {args.ssl_model}")
    log.info(f"  task        : {args.task}")
    log.info(f"  arch        : {args.arch}")
    log.info(f"  extract_only: {args.extract_only}")
    log.info(f"  probe_only  : {args.probe_only}")
    log.info(f"  epochs      : {args.epochs}")
    log.info(f"  batch_size  : {args.batch_size}")
    log.info(f"  lr          : {args.lr}")
    log.info(f"  seed        : {args.seed}")
    log.info(f"  es_patience : {args.early_stopping_patience}")
    log.info(f"  es_min_delta: {args.early_stopping_min_delta}")
    log.info(f"  val_split   : {args.val_split}")
    log.info(f"  data_dir    : {args.data_dir}")
    log.info(f"  output_dir  : {args.output_dir}")
    log.info("=" * 70)

    data = load_data(
        data_dir=args.data_dir,
        target_cols=DEFAULT_TARGET_COLS,
        exclude_cols=args.exclude_cols,
        scale=not args.no_scale,
    )
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train_dict = data["y_train"]
    y_test_dict = data["y_test"]
    task_types = data["task_types"]

    log.info(
        f"Data loaded: X_train={X_train.shape}, X_test={X_test.shape}, "
        f"tasks={list(task_types.keys())}"
    )

    if "all" in args.ssl_model:
        families = ALL_SSL_FAMILIES
    else:
        families = list(args.ssl_model)

    skip_pretrain = args.probe_only
    skip_extract = args.probe_only
    skip_probe = args.extract_only

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.arch:
        from ssl_benchmark.downstream import build_summary

        all_results = []
        for arch_name in args.arch:
            family = arch_name.split("_")[0]
            if family not in families:
                log.warning(
                    f"Architecture '{arch_name}' family '{family}' not in selected "
                    f"families {families}. Skipping."
                )
                continue
            results = run_ssl_architecture(
                ssl_family=family,
                arch_name=arch_name,
                X_train=X_train,
                X_test=X_test,
                y_train_dict=y_train_dict,
                y_test_dict=y_test_dict,
                task_types=task_types,
                output_dir=args.output_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                seed=args.seed,
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_min_delta=args.early_stopping_min_delta,
                val_split=args.val_split,
                skip_pretrain=skip_pretrain,
                skip_extract=skip_extract,
                skip_probe=skip_probe,
            )
            all_results.extend(results)
        build_summary(all_results, args.output_dir)
    else:
        run_full_benchmark(
            ssl_families=families,
            X_train=X_train,
            X_test=X_test,
            y_train_dict=y_train_dict,
            y_test_dict=y_test_dict,
            task_types=task_types,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_delta=args.early_stopping_min_delta,
            val_split=args.val_split,
            skip_pretrain=skip_pretrain,
            skip_extract=skip_extract,
            skip_probe=skip_probe,
            tasks_filter=args.task,
        )

    log.info("Done.")


if __name__ == "__main__":
    main()
