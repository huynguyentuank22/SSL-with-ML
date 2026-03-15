#!/usr/bin/env python
"""
run_ssl_probe.py
================
Standalone CLI for downstream probing using precomputed SSL embeddings.

This script does NOT run pretraining or embedding extraction.
It only reads train/test embeddings from disk and trains downstream models.

Expected embedding layout:
  <embeddings_dir>/embeddings/<ssl_family>/<arch_name>/<layer_name>/train.npy
  <embeddings_dir>/embeddings/<ssl_family>/<arch_name>/<layer_name>/test.npy

Usage examples
--------------
# Probe all discovered embeddings
  python src/run_ssl_probe.py

# Probe specific SSL family/families
  python src/run_ssl_probe.py --ssl_model vime
  python src/run_ssl_probe.py --ssl_model vime scarf

# Probe only selected tasks
  python src/run_ssl_probe.py --task duration pclass

# Probe only selected architectures
  python src/run_ssl_probe.py --arch vime_small vime_large

# Read embeddings from one place, write results to another
  python src/run_ssl_probe.py --embeddings_dir results/old_run --output_dir results/probe_only_run
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Project root inferred from this file location: <repo>/src/run_ssl_probe.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "ssl_benchmark_outputs"

# Make sure <repo>/src is importable when running this file directly
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ssl_benchmark.data_utils import load_data, DEFAULT_TARGET_COLS
from ssl_benchmark.downstream import run_downstream_probe, build_summary
from ssl_benchmark.experiment_runner import ALL_SSL_FAMILIES

log = logging.getLogger(__name__)


def _setup_logging(output_dir: Path) -> None:
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "probe.log", mode="a", encoding="utf-8"),
        ],
        force=True,
    )


def _discover_layers(
    embeddings_dir: Path,
    families: List[str],
    arch_filter: Optional[List[str]] = None,
) -> List[Tuple[str, str, str]]:
    """Return list of (family, arch_name, layer_name) with valid train/test embeddings."""
    root = embeddings_dir / "embeddings"
    discovered: List[Tuple[str, str, str]] = []

    if not root.exists():
        log.warning(f"Embeddings root does not exist: {root}")
        return discovered

    arch_filter_set = set(arch_filter) if arch_filter else None

    for family in families:
        family_dir = root / family
        if not family_dir.exists():
            log.warning(f"No embeddings directory for family '{family}': {family_dir}")
            continue

        for arch_dir in sorted([p for p in family_dir.iterdir() if p.is_dir()]):
            arch_name = arch_dir.name
            if arch_filter_set is not None and arch_name not in arch_filter_set:
                continue

            for layer_dir in sorted([p for p in arch_dir.iterdir() if p.is_dir()]):
                layer_name = layer_dir.name
                train_path = layer_dir / "train.npy"
                test_path = layer_dir / "test.npy"

                if train_path.exists() and test_path.exists():
                    discovered.append((family, arch_name, layer_name))
                else:
                    log.warning(
                        f"Missing embeddings for {family}/{arch_name}/{layer_name} "
                        f"(train={train_path.exists()}, test={test_path.exists()})"
                    )

    return discovered


def parse_args():
    p = argparse.ArgumentParser(
        description="Standalone downstream probe for SSL embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument(
        "--ssl_model", nargs="+",
        choices=ALL_SSL_FAMILIES + ["all"],
        default=["all"],
        help="Which SSL family/families to probe. Default: all.",
    )
    p.add_argument(
        "--task", nargs="+", default=None,
        help="Which prediction tasks to probe. Default: all tasks.",
    )
    p.add_argument(
        "--arch", nargs="+", default=None,
        help="Specific architecture names to probe (e.g. vime_small).",
    )

    p.add_argument(
        "--data_dir", type=Path, default=DATA_DIR,
        help="Directory containing train.parquet and test.parquet.",
    )
    p.add_argument(
        "--output_dir", type=Path, default=OUTPUT_DIR,
        help="Where to write predictions and summary tables.",
    )
    p.add_argument(
        "--embeddings_dir", type=Path, default=None,
        help="Where to read embeddings. Defaults to --output_dir when omitted.",
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

    output_dir = args.output_dir
    embeddings_dir = args.embeddings_dir if args.embeddings_dir is not None else output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(output_dir)

    if "all" in args.ssl_model:
        families = ALL_SSL_FAMILIES
    else:
        families = list(args.ssl_model)

    log.info("=" * 70)
    log.info("SSL PROBE-ONLY RUN starting ...")
    log.info(f"  ssl_model      : {families}")
    log.info(f"  task           : {args.task}")
    log.info(f"  arch           : {args.arch}")
    log.info(f"  data_dir       : {args.data_dir}")
    log.info(f"  embeddings_dir : {embeddings_dir}")
    log.info(f"  output_dir     : {output_dir}")
    log.info("=" * 70)

    data = load_data(
        data_dir=args.data_dir,
        target_cols=DEFAULT_TARGET_COLS,
        exclude_cols=args.exclude_cols,
        scale=not args.no_scale,
    )
    y_train_dict: Dict[str, np.ndarray] = data["y_train"]
    y_test_dict: Dict[str, np.ndarray] = data["y_test"]
    task_types: Dict[str, str] = data["task_types"]

    if args.task:
        selected = set(args.task)
        task_types = {k: v for k, v in task_types.items() if k in selected}
        y_train_dict = {k: v for k, v in y_train_dict.items() if k in selected}
        y_test_dict = {k: v for k, v in y_test_dict.items() if k in selected}

    discovered = _discover_layers(
        embeddings_dir=embeddings_dir,
        families=families,
        arch_filter=args.arch,
    )

    if not discovered:
        log.error("No valid embeddings discovered. Nothing to probe.")
        return

    log.info(f"Discovered {len(discovered)} layer(s) to probe.")

    all_results: List[Dict] = []

    for family, arch_name, layer_name in discovered:
        layer_dir = embeddings_dir / "embeddings" / family / arch_name / layer_name
        train_path = layer_dir / "train.npy"
        test_path = layer_dir / "test.npy"

        try:
            emb_train = np.load(train_path)
            emb_test = np.load(test_path)
        except Exception as exc:
            log.error(
                f"Failed loading embeddings for {family}/{arch_name}/{layer_name}: {exc}"
            )
            continue

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
                ssl_model_name=family,
                arch_name=arch_name,
                layer_name=layer_name,
                output_dir=output_dir,
            )
            all_results.extend(results)

        del emb_train, emb_test

    build_summary(all_results, output_dir)

    log.info("=" * 70)
    log.info("Probe-only run complete.")
    log.info(f"Total result rows: {len(all_results)}")
    log.info(f"Output directory:  {output_dir}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
