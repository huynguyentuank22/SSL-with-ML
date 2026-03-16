#!/usr/bin/env python
"""
run_ssl_grid_search.py
======================
CLI wrapper for SSL pretraining grid search over DAE, SCARF, and VIME.

Example:
  python src/run_ssl_grid_search.py --family dae
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ssl_benchmark.grid_search_ssl import main


if __name__ == "__main__":
    main()
