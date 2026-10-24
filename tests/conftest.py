"""Shared test setup: run offline and make sure trained models exist."""
import os
import sys
from pathlib import Path

os.environ.setdefault("ARRIVO_OFFLINE", "1")          # tests must never depend on the internet
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402

if not config.ETA_MODEL_FILE.exists():                # first run on a fresh clone -> quick training
    from src.pipeline import run
    run(n_trials=1, force_synthetic=True)
