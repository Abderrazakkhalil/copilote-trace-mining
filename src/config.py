"""Project-wide paths and constants.

Every module should ``from src.config import *`` (or import selectively)
to get reproducible, consistent path references.
"""

from pathlib import Path

# ── Root ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Data ────────────────────────────────────────────────────────────────
DATA_DIR      = PROJECT_ROOT / "data"
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

TRAIN_CSV = RAW_DIR / "train.csv"
TEST_CSV  = RAW_DIR / "test.csv"

# ── Outputs ─────────────────────────────────────────────────────────────
OUTPUTS_DIR      = PROJECT_ROOT / "outputs"
FIGURES_DIR      = PROJECT_ROOT / "figures"
MODEL_FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR       = OUTPUTS_DIR / "tables"
SUBMISSIONS_DIR  = OUTPUTS_DIR / "submissions"

# ── Reproducibility ────────────────────────────────────────────────────
RANDOM_STATE = 42
N_SPLITS     = 4          # stratified k-fold

# ── Auto-create directories ────────────────────────────────────────────
for _d in (RAW_DIR, PROCESSED_DIR, FIGURES_DIR, MODEL_FIGURES_DIR, TABLES_DIR, SUBMISSIONS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
