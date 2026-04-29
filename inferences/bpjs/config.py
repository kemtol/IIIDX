"""
Shared infrastructure config for BPJS inference.
Model/policy params live in variants/<name>.py.

NOTE: BPJS is currently PAUSED (v16b AUC=0.723 but return=-16%).
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # idx/

# ── Inference DB ──────────────────────────────────────────────────────────────
INFERENCE_DB    = Path(__file__).parent / "db/inference.duckdb"
DB_HISTORY_DAYS = 90

# ── Feature pipeline (L1 / L2) ───────────────────────────────────────────────
L1_SCRIPT            = REPO_ROOT / "pipeline/feature/generate_broksum_datamart.py"
L2_SCRIPT            = REPO_ROOT / "edges/bpjs_intraday/scripts/generate_datamart.py"
L2_PARQUET           = REPO_ROOT / "data/Level_2_Datamart/training_datamart_bpjs_intraday.parquet"
L1_WARMUP_MULTIPLIER = 1

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR = REPO_ROOT / "_LOG"
