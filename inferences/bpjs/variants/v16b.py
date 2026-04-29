"""
BPJS v16b — current best model (PAUSED: high AUC but negative returns).
Do not run in production until returns are positive.
"""
from pathlib import Path

REPO_ROOT    = Path(__file__).resolve().parents[3]  # idx/

MODEL_VERSION = "bpjs_v13"
MODEL_PATH    = REPO_ROOT / "model/BPJS/bpjs_v13/model_lightgbm_opening_tp3.txt"

TOP_K         = 3
RANK_WEIGHTS  = [0.6, 0.3, 0.1]
MIN_PROBA     = 0.04
