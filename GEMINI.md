# MMMACHINE / idx (BPJS & BSJP Screener)

## Project Overview
This is a fully-automated quantitative trading screener for IDX non-blue chip stocks. It executes a trading strategy (currently focusing on the **BSJP** "Beli Sore Jual Pagi" - Buy Afternoon Sell Morning strategy) driven by empirical probabilities derived from *broker summary* activity and market context.

The system is strictly divided into **two separate products** that share data layers but have distinct code, invariants, and output artifacts:
1. **Training Product (Research):** A Python-based pipeline that processes raw data into Parquet feature marts, trains LightGBM binary classifiers, and evaluates them with walk-forward validation and Monte Carlo simulations. Its deliverable is a `model_lightgbm_*.txt` file.
2. **Inference Product (Production):** A Go-based (historically Python) fast-path pipeline that reads a DuckDB feature store, scores live data against the trained model, and generates daily stock picks. Its deliverable is a ranked list of 3 tickers sent to end users.

### Main Technologies
- **Python (3.12)**: Used for data engineering, pipeline execution, and model training (`pandas`, `pyarrow`, `lightgbm`).
- **Go**: Used for the low-latency production inference binary.
- **DuckDB (1.5.2)**: Used as the production inference feature store and target for Level 0 raw data migration.
- **Parquet**: Used for historical archiving and the training feature layer.
- **LightGBM**: The core machine learning binary classifier.

### Architecture (Data Layers)
- **Level 0 (Raw Data)**: Raw *broker summary*, OHLCV (intraday & daily via yfinance), master data, and global indices.
- **Level 1 (Feature Engineering)**: Strategy-agnostic feature mart (Parquet). Uses a modular feature architecture.
- **Level 2 (Training Datamart)**: Feature aggregation explicitly tied to the trading objective (Parquet). 
- **Inference DB**: Fast DuckDB store for live feature vectors (`inferences/bsjp/db/inference.duckdb`).

---

## Building and Running

### 1. Training / Research Batch Pipeline (Python)
Executed for creating new model versions or retraining:
```bash
# Activate virtual environment
source ../.venv/bin/activate

# Fetch required data
bash pipeline/run/run_fetch_broksum.sh
bash pipeline/run/run_feature_l1.sh

# Build Level 2 datamart & Train
cd edges/bsjp_overnight_sl2/scripts
python generate_datamart.py --strategy-mode bsjp
python train_lightgbm.py \
  --output-dir ../../model/BSJP/bsjp_vNEXT \
  --feature-modules-dir ../../data/Level_1_Features/modules \
  --feature-prune-top-n 0 \
  --tp-pct 0.01 --sl-pct -0.02 \
  --oot-valid-days 100 \
  --min-data-in-leaf 100 \
  --lambda-l1 1.0 --lambda-l2 1.5
```

### 2. Production Inference (Go & Shell)
Executed daily (e.g., via cron @ 15:15 WIB):
```bash
# Build the Go binary (if needed)
cd inferences/bsjp/golang
go build -o bsjp ./cmd/bsjp/

# Run the end-to-end inference script
bash pipeline/run/run_inference_bsjp.sh

# Or run manually via the Go binary
cd inferences/bsjp/golang
./bsjp fetch --date YYYY-MM-DD
./bsjp predict --variant v19d_close10_preclose14_orb_md100_l21.5 --log
```

---

## Development Conventions & Invariants

### Critical Rules (No Lookahead Bias)
- **Zero Lookahead:** All broker/feature columns must use `.shift(1)` to ensure the model only uses data known prior to the decision point.
- **Entry/Exit Integrity:** `entry_price` (e.g., `close@15:xx`) must be known at decision time. `exit_price` (e.g., `open@10:xx T+1`) is strictly a label and NEVER a feature.
- **Walk-forward Validation:** Strict chronological splits. No randomization. The Out-of-Time (OOT) window is the most recent 100 trading days and is used strictly for final evaluation.

### Coding Standards
- **Python:** Use `pandas>=2.0` and `pyarrow>=14.0`. Use `float32` dtype for all feature columns to ensure consistency between Python training and Go inference.
- **Go:** Future inference runtime is Go. Do not add Python to the production inference path unless necessary.
- **Persistence:** Use Parquet for historical data. Use DuckDB for inference vector storage. Never use CSV/JSON for structured pipeline data.
- **Paths:** Use absolute paths or repository-root-relative paths (e.g., `idx/data/Level_0_Raw/`). Avoid hardcoded absolute machine paths (e.g., `/home-ssd/`).

### Protocol and State
- **Documentation is Truth:** `program.md` is the absolute source of truth regarding architecture, current state, and invariants.
- **Session Memory:** Always review the latest session file in `_MEMORY/YYYYMMDDHHMMSS.md` when starting, and create a new session file detailing decisions and handoffs when finishing.
- **Logs:** ALL execution logs must go into the `_LOG/` directory. 
- **Product Boundaries:** Never mix inference (production scoring) and training (research) concerns. Inference must NOT rebuild `L1/L2` datamarts.
