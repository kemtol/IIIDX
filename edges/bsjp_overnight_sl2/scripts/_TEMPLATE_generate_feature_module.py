#!/usr/bin/env python3
"""
TEMPLATE: Generate a new feature module parquet.

WORKFLOW
────────
1. Copy this file  →  generate_{your_family}_features.py
2. Change MODULE_NAME and GRAIN at the bottom of the CONFIG section
3. Implement build_features() with your feature logic
4. Run:  python generate_{your_family}_features.py
5. Output → data/Level_1_Features/modules/{your_family}_features.parquet
6. Training auto-detects via `--feature-modules-dir` — no code changes needed

REQUIREMENTS
────────────
- Output columns: date [, ticker] + numeric feature columns (float32)
- Grain: (date, ticker) for stock-level features, (date,) for market-level
- No lookahead: use shift(1) on any forward-looking data
- Feature column names must not conflict with existing modules
  (check data/Level_1_Features/modules/*_features.parquet columns)

EXAMPLES
────────
Stock-level (date, ticker):
    df = pd.DataFrame({"date": [...], "ticker": [...], "my_feat_ma_20d": [...], ...})

Market-level (date,) — e.g. macro/global indices:
    df = pd.DataFrame({"date": [...], "my_feat_rate": [...], ...})
"""

import argparse
from pathlib import Path
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════
# CONFIG  —  Change these for your new feature family
# ═══════════════════════════════════════════════════════════════════════
MODULE_NAME = "new_family"           # ← Your feature family name (no spaces)
GRAIN = "stock"                      # ← "stock" for (date,ticker), "market" for (date,)
# ═══════════════════════════════════════════════════════════════════════

DATA_DIR = Path("data")
FEATURES_DIR = DATA_DIR / "Level_1_Features"
MODULES_DIR = FEATURES_DIR / "modules"


def load_sources() -> dict[str, pd.DataFrame]:
    """Load L1 data sources needed for feature engineering.

    Customize this to load whatever sources your features need:
    - broksum_datamart.parquet   — broker aggregate + stock universe
    - vwap_features.parquet      — VWAP-derived features
    - yfinance_1h.parquet        — OHLCV 1-hour bars
    - yfinance_daily.parquet     — OHLCV daily bars
    - global_indices.parquet     — IHSG, VIX, etc.
    - existing modules/*.parquet — build on top of other feature families
    """
    return {
        "broksum": pd.read_parquet(FEATURES_DIR / "broksum_datamart.parquet"),
        "yf_1h": pd.read_parquet(DATA_DIR / "Level_0_Raw" / "yfinance_1h.parquet"),
        "yf_daily": pd.read_parquet(DATA_DIR / "Level_0_Raw" / "yfinance_daily.parquet"),
    }


def build_features(sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """╔══════════════════════════════════════════════════════════════╗
       ║  CORE LOGIC — Implement your feature engineering here.     ║
       ╚══════════════════════════════════════════════════════════════╝

    Args:
        sources: dict of DataFrames from load_sources()

    Returns:
        DataFrame with merge key column(s) + feature columns (float32)

    Rules:
        ✅ shift(1) on any data you read at current time
        ✅ .astype("float32") on every feature column
        ✅ Column names: snake_case, descriptive, no collisions
        ❌ No lookahead — only use data available BEFORE the date
        ❌ Don't include date/ticker as feature columns (they're merge keys)
    """
    # ── Stock universe: (date, ticker) pairs from broksum ────────
    universe = sources["broksum"][["date", "ticker"]].drop_duplicates()
    n_stocks = universe["ticker"].nunique()
    print(f"  Universe: {len(universe):,} rows, {n_stocks} stocks")

    # ──────────────────────────────────────────────────────────────
    #  INSERT YOUR FEATURE ENGINEERING CODE BELOW
    #
    #  Examples:
    #
    #  # MA crossover feature
    #  daily = sources["yf_daily"].copy()
    #  daily = daily.sort_values(["ticker", "date"])
    #  daily["ma_20"] = daily.groupby("ticker")["close"].transform(
    #      lambda x: x.rolling(20).mean().shift(1)
    #  )
    #  daily["ma_50"] = daily.groupby("ticker")["close"].transform(
    #      lambda x: x.rolling(50).mean().shift(1)
    #  )
    #  daily["ma_cross"] = (daily["ma_20"] / daily["ma_50"] - 1).astype("float32")
    #
    #  # Merge onto universe
    #  result = universe.merge(
    #      daily[["date", "ticker", "ma_cross"]],
    #      on=["date", "ticker"], how="left"
    #  )
    # ──────────────────────────────────────────────────────────────

    # Placeholder: return universe with one dummy feature
    result = universe.copy()
    result["dummy_feature"] = 0.0

    # ── Ensure float32 dtypes ─────────────────────────────────────
    feat_cols = [c for c in result.columns if c not in ("date", "ticker")]
    for col in feat_cols:
        result[col] = result[col].astype("float32")

    return result


def check_column_conflict(df: pd.DataFrame) -> list[str]:
    """Check if feature column names collide with existing modules."""
    feat_cols = set(c for c in df.columns if c not in ("date", "ticker"))
    conflicts = []
    for fpath in sorted(MODULES_DIR.glob("*_features.parquet")):
        existing = pd.read_parquet(fpath, nrows=0).columns
        existing_feats = set(existing) - {"date", "ticker"}
        overlap = feat_cols & existing_feats
        if overlap:
            conflicts.append(f"  ⚠️  {fpath.name}: {sorted(overlap)}")
    return conflicts


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Generate {MODULE_NAME} feature module")
    parser.add_argument("--modules-dir", type=Path, default=MODULES_DIR,
                        help=f"Output directory (default: {MODULES_DIR})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print stats without writing")
    parser.add_argument("--force", action="store_true",
                        help="Write even if column name conflicts detected")
    args = parser.parse_args()

    print(f"[{MODULE_NAME}] Loading sources...")
    sources = load_sources()

    print(f"[{MODULE_NAME}] Building features...")
    df = build_features(sources)

    # ── Validate ──────────────────────────────────────────────────
    has_ticker = "ticker" in df.columns
    merge_keys = ["date", "ticker"] if has_ticker else ["date"]
    n_feats = len([c for c in df.columns if c not in merge_keys])

    print(f"[{MODULE_NAME}] Rows: {len(df):,}")
    print(f"[{MODULE_NAME}] Features: {n_feats}")
    print(f"[{MODULE_NAME}] Grain: ({', '.join(merge_keys)})")
    print(f"[{MODULE_NAME}] Columns: {list(df.columns)}")

    # Check for NaN ratio
    for col in df.columns:
        if col in merge_keys:
            continue
        nan_pct = df[col].isna().mean() * 100
        if nan_pct > 50:
            print(f"  ⚠️  {col}: {nan_pct:.1f}% NaN — may degrade model")

    # Check column name conflicts
    conflicts = check_column_conflict(df)
    if conflicts:
        print(f"[{MODULE_NAME}] ⚠️  Column name conflicts detected:")
        for c in conflicts:
            print(c)
        if not args.force:
            print(f"[{MODULE_NAME}] Aborting. Use --force to write anyway, or rename columns.")
            return

    if args.dry_run:
        print(f"[{MODULE_NAME}] Dry run — not writing")
        return

    # ── Write ─────────────────────────────────────────────────────
    args.modules_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.modules_dir / f"{MODULE_NAME}_features.parquet"
    df.to_parquet(out_path, index=False)
    file_size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"[{MODULE_NAME}] Written to {out_path} ({file_size_mb:.1f} MB)")
    print(f"[{MODULE_NAME}] ✅ Ready for training — just re-run with --feature-modules-dir")


if __name__ == "__main__":
    main()
