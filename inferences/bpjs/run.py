"""
run.py — Score today's features and output top-k picks (BPJS).

NOTE: BPJS is currently PAUSED. This script is scaffolded but not in production.

Usage
-----
    python run.py --variant v16b [--date YYYY-MM-DD] [--top-k N] [--log-picks]
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import lightgbm as lgb
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import db as inference_db


def load_variant(name: str):
    spec_path = Path(__file__).parent / "variants" / f"{name}.py"
    if not spec_path.exists():
        raise FileNotFoundError(f"Variant not found: {spec_path}")
    spec = importlib.util.spec_from_file_location(f"variants.{name}", spec_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--variant",   required=True,         help="Variant name, e.g. v16b")
    p.add_argument("--date",      default=None,           help="Override date YYYY-MM-DD")
    p.add_argument("--top-k",     type=int, default=None, help="Override top-k")
    p.add_argument("--log-picks", action="store_true",    help="Write picks to picks_log")
    return p.parse_args()


def apply_policy(df: pd.DataFrame, top_k: int, min_proba: float, rank_weights: list) -> pd.DataFrame:
    picks = (
        df[df["pred_proba"] >= min_proba]
        .sort_values("pred_proba", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )
    picks["rank"]   = picks.index + 1
    picks["weight"] = [rank_weights[i] if i < len(rank_weights) else 0.0
                       for i in range(len(picks))]
    return picks


def format_output(picks: pd.DataFrame, inference_date, variant: str) -> str:
    lines = [
        f"\n{'═'*55}",
        f"  BPJS {variant} — Picks for {inference_date}",
        f"  Strategy: beli open ~09:00, jual close ~10:00",
        f"{'═'*55}",
        f"  {'Rank':<5} {'Ticker':<8} {'Prob':>6}  {'Weight':>7}",
        f"  {'─'*42}",
    ]
    for _, row in picks.iterrows():
        lines.append(
            f"  #{int(row['rank'])}    {row['ticker']:<8} {row['pred_proba']:.4f}  {row['weight']*100:.0f}%"
        )
    if picks.empty:
        lines.append("  (no picks above minimum probability threshold)")
    lines.append(f"{'═'*55}\n")
    return "\n".join(lines)


def main() -> None:
    args    = parse_args()
    variant = load_variant(args.variant)

    model         = lgb.Booster(model_file=str(variant.MODEL_PATH))
    feature_names = model.feature_name()
    top_k         = args.top_k or variant.TOP_K

    con = inference_db.connect(read_only=not args.log_picks)
    inference_db.init_schema(con, feature_names)

    from config import L2_PARQUET
    import warnings
    warnings.warn(
        "[run] Reading from L2 parquet directly — implement DB upsert in Phase 2",
        stacklevel=2,
    )

    target_date = args.date
    dm = pd.read_parquet(L2_PARQUET)
    if target_date:
        today = dm[dm["date"] == pd.Timestamp(target_date)]
    else:
        target_date = dm["date"].max()
        today       = dm[dm["date"] == target_date]

    if today.empty:
        print(f"[run] No data for {target_date}")
        return

    X = today.reindex(columns=feature_names)
    today = today.copy()
    today["pred_proba"] = model.predict(X)

    picks = apply_policy(today, top_k, variant.MIN_PROBA, variant.RANK_WEIGHTS)
    print(format_output(picks, target_date, args.variant))

    if args.log_picks:
        print("[run] TODO: log_picks not yet implemented")

    con.close()


if __name__ == "__main__":
    main()
