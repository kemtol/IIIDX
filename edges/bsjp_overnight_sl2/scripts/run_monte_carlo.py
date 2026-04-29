#!/usr/bin/env python3
"""
Monte Carlo Block Bootstrap Simulation for BSJP Portfolio.

Reads portfolio_daily.parquet from a grid candidate directory, performs
block bootstrap sampling (contiguous blocks of daily returns), generates
N simulated equity curves for each horizon, and produces:

  - monte_config.json       (configuration used)
  - monte_summary_metrics.csv  (aggregate risk/return stats per horizon)
  - monte_equity_fan.png    (equity curve fan chart with percentiles)
  - monte_maxdd_hist.png    (maximum drawdown histogram)
  - monte_return_cdf.png    (terminal return CDF)

Usage:
    python run_monte_carlo.py <grid_dir> [--output_dir <path>] [options]

Examples:
    python run_monte_carlo.py ../../model/BSJP/bsjp_grid_fine/md100_lam1_5_gain0_02
    python run_monte_carlo.py ../../model/BSJP/bsjp_grid_fine/md100_lam1_5_gain0_1   \\
        --n_paths 5000 --block_size 5 --horizons 100 252
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# defaults (mirror existing monte_config.json in k2_w25_guard)
# ---------------------------------------------------------------------------
DEFAULT_N_PATHS = 10000
DEFAULT_BLOCK_SIZE = 5
DEFAULT_HORIZONS = (100, 252)
DEFAULT_DD_THRESHOLDS = (0.20, 0.25, 0.30)
DEFAULT_SEED = 20260424


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monte Carlo Block Bootstrap for BSJP portfolio daily returns."
    )
    parser.add_argument(
        "grid_dir",
        type=str,
        help="Path to grid candidate directory containing portfolio_daily.parquet",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: <grid_dir>/monte_carlo/)",
    )
    parser.add_argument(
        "--n_paths",
        type=int,
        default=DEFAULT_N_PATHS,
        help=f"Number of bootstrap paths (default: {DEFAULT_N_PATHS})",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=DEFAULT_BLOCK_SIZE,
        help=f"Block size in days (default: {DEFAULT_BLOCK_SIZE})",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=list(DEFAULT_HORIZONS),
        help=f"Horizon(s) in trading days (default: {' '.join(map(str, DEFAULT_HORIZONS))})",
    )
    parser.add_argument(
        "--dd_thresholds",
        type=float,
        nargs="+",
        default=list(DEFAULT_DD_THRESHOLDS),
        help=f"Drawdown thresholds for probability estimates (default: {' '.join(f'{v:.2f}' for v in DEFAULT_DD_THRESHOLDS)})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED})",
    )
    return parser.parse_args()


def _load_portfolio(grid_dir: Path) -> pd.DataFrame:
    """Load portfolio_daily.parquet and compute daily return ratio."""
    src = grid_dir / "portfolio_daily.parquet"
    if not src.exists():
        print(f"✗ File not found: {src}")
        sys.exit(1)
    df = pd.read_parquet(src)
    # daily return: capital_end / capital_start - 1  (fractional return)
    # NOTE: capital_start is the portfolio's start-of-day capital;
    # capital_end is end-of-day. net_return is NOT the PnL ratio.
    df["daily_return"] = df["capital_end"] / df["capital_start"] - 1.0
    return df


def _block_bootstrap(
    returns: np.ndarray,
    n_paths: int,
    horizon: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate n_paths bootstrap equity curves for a given horizon.

    Each path is constructed by concatenating randomly chosen blocks of
    *block_size* contiguous daily returns until *horizon* days are reached,
    then trimmed.

    Returns:
        np.ndarray of shape (n_paths, horizon) with cumulative product of
        (1 + return) along each path.
    """
    n_obs = len(returns)
    # number of full blocks needed (ceil division)
    n_blocks = (horizon + block_size - 1) // block_size
    total_drawn = n_blocks * block_size

    paths = np.empty((n_paths, horizon), dtype=np.float64)

    for i in range(n_paths):
        # pick random block start indices (with replacement)
        starts = rng.integers(0, max(1, n_obs - block_size + 1), size=n_blocks)
        blocks = []
        for s in starts:
            blocks.append(returns[s : s + block_size])
        full = np.concatenate(blocks)[:horizon]
        paths[i] = np.cumprod(1.0 + full)

    return paths


def _compute_max_drawdowns(paths: np.ndarray) -> np.ndarray:
    """Return maximum drawdown (as positive fraction) for each path."""
    # paths shape: (n_paths, horizon)
    cummax = np.maximum.accumulate(paths, axis=1)
    dd = (paths - cummax) / cummax  # drawdown as negative fraction
    return dd.min(axis=1)  # most negative value per path


def _estimate_prob_maxdd_le(paths: np.ndarray, thresholds: list[float]) -> dict:
    """Estimate P(maxDD <= -threshold) for each threshold."""
    mdd = _compute_max_drawdowns(paths)
    return {f"prob_maxdd_le_minus_{int(t*100)}pct": float(np.mean(mdd <= -t)) for t in thresholds}


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------

def _plot_equity_fan(
    paths: np.ndarray,
    horizon: int,
    save_path: Path,
    label: str = "",
) -> None:
    """Fan chart: median + percentile bands."""
    n_paths, n_days = paths.shape
    pcts = [5, 25, 50, 75, 95]
    bands = np.percentile(paths, pcts, axis=0)  # (5, n_days)
    days = np.arange(1, n_days + 1)

    fig, ax = plt.subplots(figsize=(10, 6))

    # fill between percentiles
    ax.fill_between(days, bands[0], bands[4], alpha=0.15, color="steelblue", label="90% CI (P5–P95)")
    ax.fill_between(days, bands[1], bands[3], alpha=0.25, color="steelblue", label="50% CI (P25–P75)")
    ax.plot(days, bands[2], color="darkblue", linewidth=1.8, label="Median (P50)")

    # thin lines for extremes
    for idx, p in enumerate([5, 95]):
        ax.plot(days, bands[idx], color="steelblue", linewidth=0.8, linestyle="--")

    ax.axhline(y=1.0, color="gray", linewidth=0.7, linestyle=":", alpha=0.6)
    ax.set_xlabel("Trading Days", fontsize=11)
    ax.set_ylabel("Cumulative Return (×)", fontsize=11)
    ax.set_title(f"Monte Carlo Equity Fan — {horizon}d Horizon{label}", fontsize=12)
    ax.legend(loc="upper left", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f×"))
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")


def _plot_maxdd_hist(
    paths: np.ndarray,
    horizon: int,
    save_path: Path,
    label: str = "",
) -> None:
    """Histogram of maximum drawdown across all paths."""
    mdd = _compute_max_drawdowns(paths) * 100  # as percentage
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(mdd, bins=80, color="crimson", alpha=0.7, edgecolor="white", linewidth=0.3)
    ax.axvline(x=np.median(mdd), color="darkred", linewidth=1.5, linestyle="--",
               label=f"Median DD: {np.median(mdd):.1f}%")
    ax.axvline(x=np.percentile(mdd, 95), color="darkred", linewidth=1.2, linestyle=":",
               label=f"P95 DD: {np.percentile(mdd, 95):.1f}%")
    ax.set_xlabel("Maximum Drawdown (%)", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title(f"MaxDD Distribution — {horizon}d Horizon{label}", fontsize=12)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")


def _plot_return_cdf(
    paths: np.ndarray,
    horizon: int,
    save_path: Path,
    label: str = "",
) -> None:
    """CDF of terminal (final-day) cumulative return."""
    terminal = paths[:, -1]
    sorted_vals = np.sort(terminal)
    cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sorted_vals, cdf, color="darkgreen", linewidth=1.5)
    ax.axhline(y=0.5, color="gray", linewidth=0.7, linestyle=":", alpha=0.5)
    ax.axvline(x=np.median(terminal), color="darkgreen", linewidth=1.0, linestyle="--",
               label=f"Median: {np.median(terminal):.2f}×")
    ax.axvline(x=np.percentile(terminal, 5), color="darkgreen", linewidth=1.0, linestyle=":",
               label=f"P5: {np.percentile(terminal, 5):.2f}×")
    ax.axvline(x=1.0, color="red", linewidth=0.8, linestyle="-", alpha=0.5,
               label="Breakeven (1.0×)")
    ax.set_xlabel("Terminal Cumulative Return (×)", fontsize=11)
    ax.set_ylabel("Cumulative Probability", fontsize=11)
    ax.set_title(f"Terminal Return CDF — {horizon}d Horizon{label}", fontsize=12)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    grid_dir = Path(args.grid_dir).resolve()
    if not grid_dir.is_dir():
        print(f"✗ Grid directory not found: {grid_dir}")
        sys.exit(1)

    # output directory
    if args.output_dir:
        out_dir = Path(args.output_dir).resolve()
    else:
        out_dir = grid_dir / "monte_carlo"
    out_dir.mkdir(parents=True, exist_ok=True)

    # seed RNG
    rng = np.random.default_rng(args.seed)

    # 1. load portfolio daily returns
    print(f"[1/5] Loading portfolio from: {grid_dir / 'portfolio_daily.parquet'}")
    df = _load_portfolio(grid_dir)
    returns = df["daily_return"].values
    n_source = len(df)
    source_start = df["date"].min()
    source_end = df["date"].max()
    print(f"      Source rows: {n_source}, date range: {source_start} → {source_end}")
    print(f"      Daily return: mean={returns.mean():.6f}, std={returns.std():.6f}, "
          f"min={returns.min():.4f}, max={returns.max():.4f}")

    horizons = args.horizons
    n_paths = args.n_paths
    block_size = args.block_size
    dd_thresholds = args.dd_thresholds

    # 2. save config
    config = {
        "source_file": str(grid_dir / "portfolio_daily.parquet"),
        "source_rows": int(n_source),
        "source_start_date": str(source_start),
        "source_end_date": str(source_end),
        "seed": args.seed,
        "n_paths": n_paths,
        "block_size_days": block_size,
        "horizons_days": horizons,
        "drawdown_thresholds": dd_thresholds,
    }
    config_path = out_dir / "monte_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[2/5] Config saved: {config_path}")

    # extract a short label from directory name for plot titles
    label = f" — {grid_dir.name}"

    # 3. run bootstrap for each horizon
    records = []
    for h_idx, horizon in enumerate(horizons, 1):
        print(f"\n[3/5] Horizon {h_idx}/{len(horizons)}: {horizon}d — bootstrapping {n_paths} paths ...")

        paths = _block_bootstrap(returns, n_paths, horizon, block_size, rng)

        terminal = paths[:, -1]
        mean_t = float(np.mean(terminal))
        median_t = float(np.median(terminal))
        p05_t = float(np.percentile(terminal, 5))
        p25_t = float(np.percentile(terminal, 25))
        p75_t = float(np.percentile(terminal, 75))
        p95_t = float(np.percentile(terminal, 95))
        prob_neg = float(np.mean(terminal < 1.0))

        mdd_all = _compute_max_drawdowns(paths)
        mean_mdd = float(np.mean(mdd_all))
        median_mdd = float(np.median(mdd_all))
        p95_mdd = float(np.percentile(mdd_all, 5))  # 5th percentile = worst 5%

        dd_probs = _estimate_prob_maxdd_le(paths, dd_thresholds)

        rec = {
            "horizon_days": horizon,
            "n_paths": n_paths,
            "block_size_days": block_size,
            "mean_terminal_return": mean_t,
            "median_terminal_return": median_t,
            "p05_terminal_return": p05_t,
            "p25_terminal_return": p25_t,
            "p75_terminal_return": p75_t,
            "p95_terminal_return": p95_t,
            "prob_terminal_negative": prob_neg,
            "mean_max_drawdown": mean_mdd,
            "median_max_drawdown": median_mdd,
            "p95_worst_max_drawdown": p95_mdd,
        }
        rec.update(dd_probs)
        records.append(rec)

        # summary line
        print(f"      Terminal: mean={mean_t:.4f}×, median={median_t:.4f}×, "
              f"P5={p05_t:.4f}×, P95={p95_t:.4f}×")
        print(f"      MaxDD: mean={mean_mdd*100:.2f}%, median={median_mdd*100:.2f}%, "
              f"P95_worst={p95_mdd*100:.2f}%")
        print(f"      P(return<1.0)={prob_neg:.4f}, "
              f"P(maxDD≤-20%)={dd_probs['prob_maxdd_le_minus_20pct']:.4f}, "
              f"P(maxDD≤-30%)={dd_probs['prob_maxdd_le_minus_30pct']:.4f}")

        # 4. generate plots
        print(f"[4/5] Generating plots for {horizon}d horizon ...")
        _plot_equity_fan(paths, horizon, out_dir / f"monte_equity_fan_{horizon}d.png", label)
        _plot_maxdd_hist(paths, horizon, out_dir / f"monte_maxdd_hist_{horizon}d.png", label)
        _plot_return_cdf(paths, horizon, out_dir / f"monte_return_cdf_{horizon}d.png", label)

    # 5. save summary CSV
    summary_path = out_dir / "monte_summary_metrics.csv"
    summary_df = pd.DataFrame(records)
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[5/5] Summary metrics saved: {summary_path}")
    print(summary_df.to_string(index=False))
    print("\n✓ Monte Carlo simulation complete.")


if __name__ == "__main__":
    main()
