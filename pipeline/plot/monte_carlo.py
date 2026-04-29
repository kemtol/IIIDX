"""
monte_carlo.py — Bootstrap Monte Carlo simulation for OOT portfolio returns.

Resamples daily net returns N times to estimate distribution of outcomes.
Produces monte_carlo.png + monte_carlo.json in the model directory.

Usage
-----
    python monte_carlo.py --model-dir model/BSJP/bsjp_v12
    python monte_carlo.py --model-dir model/BSJP/bsjp_v12 --n-sims 20000
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

INITIAL_CAPITAL = 10_000_000
N_SIMS_DEFAULT  = 10_000
SEED            = 42


def load_returns(model_dir: Path) -> np.ndarray:
    path = model_dir / "portfolio_daily.parquet"
    if not path.exists():
        raise FileNotFoundError(f"portfolio_daily.parquet not found in {model_dir}")
    df = pd.read_parquet(path)
    return df["net_return"].values


def run_simulation(returns: np.ndarray, n_sims: int, rng: np.random.Generator) -> np.ndarray:
    n_days = len(returns)
    # shape: (n_sims, n_days)
    sampled = rng.choice(returns, size=(n_sims, n_days), replace=True)
    # cumulative equity paths
    equity = INITIAL_CAPITAL * np.cumprod(1 + sampled, axis=1)
    return equity


def max_drawdown_matrix(equity: np.ndarray) -> np.ndarray:
    running_max = np.maximum.accumulate(equity, axis=1)
    dd = equity / running_max - 1
    return dd.min(axis=1)


def compute_stats(equity: np.ndarray, returns: np.ndarray) -> dict:
    terminal = equity[:, -1]
    mdd      = max_drawdown_matrix(equity)

    p_loss        = (terminal < INITIAL_CAPITAL).mean()
    p_mdd_30      = (mdd < -0.30).mean()
    p_mdd_50      = (mdd < -0.50).mean()
    percentiles   = np.percentile(terminal, [5, 25, 50, 75, 95])
    actual_cum    = (1 + returns).prod() - 1

    return dict(
        n_days          = len(returns),
        n_sims          = len(terminal),
        actual_cum_ret  = float(actual_cum),
        p_loss          = float(p_loss),
        p_mdd_gt30      = float(p_mdd_30),
        p_mdd_gt50      = float(p_mdd_50),
        terminal_p5     = float(percentiles[0]),
        terminal_p25    = float(percentiles[1]),
        terminal_p50    = float(percentiles[2]),
        terminal_p75    = float(percentiles[3]),
        terminal_p95    = float(percentiles[4]),
        terminal_mean   = float(terminal.mean()),
        mdd_median      = float(np.median(mdd)),
        mdd_p95         = float(np.percentile(mdd, 5)),  # 5th pct = worst 5%
    )


def plot_simulation(equity: np.ndarray, stats: dict, model_name: str, out_path: Path) -> None:
    terminal = equity[:, -1]
    n_days   = equity.shape[1]
    x        = np.arange(1, n_days + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")
    fig.suptitle(
        f"{model_name}  |  Monte Carlo Bootstrap  |  {stats['n_sims']:,} sims × {n_days} days",
        fontsize=11
    )

    # ── Left: equity paths (sample 500) ──────────────────────────────────────
    ax1.set_facecolor("white")
    sample_idx = np.random.choice(len(equity), size=min(500, len(equity)), replace=False)
    for i in sample_idx:
        color = "#d62728" if equity[i, -1] < INITIAL_CAPITAL else "#1f77b4"
        ax1.plot(x, equity[i] / 1e6, color=color, alpha=0.04, lw=0.6)

    # Percentile bands
    p5  = np.percentile(equity, 5,  axis=0) / 1e6
    p25 = np.percentile(equity, 25, axis=0) / 1e6
    p50 = np.percentile(equity, 50, axis=0) / 1e6
    p75 = np.percentile(equity, 75, axis=0) / 1e6
    p95 = np.percentile(equity, 95, axis=0) / 1e6

    ax1.fill_between(x, p5,  p95, alpha=0.15, color="#1f77b4", label="5–95%")
    ax1.fill_between(x, p25, p75, alpha=0.25, color="#1f77b4", label="25–75%")
    ax1.plot(x, p50, color="#1f77b4", lw=2, label="Median")
    ax1.axhline(INITIAL_CAPITAL / 1e6, color="gray", lw=1, ls="--", alpha=0.7, label="Initial")

    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}M"))
    ax1.set_xlabel("Trading Days", fontsize=9)
    ax1.set_ylabel("Equity (IDR)", fontsize=9)
    ax1.set_title("Equity Path Distribution", fontsize=10)
    ax1.legend(fontsize=8)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.grid(axis="y", ls=":", alpha=0.4)

    # ── Right: terminal equity distribution ──────────────────────────────────
    ax2.set_facecolor("white")
    bins = np.linspace(np.percentile(terminal, 1), np.percentile(terminal, 99), 80)
    ax2.hist(terminal / 1e6, bins=bins / 1e6, color="#1f77b4", alpha=0.7, edgecolor="white")
    ax2.axvline(INITIAL_CAPITAL / 1e6, color="gray", lw=1.2, ls="--", label="Initial (10M)")
    ax2.axvline(stats["terminal_p50"] / 1e6, color="navy", lw=1.5, ls="-", label=f"Median {stats['terminal_p50']/1e6:.1f}M")
    ax2.axvline(stats["terminal_p5"]  / 1e6, color="crimson", lw=1.5, ls="-", label=f"P5 {stats['terminal_p5']/1e6:.1f}M")

    textstr = (
        f"P(loss)      = {stats['p_loss']*100:.1f}%\n"
        f"P(DD>30%)  = {stats['p_mdd_gt30']*100:.1f}%\n"
        f"P(DD>50%)  = {stats['p_mdd_gt50']*100:.1f}%\n"
        f"Median DD  = {stats['mdd_median']*100:.1f}%\n"
        f"Actual ret = {stats['actual_cum_ret']*100:.0f}%"
    )
    ax2.text(0.97, 0.97, textstr, transform=ax2.transAxes,
             fontsize=8.5, va="top", ha="right",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))

    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}M"))
    ax2.set_xlabel("Terminal Equity (IDR)", fontsize=9)
    ax2.set_ylabel("Frequency", fontsize=9)
    ax2.set_title("Terminal Equity Distribution", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="y", ls=":", alpha=0.4)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[mc] Saved {out_path.name}")


def run_monte_carlo(model_dir: Path, n_sims: int = N_SIMS_DEFAULT) -> dict:
    model_dir  = Path(model_dir)
    returns    = load_returns(model_dir)
    rng        = np.random.default_rng(SEED)
    equity     = run_simulation(returns, n_sims, rng)
    stats      = compute_stats(equity, returns)
    model_name = model_dir.name

    plot_simulation(equity, stats, model_name, model_dir / "monte_carlo.png")

    json_path = model_dir / "monte_carlo.json"
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[mc] Saved {json_path.name}")

    print()
    print(f"=== Monte Carlo Results: {model_name} ===")
    print(f"  Sims × days     : {stats['n_sims']:,} × {stats['n_days']}")
    print(f"  Actual cum ret  : {stats['actual_cum_ret']*100:.0f}%")
    print(f"  P(loss)         : {stats['p_loss']*100:.1f}%")
    print(f"  P(MaxDD > 30%)  : {stats['p_mdd_gt30']*100:.1f}%")
    print(f"  P(MaxDD > 50%)  : {stats['p_mdd_gt50']*100:.1f}%")
    print(f"  Terminal P5     : {stats['terminal_p5']/1e6:.1f}M  ({(stats['terminal_p5']/INITIAL_CAPITAL-1)*100:.0f}%)")
    print(f"  Terminal P50    : {stats['terminal_p50']/1e6:.1f}M  ({(stats['terminal_p50']/INITIAL_CAPITAL-1)*100:.0f}%)")
    print(f"  Terminal P95    : {stats['terminal_p95']/1e6:.1f}M  ({(stats['terminal_p95']/INITIAL_CAPITAL-1)*100:.0f}%)")
    print(f"  Median MaxDD    : {stats['mdd_median']*100:.1f}%")

    return stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--n-sims", type=int, default=N_SIMS_DEFAULT)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_monte_carlo(args.model_dir, args.n_sims)
