"""
plot_pnl.py — Generate PnL equity curve charts for a trained model.

Produces 3 PNG files in the model directory:
    pnl_20d.png   — last 20 trading days
    pnl_50d.png   — last 50 trading days
    pnl_100d.png  — full 100-day OOT window

Usage
-----
    python plot_pnl.py --model-dir idx/model/BSJP/bsjp_v13

Can also be imported and called directly:
    from plot_pnl import generate_pnl_charts
    generate_pnl_charts(model_dir)
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

INITIAL_CAPITAL = 10_000_000
WINDOWS = [20, 50, 100]


def load_portfolio(model_dir: Path) -> pd.DataFrame:
    path = model_dir / "portfolio_daily.parquet"
    if not path.exists():
        raise FileNotFoundError(f"portfolio_daily.parquet not found in {model_dir}")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["equity"] = (1 + df["net_return"]).cumprod() * INITIAL_CAPITAL
    df["drawdown"] = df["equity"] / df["equity"].cummax() - 1
    return df


def _stats(df: pd.DataFrame) -> dict:
    cum_ret = (1 + df["net_return"]).prod() - 1
    max_dd  = df["drawdown"].min()
    win_rate = (df["net_return"] > 0).mean()
    vol     = df["net_return"].std()
    sharpe  = (df["net_return"].mean() / vol * np.sqrt(252)) if vol > 0 else 0
    return dict(cum_ret=cum_ret, max_dd=max_dd, win_rate=win_rate, sharpe=sharpe)


def plot_window(df: pd.DataFrame, window: int, model_name: str, out_path: Path) -> None:
    data = df.tail(window).copy().reset_index(drop=True)
    # re-base equity to 10M from window start
    data["equity"] = (1 + data["net_return"]).cumprod() * INITIAL_CAPITAL
    data["drawdown"] = data["equity"] / data["equity"].cummax() - 1
    s = _stats(data)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 6),
        gridspec_kw={"height_ratios": [3, 1]},
        facecolor="white"
    )
    fig.subplots_adjust(hspace=0.08)

    # ── Equity curve ──────────────────────────────────────────────────────────
    ax1.set_facecolor("white")
    color = "#1f77b4"
    ax1.plot(data["date"], data["equity"] / 1e6, color=color, lw=1.8, zorder=3)
    ax1.fill_between(data["date"], INITIAL_CAPITAL / 1e6, data["equity"] / 1e6,
                     where=data["equity"] >= INITIAL_CAPITAL,
                     alpha=0.12, color=color)
    ax1.fill_between(data["date"], INITIAL_CAPITAL / 1e6, data["equity"] / 1e6,
                     where=data["equity"] < INITIAL_CAPITAL,
                     alpha=0.12, color="crimson")
    ax1.axhline(INITIAL_CAPITAL / 1e6, color="gray", lw=0.8, ls="--", alpha=0.6)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}M"))
    ax1.set_title(
        f"{model_name}  |  last {window}d  |  "
        f"CumRet {s['cum_ret']*100:+.1f}%  |  "
        f"MaxDD {s['max_dd']*100:.1f}%  |  "
        f"WinRate {s['win_rate']*100:.0f}%  |  "
        f"Sharpe {s['sharpe']:.2f}",
        fontsize=10, pad=8
    )
    ax1.set_ylabel("Equity (IDR)", fontsize=9)
    ax1.tick_params(labelbottom=False)
    ax1.grid(axis="y", ls=":", alpha=0.4)
    ax1.spines[["top", "right"]].set_visible(False)

    # ── Drawdown ──────────────────────────────────────────────────────────────
    ax2.set_facecolor("white")
    ax2.fill_between(data["date"], 0, data["drawdown"] * 100,
                     color="crimson", alpha=0.5)
    ax2.plot(data["date"], data["drawdown"] * 100, color="crimson", lw=1.2)
    ax2.axhline(0, color="gray", lw=0.6)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax2.set_ylabel("Drawdown", fontsize=9)
    ax2.tick_params(axis="x", labelsize=8, rotation=20)
    ax2.grid(axis="y", ls=":", alpha=0.4)
    ax2.spines[["top", "right"]].set_visible(False)

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[plot] Saved {out_path.name}")


def generate_pnl_charts(model_dir: Path) -> None:
    model_dir = Path(model_dir)
    df = load_portfolio(model_dir)
    model_name = model_dir.name

    for w in WINDOWS:
        out = model_dir / f"pnl_{w}d.png"
        available = min(w, len(df))
        plot_window(df, available, model_name, out)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, required=True,
                   help="Path to model directory containing portfolio_daily.parquet")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_pnl_charts(args.model_dir)
