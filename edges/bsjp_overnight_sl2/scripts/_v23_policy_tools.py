from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


IDX_DIR = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_DIR = IDX_DIR / "model/BSJP/v23b_t1audit2_clean"
DEFAULT_MODULES_DIR = IDX_DIR / "data/Level_1_Features/modules"
DEFAULT_LOG_DIR = IDX_DIR / "_LOG"

ROUNDTRIP_COST_FRAC = 0.004


@dataclass(frozen=True)
class Policy:
    max_positions: int = 3
    max_weight: float = 0.25
    max_pre14_market_cost_est: float = 0.030
    min_entry_price: float = 500.0
    adaptive_quantile: float = 0.85
    conviction_min_proba: float = 0.035
    conviction_top_k: int = 3
    score_gap_min: float = 0.0
    max_pre14_tick_pct: float | None = None
    exclude_pre14_ara_like: bool = False


def capped_weights(scores: np.ndarray, max_weight: float) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0 or scores.sum() <= 0:
        return np.zeros_like(scores)
    w = scores / scores.sum()
    if max_weight <= 0:
        return np.zeros_like(scores)
    max_weight = min(max_weight, 1.0)
    for _ in range(10):
        over = w > max_weight
        if not over.any():
            break
        excess = float((w[over] - max_weight).sum())
        w[over] = max_weight
        under = ~over
        if not under.any():
            break
        under_sum = float(w[under].sum())
        if under_sum <= 0:
            break
        w[under] += w[under] / under_sum * excess
    return np.clip(w, 0.0, max_weight)


_TICK_MULTIPLIER = {
    (0, 200): (1, 2.5),
    (200, 500): (2, 2.0),
    (500, 2000): (5, 1.5),
    (2000, 5000): (10, 1.5),
    (5000, float("inf")): (25, 1.0),
}


def estimate_spread_frac(price: float) -> float:
    if price <= 0:
        return 0.0
    for (lo, hi), (tick, mult) in _TICK_MULTIPLIER.items():
        if lo <= price < hi:
            return (tick * mult) / price
    return 0.0


def normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    if "ticker" in out.columns:
        out["ticker"] = out["ticker"].astype(str).str.replace(r"\.JK$", "", regex=True).str.upper().str.strip()
    return out


def load_predictions(model_dir: Path = DEFAULT_MODEL_DIR, modules_dir: Path = DEFAULT_MODULES_DIR) -> pd.DataFrame:
    pred = normalize_keys(pd.read_parquet(model_dir / "valid_predictions.parquet"))
    join_specs = [
        (
            "preclose14_features.parquet",
            [
                "pre14_tick_pct",
                "pre14_spread_cost_est",
                "pre14_range_pct",
                "pre14_vwap",
                "pre14_close_vs_open9",
                "pre14_pm_close_near_high",
                "pre14_orb_position",
                "pre14_pm_volume",
                "pre14_am_turnover",
                "pre14_volume_until14",
            ],
        ),
        (
            "ara_history_features.parquet",
            [
                "was_ara_tminus1",
                "ara_count_5d",
                "ara_count_20d",
                "days_since_last_ara",
                "last_ara_return",
                "max_return_5d_tminus1",
                "max_return_20d_tminus1",
            ],
        ),
        (
            "global_indices_features.parquet",
            [
                "ihsg_prev_close",
                "ihsg_prev_return",
                "ihsg_close_ma20_ratio",
                "usdidr_prev_close",
                "usdidr_5d_return",
                "vix_prev_close",
                "nasdaq_prev_return",
            ],
        ),
    ]
    for fname, cols in join_specs:
        module_path = modules_dir / fname
        if not module_path.exists():
            continue
        module = normalize_keys(pd.read_parquet(module_path, columns=["date", "ticker", *cols] if fname != "global_indices_features.parquet" else ["date", *cols]))
        if "ticker" in module.columns:
            join_cols = ["date", "ticker"]
        else:
            join_cols = ["date"]
        existing = set(pred.columns) - set(join_cols)
        use_cols = join_cols + [c for c in module.columns if c not in join_cols and c not in existing]
        pred = pred.merge(module[use_cols], on=join_cols, how="left")
    return pred


def ara_bucket(row: pd.Series) -> str:
    touched = float(row.get("pre14_ara_touched", 0) or 0) >= 0.5
    locked = float(row.get("pre14_ara_locked_proxy", 0) or 0) >= 0.5
    release = float(row.get("pre14_ara_release_wick_count", 0) or 0)
    dist = row.get("pre14_ara_distance_pct", np.nan)
    ret = row.get("pre14_return_from_prev_close", np.nan)
    was_ara = float(row.get("was_ara_tminus1", 0) or 0) >= 0.5
    ara5 = float(row.get("ara_count_5d", 0) or 0)
    if touched and release == 1:
        return "touched_single_release"
    if touched and release > 1:
        return "touched_repeated_release"
    if touched and locked:
        return "touched_locked"
    if pd.notna(dist) and 0 <= float(dist) <= 0.03 and not touched and not was_ara and ara5 == 0:
        return "virgin_near_not_touched"
    if pd.notna(dist) and 0 <= float(dist) <= 0.03 and not touched:
        return "recent_near_not_touched"
    if pd.notna(ret) and pd.notna(dist) and float(ret) >= 0.03 and float(dist) <= 0.08 and not touched:
        return "momentum_near_3_8_not_touched"
    return "far_or_other"


def add_risk_bands(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ara_bucket"] = out.apply(ara_bucket, axis=1)
    out["price_tier"] = pd.cut(
        pd.to_numeric(out["entry_price"], errors="coerce"),
        bins=[0, 200, 500, 2000, 5000, np.inf],
        labels=["lt200", "200_500", "500_2000", "2000_5000", "gt5000"],
        include_lowest=True,
    ).astype(str)
    for col in ["pre14_tick_pct", "pre14_spread_cost_est", "pre14_market_cost_est", "pre14_range_pct"]:
        if col in out.columns:
            x = pd.to_numeric(out[col], errors="coerce")
            try:
                out[f"{col}_band"] = pd.qcut(x, q=5, duplicates="drop").astype(str)
            except ValueError:
                out[f"{col}_band"] = "na"
    if "vix_prev_close" in out.columns:
        out["vix_regime"] = pd.cut(
            pd.to_numeric(out["vix_prev_close"], errors="coerce"),
            bins=[0, 15, 20, 30, np.inf],
            labels=["low", "normal", "elevated", "fear"],
            include_lowest=True,
        ).astype(str)
    return out


def simulate_policy(df: pd.DataFrame, policy: Policy) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    rows: list[dict] = []
    trades: list[pd.DataFrame] = []
    capital = 10_000_000.0
    for date, raw_day in df.groupby("date", sort=True):
        day = raw_day.copy()
        day["pred_proba"] = pd.to_numeric(day["pred_proba"], errors="coerce")
        day["risk_norm"] = pd.to_numeric(day.get("risk_norm", 0.5), errors="coerce").fillna(0.5).clip(0, 1)
        for col in ["entry_price", "exit_price", "overnight_return", "pre14_market_cost_est", "pre14_tick_pct"]:
            if col in day.columns:
                day[col] = pd.to_numeric(day[col], errors="coerce")
        day = day.dropna(subset=["pred_proba", "entry_price", "exit_price", "overnight_return"])
        day = day[day["entry_price"] > 0]
        day = day[day["entry_price"] >= policy.min_entry_price]
        day = day[day["pre14_market_cost_est"] <= policy.max_pre14_market_cost_est]
        if policy.max_pre14_tick_pct is not None and "pre14_tick_pct" in day.columns:
            day = day[day["pre14_tick_pct"] <= policy.max_pre14_tick_pct]
        if policy.exclude_pre14_ara_like and "pre14_is_ara_like" in day.columns:
            day = day[pd.to_numeric(day["pre14_is_ara_like"], errors="coerce").fillna(0) < 0.5]
        if day.empty:
            rows.append({"date": date, "positions": 0, "gross_return": 0.0, "net_return": 0.0, "effective_threshold": np.nan})
            continue
        day = day.sort_values("pred_proba", ascending=False).reset_index(drop=True)
        day["pred_rank"] = np.arange(1, len(day) + 1)
        q = float(np.clip(policy.adaptive_quantile, 0.0, 1.0))
        effective_threshold = max(policy.conviction_min_proba, float(day["pred_proba"].quantile(q)))
        if policy.score_gap_min > 0 and len(day) >= max(2, policy.max_positions):
            kth_idx = min(policy.max_positions, len(day)) - 1
            score_gap = float(day.loc[0, "pred_proba"] - day.loc[kth_idx, "pred_proba"])
            if score_gap < policy.score_gap_min:
                rows.append({"date": date, "positions": 0, "gross_return": 0.0, "net_return": 0.0, "effective_threshold": effective_threshold})
                continue
        if policy.conviction_top_k > 0:
            day = day[day["pred_rank"] <= policy.conviction_top_k]
        day = day[day["pred_proba"] >= effective_threshold]
        if day.empty:
            rows.append({"date": date, "positions": 0, "gross_return": 0.0, "net_return": 0.0, "effective_threshold": effective_threshold})
            continue
        day["score"] = np.maximum(day["pred_proba"] - effective_threshold, 0.0) * (1.0 - day["risk_norm"])
        day = day[day["score"] > 0].sort_values("score", ascending=False).head(policy.max_positions)
        if day.empty:
            rows.append({"date": date, "positions": 0, "gross_return": 0.0, "net_return": 0.0, "effective_threshold": effective_threshold})
            continue
        day["weight"] = capped_weights(day["score"].to_numpy(), policy.max_weight)
        day = day[day["weight"] > 0].copy()
        day["trade_gross_return"] = day["overnight_return"]
        entry_spread = day["entry_price"].apply(estimate_spread_frac)
        exit_spread = day["exit_price"].apply(estimate_spread_frac)
        day["spread_cost_frac"] = entry_spread + exit_spread
        day["trade_net_return"] = day["trade_gross_return"] - ROUNDTRIP_COST_FRAC - day["spread_cost_frac"]
        day["weighted_gross"] = day["weight"] * day["trade_gross_return"]
        day["weighted_net"] = day["weight"] * day["trade_net_return"]
        rows.append(
            {
                "date": date,
                "positions": int(len(day)),
                "gross_return": float(day["weighted_gross"].sum()),
                "net_return": float(day["weighted_net"].sum()),
                "effective_threshold": effective_threshold,
            }
        )
        trades.append(day.assign(effective_threshold=effective_threshold))
        capital *= 1.0 + float(day["weighted_net"].sum())
    daily = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    trade_df = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()
    if daily.empty:
        summary = {"days": 0, "trading_days": 0, "cumulative_net_return": np.nan, "max_drawdown": np.nan}
        return daily, trade_df, summary
    equity = (1.0 + daily["net_return"]).cumprod()
    dd = equity / equity.cummax() - 1.0
    summary = {
        "days": int(len(daily)),
        "trading_days": int((daily["positions"] > 0).sum()),
        "mean_daily_net_return": float(daily["net_return"].mean()),
        "mean_daily_gross_return": float(daily["gross_return"].mean()),
        "cumulative_net_return": float(equity.iloc[-1] - 1.0),
        "max_drawdown": float(dd.min()),
        "win_rate_days": float((daily["net_return"] > 0).mean()),
        "volatility_daily": float(daily["net_return"].std(ddof=0)),
        "net_expectancy_per_trade": float(trade_df["trade_net_return"].mean()) if not trade_df.empty else 0.0,
    }
    return daily, trade_df, summary


def summarize_window(daily: pd.DataFrame, tail_n: int) -> dict[str, float]:
    sub = daily.tail(tail_n).copy()
    if sub.empty:
        return {"cum_net": np.nan, "maxdd": np.nan, "mean": np.nan}
    eq = (1.0 + sub["net_return"]).cumprod()
    dd = eq / eq.cummax() - 1.0
    return {
        "cum_net": float(eq.iloc[-1] - 1.0),
        "maxdd": float(dd.min()),
        "mean": float(sub["net_return"].mean()),
        "trading_days": int((sub["positions"] > 0).sum()),
    }
