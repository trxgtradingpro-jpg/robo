"""Performance and scoring metrics."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd


@dataclass(slots=True)
class ScoreConfig:
    """Robust score hyperparameters."""

    drawdown_weight: float = 1.5
    min_trade_count: int = 20
    penalty_per_missing_trade: float = 150.0


def robust_score(
    net_profit: float,
    max_drawdown: float,
    trade_count: int,
    config: ScoreConfig,
) -> float:
    """Robust score used for train/test model selection."""
    penalty_for_low_trade_count = max(0, config.min_trade_count - int(trade_count))
    return (
        float(net_profit)
        - float(config.drawdown_weight) * float(max_drawdown)
        - float(penalty_for_low_trade_count) * float(config.penalty_per_missing_trade)
    )


def compute_metrics(
    trades: pd.DataFrame,
    equity_curve: pd.DataFrame,
    initial_capital: float,
    score_config: ScoreConfig,
) -> dict[str, float]:
    """Compute aggregated metrics from trades and equity."""
    max_drawdown = _max_drawdown(equity_curve, initial_capital)
    max_drawdown_pct = (100.0 * max_drawdown / initial_capital) if initial_capital > 0 else 0.0
    sharpe_like = _sharpe_like(equity_curve)
    monthly_stats = _monthly_return_stats(equity_curve)

    if trades.empty:
        score = robust_score(0.0, max_drawdown, 0, score_config)
        return {
            "trade_count": 0.0,
            "win_rate": 0.0,
            "net_profit": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "profit_factor": 0.0,
            "avg_trade": 0.0,
            "expectancy": 0.0,
            "payoff_ratio": 0.0,
            "win_loss_ratio": 0.0,
            "max_consecutive_wins": 0.0,
            "max_consecutive_losses": 0.0,
            "avg_trade_duration_min": 0.0,
            "exposure_pct": 0.0,
            "max_drawdown": max_drawdown,
            "max_drawdown_pct": max_drawdown_pct,
            "return_pct": 0.0,
            "recovery_factor": 0.0,
            "monthly_return_mean_pct": monthly_stats["monthly_return_mean_pct"],
            "monthly_return_std_pct": monthly_stats["monthly_return_std_pct"],
            "positive_months_pct": monthly_stats["positive_months_pct"],
            "sharpe_like": sharpe_like,
            "score": score,
        }

    pnl = trades["pnl_net"].astype(float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_profit = float(wins.sum()) if not wins.empty else 0.0
    gross_loss = float(losses.sum()) if not losses.empty else 0.0
    net_profit = float(pnl.sum())
    trade_count = int(len(pnl))
    win_rate = float((pnl > 0).mean()) if trade_count else 0.0
    expectancy = float(pnl.mean()) if trade_count else 0.0
    return_pct = 100.0 * net_profit / initial_capital if initial_capital != 0 else 0.0
    profit_factor = abs(gross_profit / gross_loss) if gross_loss < 0 else (math.inf if gross_profit > 0 else 0.0)
    avg_win = float(wins.mean()) if not wins.empty else 0.0
    avg_loss = float(losses.mean()) if not losses.empty else 0.0
    payoff_ratio = (avg_win / abs(avg_loss)) if avg_loss < 0 else 0.0
    win_loss_ratio = (float(len(wins)) / float(len(losses))) if len(losses) > 0 else float(len(wins))
    max_consecutive_wins = float(_max_streak(pnl, positive=True))
    max_consecutive_losses = float(_max_streak(pnl, positive=False))
    avg_trade_duration_min = _average_trade_duration_minutes(trades)
    exposure_pct = _exposure_pct(trades, equity_curve)
    recovery_factor = (net_profit / max_drawdown) if max_drawdown > 0 else 0.0

    score = robust_score(
        net_profit=net_profit,
        max_drawdown=max_drawdown,
        trade_count=trade_count,
        config=score_config,
    )
    return {
        "trade_count": float(trade_count),
        "win_rate": win_rate,
        "net_profit": net_profit,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": float(profit_factor) if np.isfinite(profit_factor) else 0.0,
        "avg_trade": expectancy,
        "expectancy": expectancy,
        "payoff_ratio": float(payoff_ratio),
        "win_loss_ratio": float(win_loss_ratio),
        "max_consecutive_wins": max_consecutive_wins,
        "max_consecutive_losses": max_consecutive_losses,
        "avg_trade_duration_min": avg_trade_duration_min,
        "exposure_pct": exposure_pct,
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": max_drawdown_pct,
        "return_pct": return_pct,
        "recovery_factor": float(recovery_factor),
        "monthly_return_mean_pct": monthly_stats["monthly_return_mean_pct"],
        "monthly_return_std_pct": monthly_stats["monthly_return_std_pct"],
        "positive_months_pct": monthly_stats["positive_months_pct"],
        "sharpe_like": sharpe_like,
        "score": score,
    }


def _max_drawdown(equity_curve: pd.DataFrame, initial_capital: float) -> float:
    if equity_curve.empty or "equity" not in equity_curve:
        return 0.0
    series = equity_curve["equity"].astype(float).copy()
    if series.empty:
        return 0.0
    running_peak = series.cummax()
    drawdown = running_peak - series
    max_dd = float(drawdown.max()) if not drawdown.empty else 0.0
    return max(0.0, max_dd)


def _sharpe_like(equity_curve: pd.DataFrame) -> float:
    if equity_curve.empty or len(equity_curve) < 3:
        return 0.0
    eq = equity_curve["equity"].astype(float)
    rets = eq.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if rets.empty:
        return 0.0
    std = float(rets.std(ddof=1))
    if std == 0.0:
        return 0.0
    return float((rets.mean() / std) * np.sqrt(252))


def _max_streak(pnl: pd.Series, positive: bool) -> int:
    best = 0
    current = 0
    for value in pnl.astype(float):
        condition = value > 0 if positive else value < 0
        if condition:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def _average_trade_duration_minutes(trades: pd.DataFrame) -> float:
    if trades.empty or "entry_time" not in trades or "exit_time" not in trades:
        return 0.0
    entry = pd.to_datetime(trades["entry_time"], errors="coerce")
    exit_ = pd.to_datetime(trades["exit_time"], errors="coerce")
    duration = (exit_ - entry).dt.total_seconds().dropna()
    if duration.empty:
        return 0.0
    return float(duration.mean() / 60.0)


def _exposure_pct(trades: pd.DataFrame, equity_curve: pd.DataFrame) -> float:
    if trades.empty or equity_curve.empty or "entry_time" not in trades or "exit_time" not in trades:
        return 0.0
    eq_dt = pd.to_datetime(equity_curve["datetime"], errors="coerce").dropna()
    if eq_dt.empty:
        return 0.0
    total_seconds = (eq_dt.max() - eq_dt.min()).total_seconds()
    if total_seconds <= 0:
        return 0.0
    entry = pd.to_datetime(trades["entry_time"], errors="coerce")
    exit_ = pd.to_datetime(trades["exit_time"], errors="coerce")
    held_seconds = (exit_ - entry).dt.total_seconds().fillna(0.0).clip(lower=0.0).sum()
    pct = 100.0 * float(held_seconds) / float(total_seconds)
    return float(min(max(pct, 0.0), 100.0))


def _monthly_return_stats(equity_curve: pd.DataFrame) -> dict[str, float]:
    if equity_curve.empty or "datetime" not in equity_curve or "equity" not in equity_curve:
        return {
            "monthly_return_mean_pct": 0.0,
            "monthly_return_std_pct": 0.0,
            "positive_months_pct": 0.0,
        }
    series = equity_curve.copy()
    series["datetime"] = pd.to_datetime(series["datetime"], errors="coerce")
    series = series.dropna(subset=["datetime"]).sort_values("datetime")
    if series.empty:
        return {
            "monthly_return_mean_pct": 0.0,
            "monthly_return_std_pct": 0.0,
            "positive_months_pct": 0.0,
        }
    monthly = series.set_index("datetime")["equity"].astype(float).resample("ME").last().dropna()
    returns = monthly.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if returns.empty:
        return {
            "monthly_return_mean_pct": 0.0,
            "monthly_return_std_pct": 0.0,
            "positive_months_pct": 0.0,
        }
    return {
        "monthly_return_mean_pct": float(returns.mean() * 100.0),
        "monthly_return_std_pct": float(returns.std(ddof=1) * 100.0) if len(returns) > 1 else 0.0,
        "positive_months_pct": float((returns > 0).mean() * 100.0),
    }
