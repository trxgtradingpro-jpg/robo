"""Advanced reporting utilities for walk-forward outputs."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd


def build_monthly_report(
    trades: pd.DataFrame,
    equity_curve: pd.DataFrame,
    initial_capital: float,
) -> pd.DataFrame:
    """Build monthly PnL and stability report for one strategy run."""
    if equity_curve.empty:
        return pd.DataFrame(
            columns=[
                "month",
                "start_equity",
                "end_equity",
                "net_profit",
                "return_pct",
                "trade_count",
                "win_rate",
                "profit_factor",
                "max_drawdown",
            ]
        )

    eq = equity_curve.copy()
    eq["datetime"] = pd.to_datetime(eq["datetime"], errors="coerce")
    eq = eq.dropna(subset=["datetime"]).sort_values("datetime")
    eq["month"] = eq["datetime"].dt.to_period("M").astype(str)

    if trades.empty:
        trades_local = pd.DataFrame(columns=["exit_time", "pnl_net"])
    else:
        trades_local = trades.copy()
        trades_local["exit_time"] = pd.to_datetime(trades_local["exit_time"], errors="coerce")
        trades_local["month"] = trades_local["exit_time"].dt.to_period("M").astype(str)

    rows: list[dict[str, Any]] = []
    for month, month_eq in eq.groupby("month", sort=True):
        month_eq = month_eq.sort_values("datetime")
        start_equity = float(month_eq["equity"].iloc[0])
        end_equity = float(month_eq["equity"].iloc[-1])
        net_profit = end_equity - start_equity
        denom = start_equity if abs(start_equity) > 1e-12 else max(initial_capital, 1e-12)
        return_pct = 100.0 * net_profit / denom

        month_trades = trades_local[trades_local["month"] == month] if "month" in trades_local else pd.DataFrame()
        pnl = month_trades["pnl_net"].astype(float) if not month_trades.empty and "pnl_net" in month_trades else pd.Series([], dtype=float)
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        gross_profit = float(wins.sum()) if not wins.empty else 0.0
        gross_loss = float(losses.sum()) if not losses.empty else 0.0
        profit_factor = abs(gross_profit / gross_loss) if gross_loss < 0 else (float("inf") if gross_profit > 0 else 0.0)
        win_rate = float((pnl > 0).mean()) if len(pnl) else 0.0
        max_drawdown = _max_drawdown(month_eq["equity"].astype(float))

        rows.append(
            {
                "month": month,
                "start_equity": start_equity,
                "end_equity": end_equity,
                "net_profit": net_profit,
                "return_pct": return_pct,
                "trade_count": float(len(pnl)),
                "win_rate": win_rate,
                "profit_factor": float(profit_factor) if np.isfinite(profit_factor) else 0.0,
                "max_drawdown": max_drawdown,
            }
        )

    return pd.DataFrame(rows).sort_values("month").reset_index(drop=True)


def build_robustness_report(
    window_results: pd.DataFrame,
    topk_test_results: pd.DataFrame,
    consolidated_metrics: dict[str, float],
) -> dict[str, Any]:
    """Build robustness diagnostics and alerts for one strategy run."""
    windows = window_results.copy() if not window_results.empty else pd.DataFrame()
    topk = topk_test_results.copy() if not topk_test_results.empty else pd.DataFrame()

    total_windows = int(len(windows))
    oos_score_mean = float(windows["oos_score"].mean()) if "oos_score" in windows and not windows.empty else 0.0
    oos_score_std = float(windows["oos_score"].std(ddof=1)) if "oos_score" in windows and len(windows) > 1 else 0.0
    oos_score_min = float(windows["oos_score"].min()) if "oos_score" in windows and not windows.empty else 0.0
    oos_score_max = float(windows["oos_score"].max()) if "oos_score" in windows and not windows.empty else 0.0
    positive_oos_windows_pct = (
        float((windows["oos_net_profit"].astype(float) > 0).mean() * 100.0)
        if "oos_net_profit" in windows and not windows.empty
        else 0.0
    )
    train_test_score_corr = _train_test_corr(windows)
    param_stability = _parameter_stability(windows)
    topk_dispersion = _topk_dispersion(topk)
    alerts = _robustness_alerts(
        positive_oos_windows_pct=positive_oos_windows_pct,
        oos_score_mean=oos_score_mean,
        oos_score_std=oos_score_std,
        train_test_score_corr=train_test_score_corr,
        drawdown_pct=float(consolidated_metrics.get("max_drawdown_pct", 0.0)),
        dominant_param_ratio=float(param_stability.get("dominant_ratio", 0.0)),
    )

    return {
        "total_windows": total_windows,
        "oos_score_mean": oos_score_mean,
        "oos_score_std": oos_score_std,
        "oos_score_min": oos_score_min,
        "oos_score_max": oos_score_max,
        "positive_oos_windows_pct": positive_oos_windows_pct,
        "train_test_score_corr": train_test_score_corr,
        "parameter_stability": param_stability,
        "topk_dispersion": topk_dispersion,
        "alerts": alerts,
    }


def build_parameter_sensitivity_report(topk_test_results: pd.DataFrame) -> pd.DataFrame:
    """Summarize sensitivity of top-k test scores to each parameter."""
    if topk_test_results.empty or "params_json" not in topk_test_results:
        return pd.DataFrame(
            columns=[
                "parameter",
                "value",
                "samples",
                "avg_test_score",
                "std_test_score",
                "avg_test_net_profit",
                "avg_test_max_drawdown",
            ]
        )

    rows: list[dict[str, Any]] = []
    for _, row in topk_test_results.iterrows():
        params = _safe_json_loads(row.get("params_json", "{}"))
        for key, value in params.items():
            rows.append(
                {
                    "parameter": str(key),
                    "value": str(value),
                    "test_score": float(row.get("test_score", 0.0)),
                    "test_net_profit": float(row.get("test_net_profit", 0.0)),
                    "test_max_drawdown": float(row.get("test_max_drawdown", 0.0)),
                }
            )
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    grouped = (
        df.groupby(["parameter", "value"], as_index=False)
        .agg(
            samples=("test_score", "count"),
            avg_test_score=("test_score", "mean"),
            std_test_score=("test_score", "std"),
            avg_test_net_profit=("test_net_profit", "mean"),
            avg_test_max_drawdown=("test_max_drawdown", "mean"),
        )
        .sort_values(["parameter", "avg_test_score"], ascending=[True, False])
        .reset_index(drop=True)
    )
    grouped["std_test_score"] = grouped["std_test_score"].fillna(0.0)
    return grouped


def _max_drawdown(equity_series: pd.Series) -> float:
    if equity_series.empty:
        return 0.0
    running_peak = equity_series.cummax()
    drawdown = running_peak - equity_series
    return float(drawdown.max()) if not drawdown.empty else 0.0


def _train_test_corr(windows: pd.DataFrame) -> float:
    if windows.empty or "best_train_score" not in windows or "oos_score" not in windows:
        return 0.0
    if len(windows) < 2:
        return 0.0
    corr = windows["best_train_score"].astype(float).corr(windows["oos_score"].astype(float))
    return float(corr) if pd.notna(corr) else 0.0


def _parameter_stability(windows: pd.DataFrame) -> dict[str, Any]:
    if windows.empty or "best_train_params_json" not in windows:
        return {"dominant_ratio": 0.0, "n_unique_full_sets": 0, "parameters": {}}

    params_series = windows["best_train_params_json"].astype(str)
    set_counts = params_series.value_counts(dropna=False)
    dominant_ratio = float(set_counts.iloc[0] / len(params_series)) if len(params_series) else 0.0

    parsed = [(_safe_json_loads(raw) if isinstance(raw, str) else {}) for raw in params_series]
    keys = sorted({k for item in parsed for k in item.keys()})
    by_key: dict[str, dict[str, Any]] = {}
    for key in keys:
        values = [str(item.get(key, "")) for item in parsed]
        vc = pd.Series(values).value_counts(dropna=False)
        by_key[key] = {
            "unique_values": int(vc.size),
            "most_common_value": str(vc.index[0]) if not vc.empty else "",
            "most_common_ratio": float(vc.iloc[0] / max(len(values), 1)),
        }

    return {
        "dominant_ratio": dominant_ratio,
        "n_unique_full_sets": int(set_counts.size),
        "parameters": by_key,
    }


def _topk_dispersion(topk: pd.DataFrame) -> dict[str, float]:
    if topk.empty or "window_id" not in topk or "train_rank" not in topk:
        return {
            "avg_top1_minus_topk_mean": 0.0,
            "avg_top1_test_score": 0.0,
            "avg_topk_test_score": 0.0,
        }
    rows: list[dict[str, float]] = []
    for _, grp in topk.groupby("window_id"):
        grp = grp.sort_values("train_rank")
        top1 = float(grp.iloc[0]["test_score"])
        mean_score = float(grp["test_score"].mean())
        rows.append(
            {
                "top1_minus_mean": top1 - mean_score,
                "top1": top1,
                "mean": mean_score,
            }
        )
    if not rows:
        return {
            "avg_top1_minus_topk_mean": 0.0,
            "avg_top1_test_score": 0.0,
            "avg_topk_test_score": 0.0,
        }
    frame = pd.DataFrame(rows)
    return {
        "avg_top1_minus_topk_mean": float(frame["top1_minus_mean"].mean()),
        "avg_top1_test_score": float(frame["top1"].mean()),
        "avg_topk_test_score": float(frame["mean"].mean()),
    }


def _robustness_alerts(
    positive_oos_windows_pct: float,
    oos_score_mean: float,
    oos_score_std: float,
    train_test_score_corr: float,
    drawdown_pct: float,
    dominant_param_ratio: float,
) -> list[dict[str, str]]:
    alerts: list[dict[str, str]] = []
    if positive_oos_windows_pct < 50.0:
        alerts.append(
            {
                "severity": "high",
                "code": "LOW_POSITIVE_WINDOWS",
                "message": "Menos de 50% das janelas OOS com lucro.",
            }
        )
    if abs(oos_score_mean) > 1e-9 and oos_score_std > abs(oos_score_mean) * 1.5:
        alerts.append(
            {
                "severity": "medium",
                "code": "HIGH_SCORE_VOLATILITY",
                "message": "Volatilidade de score OOS alta em relacao a media.",
            }
        )
    if train_test_score_corr < 0.0:
        alerts.append(
            {
                "severity": "high",
                "code": "NEGATIVE_TRAIN_TEST_CORR",
                "message": "Correlacao treino vs teste negativa (possivel overfit).",
            }
        )
    if drawdown_pct > 20.0:
        alerts.append(
            {
                "severity": "medium",
                "code": "HIGH_DRAWDOWN",
                "message": "Drawdown maximo acima de 20% do capital.",
            }
        )
    if dominant_param_ratio < 0.40:
        alerts.append(
            {
                "severity": "low",
                "code": "UNSTABLE_PARAMS",
                "message": "Baixa repeticao do melhor conjunto de parametros entre janelas.",
            }
        )
    return alerts


def _safe_json_loads(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}
