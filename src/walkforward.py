"""Walk-forward optimization and out-of-sample validation."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Callable

import pandas as pd

from .backtest_engine import BacktestConfig, run_backtest
from .metrics import ScoreConfig, compute_metrics
from .optimizer import (
    OptimizationCandidate,
    OptimizerConfig,
    apply_hard_limits_to_metrics,
    build_runtime_config_for_params,
    generate_signals_with_time_filter,
    optimize_strategy,
)
from .strategies import StrategySpec

ProgressCallback = Callable[[dict[str, Any]], None]
CheckpointCallback = Callable[["WalkForwardCheckpoint"], None]


@dataclass(slots=True)
class WalkForwardConfig:
    """Train/test rolling windows in trading days."""

    train_days: int = 120
    test_days: int = 30
    score_config: ScoreConfig = field(default_factory=ScoreConfig)


@dataclass(slots=True)
class WalkForwardResult:
    """Aggregated walk-forward outputs for one strategy/timeframe."""

    strategy_name: str
    window_results: pd.DataFrame
    topk_test_results: pd.DataFrame
    oos_trades: pd.DataFrame
    oos_equity: pd.DataFrame
    consolidated_metrics: dict[str, float]
    best_params_from_tests: dict[str, float | int | bool]


@dataclass(slots=True)
class WalkForwardCheckpoint:
    """Partial walk-forward state emitted after each completed window."""

    strategy_name: str
    window_results: pd.DataFrame
    topk_test_results: pd.DataFrame
    oos_trades: pd.DataFrame
    oos_equity: pd.DataFrame
    windows_completed: int
    total_windows: int
    latest_oos_score: float
    latest_oos_net_profit: float


def run_walkforward(
    df: pd.DataFrame,
    strategy: StrategySpec,
    base_config: BacktestConfig,
    optimizer_config: OptimizerConfig,
    wf_config: WalkForwardConfig,
    progress_callback: ProgressCallback | None = None,
    checkpoint_callback: CheckpointCallback | None = None,
) -> WalkForwardResult:
    """Run full walk-forward for a strategy and return OOS artifacts."""
    date_index = pd.to_datetime(df["datetime"]).dt.date
    unique_days = pd.Series(date_index.unique()).sort_values().to_list()
    if len(unique_days) < wf_config.train_days + wf_config.test_days:
        raise ValueError(
            f"Dados insuficientes para walk-forward ({len(unique_days)} dias). "
            f"Minimo: {wf_config.train_days + wf_config.test_days}."
        )

    window_rows: list[dict[str, object]] = []
    topk_rows: list[dict[str, object]] = []
    oos_trades_parts: list[pd.DataFrame] = []
    oos_equity_parts: list[pd.DataFrame] = []

    capital = base_config.initial_capital
    step = wf_config.test_days
    start = 0
    window_id = 0
    total_windows = _compute_total_windows(
        num_days=len(unique_days),
        train_days=wf_config.train_days,
        test_days=wf_config.test_days,
        step=step,
    )
    _emit(
        progress_callback,
        {
            "stage": "walkforward_start",
            "strategy": strategy.name,
            "total_windows": total_windows,
        },
    )

    while start + wf_config.train_days + wf_config.test_days <= len(unique_days):
        train_days = unique_days[start : start + wf_config.train_days]
        test_days = unique_days[
            start + wf_config.train_days : start + wf_config.train_days + wf_config.test_days
        ]
        _emit(
            progress_callback,
            {
                "stage": "window_start",
                "strategy": strategy.name,
                "window_id": window_id,
                "window_index": window_id + 1,
                "total_windows": total_windows,
                "train_start": str(min(train_days)),
                "train_end": str(max(train_days)),
                "test_start": str(min(test_days)),
                "test_end": str(max(test_days)),
                "capital_before_window": float(capital),
            },
        )
        train_df = df[pd.to_datetime(df["datetime"]).dt.date.isin(train_days)].reset_index(drop=True)
        test_df = df[pd.to_datetime(df["datetime"]).dt.date.isin(test_days)].reset_index(drop=True)

        def _on_optimizer_progress(event: dict[str, Any]) -> None:
            _emit(
                progress_callback,
                {
                    **event,
                    "window_id": window_id,
                    "window_index": window_id + 1,
                    "total_windows": total_windows,
                },
            )

        candidates = optimize_strategy(
            train_df=train_df,
            strategy=strategy,
            base_config=base_config,
            optimizer_config=optimizer_config,
            progress_callback=_on_optimizer_progress if progress_callback else None,
        )
        if not candidates:
            break

        best_train = candidates[0]
        train_start = min(train_days)
        train_end = max(train_days)
        test_start = min(test_days)
        test_end = max(test_days)

        for rank, candidate in enumerate(candidates, start=1):
            test_cfg = build_runtime_config_for_params(base_config=base_config, params=candidate.params)
            test_result = run_backtest(
                df=test_df,
                signals=generate_signals_with_time_filter(test_df, strategy, candidate.params),
                config=test_cfg,
                strategy_name=strategy.name,
                strategy_params=candidate.params,
            )
            test_metrics = compute_metrics(
                trades=test_result.trades,
                equity_curve=test_result.equity_curve,
                initial_capital=test_cfg.initial_capital,
                score_config=wf_config.score_config,
            )
            test_metrics = apply_hard_limits_to_metrics(
                trades=test_result.trades,
                metrics=test_metrics,
                optimizer_config=optimizer_config,
            )
            topk_rows.append(
                {
                    "window_id": window_id,
                    "train_rank": rank,
                    "train_score": candidate.train_score,
                    "test_score": test_metrics["score"],
                    "train_start": str(train_start),
                    "train_end": str(train_end),
                    "test_start": str(test_start),
                    "test_end": str(test_end),
                    "params_json": json.dumps(candidate.params, sort_keys=True),
                    **{f"test_{k}": v for k, v in test_metrics.items()},
                }
            )

        # OOS stream uses best params from train only (no leak from test selection).
        oos_cfg = build_runtime_config_for_params(base_config=base_config, params=best_train.params)
        oos_cfg.initial_capital = capital
        oos_result = run_backtest(
            df=test_df,
            signals=generate_signals_with_time_filter(test_df, strategy, best_train.params),
            config=oos_cfg,
            strategy_name=strategy.name,
            strategy_params=best_train.params,
        )
        oos_metrics = compute_metrics(
            trades=oos_result.trades,
            equity_curve=oos_result.equity_curve,
            initial_capital=capital,
            score_config=wf_config.score_config,
        )
        oos_metrics = apply_hard_limits_to_metrics(
            trades=oos_result.trades,
            metrics=oos_metrics,
            optimizer_config=optimizer_config,
        )
        capital = oos_result.final_capital

        trades_with_window = oos_result.trades.copy()
        if not trades_with_window.empty:
            trades_with_window["window_id"] = window_id
            oos_trades_parts.append(trades_with_window)

        equity_with_window = oos_result.equity_curve.copy()
        if not equity_with_window.empty:
            equity_with_window["window_id"] = window_id
            oos_equity_parts.append(equity_with_window)

        window_rows.append(
            {
                "window_id": window_id,
                "train_start": str(train_start),
                "train_end": str(train_end),
                "test_start": str(test_start),
                "test_end": str(test_end),
                "best_train_score": best_train.train_score,
                "best_train_params_json": json.dumps(best_train.params, sort_keys=True),
                **{f"oos_{k}": v for k, v in oos_metrics.items()},
            }
        )
        _emit(
            progress_callback,
            {
                "stage": "window_complete",
                "strategy": strategy.name,
                "window_id": window_id,
                "window_index": window_id + 1,
                "total_windows": total_windows,
                "capital_after_window": float(capital),
                "oos_score": float(oos_metrics["score"]),
                "oos_net_profit": float(oos_metrics["net_profit"]),
                "oos_profit_factor": float(oos_metrics.get("profit_factor", 0.0)),
                "oos_trade_count": float(oos_metrics["trade_count"]),
            },
        )
        _emit_checkpoint(
            callback=checkpoint_callback,
            checkpoint=WalkForwardCheckpoint(
                strategy_name=strategy.name,
                window_results=pd.DataFrame(window_rows),
                topk_test_results=pd.DataFrame(topk_rows),
                oos_trades=(
                    pd.concat(oos_trades_parts, ignore_index=True)
                    if oos_trades_parts
                    else pd.DataFrame()
                ),
                oos_equity=(
                    pd.concat(oos_equity_parts, ignore_index=True)
                    if oos_equity_parts
                    else pd.DataFrame(columns=["datetime", "equity", "cash"])
                ),
                windows_completed=int(window_id + 1),
                total_windows=int(total_windows),
                latest_oos_score=float(oos_metrics["score"]),
                latest_oos_net_profit=float(oos_metrics["net_profit"]),
            ),
        )

        window_id += 1
        start += step

    oos_trades = (
        pd.concat(oos_trades_parts, ignore_index=True)
        if oos_trades_parts
        else pd.DataFrame()
    )
    oos_equity = (
        pd.concat(oos_equity_parts, ignore_index=True)
        if oos_equity_parts
        else pd.DataFrame(columns=["datetime", "equity", "cash"])
    )
    window_df = pd.DataFrame(window_rows)
    topk_df = pd.DataFrame(topk_rows)

    consolidated = compute_metrics(
        trades=oos_trades,
        equity_curve=oos_equity,
        initial_capital=base_config.initial_capital,
        score_config=wf_config.score_config,
    )
    best_params = _best_params_by_test_score(topk_df)
    _emit(
        progress_callback,
        {
            "stage": "walkforward_done",
            "strategy": strategy.name,
            "total_windows": int(len(window_rows)),
            "final_score": float(consolidated["score"]),
            "net_profit": float(consolidated["net_profit"]),
            "profit_factor": float(consolidated.get("profit_factor", 0.0)),
            "trade_count": float(consolidated["trade_count"]),
        },
    )

    return WalkForwardResult(
        strategy_name=strategy.name,
        window_results=window_df,
        topk_test_results=topk_df,
        oos_trades=oos_trades,
        oos_equity=oos_equity,
        consolidated_metrics=consolidated,
        best_params_from_tests=best_params,
    )


def _best_params_by_test_score(topk_df: pd.DataFrame) -> dict[str, float | int | bool]:
    if topk_df.empty:
        return {}
    grouped = (
        topk_df.groupby("params_json", as_index=False)["test_score"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "avg_test_score", "count": "windows"})
        .sort_values(["avg_test_score", "windows"], ascending=[False, False])
    )
    best_json = grouped.iloc[0]["params_json"]
    return json.loads(best_json)


def _compute_total_windows(num_days: int, train_days: int, test_days: int, step: int) -> int:
    if num_days < train_days + test_days:
        return 0
    return ((num_days - train_days - test_days) // max(step, 1)) + 1


def _emit(callback: ProgressCallback | None, event: dict[str, Any]) -> None:
    if callback is None:
        return
    callback(event)


def _emit_checkpoint(
    callback: CheckpointCallback | None,
    checkpoint: WalkForwardCheckpoint,
) -> None:
    if callback is None:
        return
    callback(checkpoint)
