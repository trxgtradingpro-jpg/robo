"""Parameter optimization routines."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from datetime import date
from typing import Any, Callable

import numpy as np
import pandas as pd

from .backtest_engine import BacktestConfig, run_backtest, with_stop_take
from .metrics import ScoreConfig, compute_metrics
from .strategies import StrategySpec

ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(slots=True)
class OptimizerConfig:
    """Random-search optimizer settings."""

    n_samples: int = 200
    top_k: int = 5
    random_seed: int = 42
    train_bar_step: int = 1
    score_config: ScoreConfig = field(default_factory=ScoreConfig)
    seed_params: tuple[dict[str, float | int | bool], ...] = ()
    max_stop_points: float = 0.0
    max_daily_loss: float = 0.0
    max_drawdown_pct_hard: float = 0.0
    optimize_hours: bool = False
    hour_start_min: int = 9
    hour_start_max: int = 16
    hour_end_min: int = 10
    hour_end_max: int = 18


@dataclass(slots=True)
class OptimizationCandidate:
    """Single sampled parameter set and train metrics."""

    params: dict[str, float | int | bool]
    train_metrics: dict[str, float]
    train_score: float


def optimize_strategy(
    train_df: pd.DataFrame,
    strategy: StrategySpec,
    base_config: BacktestConfig,
    optimizer_config: OptimizerConfig,
    progress_callback: ProgressCallback | None = None,
) -> list[OptimizationCandidate]:
    """Run random search on train data and return top-k candidates."""
    if train_df.empty:
        return []

    train_local = _apply_train_step(
        train_df=train_df,
        step=max(1, int(optimizer_config.train_bar_step)),
    )

    rng = np.random.default_rng(optimizer_config.random_seed)
    seen: set[str] = set()
    candidates: list[OptimizationCandidate] = []

    for seed_idx, params in enumerate(optimizer_config.seed_params, start=1):
        params = _prepare_candidate_params(
            params=params,
            rng=rng,
            config=optimizer_config,
            sampled_random_hours=False,
        )
        key = json.dumps(params, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        metrics = _evaluate_candidate(
            train_df=train_local,
            strategy=strategy,
            params=params,
            base_config=base_config,
            optimizer_config=optimizer_config,
        )
        candidates.append(
            OptimizationCandidate(
                params=params,
                train_metrics=metrics,
                train_score=float(metrics["score"]),
            )
        )
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "optimizer_seed",
                    "strategy": strategy.name,
                    "seed_index": int(seed_idx),
                    "seed_total": int(len(optimizer_config.seed_params)),
                    "score": float(metrics["score"]),
                    "net_profit": float(metrics.get("net_profit", 0.0)),
                    "profit_factor": float(metrics.get("profit_factor", 0.0)),
                    "constraints_ok": float(metrics.get("constraints_ok", 1.0)),
                    "train_rows": int(len(train_local)),
                    "train_step": int(max(1, optimizer_config.train_bar_step)),
                }
            )

    random_sample_count = 0
    while random_sample_count < optimizer_config.n_samples:
        params = _prepare_candidate_params(
            params=strategy.sample_parameters(rng),
            rng=rng,
            config=optimizer_config,
            sampled_random_hours=True,
        )
        key = json.dumps(params, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        random_sample_count += 1

        metrics = _evaluate_candidate(
            train_df=train_local,
            strategy=strategy,
            params=params,
            base_config=base_config,
            optimizer_config=optimizer_config,
        )
        candidates.append(
            OptimizationCandidate(
                params=params,
                train_metrics=metrics,
                train_score=float(metrics["score"]),
            )
        )
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "optimizer_sample",
                    "strategy": strategy.name,
                    "sample_index": random_sample_count,
                    "samples_total": optimizer_config.n_samples,
                    "score": float(metrics["score"]),
                    "net_profit": float(metrics.get("net_profit", 0.0)),
                    "profit_factor": float(metrics.get("profit_factor", 0.0)),
                    "constraints_ok": float(metrics.get("constraints_ok", 1.0)),
                    "train_rows": int(len(train_local)),
                    "train_step": int(max(1, optimizer_config.train_bar_step)),
                }
            )

    ranked = sorted(candidates, key=lambda c: c.train_score, reverse=True)
    top = ranked[: max(1, optimizer_config.top_k)]
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "optimizer_done",
                "strategy": strategy.name,
                "samples_total": optimizer_config.n_samples,
                "top_k": len(top),
                "best_score": float(top[0].train_score) if top else 0.0,
                "train_rows": int(len(train_local)),
                "train_step": int(max(1, optimizer_config.train_bar_step)),
            }
        )
    return top


def _apply_train_step(train_df: pd.DataFrame, step: int) -> pd.DataFrame:
    if step <= 1 or len(train_df) < 500:
        return train_df
    sampled = train_df.iloc[::step].reset_index(drop=True)
    if len(sampled) < 200:
        return train_df
    return sampled


def generate_signals_with_time_filter(
    df: pd.DataFrame,
    strategy: StrategySpec,
    params: dict[str, float | int | bool],
) -> pd.Series:
    signals = strategy.generate_signals(df, params).reindex(df.index).fillna(0).astype(int)
    signals = apply_entry_hour_filter(df=df, signals=signals, params=params)
    signals = apply_license_date_filter(df=df, signals=signals, params=params)
    return signals


def apply_entry_hour_filter(
    df: pd.DataFrame,
    signals: pd.Series,
    params: dict[str, float | int | bool],
) -> pd.Series:
    start_raw = params.get("hour_start")
    end_raw = params.get("hour_end")
    if start_raw is None or end_raw is None:
        return signals.fillna(0).astype(int)
    if "datetime" not in df.columns:
        return signals.fillna(0).astype(int)
    try:
        start_hour = int(start_raw)
        end_hour = int(end_raw)
    except (TypeError, ValueError):
        return signals.fillna(0).astype(int)
    start_hour = max(0, min(23, start_hour))
    end_hour = max(0, min(23, end_hour))

    dt = pd.to_datetime(df["datetime"], errors="coerce")
    hours = dt.dt.hour
    if start_hour <= end_hour:
        in_window = (hours >= start_hour) & (hours <= end_hour)
    else:
        in_window = (hours >= start_hour) | (hours <= end_hour)

    out = signals.fillna(0).astype(int).copy()
    out.loc[~in_window.fillna(False)] = 0
    return out.astype(int)


def apply_license_date_filter(
    df: pd.DataFrame,
    signals: pd.Series,
    params: dict[str, float | int | bool],
) -> pd.Series:
    start_raw = params.get("license_start_date")
    end_raw = params.get("license_end_date")
    start_date = _coerce_date(start_raw)
    end_date = _coerce_date(end_raw)
    if start_date is None and end_date is None:
        return signals.fillna(0).astype(int)
    if "datetime" not in df.columns:
        return signals.fillna(0).astype(int)
    dt = pd.to_datetime(df["datetime"], errors="coerce").dt.date
    valid = pd.Series(True, index=signals.index)
    if start_date is not None:
        valid &= dt >= start_date
    if end_date is not None:
        valid &= dt <= end_date
    out = signals.fillna(0).astype(int).copy()
    out.loc[~valid.fillna(False)] = 0
    return out.astype(int)


def _prepare_candidate_params(
    params: dict[str, float | int | bool],
    rng: np.random.Generator,
    config: OptimizerConfig,
    sampled_random_hours: bool,
) -> dict[str, float | int | bool]:
    out = dict(params)
    if config.max_stop_points > 0 and "stop_points" in out:
        stop = float(out["stop_points"])
        stop = min(stop, float(config.max_stop_points))
        out["stop_points"] = int(round(stop)) if isinstance(out["stop_points"], int) else float(stop)
        if "break_even_trigger_points" in out:
            be = min(float(out["break_even_trigger_points"]), stop)
            out["break_even_trigger_points"] = (
                int(round(be)) if isinstance(out["break_even_trigger_points"], int) else float(be)
            )

    if config.optimize_hours:
        hsm, hsx, hem, hex_ = _normalized_hour_bounds(config)
        if sampled_random_hours:
            start_hour = int(rng.integers(hsm, hsx + 1))
            end_min = max(start_hour, hem)
            if end_min > hex_:
                end_min = start_hour
            end_hour = int(rng.integers(end_min, max(end_min, hex_) + 1))
            out["hour_start"] = start_hour
            out["hour_end"] = end_hour
        else:
            if "hour_start" in out and "hour_end" in out:
                try:
                    out["hour_start"] = int(max(0, min(23, int(out["hour_start"]))))
                    out["hour_end"] = int(max(0, min(23, int(out["hour_end"]))))
                except (TypeError, ValueError):
                    out["hour_start"] = int(hsm)
                    out["hour_end"] = int(hex_)
            else:
                out["hour_start"] = int(hsm)
                out["hour_end"] = int(hex_)

    return out


def _normalized_hour_bounds(config: OptimizerConfig) -> tuple[int, int, int, int]:
    hsm = int(max(0, min(23, config.hour_start_min)))
    hsx = int(max(0, min(23, config.hour_start_max)))
    hem = int(max(0, min(23, config.hour_end_min)))
    hex_ = int(max(0, min(23, config.hour_end_max)))
    if hsx < hsm:
        hsm, hsx = hsx, hsm
    if hex_ < hem:
        hem, hex_ = hex_, hem
    return hsm, hsx, hem, hex_


def _evaluate_candidate(
    train_df: pd.DataFrame,
    strategy: StrategySpec,
    params: dict[str, float | int | bool],
    base_config: BacktestConfig,
    optimizer_config: OptimizerConfig,
) -> dict[str, float]:
    signals = generate_signals_with_time_filter(train_df, strategy, params)
    run_cfg = _build_runtime_config(base_config=base_config, params=params)
    result = run_backtest(
        df=train_df,
        signals=signals,
        config=run_cfg,
        strategy_name=strategy.name,
        strategy_params=params,
    )
    metrics = compute_metrics(
        trades=result.trades,
        equity_curve=result.equity_curve,
        initial_capital=run_cfg.initial_capital,
        score_config=optimizer_config.score_config,
    )
    return apply_hard_limits_to_metrics(trades=result.trades, metrics=metrics, optimizer_config=optimizer_config)


def apply_hard_limits_to_metrics(
    trades: pd.DataFrame,
    metrics: dict[str, float],
    optimizer_config: OptimizerConfig,
) -> dict[str, float]:
    out = dict(metrics)
    violates, worst_daily = _violates_hard_limits(
        trades=trades,
        metrics=out,
        max_daily_loss=float(max(0.0, optimizer_config.max_daily_loss)),
        max_drawdown_pct_hard=float(max(0.0, optimizer_config.max_drawdown_pct_hard)),
    )
    out["worst_daily_pnl"] = float(worst_daily)
    out["constraints_ok"] = 0.0 if violates else 1.0
    if violates:
        out["score"] = -1e15
    return out


def _violates_hard_limits(
    trades: pd.DataFrame,
    metrics: dict[str, float],
    max_daily_loss: float,
    max_drawdown_pct_hard: float,
) -> tuple[bool, float]:
    worst_daily = _worst_daily_pnl(trades)
    if max_daily_loss > 0 and worst_daily < -abs(max_daily_loss):
        return True, worst_daily
    if max_drawdown_pct_hard > 0 and float(metrics.get("max_drawdown_pct", 0.0)) > max_drawdown_pct_hard:
        return True, worst_daily
    return False, worst_daily


def _worst_daily_pnl(trades: pd.DataFrame) -> float:
    if trades.empty or "exit_time" not in trades.columns or "pnl_net" not in trades.columns:
        return 0.0
    local = trades.copy()
    local["exit_time"] = pd.to_datetime(local["exit_time"], errors="coerce")
    local = local.dropna(subset=["exit_time"])
    if local.empty:
        return 0.0
    local["day"] = local["exit_time"].dt.date
    daily = local.groupby("day", as_index=False)["pnl_net"].sum(numeric_only=True)
    if daily.empty:
        return 0.0
    return float(daily["pnl_net"].min())


def _build_runtime_config(
    base_config: BacktestConfig,
    params: dict[str, float | int | bool],
) -> BacktestConfig:
    cfg = with_stop_take(
        base_config,
        stop_points=float(params.get("stop_points", base_config.stop_points)),
        take_points=float(params.get("take_points", base_config.take_points)),
        break_even_points=float(params.get("break_even_trigger_points", base_config.break_even_trigger_points)),
        break_even_lock_points=float(params.get("break_even_lock_points", base_config.break_even_lock_points)),
    )
    overrides: dict[str, Any] = {}
    if "session_start" in params and isinstance(params["session_start"], str):
        overrides["session_start"] = str(params["session_start"]).strip() or None
    if "session_end" in params and isinstance(params["session_end"], str):
        overrides["session_end"] = str(params["session_end"]).strip() or None
    if "close_on_session_end" in params:
        overrides["close_on_session_end"] = bool(params["close_on_session_end"])
    if "max_consecutive_losses_per_day" in params:
        try:
            overrides["max_consecutive_losses_per_day"] = int(max(0, int(params["max_consecutive_losses_per_day"])))
        except (TypeError, ValueError):
            pass
    if "enable_daily_limits" in params:
        overrides["enable_daily_limits"] = bool(params["enable_daily_limits"])
    if "daily_loss_limit" in params:
        try:
            overrides["daily_loss_limit"] = float(max(0.0, float(params["daily_loss_limit"])))
        except (TypeError, ValueError):
            pass
    if "daily_profit_target" in params:
        try:
            overrides["daily_profit_target"] = float(max(0.0, float(params["daily_profit_target"])))
        except (TypeError, ValueError):
            pass
    if overrides:
        cfg = replace(cfg, **overrides)
    return cfg


def build_runtime_config_for_params(
    base_config: BacktestConfig,
    params: dict[str, float | int | bool],
) -> BacktestConfig:
    return _build_runtime_config(base_config=base_config, params=params)


def _coerce_date(raw: Any) -> date | None:
    if raw is None:
        return None
    if isinstance(raw, date):
        return raw
    if isinstance(raw, (int, float)):
        text = str(int(raw))
        return _coerce_date(text)
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text:
        return None
    if text.isdigit() and len(text) == 7 and text.startswith("1"):
        # Profit style: 1YYMMDD (ex: 1250208 -> 2025-02-08)
        yy = int(text[1:3])
        mm = int(text[3:5])
        dd = int(text[5:7])
        try:
            return date(year=2000 + yy, month=mm, day=dd)
        except ValueError:
            return None
    if text.isdigit() and len(text) == 8:
        yyyy = int(text[0:4])
        mm = int(text[4:6])
        dd = int(text[6:8])
        try:
            return date(year=yyyy, month=mm, day=dd)
        except ValueError:
            return None
    try:
        return pd.Timestamp(text).date()
    except (TypeError, ValueError):
        return None
