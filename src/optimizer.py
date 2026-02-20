"""Parameter optimization routines."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
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
        key = json.dumps(params, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        signals = strategy.generate_signals(train_local, params)
        run_cfg = with_stop_take(
            base_config,
            stop_points=float(params.get("stop_points", base_config.stop_points)),
            take_points=float(params.get("take_points", base_config.take_points)),
            break_even_points=float(params.get("break_even_trigger_points", base_config.break_even_trigger_points)),
        )
        result = run_backtest(
            df=train_local,
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
                    "train_rows": int(len(train_local)),
                    "train_step": int(max(1, optimizer_config.train_bar_step)),
                }
            )

    random_sample_count = 0
    while random_sample_count < optimizer_config.n_samples:
        params = strategy.sample_parameters(rng)
        key = json.dumps(params, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        random_sample_count += 1

        signals = strategy.generate_signals(train_local, params)
        run_cfg = with_stop_take(
            base_config,
            stop_points=float(params.get("stop_points", base_config.stop_points)),
            take_points=float(params.get("take_points", base_config.take_points)),
            break_even_points=float(params.get("break_even_trigger_points", base_config.break_even_trigger_points)),
        )
        result = run_backtest(
            df=train_local,
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
