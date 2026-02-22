from __future__ import annotations

import pandas as pd

from src.backtest_engine import BacktestConfig
from src.optimizer import (
    OptimizerConfig,
    apply_entry_hour_filter,
    apply_license_date_filter,
    optimize_strategy,
)
from src.strategies import StrategySpec


def test_apply_entry_hour_filter_keeps_only_selected_hour() -> None:
    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    "2025-01-02 09:00:00",
                    "2025-01-02 10:00:00",
                    "2025-01-02 11:00:00",
                ]
            ),
            "open": [100.0, 100.0, 100.0],
            "high": [101.0, 101.0, 101.0],
            "low": [99.0, 99.0, 99.0],
            "close": [100.0, 100.0, 100.0],
            "volume": [10.0, 10.0, 10.0],
        }
    )
    signals = pd.Series([1, 1, 1], index=df.index, dtype=int)
    filtered = apply_entry_hour_filter(df=df, signals=signals, params={"hour_start": 10, "hour_end": 10})
    assert filtered.tolist() == [0, 1, 0]


def test_optimize_strategy_applies_daily_loss_hard_limit() -> None:
    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    "2025-01-02 09:00:00",
                    "2025-01-02 09:05:00",
                    "2025-01-02 09:10:00",
                ]
            ),
            "open": [100.0, 100.0, 99.0],
            "high": [100.5, 100.2, 99.2],
            "low": [99.8, 98.8, 98.7],
            "close": [100.0, 99.0, 99.0],
            "volume": [10.0, 10.0, 10.0],
        }
    )

    def _signals(data: pd.DataFrame, params: dict[str, float | int | bool]) -> pd.Series:
        out = pd.Series(0, index=data.index, dtype=int)
        out.iloc[0] = 1
        return out

    strategy = StrategySpec(
        name="test_strategy",
        generate_signals=_signals,
        sample_parameters=lambda rng: {"stop_points": 1, "take_points": 10},
    )
    base_cfg = BacktestConfig(
        initial_capital=10_000.0,
        entry_mode="close_slippage",
        point_value=100.0,
        contracts=1,
        stop_points=1.0,
        take_points=10.0,
    )
    opt_cfg = OptimizerConfig(
        n_samples=1,
        top_k=1,
        random_seed=42,
        max_daily_loss=50.0,
    )
    candidates = optimize_strategy(
        train_df=df,
        strategy=strategy,
        base_config=base_cfg,
        optimizer_config=opt_cfg,
    )
    assert len(candidates) == 1
    best = candidates[0]
    assert float(best.train_metrics.get("constraints_ok", 1.0)) == 0.0
    assert float(best.train_score) <= -1e14


def test_apply_license_date_filter_supports_profit_integer_format() -> None:
    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    "2025-02-07 09:00:00",
                    "2025-02-08 09:00:00",
                    "2026-03-21 09:00:00",
                    "2026-03-22 09:00:00",
                ]
            ),
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [101.0, 101.0, 101.0, 101.0],
            "low": [99.0, 99.0, 99.0, 99.0],
            "close": [100.0, 100.0, 100.0, 100.0],
            "volume": [10.0, 10.0, 10.0, 10.0],
        }
    )
    signals = pd.Series([1, 1, 1, 1], index=df.index, dtype=int)
    filtered = apply_license_date_filter(
        df=df,
        signals=signals,
        params={"license_start_date": 1250208, "license_end_date": 1260321},
    )
    assert filtered.tolist() == [0, 1, 1, 0]
