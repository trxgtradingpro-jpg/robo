from __future__ import annotations

import pandas as pd

from src.backtest_engine import BacktestConfig, run_backtest


def test_dynamic_range_cost_increases_trade_costs() -> None:
    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2025-01-02 09:00:00", "2025-01-02 09:05:00"]),
            "open": [100.0, 102.0],
            "high": [103.0, 105.0],
            "low": [99.0, 101.0],
            "close": [102.0, 104.0],
            "volume": [10.0, 10.0],
        }
    )
    signals = pd.Series([1, 0], index=df.index, dtype=int)

    cfg_fixed = BacktestConfig(
        initial_capital=10_000.0,
        entry_mode="close_slippage",
        fixed_cost_per_trade=1.0,
        cost_per_contract=0.0,
        cost_model="fixed",
    )
    fixed_result = run_backtest(df, signals, cfg_fixed, "x", {})
    fixed_cost = float(fixed_result.trades.iloc[0]["costs"])

    cfg_dynamic = BacktestConfig(
        initial_capital=10_000.0,
        entry_mode="close_slippage",
        fixed_cost_per_trade=1.0,
        cost_per_contract=0.0,
        cost_model="range_scaled",
        cost_range_factor=0.5,
    )
    dynamic_result = run_backtest(df, signals, cfg_dynamic, "x", {})
    dynamic_cost = float(dynamic_result.trades.iloc[0]["costs"])

    assert dynamic_cost > fixed_cost
    assert int(dynamic_result.trades.iloc[0]["duration_bars"]) >= 1


def test_break_even_trigger_moves_stop_to_entry() -> None:
    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    "2025-01-02 09:00:00",
                    "2025-01-02 09:05:00",
                    "2025-01-02 09:10:00",
                ]
            ),
            "open": [100.0, 101.0, 101.0],
            "high": [101.0, 107.0, 103.0],
            "low": [99.0, 101.0, 99.0],
            "close": [100.0, 106.0, 100.0],
            "volume": [10.0, 10.0, 10.0],
        }
    )
    signals = pd.Series([1, 0, 0], index=df.index, dtype=int)
    cfg = BacktestConfig(
        initial_capital=10_000.0,
        entry_mode="close_slippage",
        stop_points=10.0,
        take_points=200.0,
        break_even_trigger_points=5.0,
        fixed_cost_per_trade=0.0,
        cost_per_contract=0.0,
    )

    result = run_backtest(df, signals, cfg, "x", {})
    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert str(trade["exit_reason"]) == "stop_loss"
    assert float(trade["pnl_points"]) == 0.0
    assert float(trade["pnl_net"]) == 0.0


def test_session_end_1700_forces_flatten() -> None:
    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2025-01-02 16:55:00", "2025-01-02 17:00:00"]),
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.0, 101.0],
            "volume": [10.0, 10.0],
        }
    )
    signals = pd.Series([1, 0], index=df.index, dtype=int)
    cfg = BacktestConfig(
        initial_capital=10_000.0,
        entry_mode="close_slippage",
        stop_points=50.0,
        take_points=500.0,
        session_start="09:00",
        session_end="17:00",
        close_on_session_end=True,
    )

    result = run_backtest(df, signals, cfg, "x", {})
    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert str(trade["exit_reason"]) == "session_end"
    assert pd.Timestamp(trade["exit_time"]) == pd.Timestamp("2025-01-02 17:00:00")
