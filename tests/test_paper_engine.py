from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest_engine import BacktestConfig
from src.paper_engine import PaperEngineConfig, run_paper_engine
from src.risk import RiskLimits
from src.strategies import StrategySpec


def test_paper_engine_halts_on_risk_limit() -> None:
    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    "2026-01-10 09:00:00",
                    "2026-01-10 09:05:00",
                    "2026-01-10 09:10:00",
                    "2026-01-10 09:15:00",
                ]
            ),
            "open": [100.0, 99.0, 98.0, 97.0],
            "high": [100.5, 99.5, 98.5, 97.5],
            "low": [99.0, 98.0, 97.0, 96.0],
            "close": [99.2, 98.2, 97.2, 96.2],
            "volume": [10.0, 10.0, 10.0, 10.0],
        }
    )

    def _signals(data: pd.DataFrame, params: dict[str, float | int | bool]) -> pd.Series:
        out = pd.Series(0, index=data.index, dtype=int)
        out.iloc[0] = 1
        out.iloc[2] = 1
        return out

    strategy = StrategySpec(
        name="test_strategy",
        generate_signals=_signals,
        sample_parameters=lambda rng: {"stop_points": 1, "take_points": 2},
    )
    cfg = PaperEngineConfig(
        backtest_config=BacktestConfig(
            initial_capital=1_000.0,
            point_value=10.0,
            contracts=1,
            stop_points=1.0,
            take_points=2.0,
            entry_mode="close_slippage",
        ),
        risk_limits=RiskLimits(daily_loss_limit=5.0),
        halt_on_risk=True,
    )

    result = run_paper_engine(
        df=df,
        strategy=strategy,
        strategy_params={"stop_points": 1, "take_points": 2},
        config=cfg,
    )

    assert result.halted is True
    assert result.halt_code == "DAILY_LOSS_LIMIT"
    assert len(result.trades) >= 1
    assert len(result.alerts) >= 1


def test_paper_engine_break_even_trigger_moves_stop_to_entry() -> None:
    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    "2026-01-10 09:00:00",
                    "2026-01-10 09:05:00",
                    "2026-01-10 09:10:00",
                ]
            ),
            "open": [100.0, 101.0, 101.0],
            "high": [101.0, 107.0, 103.0],
            "low": [99.0, 101.0, 99.0],
            "close": [100.0, 106.0, 100.0],
            "volume": [10.0, 10.0, 10.0],
        }
    )

    def _signals(data: pd.DataFrame, params: dict[str, float | int | bool]) -> pd.Series:
        out = pd.Series(0, index=data.index, dtype=int)
        out.iloc[0] = 1
        return out

    strategy = StrategySpec(
        name="test_break_even",
        generate_signals=_signals,
        sample_parameters=lambda rng: {
            "stop_points": 10,
            "take_points": 200,
            "break_even_trigger_points": 5,
        },
    )
    cfg = PaperEngineConfig(
        backtest_config=BacktestConfig(
            initial_capital=1_000.0,
            point_value=1.0,
            contracts=1,
            stop_points=10.0,
            take_points=200.0,
            entry_mode="close_slippage",
        ),
        risk_limits=RiskLimits(),
        halt_on_risk=True,
    )

    result = run_paper_engine(
        df=df,
        strategy=strategy,
        strategy_params={"stop_points": 10, "take_points": 200, "break_even_trigger_points": 5},
        config=cfg,
    )

    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert str(trade["exit_reason"]) == "stop_loss"
    assert float(trade["pnl_points"]) == 0.0


def test_paper_engine_session_end_1700_forces_flatten() -> None:
    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-01-10 16:55:00", "2026-01-10 17:00:00"]),
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.0, 101.0],
            "volume": [10.0, 10.0],
        }
    )

    def _signals(data: pd.DataFrame, params: dict[str, float | int | bool]) -> pd.Series:
        out = pd.Series(0, index=data.index, dtype=int)
        out.iloc[0] = 1
        return out

    strategy = StrategySpec(
        name="test_session_end",
        generate_signals=_signals,
        sample_parameters=lambda rng: {"stop_points": 50, "take_points": 500},
    )
    cfg = PaperEngineConfig(
        backtest_config=BacktestConfig(
            initial_capital=1_000.0,
            point_value=1.0,
            contracts=1,
            stop_points=50.0,
            take_points=500.0,
            entry_mode="close_slippage",
            session_end="17:00",
            close_on_session_end=True,
        ),
        risk_limits=RiskLimits(),
        halt_on_risk=True,
    )

    result = run_paper_engine(
        df=df,
        strategy=strategy,
        strategy_params={"stop_points": 50, "take_points": 500},
        config=cfg,
    )

    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert str(trade["exit_reason"]) == "session_end"
    assert pd.Timestamp(trade["exit_time"]) == pd.Timestamp("2026-01-10 17:00:00")
