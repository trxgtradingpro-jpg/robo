from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest_engine import BacktestConfig
from src.optimizer import OptimizerConfig
from src.strategies import StrategySpec
from src.walkforward import WalkForwardCheckpoint, WalkForwardConfig, run_walkforward


def test_walkforward_emits_checkpoint_each_completed_window() -> None:
    dt = pd.date_range("2025-01-01 10:00:00", periods=8, freq="1D")
    base = np.linspace(100.0, 103.5, len(dt))
    df = pd.DataFrame(
        {
            "datetime": dt,
            "open": base,
            "high": base + 1.0,
            "low": base - 1.0,
            "close": base + 0.2,
            "volume": np.full(len(dt), 10.0),
        }
    )

    def _signals(data: pd.DataFrame, params: dict[str, float | int | bool]) -> pd.Series:
        out = pd.Series(0, index=data.index, dtype=int)
        out.iloc[:] = 1
        return out

    strategy = StrategySpec(
        name="checkpoint_strategy",
        generate_signals=_signals,
        sample_parameters=lambda rng: {
            "stop_points": float(rng.integers(1, 4)),
            "take_points": float(rng.integers(3, 7)),
        },
    )
    checkpoints: list[WalkForwardCheckpoint] = []

    result = run_walkforward(
        df=df,
        strategy=strategy,
        base_config=BacktestConfig(entry_mode="close_slippage", initial_capital=10_000.0),
        optimizer_config=OptimizerConfig(n_samples=2, top_k=1, random_seed=7),
        wf_config=WalkForwardConfig(train_days=3, test_days=1),
        checkpoint_callback=lambda cp: checkpoints.append(cp),
    )

    assert len(checkpoints) == 5
    assert checkpoints[-1].windows_completed == 5
    assert int(len(checkpoints[-1].window_results)) == 5
    assert int(len(result.window_results)) == 5
    assert checkpoints[-1].latest_oos_score == float(result.window_results.iloc[-1]["oos_score"])
