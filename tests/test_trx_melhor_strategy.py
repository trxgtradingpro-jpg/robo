from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.trx_melhor_20_02 import generate_signals, sample_parameters


def test_trx_melhor_sample_has_core_fields() -> None:
    rng = np.random.default_rng(42)
    params = sample_parameters(rng)
    required = {
        "ema_period",
        "hilo_period",
        "adx_period",
        "adx_smoothing",
        "adx_min",
        "stop_points",
        "take_points",
        "break_even_trigger_points",
        "break_even_lock_points",
        "entry_start_time",
        "entry_end_time",
        "session_end",
        "close_on_session_end",
        "max_consecutive_losses_per_day",
        "enable_daily_limits",
        "daily_loss_limit",
        "daily_profit_target",
    }
    assert required.issubset(set(params.keys()))


def test_trx_melhor_generate_signals_returns_series() -> None:
    idx = pd.date_range("2025-02-10 09:00:00", periods=180, freq="5min")
    base = np.linspace(100.0, 130.0, len(idx))
    close = base + np.sin(np.linspace(0.0, 12.0, len(idx))) * 0.8
    df = pd.DataFrame(
        {
            "datetime": idx,
            "open": close - 0.1,
            "high": close + 0.7,
            "low": close - 0.7,
            "close": close,
            "volume": np.full(len(idx), 10.0),
        }
    )
    params = sample_parameters(np.random.default_rng(1))
    signals = generate_signals(df, params)
    assert len(signals) == len(df)
    assert set(signals.dropna().unique()).issubset({-1, 0, 1})

