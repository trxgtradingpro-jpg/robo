from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.trx_htsl import generate_signals, sample_parameters


def test_trx_htsl_sample_has_core_fields() -> None:
    rng = np.random.default_rng(42)
    params = sample_parameters(rng)
    required = {
        "stop_points",
        "take_points",
        "break_even_trigger_points",
        "break_even_lock_points",
        "hour_start",
        "hour_end",
        "session_end",
        "close_on_session_end",
        "max_consecutive_losses_per_day",
        "license_start_date",
        "license_end_date",
    }
    assert required.issubset(set(params.keys()))


def test_trx_htsl_generate_signals_returns_series() -> None:
    idx = pd.date_range("2025-02-10 09:00:00", periods=120, freq="5min")
    base = np.linspace(100.0, 120.0, len(idx))
    df = pd.DataFrame(
        {
            "datetime": idx,
            "open": base + 0.1,
            "high": base + 0.6,
            "low": base - 0.6,
            "close": base + 0.2,
            "volume": np.full(len(idx), 10.0),
        }
    )
    params = sample_parameters(np.random.default_rng(1))
    signals = generate_signals(df, params)
    assert len(signals) == len(df)
    assert set(signals.dropna().unique()).issubset({-1, 0, 1})
