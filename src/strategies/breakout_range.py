"""Range breakout strategy definition."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import StrategySpec


def sample_parameters(rng: np.random.Generator) -> dict[str, float | int | bool]:
    """Random-search parameter sampler."""
    lookback = int(rng.integers(8, 81))
    min_range_points = int(rng.integers(80, 1601))
    stop_points = int(rng.integers(120, 901))
    rr = float(rng.uniform(1.0, 2.6))
    take_points = int(max(stop_points * rr, stop_points + 50))
    return {
        "lookback": lookback,
        "min_range_points": min_range_points,
        "stop_points": stop_points,
        "take_points": take_points,
    }


def generate_signals(
    df: pd.DataFrame, params: dict[str, float | int | bool]
) -> pd.Series:
    """Generate breakout signals based on prior N-candle range."""
    lookback = int(params["lookback"])
    min_range_points = float(params["min_range_points"])

    prior_high = df["high"].shift(1).rolling(lookback).max()
    prior_low = df["low"].shift(1).rolling(lookback).min()
    range_points = prior_high - prior_low

    long_signal = (df["close"] > prior_high) & (range_points >= min_range_points)
    short_signal = (df["close"] < prior_low) & (range_points >= min_range_points)

    signals = pd.Series(0, index=df.index, dtype=int)
    signals[long_signal.fillna(False)] = 1
    signals[short_signal.fillna(False)] = -1
    return signals


STRATEGY = StrategySpec(
    name="breakout_range",
    generate_signals=generate_signals,
    sample_parameters=sample_parameters,
)

