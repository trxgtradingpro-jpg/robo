"""Scalping strategy with short stop, break-even lock and longer target."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import StrategySpec


def sample_parameters(rng: np.random.Generator) -> dict[str, float | int | bool]:
    stop_points = int(rng.integers(35, 141))
    rr = float(rng.uniform(2.8, 6.2))
    take_points = int(max(stop_points * rr, stop_points + 120))
    break_even_trigger_points = int(max(10, stop_points * float(rng.uniform(0.5, 1.0))))
    ema_fast = int(rng.integers(5, 16))
    ema_slow = int(rng.integers(max(ema_fast + 8, 18), 61))
    min_range_points = int(rng.integers(20, 251))
    return {
        "ema_fast": ema_fast,
        "ema_slow": ema_slow,
        "min_range_points": min_range_points,
        "stop_points": stop_points,
        "take_points": take_points,
        "break_even_trigger_points": break_even_trigger_points,
    }


def generate_signals(df: pd.DataFrame, params: dict[str, float | int | bool]) -> pd.Series:
    ema_fast_n = int(params["ema_fast"])
    ema_slow_n = int(params["ema_slow"])
    min_range_points = float(params.get("min_range_points", 0.0))

    close = df["close"]
    open_ = df["open"]
    high = df["high"]
    low = df["low"]
    bar_range = (high - low).abs()

    ema_fast = close.ewm(span=ema_fast_n, adjust=False).mean()
    ema_slow = close.ewm(span=ema_slow_n, adjust=False).mean()

    trend_long = (ema_fast > ema_slow) & (close > ema_fast)
    trend_short = (ema_fast < ema_slow) & (close < ema_fast)

    pullback_long = (low <= ema_fast) & (close > open_) & (close > close.shift(1))
    pullback_short = (high >= ema_fast) & (close < open_) & (close < close.shift(1))

    long_signal = trend_long & pullback_long
    short_signal = trend_short & pullback_short
    if min_range_points > 0:
        long_signal &= bar_range >= min_range_points
        short_signal &= bar_range >= min_range_points

    signals = pd.Series(0, index=df.index, dtype=int)
    signals[long_signal.fillna(False)] = 1
    signals[short_signal.fillna(False)] = -1
    return signals


STRATEGY = StrategySpec(
    name="scalp_break_even",
    generate_signals=generate_signals,
    sample_parameters=sample_parameters,
)

