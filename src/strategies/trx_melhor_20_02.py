"""TRX variant focused on lower drawdown and better consistency.

Base profile mirrors the provided NTSL setup:
- EMA/Hilo/ADX trend filter
- fixed entry window 09:00-12:00
- stop/take + break-even
- max consecutive losses per day
- optional daily loss/profit limits (pause and flatten)

Compared to base TRX, this variant adds quality filters:
- ADX rising condition
- EMA slope direction condition
- max bar range filter (avoid very volatile entries)
- max distance from EMA (avoid chasing stretched candles)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import StrategySpec
from .trx_htsl import _adx, _apply_exact_entry_time_window, _hilo_activator


def sample_parameters(rng: np.random.Generator) -> dict[str, float | int | bool]:
    ema_period = int(rng.integers(19, 26))
    hilo_period = int(rng.integers(8, 12))
    adx_period = int(rng.integers(12, 16))
    adx_smoothing = int(rng.integers(12, 16))
    adx_min = float(rng.uniform(19.0, 24.0))

    stop_points = int(rng.integers(300, 381))
    take_points = int(max(stop_points * float(rng.uniform(1.65, 2.05)), stop_points + 230))
    break_even_trigger_points = int(min(stop_points, rng.integers(170, 231)))
    break_even_lock_points = int(rng.integers(10, 21))

    max_bar_range_points = int(rng.integers(280, 621))
    max_ema_distance_points = int(rng.integers(140, 361))

    max_consecutive_losses_per_day = int(rng.integers(10, 21))
    daily_loss_limit = float(rng.uniform(300.0, 600.0))
    daily_profit_target = float(rng.uniform(420.0, 780.0))

    return {
        "ema_period": ema_period,
        "hilo_period": hilo_period,
        "adx_period": adx_period,
        "adx_smoothing": adx_smoothing,
        "adx_min": adx_min,
        "stop_points": stop_points,
        "take_points": take_points,
        "break_even_trigger_points": break_even_trigger_points,
        "break_even_lock_points": break_even_lock_points,
        "entry_start_time": "09:00",
        "entry_end_time": "12:00",
        "session_end": "17:40",
        "close_on_session_end": True,
        "max_consecutive_losses_per_day": max_consecutive_losses_per_day,
        "enable_daily_limits": True,
        "daily_loss_limit": daily_loss_limit,
        "daily_profit_target": daily_profit_target,
        "max_bar_range_points": max_bar_range_points,
        "max_ema_distance_points": max_ema_distance_points,
        "require_adx_rising": True,
    }


def generate_signals(df: pd.DataFrame, params: dict[str, float | int | bool]) -> pd.Series:
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    ema_period = int(params.get("ema_period", 22))
    hilo_period = int(params.get("hilo_period", 9))
    adx_period = int(params.get("adx_period", 13))
    adx_smoothing = int(params.get("adx_smoothing", 14))
    adx_min = float(params.get("adx_min", 20.9))
    max_bar_range_points = float(params.get("max_bar_range_points", 500.0))
    max_ema_distance_points = float(params.get("max_ema_distance_points", 250.0))
    require_adx_rising = bool(params.get("require_adx_rising", True))

    ema = close.ewm(span=max(2, ema_period), adjust=False).mean()
    ema_slope = ema.diff()
    hilo = _hilo_activator(high=high, low=low, close=close, period=max(2, hilo_period))
    adx = _adx(df, period=max(2, adx_period), smoothing=max(2, adx_smoothing))
    bar_range = (high - low).abs()
    ema_distance = (close - ema).abs()

    trend_strong = adx >= adx_min
    adx_filter = (adx.diff() >= 0) if require_adx_rising else pd.Series(True, index=df.index)
    volatility_filter = bar_range <= max_bar_range_points
    distance_filter = ema_distance <= max_ema_distance_points

    long_signal = (
        trend_strong
        & adx_filter
        & (ema_slope > 0)
        & (close > ema)
        & (close > hilo)
        & volatility_filter
        & distance_filter
    )
    short_signal = (
        trend_strong
        & adx_filter
        & (ema_slope < 0)
        & (close < ema)
        & (close < hilo)
        & volatility_filter
        & distance_filter
    )

    signals = pd.Series(0, index=df.index, dtype=int)
    signals[long_signal.fillna(False)] = 1
    signals[short_signal.fillna(False)] = -1
    signals = _apply_exact_entry_time_window(df=df, signals=signals, params=params)
    return signals.astype(int)


STRATEGY = StrategySpec(
    name="trx_melhor_20_02",
    generate_signals=generate_signals,
    sample_parameters=sample_parameters,
)

