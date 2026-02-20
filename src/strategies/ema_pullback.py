"""EMA pullback strategy definition."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import StrategySpec


def sample_parameters(rng: np.random.Generator) -> dict[str, float | int | bool]:
    """Random-search parameter sampler."""
    ema_fast = int(rng.integers(10, 41))
    ema_slow = int(rng.integers(max(ema_fast + 40, 80), 301))
    stop_points = int(rng.integers(120, 801))
    rr = float(rng.uniform(1.0, 2.6))
    take_points = int(max(stop_points * rr, stop_points + 40))
    atr_period = int(rng.integers(8, 31))
    atr_min = float(rng.choice([0, 40, 60, 80, 100, 120, 150]))
    use_adx = bool(rng.integers(0, 2))
    adx_period = int(rng.integers(8, 31))
    adx_min = float(rng.choice([15, 20, 25, 30, 35]))

    return {
        "ema_fast": ema_fast,
        "ema_slow": ema_slow,
        "stop_points": stop_points,
        "take_points": take_points,
        "atr_period": atr_period,
        "atr_min": atr_min,
        "use_adx": use_adx,
        "adx_period": adx_period,
        "adx_min": adx_min,
    }


def generate_signals(
    df: pd.DataFrame, params: dict[str, float | int | bool]
) -> pd.Series:
    """Generate close-based entry signals (+1 long, -1 short, 0 none)."""
    ema_fast_n = int(params["ema_fast"])
    ema_slow_n = int(params["ema_slow"])
    atr_period = int(params["atr_period"])
    atr_min = float(params["atr_min"])
    use_adx = bool(params.get("use_adx", False))
    adx_period = int(params.get("adx_period", 14))
    adx_min = float(params.get("adx_min", 20.0))

    close = df["close"]
    open_ = df["open"]
    high = df["high"]
    low = df["low"]

    ema_fast = close.ewm(span=ema_fast_n, adjust=False).mean()
    ema_slow = close.ewm(span=ema_slow_n, adjust=False).mean()
    atr = _atr(df, period=atr_period)

    trend_long = close > ema_slow
    trend_short = close < ema_slow

    pullback_long = low <= ema_fast
    pullback_short = high >= ema_fast

    # "Volta a favor": candle fecha na direcao da tendencia e melhora vs candle anterior.
    momentum_long = (close > open_) & (close > close.shift(1)) & (close >= ema_fast)
    momentum_short = (close < open_) & (close < close.shift(1)) & (close <= ema_fast)

    long_signal = trend_long & pullback_long & momentum_long
    short_signal = trend_short & pullback_short & momentum_short

    if atr_min > 0:
        long_signal &= atr >= atr_min
        short_signal &= atr >= atr_min

    if use_adx:
        adx = _adx(df, period=adx_period)
        long_signal &= adx >= adx_min
        short_signal &= adx >= adx_min

    signals = pd.Series(0, index=df.index, dtype=int)
    signals[long_signal.fillna(False)] = 1
    signals[short_signal.fillna(False)] = -1
    return signals


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def _adx(df: pd.DataFrame, period: int) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr.replace(0, np.nan))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.rolling(period).mean()


STRATEGY = StrategySpec(
    name="ema_pullback",
    generate_signals=generate_signals,
    sample_parameters=sample_parameters,
)

