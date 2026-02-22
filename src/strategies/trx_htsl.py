"""TRX HTSL strategy (EMA20 + HiloActivator + ADX) adapted for this framework."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import StrategySpec


def sample_parameters(rng: np.random.Generator) -> dict[str, float | int | bool]:
    ema_period = int(rng.integers(16, 29))
    hilo_period = int(rng.integers(8, 15))
    adx_period = int(rng.integers(12, 19))
    adx_smoothing = int(rng.integers(12, 19))
    adx_min = float(rng.uniform(18.0, 30.0))

    stop_points = int(rng.integers(240, 361))
    take_points = int(rng.integers(520, 760))
    break_even_trigger_points = int(rng.integers(170, 231))
    break_even_lock_points = int(rng.integers(5, 21))

    start_hour = int(rng.integers(9, 11))
    end_hour = int(rng.integers(max(start_hour, 11), 13))

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
        "hour_start": start_hour,
        "hour_end": end_hour,
        "entry_start_time": "09:00",
        "entry_end_time": "12:00",
        "session_end": "17:40",
        "close_on_session_end": True,
        "max_consecutive_losses_per_day": 20,
        "license_start_date": "2025-02-08",
        "license_end_date": "2026-03-21",
    }


def generate_signals(df: pd.DataFrame, params: dict[str, float | int | bool]) -> pd.Series:
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    ema_period = int(params.get("ema_period", 20))
    hilo_period = int(params.get("hilo_period", 10))
    adx_period = int(params.get("adx_period", 14))
    adx_smoothing = int(params.get("adx_smoothing", 14))
    adx_min = float(params.get("adx_min", 20.0))

    ema = close.ewm(span=max(2, ema_period), adjust=False).mean()
    hilo = _hilo_activator(high=high, low=low, close=close, period=max(2, hilo_period))
    adx = _adx(df, period=max(2, adx_period), smoothing=max(2, adx_smoothing))

    trend_strong = adx >= adx_min
    long_signal = trend_strong & (close > ema) & (close > hilo)
    short_signal = trend_strong & (close < ema) & (close < hilo)

    signals = pd.Series(0, index=df.index, dtype=int)
    signals[long_signal.fillna(False)] = 1
    signals[short_signal.fillna(False)] = -1
    signals = _apply_exact_entry_time_window(df=df, signals=signals, params=params)
    return signals


def _apply_exact_entry_time_window(
    df: pd.DataFrame,
    signals: pd.Series,
    params: dict[str, float | int | bool],
) -> pd.Series:
    start_raw = str(params.get("entry_start_time", "09:00"))
    end_raw = str(params.get("entry_end_time", "12:00"))
    try:
        start_minutes = _parse_hhmm_to_minutes(start_raw)
        end_minutes = _parse_hhmm_to_minutes(end_raw)
    except ValueError:
        return signals

    dt = pd.to_datetime(df["datetime"], errors="coerce")
    now_minutes = dt.dt.hour * 60 + dt.dt.minute
    # NTSL equivalente: Time >= start e Time <= end (inclusive).
    in_window = (now_minutes >= start_minutes) & (now_minutes <= end_minutes)
    out = signals.copy()
    out.loc[~in_window.fillna(False)] = 0
    return out.astype(int)


def _parse_hhmm_to_minutes(value: str) -> int:
    text = str(value).strip()
    parts = text.split(":")
    if len(parts) < 2:
        raise ValueError(f"Horario invalido: {value}")
    hour = int(parts[0])
    minute = int(parts[1])
    hour = max(0, min(23, hour))
    minute = max(0, min(59, minute))
    return hour * 60 + minute


def _hilo_activator(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    high_ma = high.rolling(period, min_periods=period).mean()
    low_ma = low.rolling(period, min_periods=period).mean()
    hilo = pd.Series(np.nan, index=close.index, dtype=float)
    direction = 0

    for i in range(len(close)):
        hi = high_ma.iloc[i]
        lo = low_ma.iloc[i]
        if pd.isna(hi) or pd.isna(lo):
            continue
        c = float(close.iloc[i])
        if i > 0 and pd.notna(hilo.iloc[i - 1]):
            prev_hilo = float(hilo.iloc[i - 1])
            if c > prev_hilo:
                direction = 1
            elif c < prev_hilo:
                direction = -1
        if direction == 0:
            direction = 1 if c >= float(lo) else -1
        hilo.iloc[i] = float(lo) if direction > 0 else float(hi)

    return hilo.ffill()


def _adx(df: pd.DataFrame, period: int, smoothing: int) -> pd.Series:
    high = df["high"].astype(float).to_numpy()
    low = df["low"].astype(float).to_numpy()
    close = df["close"].astype(float).to_numpy()
    n = len(df)
    if n == 0:
        return pd.Series(dtype=float)

    tr = np.zeros(n, dtype=float)
    plus_dm = np.zeros(n, dtype=float)
    minus_dm = np.zeros(n, dtype=float)
    for i in range(1, n):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        plus_dm[i] = up_move if (up_move > down_move and up_move > 0) else 0.0
        minus_dm[i] = down_move if (down_move > up_move and down_move > 0) else 0.0
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    tr[0] = high[0] - low[0]

    period = max(2, int(period))
    smoothing = max(2, int(smoothing))

    atr = np.full(n, np.nan, dtype=float)
    plus_sm = np.full(n, np.nan, dtype=float)
    minus_sm = np.full(n, np.nan, dtype=float)
    if n > period:
        atr[period] = np.nansum(tr[1 : period + 1])
        plus_sm[period] = np.nansum(plus_dm[1 : period + 1])
        minus_sm[period] = np.nansum(minus_dm[1 : period + 1])
        for i in range(period + 1, n):
            atr[i] = atr[i - 1] - (atr[i - 1] / period) + tr[i]
            plus_sm[i] = plus_sm[i - 1] - (plus_sm[i - 1] / period) + plus_dm[i]
            minus_sm[i] = minus_sm[i - 1] - (minus_sm[i - 1] / period) + minus_dm[i]

    plus_di = np.full(n, np.nan, dtype=float)
    minus_di = np.full(n, np.nan, dtype=float)
    valid = ~np.isnan(atr) & (atr != 0)
    plus_di[valid] = 100.0 * (plus_sm[valid] / atr[valid])
    minus_di[valid] = 100.0 * (minus_sm[valid] / atr[valid])
    denom = plus_di + minus_di
    dx = np.full(n, np.nan, dtype=float)
    valid_dx = ~np.isnan(denom) & (denom != 0)
    dx[valid_dx] = 100.0 * np.abs(plus_di[valid_dx] - minus_di[valid_dx]) / denom[valid_dx]

    adx = np.full(n, np.nan, dtype=float)
    first = period + smoothing
    if n > first:
        adx[first] = np.nanmean(dx[period + 1 : first + 1])
        for i in range(first + 1, n):
            adx[i] = ((adx[i - 1] * (smoothing - 1)) + dx[i]) / smoothing
    return pd.Series(adx, index=df.index, dtype=float)


STRATEGY = StrategySpec(
    name="trx_htsl",
    generate_signals=generate_signals,
    sample_parameters=sample_parameters,
)
