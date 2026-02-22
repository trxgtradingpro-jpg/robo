"""Estrategia "ganhador_80" focada em alta taxa de acerto com risco curto.

Ideia:
- Regras simples: opera só a favor da micro-tendencia (EMA curta > EMA longa para compra, < para venda).
- Filtro de volatilidade baixa: ignora barras com range muito grande (evita stops largos) e opera somente se range >= range_min.
- Entrada no fechamento (entry_mode=close_slippage) ou proxima abertura (next_open) definido na config externa.
- Stop curto, alvo 2-4x o stop. Break-even trava no 0x0 quando anda trigger pontos.
- Horario fixo configuravel: por padrão 09:05-12:00 para entrar; sempre zera até 17:40 via engine.

Parâmetros sorteados/otimizáveis:
- ema_fast: 4-10
- ema_slow: 14-30 (sempre > ema_fast)
- range_min_points: 10-120 (mínimo de range da barra pra validar sinal)
- stop_points: 40-140
- rr_mult (target = stop * rr_mult): 2.0-4.0
- break_even_trigger_points: 40-120 (máx stop; se menor que stop usa ele)
- break_even_lock_points: 2-8
- entry_start_time / entry_end_time: 09:05-11:55

Expectativa: alta taxa de acerto, lucro por trade modesto, perdas pequenas; ideal para reforçar win-rate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import StrategySpec


def sample_parameters(rng: np.random.Generator) -> dict[str, float | int | bool]:
    ema_fast = int(rng.integers(4, 11))
    ema_slow = int(rng.integers(max(ema_fast + 6, 14), 31))
    range_min_points = int(rng.integers(10, 121))

    stop_points = int(rng.integers(40, 141))
    rr_mult = float(rng.uniform(2.0, 4.1))
    take_points = int(max(stop_points * rr_mult, stop_points + 80))

    be_trigger = int(rng.integers(40, 121))
    be_trigger = min(be_trigger, stop_points)
    be_lock = int(rng.integers(2, 9))

    start_hour = int(rng.integers(9, 10))
    start_min = int(rng.integers(2, 45))
    end_hour = int(rng.integers(max(start_hour, 11), 13))
    end_min = int(rng.integers(0, 55))

    def _fmt(h: int, m: int) -> str:
        return f"{h:02d}:{m:02d}"

    return {
        "ema_fast": ema_fast,
        "ema_slow": ema_slow,
        "range_min_points": range_min_points,
        "stop_points": stop_points,
        "take_points": take_points,
        "break_even_trigger_points": be_trigger,
        "break_even_lock_points": be_lock,
        "entry_start_time": _fmt(start_hour, start_min),
        "entry_end_time": _fmt(end_hour, end_min),
        "session_end": "17:40",
        "close_on_session_end": True,
    }


def generate_signals(df: pd.DataFrame, params: dict[str, float | int | bool]) -> pd.Series:
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    open_ = df["open"].astype(float)

    ema_fast_n = int(params.get("ema_fast", 8))
    ema_slow_n = int(params.get("ema_slow", 21))
    range_min_points = float(params.get("range_min_points", 0.0))

    ema_fast = close.ewm(span=max(2, ema_fast_n), adjust=False).mean()
    ema_slow = close.ewm(span=max(3, ema_slow_n, ema_fast_n + 1), adjust=False).mean()

    bar_range = (high - low).abs()

    # Cruzamento EMA rápida/lenta + corpo na direção + close acima/abaixo de ambas
    up_cross = (ema_fast > ema_slow) & (ema_fast.shift(1) <= ema_slow.shift(1))
    down_cross = (ema_fast < ema_slow) & (ema_fast.shift(1) >= ema_slow.shift(1))

    long_body = close > open_
    short_body = close < open_

    long_signal = up_cross & long_body & (close > ema_fast) & (close > ema_slow)
    short_signal = down_cross & short_body & (close < ema_fast) & (close < ema_slow)

    if range_min_points > 0:
        long_signal &= bar_range >= range_min_points
        short_signal &= bar_range >= range_min_points

    signals = pd.Series(0, index=df.index, dtype=int)
    signals[long_signal.fillna(False)] = 1
    signals[short_signal.fillna(False)] = -1

    # Respeita janela de horario de entrada.
    start_raw = str(params.get("entry_start_time", "09:05"))
    end_raw = str(params.get("entry_end_time", "12:00"))
    try:
        start_minutes = _parse_hhmm_to_minutes(start_raw)
        end_minutes = _parse_hhmm_to_minutes(end_raw)
        dt = pd.to_datetime(df["datetime"], errors="coerce")
        now_minutes = dt.dt.hour * 60 + dt.dt.minute
        in_window = (now_minutes >= start_minutes) & (now_minutes <= end_minutes)
        signals = signals.where(in_window.fillna(False), 0)
    except Exception:
        pass

    return signals.astype(int)


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


STRATEGY = StrategySpec(
    name="ganhador_80",
    generate_signals=generate_signals,
    sample_parameters=sample_parameters,
)
