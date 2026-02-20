"""Strategy registry and interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


SignalFunc = Callable[[pd.DataFrame, dict[str, float | int | bool]], pd.Series]
SampleFunc = Callable[[np.random.Generator], dict[str, float | int | bool]]


@dataclass(frozen=True, slots=True)
class StrategySpec:
    """Declarative strategy container used by optimizer and walk-forward."""

    name: str
    generate_signals: SignalFunc
    sample_parameters: SampleFunc


from .breakout_range import STRATEGY as BREAKOUT_RANGE_STRATEGY
from .ema_pullback import STRATEGY as EMA_PULLBACK_STRATEGY
from .scalp_break_even import STRATEGY as SCALP_BREAK_EVEN_STRATEGY


STRATEGIES: dict[str, StrategySpec] = {
    EMA_PULLBACK_STRATEGY.name: EMA_PULLBACK_STRATEGY,
    BREAKOUT_RANGE_STRATEGY.name: BREAKOUT_RANGE_STRATEGY,
    SCALP_BREAK_EVEN_STRATEGY.name: SCALP_BREAK_EVEN_STRATEGY,
}
