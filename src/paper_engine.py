"""Event-driven paper trading engine (candle replay)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time
import time as time_mod
from typing import Any, Callable

import numpy as np
import pandas as pd

from .backtest_engine import BacktestConfig, default_trade_columns
from .metrics import ScoreConfig, compute_metrics
from .monitoring import AlertBus
from .risk import RiskLimits, RiskManager
from .strategies import StrategySpec


EventCallback = Callable[[dict[str, Any]], None]


@dataclass(slots=True)
class PaperEngineConfig:
    """Execution and operational settings for paper replay."""

    backtest_config: BacktestConfig
    risk_limits: RiskLimits = field(default_factory=RiskLimits)
    halt_on_risk: bool = True
    replay_delay_ms: int = 0
    emit_every_bars: int = 1


@dataclass(slots=True)
class PaperEngineResult:
    """Paper trading replay outputs."""

    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    alerts: pd.DataFrame
    metrics: dict[str, float]
    halted: bool
    halt_code: str
    halt_message: str


@dataclass(slots=True)
class _Position:
    direction: int
    entry_time: pd.Timestamp
    entry_price: float
    stop_price: float
    take_price: float
    break_even_armed: bool
    entry_idx: int
    entry_range: float
    entry_open_auction: bool


@dataclass(slots=True)
class _PendingEntry:
    execute_idx: int
    direction: int
    signal_time: pd.Timestamp


def run_paper_engine(
    df: pd.DataFrame,
    strategy: StrategySpec,
    strategy_params: dict[str, float | int | bool],
    config: PaperEngineConfig,
    callback: EventCallback | None = None,
) -> PaperEngineResult:
    """Replay OHLC bars as event-driven paper trading."""
    if df.empty:
        return PaperEngineResult(
            trades=pd.DataFrame(columns=default_trade_columns()),
            equity_curve=pd.DataFrame(columns=["datetime", "equity", "cash"]),
            alerts=pd.DataFrame(columns=["timestamp", "level", "code", "message", "context_json"]),
            metrics=compute_metrics(
                trades=pd.DataFrame(),
                equity_curve=pd.DataFrame(columns=["datetime", "equity", "cash"]),
                initial_capital=config.backtest_config.initial_capital,
                score_config=ScoreConfig(),
            ),
            halted=False,
            halt_code="",
            halt_message="",
        )

    ordered = df.sort_values("datetime").reset_index(drop=True).copy()
    signals = strategy.generate_signals(ordered, strategy_params).reindex(ordered.index).fillna(0).astype(int)

    bt = config.backtest_config
    session_start = _parse_time(bt.session_start)
    session_end = _parse_time(bt.session_end)
    stop_points = float(strategy_params.get("stop_points", bt.stop_points))
    take_points = float(strategy_params.get("take_points", bt.take_points))
    break_even_points = float(
        max(0.0, strategy_params.get("break_even_trigger_points", bt.break_even_trigger_points))
    )

    alert_bus = AlertBus()
    risk_manager = RiskManager(
        limits=config.risk_limits,
        initial_capital=bt.initial_capital,
        alert_bus=alert_bus,
    )

    cash = float(bt.initial_capital)
    position: _Position | None = None
    pending_entry: _PendingEntry | None = None
    trades: list[dict[str, object]] = []
    equity_rows: list[dict[str, object]] = []

    _emit(
        callback,
        {
            "stage": "paper_start",
            "bars_total": int(len(ordered)),
            "strategy": strategy.name,
        },
    )

    for i in range(len(ordered)):
        row = ordered.iloc[i]
        dt: pd.Timestamp = pd.to_datetime(row["datetime"])

        if i % max(config.emit_every_bars, 1) == 0:
            _emit(
                callback,
                {
                    "stage": "bar",
                    "bar_index": int(i + 1),
                    "bars_total": int(len(ordered)),
                    "datetime": dt.isoformat(),
                    "cash": float(cash),
                    "has_position": bool(position is not None),
                },
            )

        # Mark-to-market and risk check at bar start.
        mtm_equity = cash + _unrealized(position, float(row["open"]), bt)
        risk_manager.on_bar(dt=dt.isoformat(), equity=mtm_equity)

        # Flatten immediately if halted and requested.
        if position is not None and config.halt_on_risk and risk_manager.state.halted:
            trade = _close_trade(
                position=position,
                exit_time=dt,
                exit_idx=i,
                exit_price=_apply_slippage(
                    float(row["open"]),
                    position.direction,
                    _effective_slippage_points(row, dt, bt, session_start),
                    is_entry=False,
                ),
                exit_range=_candle_range(row),
                exit_open_auction=_is_open_auction(dt, session_start, bt.open_auction_minutes),
                exit_reason="risk_halt",
                strategy_name=strategy.name,
                strategy_params=strategy_params,
                bt=bt,
            )
            position = None
            cash += float(trade["pnl_net"])
            trades.append(trade)

        # Pending entry executes at bar open.
        if (
            pending_entry is not None
            and pending_entry.execute_idx == i
            and position is None
            and not risk_manager.state.halted
        ):
            if _in_session(dt, session_start, session_end):
                if _allow_direction(pending_entry.direction, bt):
                    position = _open_position(
                        direction=pending_entry.direction,
                        dt=dt,
                        i=i,
                        price=float(row["open"]),
                        row=row,
                        bt=bt,
                        session_start=session_start,
                        stop_points=stop_points,
                        take_points=take_points,
                    )
            pending_entry = None

        # Position management (session stop/take).
        if position is not None:
            closed_trade: dict[str, object] | None = None

            if bt.close_on_session_end and not _in_session(dt, session_start, session_end):
                closed_trade = _close_trade(
                    position=position,
                    exit_time=dt,
                    exit_idx=i,
                    exit_price=_apply_slippage(
                        float(row["open"]),
                        position.direction,
                        _effective_slippage_points(row, dt, bt, session_start),
                        is_entry=False,
                    ),
                    exit_range=_candle_range(row),
                    exit_open_auction=_is_open_auction(dt, session_start, bt.open_auction_minutes),
                    exit_reason="session_end",
                    strategy_name=strategy.name,
                    strategy_params=strategy_params,
                    bt=bt,
                )
            else:
                stop_hit, take_hit = _check_hits(
                    direction=position.direction,
                    low=float(row["low"]),
                    high=float(row["high"]),
                    stop_price=position.stop_price,
                    take_price=position.take_price,
                )
                if stop_hit or take_hit:
                    if stop_hit and take_hit:
                        raw_exit = position.stop_price
                        reason = "stop_and_take_hit_conservative_stop"
                    elif stop_hit:
                        raw_exit = position.stop_price
                        reason = "stop_loss"
                    else:
                        raw_exit = position.take_price
                        reason = "take_profit"

                    closed_trade = _close_trade(
                        position=position,
                        exit_time=dt,
                        exit_idx=i,
                        exit_price=_apply_slippage(
                            raw_exit,
                            position.direction,
                            _effective_slippage_points(row, dt, bt, session_start),
                            is_entry=False,
                        ),
                        exit_range=_candle_range(row),
                        exit_open_auction=_is_open_auction(dt, session_start, bt.open_auction_minutes),
                        exit_reason=reason,
                        strategy_name=strategy.name,
                        strategy_params=strategy_params,
                        bt=bt,
                    )

            if closed_trade is not None:
                position = None
                pnl_net = float(closed_trade["pnl_net"])
                cash += pnl_net
                trades.append(closed_trade)
                risk_manager.on_trade_close(pnl_net=pnl_net, dt=dt.isoformat(), equity=cash)
                _emit(
                    callback,
                    {
                        "stage": "trade_close",
                        "datetime": dt.isoformat(),
                        "reason": str(closed_trade["exit_reason"]),
                        "pnl_net": pnl_net,
                        "cash": float(cash),
                    },
                )
            elif break_even_points > 0:
                _maybe_arm_break_even(
                    position=position,
                    close_price=float(row["close"]),
                    trigger_points=break_even_points,
                )

        # Close-generated signal.
        signal = int(signals.iloc[i])
        if position is None and not risk_manager.state.halted and signal != 0 and _allow_direction(signal, bt):
            if bt.entry_mode == "close_slippage":
                if _in_session(dt, session_start, session_end):
                    position = _open_position(
                        direction=signal,
                        dt=dt,
                        i=i,
                        price=float(row["close"]),
                        row=row,
                        bt=bt,
                        session_start=session_start,
                        stop_points=stop_points,
                        take_points=take_points,
                    )
            else:
                if i + 1 < len(ordered):
                    pending_entry = _PendingEntry(execute_idx=i + 1, direction=signal, signal_time=dt)

        equity = cash + _unrealized(position, float(row["close"]), bt)
        risk_manager.on_bar(dt=dt.isoformat(), equity=equity)

        # If risk breached after close mtm, force flatten at close.
        if position is not None and config.halt_on_risk and risk_manager.state.halted:
            closed_trade = _close_trade(
                position=position,
                exit_time=dt,
                exit_idx=i,
                exit_price=_apply_slippage(
                    float(row["close"]),
                    position.direction,
                    _effective_slippage_points(row, dt, bt, session_start),
                    is_entry=False,
                ),
                exit_range=_candle_range(row),
                exit_open_auction=_is_open_auction(dt, session_start, bt.open_auction_minutes),
                exit_reason="risk_halt_close",
                strategy_name=strategy.name,
                strategy_params=strategy_params,
                bt=bt,
            )
            position = None
            cash += float(closed_trade["pnl_net"])
            trades.append(closed_trade)
            equity = cash

        equity_rows.append({"datetime": dt, "equity": float(equity), "cash": float(cash)})
        if config.replay_delay_ms > 0:
            time_mod.sleep(config.replay_delay_ms / 1000.0)

    # Finalize open position at end-of-data.
    if position is not None:
        last = ordered.iloc[-1]
        dt = pd.to_datetime(last["datetime"])
        trade = _close_trade(
            position=position,
            exit_time=dt,
            exit_idx=len(ordered) - 1,
            exit_price=_apply_slippage(
                float(last["close"]),
                position.direction,
                _effective_slippage_points(last, dt, bt, session_start),
                is_entry=False,
            ),
            exit_range=_candle_range(last),
            exit_open_auction=_is_open_auction(dt, session_start, bt.open_auction_minutes),
            exit_reason="end_of_data",
            strategy_name=strategy.name,
            strategy_params=strategy_params,
            bt=bt,
        )
        cash += float(trade["pnl_net"])
        trades.append(trade)
        if equity_rows:
            equity_rows[-1]["equity"] = float(cash)
            equity_rows[-1]["cash"] = float(cash)

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        trades_df = pd.DataFrame(columns=default_trade_columns())
    equity_df = pd.DataFrame(equity_rows)
    alerts_df = alert_bus.to_frame()
    metrics = compute_metrics(
        trades=trades_df,
        equity_curve=equity_df,
        initial_capital=bt.initial_capital,
        score_config=ScoreConfig(),
    )

    _emit(
        callback,
        {
            "stage": "paper_done",
            "bars_total": int(len(ordered)),
            "halted": bool(risk_manager.state.halted),
            "halt_code": risk_manager.state.halt_code,
            "net_profit": float(metrics.get("net_profit", 0.0)),
            "trade_count": float(metrics.get("trade_count", 0.0)),
        },
    )
    return PaperEngineResult(
        trades=trades_df,
        equity_curve=equity_df,
        alerts=alerts_df,
        metrics=metrics,
        halted=bool(risk_manager.state.halted),
        halt_code=risk_manager.state.halt_code,
        halt_message=risk_manager.state.halt_message,
    )


def _open_position(
    direction: int,
    dt: pd.Timestamp,
    i: int,
    price: float,
    row: pd.Series,
    bt: BacktestConfig,
    session_start: time | None,
    stop_points: float,
    take_points: float,
) -> _Position:
    slippage = _effective_slippage_points(row, dt, bt, session_start)
    entry_price = _apply_slippage(price, direction, slippage, is_entry=True)
    if direction > 0:
        stop_price = entry_price - float(stop_points)
        take_price = entry_price + float(take_points)
    else:
        stop_price = entry_price + float(stop_points)
        take_price = entry_price - float(take_points)
    return _Position(
        direction=int(direction),
        entry_time=dt,
        entry_price=float(entry_price),
        stop_price=float(stop_price),
        take_price=float(take_price),
        break_even_armed=False,
        entry_idx=int(i),
        entry_range=_candle_range(row),
        entry_open_auction=_is_open_auction(dt, session_start, bt.open_auction_minutes),
    )


def _close_trade(
    position: _Position,
    exit_time: pd.Timestamp,
    exit_idx: int,
    exit_price: float,
    exit_range: float,
    exit_open_auction: bool,
    exit_reason: str,
    strategy_name: str,
    strategy_params: dict[str, float | int | bool],
    bt: BacktestConfig,
) -> dict[str, object]:
    pnl_points = (float(exit_price) - float(position.entry_price)) * float(position.direction)
    pnl_gross = pnl_points * float(bt.point_value) * float(bt.contracts)
    costs = _compute_costs(
        bt=bt,
        entry_range=position.entry_range,
        exit_range=exit_range,
        entry_open_auction=position.entry_open_auction,
        exit_open_auction=exit_open_auction,
    )
    pnl_net = pnl_gross - costs
    return {
        "strategy": strategy_name,
        "entry_time": position.entry_time,
        "exit_time": exit_time,
        "direction": "long" if position.direction > 0 else "short",
        "entry_price": float(position.entry_price),
        "exit_price": float(exit_price),
        "pnl_points": float(pnl_points),
        "pnl_gross": float(pnl_gross),
        "costs": float(costs),
        "pnl_net": float(pnl_net),
        "duration_bars": int(max(1, int(exit_idx) - int(position.entry_idx) + 1)),
        "exit_reason": str(exit_reason),
        "params": dict(strategy_params),
    }


def _compute_costs(
    bt: BacktestConfig,
    entry_range: float,
    exit_range: float,
    entry_open_auction: bool,
    exit_open_auction: bool,
) -> float:
    costs = float(bt.fixed_cost_per_trade) + float(bt.cost_per_contract) * float(bt.contracts)
    if bt.cost_model == "range_scaled":
        avg_range = max(0.0, (float(entry_range) + float(exit_range)) / 2.0)
        costs += avg_range * float(max(bt.cost_range_factor, 0.0)) * float(bt.point_value) * float(bt.contracts)
    if entry_open_auction or exit_open_auction:
        costs *= float(max(bt.open_auction_cost_multiplier, 0.0))
    return float(max(costs, 0.0))


def _effective_slippage_points(
    row: pd.Series,
    dt: pd.Timestamp,
    bt: BacktestConfig,
    session_start: time | None,
) -> float:
    slippage = float(max(0.0, bt.slippage_points))
    if bt.slippage_model == "range_scaled":
        slippage += _candle_range(row) * float(max(0.0, bt.slippage_range_factor))
    if _is_open_auction(dt, session_start, bt.open_auction_minutes):
        slippage *= float(max(bt.open_auction_slippage_multiplier, 0.0))
    return float(max(slippage, 0.0))


def _unrealized(position: _Position | None, close_price: float, bt: BacktestConfig) -> float:
    if position is None:
        return 0.0
    pnl_points = (float(close_price) - float(position.entry_price)) * float(position.direction)
    return float(pnl_points * float(bt.point_value) * float(bt.contracts))


def _allow_direction(signal: int, bt: BacktestConfig) -> bool:
    if signal > 0 and not bt.allow_long:
        return False
    if signal < 0 and not bt.allow_short:
        return False
    return True


def _check_hits(direction: int, low: float, high: float, stop_price: float, take_price: float) -> tuple[bool, bool]:
    if direction > 0:
        return low <= stop_price, high >= take_price
    return high >= stop_price, low <= take_price


def _maybe_arm_break_even(position: _Position, close_price: float, trigger_points: float) -> None:
    trigger = float(max(0.0, trigger_points))
    if trigger <= 0 or position.break_even_armed:
        return
    if position.direction > 0:
        if close_price >= position.entry_price + trigger:
            position.stop_price = max(position.stop_price, position.entry_price)
            position.break_even_armed = True
        return
    if close_price <= position.entry_price - trigger:
        position.stop_price = min(position.stop_price, position.entry_price)
        position.break_even_armed = True


def _apply_slippage(price: float, direction: int, slippage_points: float, is_entry: bool) -> float:
    if slippage_points <= 0:
        return float(price)
    if is_entry:
        return float(price + slippage_points) if direction > 0 else float(price - slippage_points)
    return float(price - slippage_points) if direction > 0 else float(price + slippage_points)


def _parse_time(value: str | None) -> time | None:
    if not value:
        return None
    parts = value.strip().split(":")
    if len(parts) < 2:
        raise ValueError(f"Horario invalido: {value}. Use HH:MM.")
    hour = int(parts[0])
    minute = int(parts[1])
    second = int(parts[2]) if len(parts) > 2 else 0
    return time(hour=hour, minute=minute, second=second)


def _in_session(dt: pd.Timestamp, start: time | None, end: time | None) -> bool:
    if start is None and end is None:
        return True
    current = dt.time()
    if start and current < start:
        return False
    if end and current >= end:
        return False
    return True


def _is_open_auction(dt: pd.Timestamp, session_start: time | None, window_minutes: int) -> bool:
    if session_start is None or window_minutes <= 0:
        return False
    current_minutes = dt.hour * 60 + dt.minute
    start_minutes = session_start.hour * 60 + session_start.minute
    delta = current_minutes - start_minutes
    return 0 <= delta < int(window_minutes)


def _candle_range(row: pd.Series) -> float:
    return float(max(0.0, float(row["high"]) - float(row["low"])))


def _emit(callback: EventCallback | None, event: dict[str, Any]) -> None:
    if callback is None:
        return
    callback(event)
