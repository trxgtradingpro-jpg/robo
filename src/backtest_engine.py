"""Candle-based backtest engine with costs and slippage."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import time
from typing import Literal

import numpy as np
import pandas as pd


EntryMode = Literal["next_open", "close_slippage"]
SlippageModel = Literal["fixed", "range_scaled"]
CostModel = Literal["fixed", "range_scaled"]


@dataclass(slots=True)
class BacktestConfig:
    """Core execution/cost settings for backtest."""

    initial_capital: float = 100_000.0
    contracts: int = 1
    point_value: float = 0.2
    slippage_points: float = 0.0
    slippage_model: SlippageModel = "fixed"
    slippage_range_factor: float = 0.0
    open_auction_minutes: int = 0
    open_auction_slippage_multiplier: float = 1.0
    fixed_cost_per_trade: float = 0.0
    cost_per_contract: float = 0.0
    cost_model: CostModel = "fixed"
    cost_range_factor: float = 0.0
    open_auction_cost_multiplier: float = 1.0
    stop_points: float = 300.0
    take_points: float = 600.0
    break_even_trigger_points: float = 0.0
    allow_long: bool = True
    allow_short: bool = True
    max_positions: int = 1
    entry_mode: EntryMode = "next_open"
    session_start: str | None = None
    session_end: str | None = None
    close_on_session_end: bool = False


@dataclass(slots=True)
class BacktestResult:
    """Container for backtest outputs."""

    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    final_capital: float


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
    strategy: str
    params: dict[str, float | int | bool]


@dataclass(slots=True)
class _PendingEntry:
    execute_idx: int
    direction: int
    signal_time: pd.Timestamp
    strategy: str
    params: dict[str, float | int | bool]


def run_backtest(
    df: pd.DataFrame,
    signals: pd.Series,
    config: BacktestConfig,
    strategy_name: str,
    strategy_params: dict[str, float | int | bool],
) -> BacktestResult:
    """Run OHLC backtest using close-generated signals with configurable entry model."""
    if df.empty:
        return BacktestResult(
            trades=pd.DataFrame(),
            equity_curve=pd.DataFrame(columns=["datetime", "equity", "cash"]),
            final_capital=config.initial_capital,
        )

    ordered = df.sort_values("datetime").reset_index(drop=True).copy()
    signal_series = signals.reindex(ordered.index).fillna(0).astype(int)

    session_start = _parse_time(config.session_start)
    session_end = _parse_time(config.session_end)

    cash = float(config.initial_capital)
    open_positions: list[_Position] = []
    pending_entries: list[_PendingEntry] = []
    trades: list[dict[str, object]] = []
    equity_rows: list[dict[str, object]] = []

    for i in range(len(ordered)):
        row = ordered.iloc[i]
        dt: pd.Timestamp = row["datetime"]

        # Execute entries scheduled for this candle at candle open.
        due_entries = [pe for pe in pending_entries if pe.execute_idx == i]
        pending_entries = [pe for pe in pending_entries if pe.execute_idx != i]
        for pending in due_entries:
            if len(open_positions) >= max(config.max_positions, 1):
                continue
            if not _in_session(dt, session_start, session_end):
                continue
            if pending.direction > 0 and not config.allow_long:
                continue
            if pending.direction < 0 and not config.allow_short:
                continue

            slippage = _effective_slippage_points(
                row=row,
                dt=dt,
                config=config,
                session_start=session_start,
            )
            entry_price = _apply_slippage(
                row["open"],
                pending.direction,
                slippage,
                is_entry=True,
            )
            position = _build_position(
                direction=pending.direction,
                entry_time=dt,
                entry_price=entry_price,
                entry_idx=i,
                entry_range=_candle_range(row),
                entry_open_auction=_is_open_auction(dt, session_start, config.open_auction_minutes),
                config=config,
                strategy=pending.strategy,
                params=pending.params,
            )
            open_positions.append(position)

        # Manage open positions: session close then stop/take intrabar.
        closed_positions: list[tuple[_Position, float, str, float, bool]] = []
        for pos in open_positions:
            if config.close_on_session_end and not _in_session(dt, session_start, session_end):
                exit_slippage = _effective_slippage_points(
                    row=row,
                    dt=dt,
                    config=config,
                    session_start=session_start,
                )
                exit_price = _apply_slippage(
                    row["open"], pos.direction, exit_slippage, is_entry=False
                )
                closed_positions.append(
                    (pos, exit_price, "session_end", _candle_range(row), _is_open_auction(dt, session_start, config.open_auction_minutes))
                )
                continue

            stop_hit, take_hit = _check_intrabar_hits(
                direction=pos.direction,
                low=row["low"],
                high=row["high"],
                stop_price=pos.stop_price,
                take_price=pos.take_price,
            )
            if not (stop_hit or take_hit):
                continue

            # Conservative rule when both are hit in same candle.
            if stop_hit and take_hit:
                raw_exit = pos.stop_price
                reason = "stop_and_take_hit_conservative_stop"
            elif stop_hit:
                raw_exit = pos.stop_price
                reason = "stop_loss"
            else:
                raw_exit = pos.take_price
                reason = "take_profit"

            exit_slippage = _effective_slippage_points(
                row=row,
                dt=dt,
                config=config,
                session_start=session_start,
            )
            exit_price = _apply_slippage(raw_exit, pos.direction, exit_slippage, is_entry=False)
            closed_positions.append(
                (pos, exit_price, reason, _candle_range(row), _is_open_auction(dt, session_start, config.open_auction_minutes))
            )

        if closed_positions:
            survivors = []
            to_close = {id(item[0]) for item in closed_positions}
            for pos in open_positions:
                if id(pos) not in to_close:
                    survivors.append(pos)
            open_positions = survivors

            for pos, exit_price, reason, exit_range, exit_open_auction in closed_positions:
                trade = _close_trade(
                    position=pos,
                    exit_time=dt,
                    exit_idx=i,
                    exit_price=exit_price,
                    exit_range=exit_range,
                    exit_open_auction=exit_open_auction,
                    reason=reason,
                    config=config,
                )
                cash += float(trade["pnl_net"])
                trades.append(trade)

        # Break-even ("trava no 0x0"): after close confirms favorable move.
        if config.break_even_trigger_points > 0:
            for pos in open_positions:
                _maybe_arm_break_even(
                    position=pos,
                    close_price=float(row["close"]),
                    trigger_points=float(config.break_even_trigger_points),
                )

        # Build signal at candle close.
        signal = int(signal_series.iloc[i])
        if signal != 0 and len(open_positions) < max(config.max_positions, 1):
            if signal > 0 and config.allow_long:
                _schedule_or_execute_entry(
                    i=i,
                    dt=dt,
                    row=row,
                    signal=1,
                    ordered=ordered,
                    config=config,
                    strategy_name=strategy_name,
                    strategy_params=strategy_params,
                    pending_entries=pending_entries,
                    open_positions=open_positions,
                    session_start=session_start,
                    session_end=session_end,
                )
            elif signal < 0 and config.allow_short:
                _schedule_or_execute_entry(
                    i=i,
                    dt=dt,
                    row=row,
                    signal=-1,
                    ordered=ordered,
                    config=config,
                    strategy_name=strategy_name,
                    strategy_params=strategy_params,
                    pending_entries=pending_entries,
                    open_positions=open_positions,
                    session_start=session_start,
                    session_end=session_end,
                )

        unrealized = _mark_to_market(
            positions=open_positions,
            close_price=row["close"],
            point_value=config.point_value,
            contracts=config.contracts,
        )
        equity_rows.append({"datetime": dt, "equity": cash + unrealized, "cash": cash})

    # Force-close positions at last close for deterministic report completeness.
    if open_positions:
        last_row = ordered.iloc[-1]
        dt = last_row["datetime"]
        for pos in open_positions:
            exit_price = _apply_slippage(
                last_row["close"],
                pos.direction,
                _effective_slippage_points(
                    row=last_row,
                    dt=dt,
                    config=config,
                    session_start=session_start,
                ),
                is_entry=False,
            )
            trade = _close_trade(
                position=pos,
                exit_time=dt,
                exit_idx=len(ordered) - 1,
                exit_price=exit_price,
                exit_range=_candle_range(last_row),
                exit_open_auction=_is_open_auction(dt, session_start, config.open_auction_minutes),
                reason="end_of_data",
                config=config,
            )
            cash += float(trade["pnl_net"])
            trades.append(trade)
        if equity_rows:
            equity_rows[-1]["equity"] = cash
            equity_rows[-1]["cash"] = cash

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_rows)
    return BacktestResult(trades=trades_df, equity_curve=equity_df, final_capital=cash)


def with_stop_take(
    config: BacktestConfig,
    stop_points: float,
    take_points: float,
    break_even_points: float | None = None,
) -> BacktestConfig:
    """Return a config copy with strategy-specific stop/take values."""
    if break_even_points is None:
        return replace(config, stop_points=float(stop_points), take_points=float(take_points))
    return replace(
        config,
        stop_points=float(stop_points),
        take_points=float(take_points),
        break_even_trigger_points=float(max(0.0, break_even_points)),
    )


def default_trade_columns() -> list[str]:
    return [
        "strategy",
        "entry_time",
        "exit_time",
        "direction",
        "entry_price",
        "exit_price",
        "pnl_points",
        "pnl_gross",
        "costs",
        "pnl_net",
        "duration_bars",
        "exit_reason",
        "params",
    ]


def _schedule_or_execute_entry(
    i: int,
    dt: pd.Timestamp,
    row: pd.Series,
    signal: int,
    ordered: pd.DataFrame,
    config: BacktestConfig,
    strategy_name: str,
    strategy_params: dict[str, float | int | bool],
    pending_entries: list[_PendingEntry],
    open_positions: list[_Position],
    session_start: time | None,
    session_end: time | None,
) -> None:
    if config.entry_mode == "close_slippage":
        if not _in_session(dt, session_start, session_end):
            return
        slippage = _effective_slippage_points(
            row=row,
            dt=dt,
            config=config,
            session_start=session_start,
        )
        entry_price = _apply_slippage(
            row["close"], signal, slippage, is_entry=True
        )
        open_positions.append(
            _build_position(
                direction=signal,
                entry_time=dt,
                entry_price=entry_price,
                entry_idx=i,
                entry_range=_candle_range(row),
                entry_open_auction=_is_open_auction(dt, session_start, config.open_auction_minutes),
                config=config,
                strategy=strategy_name,
                params=strategy_params,
            )
        )
        return

    next_idx = i + 1
    if next_idx >= len(ordered):
        return
    pending_entries.append(
        _PendingEntry(
            execute_idx=next_idx,
            direction=signal,
            signal_time=dt,
            strategy=strategy_name,
            params=strategy_params,
        )
    )


def _build_position(
    direction: int,
    entry_time: pd.Timestamp,
    entry_price: float,
    entry_idx: int,
    entry_range: float,
    entry_open_auction: bool,
    config: BacktestConfig,
    strategy: str,
    params: dict[str, float | int | bool],
) -> _Position:
    if direction > 0:
        stop_price = entry_price - config.stop_points
        take_price = entry_price + config.take_points
    else:
        stop_price = entry_price + config.stop_points
        take_price = entry_price - config.take_points
    return _Position(
        direction=direction,
        entry_time=entry_time,
        entry_price=float(entry_price),
        stop_price=float(stop_price),
        take_price=float(take_price),
        break_even_armed=False,
        entry_idx=int(entry_idx),
        entry_range=float(entry_range),
        entry_open_auction=bool(entry_open_auction),
        strategy=strategy,
        params=dict(params),
    )


def _close_trade(
    position: _Position,
    exit_time: pd.Timestamp,
    exit_idx: int,
    exit_price: float,
    exit_range: float,
    exit_open_auction: bool,
    reason: str,
    config: BacktestConfig,
) -> dict[str, object]:
    pnl_points = (exit_price - position.entry_price) * position.direction
    pnl_gross = pnl_points * config.point_value * config.contracts
    costs = _compute_trade_costs(
        config=config,
        entry_range=position.entry_range,
        exit_range=exit_range,
        entry_open_auction=position.entry_open_auction,
        exit_open_auction=exit_open_auction,
    )
    pnl_net = pnl_gross - costs
    return {
        "strategy": position.strategy,
        "entry_time": position.entry_time,
        "exit_time": exit_time,
        "direction": "long" if position.direction > 0 else "short",
        "entry_price": position.entry_price,
        "exit_price": float(exit_price),
        "pnl_points": float(pnl_points),
        "pnl_gross": float(pnl_gross),
        "costs": float(costs),
        "pnl_net": float(pnl_net),
        "duration_bars": int(max(1, exit_idx - position.entry_idx + 1)),
        "exit_reason": reason,
        "params": position.params,
    }


def _mark_to_market(
    positions: list[_Position],
    close_price: float,
    point_value: float,
    contracts: int,
) -> float:
    if not positions:
        return 0.0
    total = 0.0
    for pos in positions:
        pnl_points = (close_price - pos.entry_price) * pos.direction
        total += pnl_points * point_value * contracts
    return float(total)


def _apply_slippage(price: float, direction: int, slippage_points: float, is_entry: bool) -> float:
    """Apply slippage in points in unfavorable direction."""
    if slippage_points <= 0:
        return float(price)
    if is_entry:
        # Long entry buys higher; short entry sells lower.
        return float(price + slippage_points) if direction > 0 else float(price - slippage_points)
    # Long exit sells lower; short exit buys higher.
    return float(price - slippage_points) if direction > 0 else float(price + slippage_points)


def _effective_slippage_points(
    row: pd.Series,
    dt: pd.Timestamp,
    config: BacktestConfig,
    session_start: time | None,
) -> float:
    slippage = float(max(0.0, config.slippage_points))
    if config.slippage_model == "range_scaled":
        slippage += _candle_range(row) * float(max(0.0, config.slippage_range_factor))
    if _is_open_auction(dt, session_start, config.open_auction_minutes):
        slippage *= float(max(config.open_auction_slippage_multiplier, 0.0))
    return float(max(slippage, 0.0))


def _compute_trade_costs(
    config: BacktestConfig,
    entry_range: float,
    exit_range: float,
    entry_open_auction: bool,
    exit_open_auction: bool,
) -> float:
    costs = float(config.fixed_cost_per_trade) + float(config.cost_per_contract) * float(config.contracts)
    if config.cost_model == "range_scaled":
        avg_range = max(0.0, (float(entry_range) + float(exit_range)) / 2.0)
        impact = avg_range * float(max(config.cost_range_factor, 0.0)) * float(config.point_value) * float(config.contracts)
        costs += impact
    if entry_open_auction or exit_open_auction:
        costs *= float(max(config.open_auction_cost_multiplier, 0.0))
    return float(max(costs, 0.0))


def _candle_range(row: pd.Series) -> float:
    return float(max(0.0, float(row["high"]) - float(row["low"])))


def _check_intrabar_hits(
    direction: int,
    low: float,
    high: float,
    stop_price: float,
    take_price: float,
) -> tuple[bool, bool]:
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
