"""Risk controls for paper/live execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .monitoring import AlertBus


@dataclass(slots=True)
class RiskLimits:
    """Risk thresholds used to halt execution."""

    daily_loss_limit: float = 0.0
    max_drawdown_pct: float = 0.0
    max_consecutive_losses: int = 0
    kill_switch_file: str | None = None


@dataclass(slots=True)
class RiskState:
    """Mutable risk state during execution."""

    halted: bool = False
    halt_code: str = ""
    halt_message: str = ""
    current_day: str = ""
    day_pnl: float = 0.0
    consecutive_losses: int = 0
    equity_peak: float = 0.0
    max_drawdown: float = 0.0


class RiskManager:
    """Risk evaluator called on each bar and trade close."""

    def __init__(
        self,
        limits: RiskLimits,
        initial_capital: float,
        alert_bus: AlertBus,
    ) -> None:
        self.limits = limits
        self.state = RiskState(equity_peak=float(initial_capital))
        self.alert_bus = alert_bus

    def on_bar(self, dt: str, equity: float) -> None:
        if self.state.halted:
            return
        day = str(dt)[:10]
        if day != self.state.current_day:
            self.state.current_day = day
            self.state.day_pnl = 0.0
            self.state.consecutive_losses = 0

        equity = float(equity)
        self.state.equity_peak = max(self.state.equity_peak, equity)
        drawdown = max(0.0, self.state.equity_peak - equity)
        self.state.max_drawdown = max(self.state.max_drawdown, drawdown)

        if self.limits.max_drawdown_pct > 0 and self.state.equity_peak > 0:
            dd_pct = 100.0 * drawdown / self.state.equity_peak
            if dd_pct >= self.limits.max_drawdown_pct:
                self._halt(
                    code="MAX_DRAWDOWN_PCT",
                    message=f"Drawdown {dd_pct:.2f}% >= limite {self.limits.max_drawdown_pct:.2f}%",
                    context={"drawdown_pct": dd_pct},
                )
                return

        if self.limits.kill_switch_file:
            file_path = Path(self.limits.kill_switch_file)
            if file_path.exists():
                self._halt(
                    code="KILL_SWITCH_FILE",
                    message=f"Kill switch detectado em {file_path}",
                    context={"kill_switch_file": str(file_path)},
                )

    def on_trade_close(self, pnl_net: float, dt: str, equity: float) -> None:
        if self.state.halted:
            return
        self.on_bar(dt=dt, equity=equity)
        pnl_net = float(pnl_net)
        self.state.day_pnl += pnl_net
        if pnl_net < 0:
            self.state.consecutive_losses += 1
        elif pnl_net > 0:
            self.state.consecutive_losses = 0

        if self.limits.daily_loss_limit > 0 and self.state.day_pnl <= -abs(self.limits.daily_loss_limit):
            self._halt(
                code="DAILY_LOSS_LIMIT",
                message=f"Perda diaria {self.state.day_pnl:.2f} <= -{abs(self.limits.daily_loss_limit):.2f}",
                context={"day_pnl": self.state.day_pnl},
            )
            return

        if (
            self.limits.max_consecutive_losses > 0
            and self.state.consecutive_losses >= self.limits.max_consecutive_losses
        ):
            self._halt(
                code="MAX_CONSECUTIVE_LOSSES",
                message=(
                    f"Sequencia de perdas {self.state.consecutive_losses} "
                    f">= limite {self.limits.max_consecutive_losses}"
                ),
                context={"consecutive_losses": self.state.consecutive_losses},
            )

    def _halt(self, code: str, message: str, context: dict[str, float | str]) -> None:
        if self.state.halted:
            return
        self.state.halted = True
        self.state.halt_code = code
        self.state.halt_message = message
        self.alert_bus.emit("high", code, message, context)
