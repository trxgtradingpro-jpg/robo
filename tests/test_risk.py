from __future__ import annotations

from pathlib import Path

from src.monitoring import AlertBus
from src.risk import RiskLimits, RiskManager


def test_risk_manager_halts_on_daily_loss() -> None:
    alerts = AlertBus()
    manager = RiskManager(
        limits=RiskLimits(daily_loss_limit=100.0),
        initial_capital=10_000.0,
        alert_bus=alerts,
    )
    manager.on_bar(dt="2026-01-10T09:00:00", equity=10_000.0)
    manager.on_trade_close(pnl_net=-120.0, dt="2026-01-10T10:00:00", equity=9_880.0)

    assert manager.state.halted is True
    assert manager.state.halt_code == "DAILY_LOSS_LIMIT"
    assert not alerts.to_frame().empty


def test_risk_manager_halts_on_kill_switch(tmp_path: Path) -> None:
    kill_file = tmp_path / "KILL_SWITCH"
    kill_file.write_text("1", encoding="utf-8")
    alerts = AlertBus()
    manager = RiskManager(
        limits=RiskLimits(kill_switch_file=str(kill_file)),
        initial_capital=10_000.0,
        alert_bus=alerts,
    )
    manager.on_bar(dt="2026-01-10T09:00:00", equity=10_000.0)

    assert manager.state.halted is True
    assert manager.state.halt_code == "KILL_SWITCH_FILE"
