from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest_engine import BacktestConfig
from src.trx_profit_parity import (
    build_profit_style_summary,
    run_trx_profit_parity,
    trx_profit_reference_params,
)


def test_trx_profit_reference_params_match_ntsl_baseline() -> None:
    params = trx_profit_reference_params()
    assert int(params["ema_period"]) == 20
    assert int(params["hilo_period"]) == 10
    assert int(params["adx_period"]) == 14
    assert int(params["adx_smoothing"]) == 14
    assert float(params["adx_min"]) == 20.0
    assert float(params["stop_points"]) == 300.0
    assert float(params["take_points"]) == 600.0
    assert float(params["break_even_trigger_points"]) == 190.0
    assert float(params["break_even_lock_points"]) == 10.0
    assert int(params["hour_start"]) == 9
    assert int(params["hour_end"]) == 12
    assert str(params["entry_start_time"]) == "09:00"
    assert str(params["entry_end_time"]) == "12:00"
    assert str(params["session_end"]) == "17:40"
    assert bool(params["close_on_session_end"]) is True
    assert int(params["max_consecutive_losses_per_day"]) == 20
    assert int(params["license_start_date"]) == 1250208
    assert int(params["license_end_date"]) == 1260321


def test_run_trx_profit_parity_returns_summary_and_metrics() -> None:
    dt = pd.date_range("2025-02-10 09:00:00", periods=400, freq="5min")
    base = np.linspace(100.0, 120.0, len(dt))
    wave = np.sin(np.linspace(0, 25, len(dt)))
    close = base + wave
    df = pd.DataFrame(
        {
            "datetime": dt,
            "open": close - 0.2,
            "high": close + 0.6,
            "low": close - 0.6,
            "close": close,
            "volume": np.full(len(dt), 10.0),
        }
    )

    output = run_trx_profit_parity(
        df=df,
        base_config=BacktestConfig(
            initial_capital=100_000.0,
            contracts=5,
            point_value=0.2,
            entry_mode="next_open",
            session_start="09:00",
            session_end="17:40",
            close_on_session_end=True,
            max_positions=1,
        ),
    )
    summary = output.summary_profit_style
    assert isinstance(output.metrics, dict)
    assert "net_profit" in output.metrics
    assert "saldo_liquido_total" in summary
    assert "fator_lucro" in summary
    assert int(summary["maximo_contratos"]) == 5


def test_build_profit_style_summary_empty() -> None:
    summary = build_profit_style_summary(
        trades=pd.DataFrame(),
        equity=pd.DataFrame(columns=["datetime", "equity", "cash"]),
        initial_capital=100_000.0,
        contracts=1,
    )
    assert float(summary["saldo_liquido_total"]) == 0.0
    assert int(summary["numero_operacoes"]) == 0
