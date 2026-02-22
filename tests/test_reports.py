from __future__ import annotations

import pandas as pd

from src.reports import (
    build_hourly_report,
    build_monthly_report,
    build_parameter_sensitivity_report,
    build_robustness_report,
)


def test_monthly_report_has_expected_rows() -> None:
    equity = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    "2025-01-10 10:00:00",
                    "2025-01-31 18:00:00",
                    "2025-02-10 10:00:00",
                    "2025-02-28 18:00:00",
                ]
            ),
            "equity": [100_000.0, 100_500.0, 100_450.0, 101_000.0],
            "cash": [100_000.0, 100_500.0, 100_450.0, 101_000.0],
        }
    )
    trades = pd.DataFrame(
        {
            "exit_time": pd.to_datetime(
                ["2025-01-20 10:00:00", "2025-02-15 10:00:00", "2025-02-20 10:00:00"]
            ),
            "pnl_net": [500.0, -200.0, 700.0],
        }
    )
    monthly = build_monthly_report(trades=trades, equity_curve=equity, initial_capital=100_000.0)
    assert len(monthly) == 2
    assert monthly.iloc[0]["month"] == "2025-01"
    assert monthly.iloc[1]["month"] == "2025-02"


def test_robustness_and_sensitivity_reports_have_core_fields() -> None:
    windows = pd.DataFrame(
        {
            "best_train_score": [10.0, 5.0, -2.0],
            "oos_score": [8.0, 2.0, -5.0],
            "oos_net_profit": [100.0, -50.0, -20.0],
            "best_train_params_json": [
                '{"a": 1, "b": 2}',
                '{"a": 1, "b": 3}',
                '{"a": 2, "b": 3}',
            ],
        }
    )
    topk = pd.DataFrame(
        {
            "window_id": [0, 0, 1, 1],
            "train_rank": [1, 2, 1, 2],
            "test_score": [8.0, 1.0, 2.0, 0.0],
            "params_json": ['{"a": 1}', '{"a": 2}', '{"a": 1}', '{"a": 2}'],
            "test_net_profit": [100.0, 20.0, 30.0, 10.0],
            "test_max_drawdown": [50.0, 80.0, 60.0, 70.0],
        }
    )
    robustness = build_robustness_report(
        window_results=windows,
        topk_test_results=topk,
        consolidated_metrics={"max_drawdown_pct": 12.0},
    )
    assert "alerts" in robustness
    assert "parameter_stability" in robustness
    assert "topk_dispersion" in robustness

    sensitivity = build_parameter_sensitivity_report(topk)
    assert not sensitivity.empty
    assert set(["parameter", "value", "avg_test_score"]).issubset(set(sensitivity.columns))


def test_hourly_report_identifies_best_hour() -> None:
    trades = pd.DataFrame(
        {
            "entry_time": pd.to_datetime(
                [
                    "2025-01-10 09:01:00",
                    "2025-01-10 09:10:00",
                    "2025-01-10 10:05:00",
                    "2025-01-10 11:15:00",
                ]
            ),
            "pnl_net": [100.0, -20.0, 50.0, -30.0],
        }
    )
    hourly = build_hourly_report(trades)
    assert not hourly.empty
    assert int(hourly.iloc[0]["hour"]) == 9
    assert float(hourly.iloc[0]["net_profit"]) == 80.0
