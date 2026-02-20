from __future__ import annotations

import pandas as pd
import pytest

from src.metrics import ScoreConfig, compute_metrics


def test_compute_metrics_includes_advanced_fields() -> None:
    trades = pd.DataFrame(
        {
            "entry_time": pd.to_datetime(
                [
                    "2025-01-10 09:00:00",
                    "2025-01-10 10:00:00",
                    "2025-02-12 11:00:00",
                    "2025-02-12 12:00:00",
                ]
            ),
            "exit_time": pd.to_datetime(
                [
                    "2025-01-10 09:20:00",
                    "2025-01-10 10:15:00",
                    "2025-02-12 11:45:00",
                    "2025-02-12 12:30:00",
                ]
            ),
            "pnl_net": [10.0, -5.0, 15.0, -5.0],
            "pnl_points": [50.0, -25.0, 75.0, -25.0],
            "costs": [1.0, 1.0, 1.0, 1.0],
        }
    )
    equity = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    "2025-01-10 09:00:00",
                    "2025-01-31 18:00:00",
                    "2025-02-12 12:30:00",
                    "2025-02-28 18:00:00",
                ]
            ),
            "equity": [100_000.0, 100_010.0, 100_020.0, 100_015.0],
            "cash": [100_000.0, 100_010.0, 100_020.0, 100_015.0],
        }
    )

    metrics = compute_metrics(
        trades=trades,
        equity_curve=equity,
        initial_capital=100_000.0,
        score_config=ScoreConfig(),
    )

    assert metrics["trade_count"] == 4.0
    assert metrics["win_rate"] == 0.5
    assert metrics["expectancy"] == pytest.approx(3.75)
    assert metrics["payoff_ratio"] == pytest.approx(2.5)
    assert metrics["max_consecutive_wins"] == 1.0
    assert metrics["max_consecutive_losses"] == 1.0
    assert "monthly_return_mean_pct" in metrics
    assert "monthly_return_std_pct" in metrics
    assert "positive_months_pct" in metrics
