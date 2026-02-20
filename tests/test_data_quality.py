from __future__ import annotations

import pandas as pd

from src.data_loader import build_data_quality_report


def test_data_quality_report_estimates_intraday_gaps() -> None:
    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    "2025-01-02 09:00:00",
                    "2025-01-02 09:05:00",
                    "2025-01-02 09:15:00",
                    "2025-01-03 09:00:00",
                    "2025-01-03 09:05:00",
                ]
            ),
            "open": [100, 101, 102, 103, 104],
            "high": [101, 102, 103, 104, 105],
            "low": [99, 100, 101, 102, 103],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5],
            "volume": [10, 20, 30, 40, 50],
        }
    )
    report = build_data_quality_report(df=df, symbol="WINFUT", timeframe="5m")

    assert report.rows == 5
    assert report.days == 2
    assert report.missing_intraday_bars_estimate == 1
    assert report.duplicate_timestamps == 0
    assert report.ohlc_inconsistent_rows == 0
