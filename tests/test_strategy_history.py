from pathlib import Path

import pandas as pd

from src.cli import _append_strategy_history_rows


def _summary_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timeframe": "5m",
                "strategy": "trx_htsl",
                "score": 10.0,
                "net_profit": 500.0,
                "trade_count": 100,
                "win_rate": 0.55,
                "max_drawdown": 200.0,
                "max_drawdown_pct": 0.2,
                "windows": 4,
            },
            {
                "timeframe": "5m",
                "strategy": "ema_pullback",
                "score": 8.0,
                "net_profit": 300.0,
                "trade_count": 80,
                "win_rate": 0.5,
                "max_drawdown": 250.0,
                "max_drawdown_pct": 0.25,
                "windows": 4,
            },
        ]
    )


def test_append_strategy_history_rows_upsert(tmp_path: Path) -> None:
    tf_output = tmp_path / "WINFUT" / "5m"
    tf_output.mkdir(parents=True)

    history_file = tf_output / "strategy_history_5m.csv"
    summary_latest = tf_output / "summary_5m.csv"
    summary_snapshot = tf_output / "summary_5m_run1.csv"
    best_params_latest = tf_output / "best_params_5m.json"
    best_params_snapshot = tf_output / "best_params_5m_run1.json"

    df = _summary_frame()
    _append_strategy_history_rows(
        history_file=history_file,
        summary_df=df,
        run_tag="run1",
        created_at_utc="2026-02-21T03:00:00+00:00",
        symbol="WINFUT",
        timeframe="5m",
        best_strategy="trx_htsl",
        summary_latest=summary_latest,
        summary_snapshot=summary_snapshot,
        best_params_latest=best_params_latest,
        best_params_snapshot=best_params_snapshot,
        tf_output=tf_output,
    )

    saved = pd.read_csv(history_file)
    assert len(saved) == 2
    assert set(saved["strategy"].astype(str)) == {"trx_htsl", "ema_pullback"}
    assert int(saved.loc[saved["strategy"] == "trx_htsl", "is_best"].iloc[0]) == 1
    assert int(saved.loc[saved["strategy"] == "ema_pullback", "rank_in_run"].iloc[0]) == 2

    df2 = _summary_frame()
    df2.loc[df2["strategy"] == "ema_pullback", "net_profit"] = 999.0
    _append_strategy_history_rows(
        history_file=history_file,
        summary_df=df2,
        run_tag="run1",
        created_at_utc="2026-02-21T03:01:00+00:00",
        symbol="WINFUT",
        timeframe="5m",
        best_strategy="trx_htsl",
        summary_latest=summary_latest,
        summary_snapshot=summary_snapshot,
        best_params_latest=best_params_latest,
        best_params_snapshot=best_params_snapshot,
        tf_output=tf_output,
    )

    saved2 = pd.read_csv(history_file)
    assert len(saved2) == 2
    ema = saved2.loc[saved2["strategy"] == "ema_pullback"].iloc[0]
    assert float(ema["net_profit"]) == 999.0
