from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.study_setup_cli import consolidate_outputs


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_consolidate_outputs_merges_histories_and_param_bank(tmp_path: Path) -> None:
    src_a = tmp_path / "outputs_a"
    src_b = tmp_path / "outputs_b"
    target = tmp_path / "outputs_master"

    _write_csv(
        src_a / "WINFUT" / "5m" / "best_history_5m.csv",
        [
            {
                "run_tag": "run_1",
                "timeframe": "5m",
                "created_at_utc": "2026-02-21T01:00:00+00:00",
                "best_strategy": "trx_htsl",
                "best_score": 10.0,
                "best_net_profit": 100.0,
            }
        ],
    )
    _write_csv(
        src_b / "WINFUT" / "5m" / "best_history_5m.csv",
        [
            {
                "run_tag": "run_1",
                "timeframe": "5m",
                "created_at_utc": "2026-02-21T02:00:00+00:00",
                "best_strategy": "trx_htsl",
                "best_score": 11.0,
                "best_net_profit": 120.0,
            },
            {
                "run_tag": "run_2",
                "timeframe": "5m",
                "created_at_utc": "2026-02-21T03:00:00+00:00",
                "best_strategy": "ema_pullback",
                "best_score": 5.0,
                "best_net_profit": 50.0,
            },
        ],
    )

    _write_csv(
        src_a / "WINFUT" / "5m" / "strategy_history_5m.csv",
        [
            {
                "run_tag": "run_1",
                "timeframe": "5m",
                "strategy": "trx_htsl",
                "created_at_utc": "2026-02-21T01:00:00+00:00",
                "rank_in_run": 1,
                "score": 10.0,
                "net_profit": 100.0,
            }
        ],
    )
    _write_csv(
        src_b / "WINFUT" / "5m" / "strategy_history_5m.csv",
        [
            {
                "run_tag": "run_1",
                "timeframe": "5m",
                "strategy": "trx_htsl",
                "created_at_utc": "2026-02-21T02:00:00+00:00",
                "rank_in_run": 1,
                "score": 11.0,
                "net_profit": 120.0,
            },
            {
                "run_tag": "run_1",
                "timeframe": "5m",
                "strategy": "ema_pullback",
                "created_at_utc": "2026-02-21T02:00:00+00:00",
                "rank_in_run": 2,
                "score": 8.0,
                "net_profit": 60.0,
            },
        ],
    )

    bank_a = src_a / "WINFUT" / "5m" / "params_bank_5m_trx_htsl.jsonl"
    bank_a.parent.mkdir(parents=True, exist_ok=True)
    bank_a.write_text(
        "\n".join(
            [
                json.dumps({"test_score": 10.0, "params": {"a": 1}}, ensure_ascii=False),
                json.dumps({"test_score": 2.0, "params": {"a": 2}}, ensure_ascii=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    bank_b = src_b / "WINFUT" / "5m" / "params_bank_5m_trx_htsl.jsonl"
    bank_b.parent.mkdir(parents=True, exist_ok=True)
    bank_b.write_text(
        "\n".join(
            [
                json.dumps({"test_score": 12.0, "params": {"a": 1}}, ensure_ascii=False),
                json.dumps({"test_score": 9.0, "params": {"a": 3}}, ensure_ascii=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = consolidate_outputs(target=target, sources=[src_a, src_b], copy_manifests=False)
    assert report["sources_count"] == 2
    assert int(report["merged_best_history_files"]) == 2
    assert int(report["merged_strategy_history_files"]) == 2
    assert int(report["merged_params_bank_files"]) == 2

    merged_best = pd.read_csv(target / "WINFUT" / "5m" / "best_history_5m.csv")
    assert len(merged_best) == 2
    row_run1 = merged_best.loc[merged_best["run_tag"].astype(str) == "run_1"].iloc[0]
    assert float(row_run1["best_net_profit"]) == 120.0

    merged_strategy = pd.read_csv(target / "WINFUT" / "5m" / "strategy_history_5m.csv")
    assert len(merged_strategy) == 2
    assert set(merged_strategy["strategy"].astype(str)) == {"trx_htsl", "ema_pullback"}
    trx = merged_strategy.loc[merged_strategy["strategy"].astype(str) == "trx_htsl"].iloc[0]
    assert float(trx["score"]) == 11.0

    merged_bank = target / "WINFUT" / "5m" / "params_bank_5m_trx_htsl.jsonl"
    lines = [ln for ln in merged_bank.read_text(encoding="utf-8").splitlines() if ln.strip()]
    payloads = [json.loads(ln) for ln in lines]
    assert len(payloads) == 3
    top = payloads[0]
    assert float(top["test_score"]) == 12.0
    assert top["params"] == {"a": 1}
