from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.reproducibility import RunManifest, dataframe_sha256, write_manifest


def test_dataframe_sha256_is_order_invariant() -> None:
    df_a = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2025-01-01 09:05:00", "2025-01-01 09:00:00"]),
            "open": [101.0, 100.0],
            "high": [102.0, 101.0],
            "low": [100.0, 99.0],
            "close": [101.5, 100.5],
            "volume": [20.0, 10.0],
        }
    )
    df_b = df_a.iloc[::-1].reset_index(drop=True)

    assert dataframe_sha256(df_a) == dataframe_sha256(df_b)


def test_write_manifest_creates_json(tmp_path: Path) -> None:
    manifest = RunManifest(
        run_id="wf_20260220_000001",
        created_at_utc="2026-02-20T00:00:01+00:00",
        symbol="WINFUT",
        start="2025-01-01T00:00:00",
        end="2025-12-31T23:59:59",
        timeframes=["5m"],
        args={"seed": 42},
        environment={"python_version": "x"},
        data_hashes={"5m": "abc"},
        generated_files=["outputs/WINFUT/5m/summary_5m.csv"],
        errors=[],
    )
    out = tmp_path / "manifest.json"
    write_manifest(manifest, out)

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["run_id"] == "wf_20260220_000001"
    assert payload["data_hashes"]["5m"] == "abc"
