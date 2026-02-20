"""Run reproducibility helpers: data hashing and manifest generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import sys
from typing import Any

import pandas as pd


@dataclass(slots=True)
class RunManifest:
    """Serializable execution manifest for reproducible runs."""

    run_id: str
    created_at_utc: str
    symbol: str
    start: str
    end: str
    timeframes: list[str]
    args: dict[str, Any]
    environment: dict[str, str]
    data_hashes: dict[str, str]
    generated_files: list[str]
    errors: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def build_run_id(prefix: str = "run") -> str:
    """Generate deterministic-like run ID for artifacts."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}"


def dataframe_sha256(df: pd.DataFrame) -> str:
    """Compute a stable sha256 hash from canonical OHLCV dataframe."""
    if df.empty:
        return hashlib.sha256(b"").hexdigest()

    cols = [col for col in ["datetime", "open", "high", "low", "close", "volume"] if col in df.columns]
    canonical = df[cols].copy()
    canonical = canonical.sort_values("datetime").reset_index(drop=True)

    if "datetime" in canonical:
        canonical["datetime"] = pd.to_datetime(canonical["datetime"], errors="coerce").dt.strftime(
            "%Y-%m-%dT%H:%M:%S"
        )
    for col in ["open", "high", "low", "close", "volume"]:
        if col in canonical:
            canonical[col] = pd.to_numeric(canonical[col], errors="coerce").round(10)

    payload = canonical.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def environment_snapshot() -> dict[str, str]:
    """Capture runtime environment metadata relevant to reproducibility."""
    return {
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "implementation": platform.python_implementation(),
    }


def write_manifest(manifest: RunManifest, output_file: Path) -> None:
    """Write run manifest to disk as UTF-8 JSON."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
