"""Operational monitoring primitives for paper/live engines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from typing import Callable
from typing import Any

import pandas as pd


AlertLevel = str


@dataclass(slots=True)
class Alert:
    """Single operational alert."""

    timestamp: str
    level: AlertLevel
    code: str
    message: str
    context: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AlertBus:
    """In-memory alert collector with optional callback."""

    def __init__(self, callback: Callable[[Alert], None] | None = None) -> None:
        self._alerts: list[Alert] = []
        self._callback = callback

    def emit(self, level: AlertLevel, code: str, message: str, context: dict[str, Any] | None = None) -> None:
        alert = Alert(
            timestamp=datetime.now(timezone.utc).isoformat(),
            level=str(level).lower(),
            code=str(code),
            message=str(message),
            context=context or {},
        )
        self._alerts.append(alert)
        if self._callback is not None:
            self._callback(alert)

    def to_frame(self) -> pd.DataFrame:
        if not self._alerts:
            return pd.DataFrame(columns=["timestamp", "level", "code", "message", "context_json"])
        rows = []
        for alert in self._alerts:
            rows.append(
                {
                    "timestamp": alert.timestamp,
                    "level": alert.level,
                    "code": alert.code,
                    "message": alert.message,
                    "context_json": json.dumps(alert.context, ensure_ascii=False, sort_keys=True),
                }
            )
        return pd.DataFrame(rows)

    def has_high_severity(self) -> bool:
        return any(a.level == "high" for a in self._alerts)
