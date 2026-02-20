"""Data loading and normalization for Profit CSV exports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable
import unicodedata

import numpy as np
import pandas as pd


TIMEFRAME_ALIASES: dict[str, str] = {
    "1m": "1m",
    "5m": "5m",
    "10m": "10m",
    "15m": "15m",
    "30m": "30m",
    "60m": "60m",
    "d": "daily",
    "1d": "daily",
    "daily": "daily",
    "diario": "daily",
    "w": "weekly",
    "1w": "weekly",
    "weekly": "weekly",
    "semanal": "weekly",
}


class DataValidationError(ValueError):
    """Raised when CSV does not contain valid OHLC data."""


@dataclass(slots=True)
class LoaderConfig:
    """Configuration for loading timeframe data."""

    data_root: Path = Path("data")
    symbol: str = "WINFUT"
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None


@dataclass(slots=True)
class DataQualityReport:
    """Quality snapshot for one symbol/timeframe dataframe."""

    symbol: str
    timeframe: str
    rows: int
    days: int
    start: str
    end: str
    duplicate_timestamps: int
    non_monotonic_timestamps: int
    ohlc_inconsistent_rows: int
    missing_intraday_bars_estimate: int
    median_bars_per_day: float
    p05_bars_per_day: float
    p95_bars_per_day: float

    def to_dict(self) -> dict[str, str | int | float]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "rows": int(self.rows),
            "days": int(self.days),
            "start": self.start,
            "end": self.end,
            "duplicate_timestamps": int(self.duplicate_timestamps),
            "non_monotonic_timestamps": int(self.non_monotonic_timestamps),
            "ohlc_inconsistent_rows": int(self.ohlc_inconsistent_rows),
            "missing_intraday_bars_estimate": int(self.missing_intraday_bars_estimate),
            "median_bars_per_day": float(self.median_bars_per_day),
            "p05_bars_per_day": float(self.p05_bars_per_day),
            "p95_bars_per_day": float(self.p95_bars_per_day),
        }


def normalize_timeframe_label(label: str) -> str:
    """Normalize user timeframe labels to expected folder names."""
    key = _normalize_text(label)
    if key not in TIMEFRAME_ALIASES:
        valid = ", ".join(sorted(set(TIMEFRAME_ALIASES.values())))
        raise ValueError(f"Timeframe '{label}' nao suportado. Use: {valid}.")
    return TIMEFRAME_ALIASES[key]


def load_timeframe_data(config: LoaderConfig, timeframe: str) -> pd.DataFrame:
    """Load and normalize all CSV files for a symbol/timeframe folder."""
    folder_name = normalize_timeframe_label(timeframe)
    folder = config.data_root / config.symbol / folder_name
    if not folder.exists():
        raise FileNotFoundError(f"Pasta nao encontrada: {folder}")

    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Nenhum CSV encontrado em: {folder}")

    frames: list[pd.DataFrame] = []
    for file_path in csv_files:
        frame = load_profit_csv(file_path)
        if not frame.empty:
            frames.append(frame)

    if not frames:
        raise DataValidationError(f"Nenhuma linha valida encontrada em: {folder}")

    df = pd.concat(frames, ignore_index=True)
    df = _clean_ohlc_dataframe(df)

    if config.start is not None:
        df = df[df["datetime"] >= config.start]
    if config.end is not None:
        df = df[df["datetime"] <= config.end]

    if df.empty:
        raise DataValidationError(
            f"Sem dados no range solicitado para {config.symbol} {timeframe}."
        )

    return df.reset_index(drop=True)


def build_data_quality_report(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
) -> DataQualityReport:
    """Build a deterministic quality report from cleaned OHLC data."""
    if df.empty:
        return DataQualityReport(
            symbol=symbol,
            timeframe=timeframe,
            rows=0,
            days=0,
            start="",
            end="",
            duplicate_timestamps=0,
            non_monotonic_timestamps=0,
            ohlc_inconsistent_rows=0,
            missing_intraday_bars_estimate=0,
            median_bars_per_day=0.0,
            p05_bars_per_day=0.0,
            p95_bars_per_day=0.0,
        )

    ordered = df.sort_values("datetime").reset_index(drop=True).copy()
    dt = pd.to_datetime(ordered["datetime"])
    start = dt.iloc[0].isoformat()
    end = dt.iloc[-1].isoformat()
    duplicate_timestamps = int(dt.duplicated().sum())
    non_monotonic_timestamps = int((dt.diff().dt.total_seconds().fillna(1) <= 0).sum())

    ohlc_inconsistent = int(
        (
            (ordered["high"] < ordered[["open", "close", "low"]].max(axis=1))
            | (ordered["low"] > ordered[["open", "close", "high"]].min(axis=1))
        ).sum()
    )

    bars_per_day = dt.dt.date.value_counts().sort_index()
    median_bars = float(bars_per_day.median()) if not bars_per_day.empty else 0.0
    p05_bars = float(bars_per_day.quantile(0.05)) if not bars_per_day.empty else 0.0
    p95_bars = float(bars_per_day.quantile(0.95)) if not bars_per_day.empty else 0.0

    tf_minutes = _timeframe_minutes(timeframe)
    missing_bars_estimate = _estimate_intraday_missing_bars(dt, tf_minutes)

    return DataQualityReport(
        symbol=symbol,
        timeframe=timeframe,
        rows=int(len(ordered)),
        days=int(len(bars_per_day)),
        start=start,
        end=end,
        duplicate_timestamps=duplicate_timestamps,
        non_monotonic_timestamps=non_monotonic_timestamps,
        ohlc_inconsistent_rows=ohlc_inconsistent,
        missing_intraday_bars_estimate=missing_bars_estimate,
        median_bars_per_day=median_bars,
        p05_bars_per_day=p05_bars,
        p95_bars_per_day=p95_bars,
    )


def load_profit_csv(file_path: Path) -> pd.DataFrame:
    """Load one CSV from Profit and normalize to a canonical schema."""
    raw = _read_csv_with_fallback(file_path)
    if raw.empty:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])

    normalized_columns = {_normalize_text(col): col for col in raw.columns}

    dt_col = _find_first(normalized_columns, ("datahora", "datetime", "timestamp"))
    date_col = _find_first(normalized_columns, ("data", "date"))
    time_col = _find_first(normalized_columns, ("hora", "time"))

    open_col = _find_first(normalized_columns, ("abertura", "open"))
    high_col = _find_first(normalized_columns, ("maxima", "maximo", "high", "max"))
    low_col = _find_first(normalized_columns, ("minima", "minimo", "low", "min"))
    close_col = _find_first(normalized_columns, ("fechamento", "close", "ultimo", "preco"))
    volume_col = _find_first(normalized_columns, ("volume", "quantidade", "qtd"))

    if not open_col or not high_col or not low_col or not close_col:
        raise DataValidationError(
            f"Colunas OHLC nao encontradas em {file_path.name}. Colunas: {list(raw.columns)}"
        )

    if dt_col:
        dt_series = raw[dt_col].astype(str).str.strip()
    elif date_col:
        time_series = raw[time_col].astype(str).str.strip() if time_col else "00:00:00"
        dt_series = raw[date_col].astype(str).str.strip() + " " + time_series
    else:
        raise DataValidationError(
            f"Coluna de data/hora nao encontrada em {file_path.name}. Colunas: {list(raw.columns)}"
        )

    out = pd.DataFrame(
        {
            "datetime": _parse_datetime(dt_series),
            "open": _parse_numeric(raw[open_col]),
            "high": _parse_numeric(raw[high_col]),
            "low": _parse_numeric(raw[low_col]),
            "close": _parse_numeric(raw[close_col]),
        }
    )

    if volume_col:
        out["volume"] = _parse_numeric(raw[volume_col]).fillna(0.0)
    else:
        out["volume"] = 0.0

    return out


def _read_csv_with_fallback(file_path: Path) -> pd.DataFrame:
    encodings = ("utf-8-sig", "cp1252", "latin1")
    last_error: Exception | None = None
    for enc in encodings:
        try:
            return pd.read_csv(
                file_path,
                sep=None,
                engine="python",
                encoding=enc,
                dtype=str,
            )
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise DataValidationError(f"Falha ao ler arquivo CSV: {file_path}")


def _parse_datetime(series: pd.Series) -> pd.Series:
    """Parse datetime in common Profit formats."""
    clean = series.astype(str).str.strip()
    parsed = pd.to_datetime(clean, errors="coerce", dayfirst=True)
    if parsed.isna().all():
        parsed = pd.to_datetime(clean, errors="coerce", dayfirst=False)
    return parsed


def _parse_numeric(series: pd.Series) -> pd.Series:
    """Parse numeric fields supporting PT-BR and EN formats."""
    text = series.astype(str).str.strip()
    text = text.replace({"": np.nan, "nan": np.nan, "None": np.nan, "-": np.nan})
    sample = text.dropna().head(200)

    has_comma = sample.str.contains(",", regex=False).any()
    has_dot = sample.str.contains(".", regex=False).any()

    normalized = text.copy()
    if has_comma and has_dot:
        normalized = normalized.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    elif has_comma:
        normalized = normalized.str.replace(",", ".", regex=False)
    elif has_dot:
        thousands_like = sample.str.match(r"^-?\d{1,3}(\.\d{3})+$").mean() if not sample.empty else 0.0
        if thousands_like > 0.6:
            normalized = normalized.str.replace(".", "", regex=False)

    return pd.to_numeric(normalized, errors="coerce")


def _clean_ohlc_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Remove invalid rows, duplicates and enforce time ordering."""
    clean = df.copy()
    clean = clean.dropna(subset=["datetime", "open", "high", "low", "close"])
    clean = clean[np.isfinite(clean["open"]) & np.isfinite(clean["high"]) & np.isfinite(clean["low"]) & np.isfinite(clean["close"])]
    clean = clean[clean["high"] >= clean["low"]]
    clean = clean[(clean["open"] > 0) & (clean["high"] > 0) & (clean["low"] > 0) & (clean["close"] > 0)]

    # Keep the latest duplicate row by timestamp.
    clean = clean.sort_values("datetime").drop_duplicates(subset=["datetime"], keep="last")
    clean = clean.reset_index(drop=True)
    clean["volume"] = clean.get("volume", pd.Series(index=clean.index, dtype=float)).fillna(0.0)
    return clean[["datetime", "open", "high", "low", "close", "volume"]]


def _find_first(columns_map: dict[str, str], options: Iterable[str]) -> str | None:
    for candidate in options:
        key = _normalize_text(candidate)
        if key in columns_map:
            return columns_map[key]
    return None


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return "".join(ch for ch in ascii_text.lower() if ch.isalnum())


def _timeframe_minutes(timeframe: str) -> int | None:
    if timeframe in {"daily", "weekly"}:
        return None
    match = re.match(r"^(\d+)m$", timeframe.strip().lower())
    if not match:
        return None
    value = int(match.group(1))
    return value if value > 0 else None


def _estimate_intraday_missing_bars(dt: pd.Series, timeframe_minutes: int | None) -> int:
    if timeframe_minutes is None or dt.empty:
        return 0
    expected = timeframe_minutes * 60
    missing = 0
    dt_series = pd.to_datetime(dt).sort_values()
    grouped = dt_series.groupby(dt_series.dt.date)
    for _, group in grouped:
        diffs = group.diff().dt.total_seconds().dropna()
        if diffs.empty:
            continue
        expected_bars = ((diffs / expected).round().astype(int) - 1).clip(lower=0)
        missing += int(expected_bars.sum())
    return missing
