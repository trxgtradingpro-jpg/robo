"""Tick-by-tick trade loader for Profit exports."""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path
import re
from typing import Iterable
import unicodedata

import pandas as pd


_DATE_RE = re.compile(r".*_(\d{2})-(\d{2})-(\d{4})\.csv$", re.IGNORECASE)


def load_ticks_between(
    tick_root: Path,
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Load and concatenate ticks for the requested date range."""
    root = tick_root.resolve()
    symbol_key = str(symbol).strip().upper()
    if not root.exists():
        return pd.DataFrame(columns=["datetime", "price", "volume"])

    day_map = _discover_tick_files(str(root), symbol_key)
    if not day_map:
        return pd.DataFrame(columns=["datetime", "price", "volume"])

    days = pd.date_range(start.normalize(), end.normalize(), freq="D")
    parts: list[pd.DataFrame] = []
    for day_ts in days:
        file_path = day_map.get(day_ts.date())
        if not file_path:
            continue
        day_df = _load_tick_file_cached(file_path)
        if not day_df.empty:
            parts.append(day_df)

    if not parts:
        return pd.DataFrame(columns=["datetime", "price", "volume"])

    out = pd.concat(parts, ignore_index=True)
    out = out[(out["datetime"] >= start) & (out["datetime"] <= end)].copy()
    if out.empty:
        return pd.DataFrame(columns=["datetime", "price", "volume"])
    return out.sort_values("datetime").reset_index(drop=True)


@lru_cache(maxsize=64)
def _discover_tick_files(root_str: str, symbol: str) -> dict[date, str]:
    root = Path(root_str)
    files = sorted(root.glob("*.csv"))
    out: dict[date, str] = {}
    for file_path in files:
        name_upper = file_path.name.upper()
        if symbol and not name_upper.startswith(symbol):
            continue
        match = _DATE_RE.match(file_path.name)
        if not match:
            continue
        dd, mm, yyyy = match.groups()
        try:
            d = date(int(yyyy), int(mm), int(dd))
        except ValueError:
            continue
        out[d] = str(file_path.resolve())
    return out


@lru_cache(maxsize=512)
def _load_tick_file_cached(file_path_str: str) -> pd.DataFrame:
    file_path = Path(file_path_str)
    raw = _read_csv_with_fallback(file_path)
    if raw.empty:
        return pd.DataFrame(columns=["datetime", "price", "volume"])

    normalized_columns = {_normalize_text(col): col for col in raw.columns}
    date_col = _find_first(normalized_columns, ("data", "date"))
    time_col = _find_first(normalized_columns, ("hora", "time"))
    price_col = _find_first(normalized_columns, ("preco", "price", "ultimo", "close"))
    volume_col = _find_first(normalized_columns, ("quantidade", "qtd", "volume"))
    if not date_col or not time_col or not price_col:
        return pd.DataFrame(columns=["datetime", "price", "volume"])

    dt_series = raw[date_col].astype(str).str.strip() + " " + raw[time_col].astype(str).str.strip()
    out = pd.DataFrame(
        {
            "datetime": pd.to_datetime(dt_series, errors="coerce", dayfirst=True),
            "price": _parse_numeric(raw[price_col]),
        }
    )
    if volume_col:
        out["volume"] = _parse_numeric(raw[volume_col]).fillna(0.0)
    else:
        out["volume"] = 0.0
    out = out.dropna(subset=["datetime", "price"]).copy()
    out = out[out["price"] > 0].copy()
    if out.empty:
        return pd.DataFrame(columns=["datetime", "price", "volume"])
    return out.sort_values("datetime").reset_index(drop=True)


def _read_csv_with_fallback(file_path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "cp1252", "latin1"):
        try:
            return pd.read_csv(
                file_path,
                sep=";",
                encoding=enc,
                dtype=str,
            )
        except UnicodeDecodeError:
            continue
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def _parse_numeric(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip()
    text = text.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "-": pd.NA})
    sample = text.dropna().head(200)
    has_comma = sample.str.contains(",", regex=False).any()
    has_dot = sample.str.contains(".", regex=False).any()
    normalized = text.copy()
    if has_comma and has_dot:
        normalized = normalized.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    elif has_comma:
        normalized = normalized.str.replace(",", ".", regex=False)
    return pd.to_numeric(normalized, errors="coerce")


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
