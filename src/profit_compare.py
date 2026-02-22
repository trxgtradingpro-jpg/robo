"""Compare robot tradebook CSV against Profit operations CSV."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import unicodedata

import numpy as np
import pandas as pd


@dataclass(slots=True)
class CompareSummary:
    robot_trade_count: int
    profit_trade_count: int
    matched_trade_count: int
    only_robot_count: int
    only_profit_count: int
    robot_net: float
    profit_net: float
    net_diff: float
    match_rate_robot_pct: float
    match_rate_profit_pct: float
    equivalent_matches_pct: float
    strict_parity: bool

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


@dataclass(slots=True)
class CompareResult:
    summary: CompareSummary
    matched: pd.DataFrame
    only_robot: pd.DataFrame
    only_profit: pd.DataFrame


def load_robot_trades(file_path: Path) -> pd.DataFrame:
    raw = _read_csv_with_fallback(file_path)
    if raw.empty:
        return pd.DataFrame(columns=_canonical_columns())

    cols = {_normalize_text(c): c for c in raw.columns}
    entry_col = _find_first(cols, ("entrytime", "abertura", "entrada", "open_time"))
    exit_col = _find_first(cols, ("exittime", "fechamento", "saida", "close_time"))
    direction_col = _find_first(cols, ("direction", "lado", "side", "direcao"))
    qty_col = _find_first(cols, ("qtd", "quantidade", "qty", "contracts", "contratos"))
    entry_price_col = _find_first(cols, ("entryprice", "precocompra", "precoentrada"))
    exit_price_col = _find_first(cols, ("exitprice", "precovenda", "precosaida"))
    pnl_col = _find_first(cols, ("pnlnet", "resultado", "pnl", "netprofit", "lucroliquido"))

    if not entry_col or not exit_col:
        raise ValueError(f"CSV do robo sem colunas de entrada/saida validas: {list(raw.columns)}")

    out = pd.DataFrame(
        {
            "entry_time": _parse_datetime(raw[entry_col]),
            "exit_time": _parse_datetime(raw[exit_col]),
            "direction": _parse_direction(raw[direction_col]) if direction_col else "unknown",
            "qty": _parse_qty(raw[qty_col]) if qty_col else 1.0,
            "entry_price": _parse_numeric(raw[entry_price_col]) if entry_price_col else np.nan,
            "exit_price": _parse_numeric(raw[exit_price_col]) if exit_price_col else np.nan,
            "pnl_net": _parse_numeric(raw[pnl_col]) if pnl_col else np.nan,
        }
    )
    out = out.dropna(subset=["entry_time", "exit_time"]).sort_values(["exit_time", "entry_time"]).reset_index(drop=True)
    out["trade_id"] = out.index.astype(int)
    return out[_canonical_columns()]


def load_profit_operations(file_path: Path) -> pd.DataFrame:
    raw = _read_csv_with_fallback(file_path)
    if raw.empty:
        return pd.DataFrame(columns=_canonical_columns())

    cols = {_normalize_text(c): c for c in raw.columns}
    entry_col = _find_first(cols, ("abertura", "entrada", "entrytime", "datahoraabertura"))
    exit_col = _find_first(cols, ("fechamento", "saida", "exittime", "datahorafechamento"))
    direction_col = _find_first(cols, ("lado", "direcao", "side"))
    qty_col = _find_first(cols, ("qtd", "quantidade", "contratos", "qty"))
    entry_price_col = _find_first(cols, ("precocompra", "precoentrada", "entryprice", "buyprice"))
    exit_price_col = _find_first(cols, ("precovenda", "precosaida", "exitprice", "sellprice"))
    pnl_col = _find_first(cols, ("resultado", "pnl", "lucro", "lucroliquido", "resliquido"))

    if not entry_col or not exit_col:
        raise ValueError(f"CSV do Profit sem colunas de abertura/fechamento validas: {list(raw.columns)}")

    qty = _parse_qty(raw[qty_col]) if qty_col else pd.Series(1.0, index=raw.index, dtype=float)
    side = _parse_direction(raw[direction_col]) if direction_col else pd.Series("unknown", index=raw.index, dtype="object")
    direction = side.where(side != "unknown", np.where(qty < 0, "short", "long"))

    out = pd.DataFrame(
        {
            "entry_time": _parse_datetime(raw[entry_col]),
            "exit_time": _parse_datetime(raw[exit_col]),
            "direction": direction,
            "qty": qty.abs().replace(0.0, 1.0),
            "entry_price": _parse_numeric(raw[entry_price_col]) if entry_price_col else np.nan,
            "exit_price": _parse_numeric(raw[exit_price_col]) if exit_price_col else np.nan,
            "pnl_net": _parse_numeric(raw[pnl_col]) if pnl_col else np.nan,
        }
    )
    out = out.dropna(subset=["entry_time", "exit_time"]).sort_values(["exit_time", "entry_time"]).reset_index(drop=True)
    out["trade_id"] = out.index.astype(int)
    return out[_canonical_columns()]


def compare_tradebooks(
    robot_trades: pd.DataFrame,
    profit_trades: pd.DataFrame,
    time_tolerance_seconds: int = 300,
    pnl_tolerance: float = 5.0,
    price_tolerance: float = 5.0,
) -> CompareResult:
    robot = robot_trades.copy().reset_index(drop=True)
    profit = profit_trades.copy().reset_index(drop=True)
    if "trade_id" not in robot.columns:
        robot["trade_id"] = robot.index.astype(int)
    if "trade_id" not in profit.columns:
        profit["trade_id"] = profit.index.astype(int)

    profit_index = set(profit.index.tolist())
    matched_rows: list[dict[str, object]] = []
    used_profit: set[int] = set()

    for ridx, r in robot.iterrows():
        best_idx: int | None = None
        best_score = float("inf")
        best_entry_diff = np.nan
        best_exit_diff = np.nan

        for pidx in profit_index:
            if pidx in used_profit:
                continue
            p = profit.iloc[pidx]

            if not _direction_compatible(str(r.get("direction", "unknown")), str(p.get("direction", "unknown"))):
                continue

            entry_diff = _time_diff_seconds(r.get("entry_time"), p.get("entry_time"))
            exit_diff = _time_diff_seconds(r.get("exit_time"), p.get("exit_time"))
            nearest_diff = min(entry_diff, exit_diff)
            if nearest_diff > float(max(0, time_tolerance_seconds)):
                continue

            pnl_diff = abs(float(r.get("pnl_net", 0.0)) - float(p.get("pnl_net", 0.0)))
            score = nearest_diff * 1000.0 + pnl_diff
            if score < best_score:
                best_score = score
                best_idx = int(pidx)
                best_entry_diff = entry_diff
                best_exit_diff = exit_diff

        if best_idx is None:
            continue

        used_profit.add(best_idx)
        p = profit.iloc[best_idx]

        entry_price_diff = _abs_diff(r.get("entry_price"), p.get("entry_price"))
        exit_price_diff = _abs_diff(r.get("exit_price"), p.get("exit_price"))
        pnl_diff = _abs_diff(r.get("pnl_net"), p.get("pnl_net"))

        within_time = bool(
            min(float(best_entry_diff), float(best_exit_diff)) <= float(max(0, time_tolerance_seconds))
        )
        within_price = bool(entry_price_diff <= price_tolerance and exit_price_diff <= price_tolerance)
        within_pnl = bool(pnl_diff <= pnl_tolerance)

        matched_rows.append(
            {
                "robot_trade_id": int(r["trade_id"]),
                "profit_trade_id": int(p["trade_id"]),
                "robot_entry_time": r.get("entry_time"),
                "profit_entry_time": p.get("entry_time"),
                "entry_time_diff_sec": float(best_entry_diff),
                "robot_exit_time": r.get("exit_time"),
                "profit_exit_time": p.get("exit_time"),
                "exit_time_diff_sec": float(best_exit_diff),
                "robot_direction": str(r.get("direction", "unknown")),
                "profit_direction": str(p.get("direction", "unknown")),
                "robot_qty": float(r.get("qty", 1.0)),
                "profit_qty": float(p.get("qty", 1.0)),
                "robot_entry_price": float(r.get("entry_price", np.nan)),
                "profit_entry_price": float(p.get("entry_price", np.nan)),
                "entry_price_diff": float(entry_price_diff),
                "robot_exit_price": float(r.get("exit_price", np.nan)),
                "profit_exit_price": float(p.get("exit_price", np.nan)),
                "exit_price_diff": float(exit_price_diff),
                "robot_pnl_net": float(r.get("pnl_net", np.nan)),
                "profit_pnl_net": float(p.get("pnl_net", np.nan)),
                "pnl_diff": float(pnl_diff),
                "within_time_tol": within_time,
                "within_price_tol": within_price,
                "within_pnl_tol": within_pnl,
                "equivalent": bool(within_time and within_price and within_pnl),
            }
        )

    matched = pd.DataFrame(matched_rows)
    only_robot = robot.loc[~robot.index.isin(matched["robot_trade_id"] if not matched.empty else [])].reset_index(drop=True)
    only_profit = profit.loc[~profit.index.isin(matched["profit_trade_id"] if not matched.empty else [])].reset_index(drop=True)

    robot_count = int(len(robot))
    profit_count = int(len(profit))
    matched_count = int(len(matched))
    equivalent_ratio = float(matched["equivalent"].mean() * 100.0) if not matched.empty else 0.0
    robot_net = float(robot["pnl_net"].fillna(0.0).sum()) if not robot.empty else 0.0
    profit_net = float(profit["pnl_net"].fillna(0.0).sum()) if not profit.empty else 0.0
    net_diff = float(robot_net - profit_net)

    summary = CompareSummary(
        robot_trade_count=robot_count,
        profit_trade_count=profit_count,
        matched_trade_count=matched_count,
        only_robot_count=int(len(only_robot)),
        only_profit_count=int(len(only_profit)),
        robot_net=robot_net,
        profit_net=profit_net,
        net_diff=net_diff,
        match_rate_robot_pct=(100.0 * matched_count / robot_count) if robot_count else 0.0,
        match_rate_profit_pct=(100.0 * matched_count / profit_count) if profit_count else 0.0,
        equivalent_matches_pct=equivalent_ratio,
        strict_parity=bool(
            robot_count == profit_count
            and matched_count == robot_count
            and int(len(only_robot)) == 0
            and int(len(only_profit)) == 0
            and abs(net_diff) <= float(pnl_tolerance)
            and (not matched.empty and bool(matched["equivalent"].all()))
        ),
    )
    return CompareResult(
        summary=summary,
        matched=matched,
        only_robot=only_robot,
        only_profit=only_profit,
    )


def save_compare_outputs(result: CompareResult, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_file = output_dir / "compare_summary.json"
    matched_file = output_dir / "compare_matched.csv"
    only_robot_file = output_dir / "compare_only_robot.csv"
    only_profit_file = output_dir / "compare_only_profit.csv"

    summary_file.write_text(json.dumps(result.summary.to_dict(), indent=2), encoding="utf-8")
    result.matched.to_csv(matched_file, index=False)
    result.only_robot.to_csv(only_robot_file, index=False)
    result.only_profit.to_csv(only_profit_file, index=False)
    return {
        "summary": str(summary_file),
        "matched": str(matched_file),
        "only_robot": str(only_robot_file),
        "only_profit": str(only_profit_file),
    }


def _canonical_columns() -> list[str]:
    return [
        "trade_id",
        "entry_time",
        "exit_time",
        "direction",
        "qty",
        "entry_price",
        "exit_price",
        "pnl_net",
    ]


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
    raise ValueError(f"Falha ao ler CSV: {file_path}")


def _find_first(columns_map: dict[str, str], options: tuple[str, ...]) -> str | None:
    for candidate in options:
        key = _normalize_text(candidate)
        if key in columns_map:
            return columns_map[key]
    return None


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value))
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return "".join(ch for ch in ascii_text.lower() if ch.isalnum())


def _parse_datetime(series: pd.Series) -> pd.Series:
    clean = series.astype(str).str.strip()
    parsed = pd.Series(pd.NaT, index=clean.index, dtype="datetime64[ns]")
    iso_like = clean.str.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}")
    br_like = clean.str.match(r"^\d{1,2}/\d{1,2}/\d{2,4}")

    if bool(iso_like.any()):
        parsed.loc[iso_like] = pd.to_datetime(clean[iso_like], errors="coerce", dayfirst=False)
    if bool(br_like.any()):
        parsed.loc[br_like] = pd.to_datetime(clean[br_like], errors="coerce", dayfirst=True)

    missing = parsed.isna()
    if bool(missing.any()):
        parsed.loc[missing] = pd.to_datetime(clean[missing], errors="coerce", dayfirst=False)
    missing = parsed.isna()
    if bool(missing.any()):
        parsed.loc[missing] = pd.to_datetime(clean[missing], errors="coerce", dayfirst=True)
    return parsed


def _parse_numeric(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip()
    text = text.replace({"": np.nan, "nan": np.nan, "None": np.nan, "-": np.nan})
    text = text.str.replace("R$", "", regex=False)
    text = text.str.replace("pts", "", case=False, regex=False)
    text = text.str.replace("%", "", regex=False)
    text = text.str.replace(" ", "", regex=False)
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


def _parse_direction(series: pd.Series) -> pd.Series:
    values = series.astype(str).str.strip().str.lower()
    mapped = []
    for v in values.tolist():
        key = _normalize_text(v)
        if key in {"c", "compra", "buy", "long", "l"}:
            mapped.append("long")
        elif key in {"v", "venda", "sell", "short", "s"}:
            mapped.append("short")
        else:
            mapped.append("unknown")
    return pd.Series(mapped, index=series.index, dtype="object")


def _parse_qty(series: pd.Series) -> pd.Series:
    qty = _parse_numeric(series)
    return qty.fillna(1.0)


def _time_diff_seconds(a: object, b: object) -> float:
    dt_a = pd.to_datetime(a, errors="coerce")
    dt_b = pd.to_datetime(b, errors="coerce")
    if pd.isna(dt_a) or pd.isna(dt_b):
        return float("inf")
    return float(abs((dt_a - dt_b).total_seconds()))


def _abs_diff(a: object, b: object) -> float:
    f1 = pd.to_numeric(pd.Series([a]), errors="coerce").iloc[0]
    f2 = pd.to_numeric(pd.Series([b]), errors="coerce").iloc[0]
    if pd.isna(f1) or pd.isna(f2):
        return float("inf")
    return float(abs(float(f1) - float(f2)))


def _direction_compatible(left: str, right: str) -> bool:
    if left == "unknown" or right == "unknown":
        return True
    return left == right
