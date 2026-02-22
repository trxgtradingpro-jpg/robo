"""TRX HTSL parity runner to mirror NTSL/Profit settings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from .backtest_engine import BacktestConfig, BacktestResult, default_trade_columns, run_backtest
from .metrics import ScoreConfig, compute_metrics
from .optimizer import build_runtime_config_for_params, generate_signals_with_time_filter
from .strategies import STRATEGIES


def trx_profit_reference_params() -> dict[str, float | int | bool]:
    """Reference params from the NTSL code shared by user."""
    return {
        "ema_period": 20,
        "hilo_period": 10,
        "adx_period": 14,
        "adx_smoothing": 14,
        "adx_min": 20.0,
        "stop_points": 300.0,
        "take_points": 600.0,
        "break_even_trigger_points": 190.0,
        "break_even_lock_points": 10.0,
        "hour_start": 9,
        "hour_end": 12,
        "entry_start_time": "09:00",
        "entry_end_time": "12:00",
        "session_end": "17:40",
        "close_on_session_end": True,
        "max_consecutive_losses_per_day": 20,
        "license_start_date": 1250208,
        "license_end_date": 1260321,
    }


@dataclass(slots=True)
class TrxParityOutput:
    params: dict[str, float | int | bool]
    backtest_result: BacktestResult
    metrics: dict[str, float]
    summary_profit_style: dict[str, float | int | str]
    operations: pd.DataFrame


def run_trx_profit_parity(
    df: pd.DataFrame,
    base_config: BacktestConfig,
    params_override: dict[str, float | int | bool] | None = None,
) -> TrxParityOutput:
    if "trx_htsl" not in STRATEGIES:
        raise ValueError("Estrategia 'trx_htsl' nao encontrada no registro.")
    strategy = STRATEGIES["trx_htsl"]
    params = trx_profit_reference_params()
    if params_override:
        params.update(params_override)

    run_cfg = build_runtime_config_for_params(base_config=base_config, params=params)
    signals = generate_signals_with_time_filter(df=df, strategy=strategy, params=params)
    result = run_backtest(
        df=df,
        signals=signals,
        config=run_cfg,
        strategy_name=strategy.name,
        strategy_params=params,
    )
    metrics = compute_metrics(
        trades=result.trades,
        equity_curve=result.equity_curve,
        initial_capital=run_cfg.initial_capital,
        score_config=ScoreConfig(drawdown_weight=1.5, min_trade_count=0, penalty_per_missing_trade=0.0),
    )
    summary = build_profit_style_summary(
        trades=result.trades,
        equity=result.equity_curve,
        initial_capital=run_cfg.initial_capital,
        contracts=int(run_cfg.contracts),
    )
    operations = build_operations_table(
        trades=result.trades,
        initial_capital=run_cfg.initial_capital,
        contracts=int(run_cfg.contracts),
    )
    return TrxParityOutput(
        params=params,
        backtest_result=result,
        metrics=metrics,
        summary_profit_style=summary,
        operations=operations,
    )


def save_trx_parity_outputs(
    output: TrxParityOutput,
    output_dir: Path,
    symbol: str,
    timeframe: str,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    trades_file = output_dir / f"trades_{timeframe}_trx_htsl_profit.csv"
    equity_file = output_dir / f"equity_curve_{timeframe}_trx_htsl_profit.csv"
    summary_file = output_dir / f"summary_{timeframe}_trx_htsl_profit.json"
    operations_file = output_dir / f"operations_{timeframe}_trx_htsl_profit.csv"
    png_file = output_dir / f"equity_{timeframe}_trx_htsl_profit.png"

    trades = output.backtest_result.trades.copy()
    if trades.empty:
        pd.DataFrame(columns=default_trade_columns()).to_csv(trades_file, index=False)
    else:
        out = trades.copy()
        out["params"] = out["params"].apply(lambda p: json.dumps(p, sort_keys=True))
        out.to_csv(trades_file, index=False)
    output.backtest_result.equity_curve.to_csv(equity_file, index=False)
    output.operations.to_csv(operations_file, index=False)
    summary_payload = {
        "symbol": symbol,
        "timeframe": timeframe,
        "strategy": "trx_htsl",
        "params": output.params,
        "metrics": output.metrics,
        "summary_profit_style": output.summary_profit_style,
    }
    summary_file.write_text(json.dumps(summary_payload, indent=2, default=_json_default), encoding="utf-8")
    _plot_equity(output.backtest_result.equity_curve, f"{symbol} {timeframe} trx_htsl parity", png_file)
    return {
        "trades": str(trades_file),
        "equity": str(equity_file),
        "summary": str(summary_file),
        "operations": str(operations_file),
        "equity_png": str(png_file),
    }


def build_operations_table(
    trades: pd.DataFrame,
    initial_capital: float,
    contracts: int,
) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()

    out = trades.copy()
    out["entry_time"] = pd.to_datetime(out["entry_time"], errors="coerce")
    out["exit_time"] = pd.to_datetime(out["exit_time"], errors="coerce")
    out = out.sort_values("exit_time").reset_index(drop=True)
    out["duration"] = out["exit_time"] - out["entry_time"]
    out["Lado"] = out["direction"].map({"long": "C", "short": "V"}).fillna("-")
    out["Qtd"] = int(max(1, contracts))

    out["Preco Compra"] = out.apply(
        lambda r: r["entry_price"] if r["direction"] == "long" else r["exit_price"],
        axis=1,
    )
    out["Preco Venda"] = out.apply(
        lambda r: r["exit_price"] if r["direction"] == "long" else r["entry_price"],
        axis=1,
    )
    out["Resultado_Num"] = out["pnl_net"].astype(float)
    out["Resultado"] = out["Resultado_Num"].apply(format_brl)
    out["Resultado(pts)"] = out["pnl_points"].astype(float).map(lambda x: f"{x:.2f}")
    out["Resultado(%)"] = (
        100.0 * out["Resultado_Num"] / max(float(initial_capital), 1e-9)
    ).map(lambda x: f"{x:.2f}%")
    out["Total_Num"] = float(initial_capital) + out["Resultado_Num"].cumsum()
    out["Total"] = out["Total_Num"].apply(format_brl)
    out["Tempo Op"] = out["duration"].dt.total_seconds().apply(_format_timedelta_seconds)
    show = out[
        [
            "entry_time",
            "exit_time",
            "Tempo Op",
            "Qtd",
            "Lado",
            "Preco Compra",
            "Preco Venda",
            "Resultado",
            "Resultado(pts)",
            "Resultado(%)",
            "Total",
            "Resultado_Num",
            "Total_Num",
        ]
    ].copy()
    show.rename(columns={"entry_time": "Abertura", "exit_time": "Fechamento"}, inplace=True)
    show["Abertura"] = pd.to_datetime(show["Abertura"]).dt.strftime("%d/%m/%Y %H:%M:%S")
    show["Fechamento"] = pd.to_datetime(show["Fechamento"]).dt.strftime("%d/%m/%Y %H:%M:%S")
    show["Preco Compra"] = show["Preco Compra"].map(lambda x: f"{float(x):.3f}")
    show["Preco Venda"] = show["Preco Venda"].map(lambda x: f"{float(x):.3f}")
    return show.sort_values("Fechamento", ascending=False).reset_index(drop=True)


def build_profit_style_summary(
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    initial_capital: float,
    contracts: int,
) -> dict[str, float | int | str]:
    if trades.empty:
        return {
            "saldo_liquido_total": 0.0,
            "lucro_bruto": 0.0,
            "prejuizo_bruto": 0.0,
            "fator_lucro": 0.0,
            "numero_operacoes": 0,
            "operacoes_vencedoras": 0,
            "operacoes_perdedoras": 0,
            "operacoes_zeradas": 0,
            "media_lucro_prejuizo": 0.0,
            "media_operacoes_vencedoras": 0.0,
            "media_operacoes_perdedoras": 0.0,
            "maior_operacao_vencedora": 0.0,
            "maior_operacao_perdedora": 0.0,
            "maior_sequencia_vencedora": 0,
            "maior_sequencia_perdedora": 0,
            "media_tempo_operacao_vencedora": "0min",
            "media_tempo_operacao_perdedora": "0min",
            "tempo_medio_operacao_total": "0min",
            "maximo_contratos": int(max(1, contracts)),
            "retorno_capital_inicial_pct": 0.0,
            "patrimonio_maximo": float(initial_capital),
            "declinio_maximo_topo_fundo_valor": 0.0,
            "declinio_maximo_topo_fundo_pct": 0.0,
            "declinio_maximo_trade_a_trade_valor": 0.0,
            "declinio_maximo_trade_a_trade_pct": 0.0,
        }

    pnl = trades["pnl_net"].astype(float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    flats = pnl[pnl == 0]

    gross_profit = float(wins.sum()) if not wins.empty else 0.0
    gross_loss = float(losses.sum()) if not losses.empty else 0.0
    net = float(pnl.sum())
    trade_count = int(len(pnl))
    factor = abs(gross_profit / gross_loss) if gross_loss < 0 else (999.0 if gross_profit > 0 else 0.0)

    durations = _trade_durations(trades)
    avg_win_dur = durations[pnl > 0]
    avg_loss_dur = durations[pnl < 0]
    avg_total_dur = durations

    equity_series = equity["equity"].astype(float) if not equity.empty and "equity" in equity else pd.Series([], dtype=float)
    peak = equity_series.cummax() if not equity_series.empty else pd.Series([float(initial_capital)], dtype=float)
    drawdown = (peak - equity_series) if not equity_series.empty else pd.Series([0.0], dtype=float)
    max_dd = float(drawdown.max()) if not drawdown.empty else 0.0
    max_dd_pct = (100.0 * max_dd / max(float(initial_capital), 1e-9)) if initial_capital > 0 else 0.0

    trade_curve = float(initial_capital) + pnl.cumsum()
    trade_peak = trade_curve.cummax()
    trade_dd = trade_peak - trade_curve
    trade_dd_max = float(trade_dd.max()) if not trade_dd.empty else 0.0
    trade_dd_pct = (100.0 * trade_dd_max / max(float(initial_capital), 1e-9)) if initial_capital > 0 else 0.0

    return {
        "saldo_liquido_total": net,
        "lucro_bruto": gross_profit,
        "prejuizo_bruto": gross_loss,
        "fator_lucro": float(factor),
        "numero_operacoes": trade_count,
        "operacoes_vencedoras": int((pnl > 0).sum()),
        "operacoes_perdedoras": int((pnl < 0).sum()),
        "operacoes_zeradas": int(len(flats)),
        "media_lucro_prejuizo": float(pnl.mean()) if trade_count else 0.0,
        "media_operacoes_vencedoras": float(wins.mean()) if not wins.empty else 0.0,
        "media_operacoes_perdedoras": float(losses.mean()) if not losses.empty else 0.0,
        "maior_operacao_vencedora": float(wins.max()) if not wins.empty else 0.0,
        "maior_operacao_perdedora": float(losses.min()) if not losses.empty else 0.0,
        "maior_sequencia_vencedora": _max_streak(pnl, positive=True),
        "maior_sequencia_perdedora": _max_streak(pnl, positive=False),
        "media_tempo_operacao_vencedora": _format_timedelta_seconds(float(avg_win_dur.mean())) if not avg_win_dur.empty else "0min",
        "media_tempo_operacao_perdedora": _format_timedelta_seconds(float(avg_loss_dur.mean())) if not avg_loss_dur.empty else "0min",
        "tempo_medio_operacao_total": _format_timedelta_seconds(float(avg_total_dur.mean())) if not avg_total_dur.empty else "0min",
        "maximo_contratos": int(max(1, contracts)),
        "retorno_capital_inicial_pct": (100.0 * net / max(float(initial_capital), 1e-9)),
        "patrimonio_maximo": float(equity_series.max()) if not equity_series.empty else float(initial_capital),
        "declinio_maximo_topo_fundo_valor": max_dd,
        "declinio_maximo_topo_fundo_pct": max_dd_pct,
        "declinio_maximo_trade_a_trade_valor": trade_dd_max,
        "declinio_maximo_trade_a_trade_pct": trade_dd_pct,
    }


def format_brl(value: float) -> str:
    sign = "-" if value < 0 else ""
    raw = f"{abs(value):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{sign}R$ {raw}"


def _trade_durations(trades: pd.DataFrame) -> pd.Series:
    entry = pd.to_datetime(trades["entry_time"], errors="coerce")
    exit_ = pd.to_datetime(trades["exit_time"], errors="coerce")
    return (exit_ - entry).dt.total_seconds().fillna(0.0)


def _max_streak(pnl: pd.Series, positive: bool) -> int:
    best = 0
    cur = 0
    for value in pnl.astype(float):
        cond = value > 0 if positive else value < 0
        if cond:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def _format_timedelta_seconds(seconds: float) -> str:
    total = int(max(seconds, 0))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    if h > 0:
        return f"{h}h{m:02d}min{s:02d}s"
    if m > 0:
        return f"{m}min{s:02d}s"
    return f"{s}s"


def _plot_equity(equity_df: pd.DataFrame, title: str, output_file: Path) -> None:
    if equity_df.empty:
        return
    plot_df = equity_df.sort_values("datetime")
    x = pd.to_datetime(plot_df["datetime"])
    y = plot_df["equity"].astype(float)
    plt.figure(figsize=(11, 5))
    plt.plot(x, y, color="#25f08a", linewidth=1.6)
    plt.fill_between(x, y, y2=min(0.0, float(y.min())), color="#25f08a", alpha=0.16)
    plt.title(title)
    plt.xlabel("Data")
    plt.ylabel("Saldo Bruto (R$)")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_file, dpi=120)
    plt.close()


def _json_default(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)
