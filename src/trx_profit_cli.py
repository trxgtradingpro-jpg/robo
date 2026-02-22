"""CLI runner for TRX HTSL parity against Profit/NTSL settings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .backtest_engine import BacktestConfig
from .data_loader import LoaderConfig, load_timeframe_data, normalize_timeframe_label
from .profit_compare import compare_tradebooks, load_profit_operations, load_robot_trades, save_compare_outputs
from .trx_profit_parity import (
    run_trx_profit_parity,
    save_trx_parity_outputs,
    trx_profit_reference_params,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Executa TRX HTSL com parametros NTSL de referencia (paridade Profit).")
    parser.add_argument("--symbol", default="WINFUT")
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--outputs", default="outputs_parity")

    parser.add_argument("--initial-capital", type=float, default=193_620.0)
    parser.add_argument("--contracts", type=int, default=5)
    parser.add_argument("--point-value", type=float, default=0.2)
    parser.add_argument("--slippage", type=float, default=0.0)
    parser.add_argument("--entry-model", choices=["next_open", "close_slippage"], default="next_open")
    parser.add_argument("--session-start", default="09:00")
    parser.add_argument("--session-end", default="17:40")
    parser.add_argument("--close-on-session-end", dest="close_on_session_end", action="store_true", default=True)
    parser.add_argument("--no-close-on-session-end", dest="close_on_session_end", action="store_false")
    parser.add_argument("--params-json", default=None, help="Override de parametros em JSON")
    parser.add_argument("--params-file", default=None, help="Arquivo JSON com override de parametros")

    parser.add_argument("--profit-ops", default=None, help="CSV de operacoes do Profit para comparar")
    parser.add_argument("--time-tolerance-sec", type=int, default=300)
    parser.add_argument("--pnl-tolerance", type=float, default=5.0)
    parser.add_argument("--price-tolerance", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = _parse_start_timestamp(args.start)
    end = _parse_end_timestamp(args.end)
    if end < start:
        raise ValueError(f"Range invalido: end {args.end} < start {args.start}")

    timeframe = normalize_timeframe_label(args.timeframe)
    loader_cfg = LoaderConfig(
        data_root=Path(args.data_root),
        symbol=args.symbol,
        start=start,
        end=end,
    )
    df = load_timeframe_data(loader_cfg, timeframe)
    data_start = pd.to_datetime(df["datetime"], errors="coerce").min()
    data_end = pd.to_datetime(df["datetime"], errors="coerce").max()
    print(
        "[TRX PARITY]",
        f"dados carregados: {data_start} -> {data_end}",
    )
    if data_end < end:
        print(
            "[WARN]",
            "Fim solicitado maior que ultimo candle disponivel.",
            f"solicitado={end} ultimo_disponivel={data_end}",
        )
    bt_cfg = BacktestConfig(
        initial_capital=float(args.initial_capital),
        contracts=int(max(1, args.contracts)),
        point_value=float(args.point_value),
        slippage_points=float(max(0.0, args.slippage)),
        entry_mode=str(args.entry_model),
        session_start=str(args.session_start).strip() or None,
        session_end=str(args.session_end).strip() or None,
        close_on_session_end=bool(args.close_on_session_end),
        max_positions=1,
    )
    params_override = _read_params_override(args.params_json, args.params_file)
    output = run_trx_profit_parity(df=df, base_config=bt_cfg, params_override=params_override)

    out_dir = Path(args.outputs) / args.symbol / timeframe / "trx_htsl_profit"
    files = save_trx_parity_outputs(output=output, output_dir=out_dir, symbol=args.symbol, timeframe=timeframe)

    summary = output.summary_profit_style
    print("[TRX PARITY] concluido")
    print(
        "[TRX PARITY]",
        f"trades={int(summary['numero_operacoes'])}",
        f"net={float(summary['saldo_liquido_total']):.2f}",
        f"gross_profit={float(summary['lucro_bruto']):.2f}",
        f"gross_loss={float(summary['prejuizo_bruto']):.2f}",
        f"pf={float(summary['fator_lucro']):.2f}",
    )
    _print_profit_style_summary(summary)
    print("[INFO] summary:", files["summary"])
    print("[INFO] trades:", files["trades"])
    print("[INFO] equity:", files["equity"])
    print("[INFO] operations:", files["operations"])
    print("[INFO] equity_png:", files["equity_png"])

    if args.profit_ops:
        compare_dir = out_dir / "compare_profit"
        compare_result = compare_tradebooks(
            robot_trades=load_robot_trades(Path(files["trades"])),
            profit_trades=load_profit_operations(Path(args.profit_ops)),
            time_tolerance_seconds=int(max(0, args.time_tolerance_sec)),
            pnl_tolerance=float(max(0.0, args.pnl_tolerance)),
            price_tolerance=float(max(0.0, args.price_tolerance)),
        )
        compare_files = save_compare_outputs(compare_result, compare_dir)
        s = compare_result.summary
        print(
            "[COMPARE]",
            f"matched={s.matched_trade_count}",
            f"only_robot={s.only_robot_count}",
            f"only_profit={s.only_profit_count}",
            f"net_diff={s.net_diff:.2f}",
            f"strict_parity={s.strict_parity}",
        )
        print("[INFO] compare_summary:", compare_files["summary"])


def _read_params_override(raw_json: str | None, params_file: str | None) -> dict[str, float | int | bool]:
    if raw_json:
        payload = json.loads(raw_json)
        if not isinstance(payload, dict):
            raise ValueError("--params-json deve ser objeto JSON.")
        return _coerce_params(payload)
    if params_file:
        payload = json.loads(Path(params_file).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("--params-file deve apontar para JSON objeto.")
        return _coerce_params(payload)
    return {}


def _coerce_params(params: dict[str, object]) -> dict[str, float | int | bool]:
    out = trx_profit_reference_params()
    for key, value in params.items():
        if isinstance(value, bool):
            out[str(key)] = value
        elif isinstance(value, int):
            out[str(key)] = int(value)
        elif isinstance(value, float):
            out[str(key)] = float(value)
        else:
            text = str(value).strip()
            low = text.lower()
            if low in {"true", "false"}:
                out[str(key)] = (low == "true")
            else:
                try:
                    if "." in text:
                        out[str(key)] = float(text)
                    else:
                        out[str(key)] = int(text)
                except ValueError:
                    out[str(key)] = text
    return out


def _parse_start_timestamp(raw: str):
    return pd.Timestamp(raw)


def _parse_end_timestamp(raw: str):
    ts = pd.Timestamp(raw)
    if len(raw.strip()) <= 10:
        return ts + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return ts


def _print_profit_style_summary(summary: dict[str, float | int | str]) -> None:
    print("")
    print("=== RESUMO ESTILO PROFIT ===")
    print(
        f"Saldo Liquido Total: {float(summary['saldo_liquido_total']):.2f} | "
        f"Lucro Bruto: {float(summary['lucro_bruto']):.2f} | "
        f"Prejuizo Bruto: {float(summary['prejuizo_bruto']):.2f} | "
        f"Fator de Lucro: {float(summary['fator_lucro']):.2f}"
    )
    print(
        f"Numero de Operacoes: {int(summary['numero_operacoes'])} | "
        f"Vencedoras: {int(summary['operacoes_vencedoras'])} | "
        f"Perdedoras: {int(summary['operacoes_perdedoras'])} | "
        f"Zeradas: {int(summary['operacoes_zeradas'])}"
    )
    print(
        f"Media por Operacao: {float(summary['media_lucro_prejuizo']):.2f} | "
        f"Media Vencedoras: {float(summary['media_operacoes_vencedoras']):.2f} | "
        f"Media Perdedoras: {float(summary['media_operacoes_perdedoras']):.2f}"
    )
    print(
        f"Maior Vencedora: {float(summary['maior_operacao_vencedora']):.2f} | "
        f"Maior Perdedora: {float(summary['maior_operacao_perdedora']):.2f}"
    )
    print(
        f"Maior Seq Vencedora: {int(summary['maior_sequencia_vencedora'])} | "
        f"Maior Seq Perdedora: {int(summary['maior_sequencia_perdedora'])}"
    )
    print(
        f"Tempo Medio Vencedoras: {summary['media_tempo_operacao_vencedora']} | "
        f"Tempo Medio Perdedoras: {summary['media_tempo_operacao_perdedora']} | "
        f"Tempo Medio Total: {summary['tempo_medio_operacao_total']}"
    )
    print(
        f"Max Contratos: {int(summary['maximo_contratos'])} | "
        f"Retorno Capital Inicial: {float(summary['retorno_capital_inicial_pct']):.2f}% | "
        f"Patrimonio Maximo: {float(summary['patrimonio_maximo']):.2f}"
    )
    print(
        f"DD Topo-Fundo: {float(summary['declinio_maximo_topo_fundo_valor']):.2f} "
        f"({float(summary['declinio_maximo_topo_fundo_pct']):.2f}%) | "
        f"DD Trade-a-Trade: {float(summary['declinio_maximo_trade_a_trade_valor']):.2f} "
        f"({float(summary['declinio_maximo_trade_a_trade_pct']):.2f}%)"
    )
    print("============================")


if __name__ == "__main__":
    main()
