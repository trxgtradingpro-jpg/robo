"""Command-line interface to run walk-forward backtests."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import date
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from .backtest_engine import BacktestConfig, default_trade_columns
from .data_loader import (
    LoaderConfig,
    build_data_quality_report,
    load_timeframe_data,
    normalize_timeframe_label,
)
from .metrics import ScoreConfig
from .optimizer import OptimizerConfig
from .reports import (
    build_hourly_report,
    build_monthly_report,
    build_parameter_sensitivity_report,
    build_robustness_report,
)
from .reproducibility import (
    RunManifest,
    build_run_id,
    dataframe_sha256,
    environment_snapshot,
    write_manifest,
)
from .strategies import STRATEGIES, StrategySpec
from .walkforward import (
    WalkForwardCheckpoint,
    WalkForwardConfig,
    WalkForwardResult,
    run_walkforward,
)


class StopRequested(RuntimeError):
    """Raised when an external stop signal was requested."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest e walk-forward para WINFUT.")
    parser.add_argument("--symbol", default="WINFUT", help="Ativo (default: WINFUT)")
    parser.add_argument(
        "--timeframes",
        nargs="+",
        required=True,
        help="Lista de timeframes (ex.: 1m 5m 15m 30m daily weekly)",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=None,
        help=f"Estrategias a executar (default: todas). Disponiveis: {', '.join(sorted(STRATEGIES.keys()))}",
    )
    parser.add_argument("--start", required=True, help="Data inicial YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="Data final YYYY-MM-DD")
    parser.add_argument("--data-root", default="data", help="Raiz dos dados locais")
    parser.add_argument("--outputs", default="outputs_master", help="Pasta de resultados")
    parser.add_argument(
        "--execution-mode",
        choices=["ohlc", "tick"],
        default="ohlc",
        help="Modo de execucao do backtest (OHLC ou Tick a Tick).",
    )
    parser.add_argument(
        "--tick-data-root",
        default="assets/dados-historicos-winfut/tik-a-tik",
        help="Pasta com CSVs de trades tick-a-tick (formato Profit).",
    )
    parser.add_argument(
        "--stop-file",
        default=None,
        help="Arquivo sentinela para parada graciosa (se existir, interrompe no proximo checkpoint).",
    )

    parser.add_argument("--train-days", type=int, default=120)
    parser.add_argument("--test-days", type=int, default=30)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--evolution-cycles",
        type=int,
        default=1,
        help="Numero de ciclos evolutivos (cada ciclo reaproveita historico e testa novos parametros).",
    )
    parser.add_argument(
        "--seed-step",
        type=int,
        default=1000,
        help="Incremento de seed por ciclo evolutivo.",
    )
    parser.add_argument(
        "--param-bank-top",
        type=int,
        default=30,
        help="Quantidade de parametros historicos usados como sementes por estrategia.",
    )
    parser.add_argument(
        "--no-param-bank",
        dest="use_param_bank",
        action="store_false",
        help="Desliga reaproveitamento de parametros historicos.",
    )
    parser.set_defaults(use_param_bank=True)
    parser.add_argument(
        "--train-bar-step",
        type=int,
        default=1,
        help="Usa uma barra a cada N no treino da otimizacao (1 = sem downsample).",
    )
    parser.add_argument(
        "--max-stop-points",
        type=float,
        default=0.0,
        help="Limite maximo global de stop em pontos (0 desliga).",
    )
    parser.add_argument(
        "--max-daily-loss",
        type=float,
        default=0.0,
        help="Perda maxima por dia (R$) para aceitar parametros no treino (0 desliga).",
    )
    parser.add_argument(
        "--max-drawdown-pct-hard",
        type=float,
        default=0.0,
        help="Drawdown maximo em percentual para aceitar parametros no treino (0 desliga).",
    )
    parser.add_argument(
        "--optimize-hours",
        action="store_true",
        help="Ativa busca automatica de janela horaria (hour_start/hour_end).",
    )
    parser.add_argument("--hour-start-min", type=int, default=9, help="Hora minima para inicio da janela (0-23).")
    parser.add_argument("--hour-start-max", type=int, default=16, help="Hora maxima para inicio da janela (0-23).")
    parser.add_argument("--hour-end-min", type=int, default=10, help="Hora minima para fim da janela (0-23).")
    parser.add_argument("--hour-end-max", type=int, default=18, help="Hora maxima para fim da janela (0-23).")

    parser.add_argument("--initial-capital", type=float, default=100_000.0)
    parser.add_argument("--contracts", type=int, default=1)
    parser.add_argument("--point-value", type=float, default=0.2)
    parser.add_argument("--slippage", type=float, default=0.0, help="Slippage em pontos por ordem")
    parser.add_argument(
        "--slippage-model",
        choices=["fixed", "range_scaled"],
        default="fixed",
        help="Modelo de slippage (fixo ou escalado por range da barra).",
    )
    parser.add_argument(
        "--slippage-range-factor",
        type=float,
        default=0.0,
        help="Fator do range da barra adicionado ao slippage em pontos.",
    )
    parser.add_argument(
        "--open-auction-minutes",
        type=int,
        default=0,
        help="Janela inicial da sessao com multiplicador de slippage/custo.",
    )
    parser.add_argument(
        "--open-auction-slippage-multiplier",
        type=float,
        default=1.0,
        help="Multiplicador de slippage na janela inicial da sessao.",
    )
    parser.add_argument("--fixed-cost", type=float, default=0.0, help="Custo fixo por trade")
    parser.add_argument("--cost-per-contract", type=float, default=0.0, help="Custo por contrato")
    parser.add_argument(
        "--cost-model",
        choices=["fixed", "range_scaled"],
        default="fixed",
        help="Modelo de custos (fixo ou escalado por range da barra).",
    )
    parser.add_argument(
        "--cost-range-factor",
        type=float,
        default=0.0,
        help="Fator de custo adicional pelo range medio entrada/saida.",
    )
    parser.add_argument(
        "--open-auction-cost-multiplier",
        type=float,
        default=1.0,
        help="Multiplicador de custo na janela inicial da sessao.",
    )
    parser.add_argument("--max-positions", type=int, default=1)
    parser.add_argument(
        "--entry-model",
        choices=["next_open", "close_slippage"],
        default="next_open",
    )
    parser.add_argument("--session-start", default=None, help="Horario HH:MM")
    parser.add_argument("--session-end", default="17:00", help="Horario HH:MM")
    parser.add_argument(
        "--close-on-session-end",
        dest="close_on_session_end",
        action="store_true",
        default=True,
        help="Forca fechamento ao sair da sessao (padrao: ligado).",
    )
    parser.add_argument(
        "--no-close-on-session-end",
        dest="close_on_session_end",
        action="store_false",
        help="Desliga fechamento automatico no fim da sessao.",
    )

    parser.add_argument("--drawdown-weight", type=float, default=1.5)
    parser.add_argument("--min-trades", type=int, default=20)
    parser.add_argument("--penalty-per-missing-trade", type=float, default=150.0)
    parser.add_argument(
        "--verbose-progress",
        action="store_true",
        help="Imprime progresso detalhado por janela/amostras.",
    )
    parser.add_argument(
        "--progress-print-every",
        type=int,
        default=25,
        help="Frequencia de log para eventos de amostras no progresso.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Nao gera PNGs de equity (modo mais rapido).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    start = _parse_start_timestamp(args.start)
    end = _parse_end_timestamp(args.end)
    if end < start:
        raise ValueError(f"Range invalido: end {args.end} < start {args.start}")

    data_root = Path(args.data_root)
    outputs_root = Path(args.outputs)
    outputs_root.mkdir(parents=True, exist_ok=True)

    score_cfg = ScoreConfig(
        drawdown_weight=args.drawdown_weight,
        min_trade_count=args.min_trades,
        penalty_per_missing_trade=args.penalty_per_missing_trade,
    )
    base_bt_cfg = BacktestConfig(
        initial_capital=args.initial_capital,
        contracts=args.contracts,
        point_value=args.point_value,
        slippage_points=args.slippage,
        slippage_model=args.slippage_model,
        slippage_range_factor=max(0.0, args.slippage_range_factor),
        open_auction_minutes=max(0, args.open_auction_minutes),
        open_auction_slippage_multiplier=max(0.0, args.open_auction_slippage_multiplier),
        fixed_cost_per_trade=args.fixed_cost,
        cost_per_contract=args.cost_per_contract,
        cost_model=args.cost_model,
        cost_range_factor=max(0.0, args.cost_range_factor),
        open_auction_cost_multiplier=max(0.0, args.open_auction_cost_multiplier),
        max_positions=max(1, args.max_positions),
        entry_mode=args.entry_model,
        execution_mode=str(args.execution_mode),
        tick_data_root=((str(args.tick_data_root).strip() or None) if str(args.execution_mode) == "tick" else None),
        tick_symbol=str(args.symbol).strip() or None,
        session_start=args.session_start,
        session_end=args.session_end,
        close_on_session_end=args.close_on_session_end,
    )
    optimizer_cfg = OptimizerConfig(
        n_samples=args.samples,
        top_k=args.top_k,
        random_seed=args.seed,
        train_bar_step=max(1, int(args.train_bar_step)),
        score_config=score_cfg,
        max_stop_points=float(max(0.0, args.max_stop_points)),
        max_daily_loss=float(max(0.0, args.max_daily_loss)),
        max_drawdown_pct_hard=float(max(0.0, args.max_drawdown_pct_hard)),
        optimize_hours=bool(args.optimize_hours),
        hour_start_min=int(max(0, min(23, args.hour_start_min))),
        hour_start_max=int(max(0, min(23, args.hour_start_max))),
        hour_end_min=int(max(0, min(23, args.hour_end_min))),
        hour_end_max=int(max(0, min(23, args.hour_end_max))),
    )
    wf_cfg = WalkForwardConfig(
        train_days=args.train_days,
        test_days=args.test_days,
        score_config=score_cfg,
    )
    loader_cfg = LoaderConfig(
        data_root=data_root,
        symbol=args.symbol,
        start=start,
        end=end,
    )
    stop_file = Path(args.stop_file).resolve() if args.stop_file else None
    if stop_file is not None and stop_file.exists():
        stop_file.unlink(missing_ok=True)

    run_id = build_run_id(prefix="wf")
    data_hashes: dict[str, str] = {}
    generated_files: list[str] = []
    errors: list[str] = []
    selected_strategies = _select_strategies(args.strategies)
    interrupted = False
    interrupt_message = ""

    total_cycles = max(1, int(args.evolution_cycles))
    for cycle_idx in range(total_cycles):
        if total_cycles > 1:
            print(f"[INFO] Ciclo evolutivo {cycle_idx + 1}/{total_cycles}")
        for tf in args.timeframes:
            if _has_stop_request(stop_file):
                interrupted = True
                interrupt_message = "Sinal de parada detectado antes do timeframe."
                break
            try:
                artifacts = run_for_timeframe(
                    timeframe=tf,
                    loader_cfg=loader_cfg,
                    outputs_root=outputs_root,
                    base_bt_cfg=base_bt_cfg,
                    optimizer_cfg=optimizer_cfg,
                    wf_cfg=wf_cfg,
                    strategies=selected_strategies,
                    verbose_progress=bool(args.verbose_progress),
                    progress_print_every=max(1, int(args.progress_print_every)),
                    skip_plots=bool(args.skip_plots),
                    cycle_index=cycle_idx,
                    cycle_seed_step=max(1, int(args.seed_step)),
                    use_param_bank=bool(args.use_param_bank),
                    param_bank_top=max(0, int(args.param_bank_top)),
                    stop_file=stop_file,
                    run_tag=f"{run_id}_c{cycle_idx + 1}",
                )
                data_hashes[str(artifacts["timeframe"])] = str(artifacts["data_hash"])
                generated_files.extend([str(path) for path in artifacts["generated_files"]])
            except StopRequested as exc:
                interrupted = True
                interrupt_message = str(exc)
                break
            except (FileNotFoundError, ValueError) as exc:
                err = f"[ERRO] timeframe={tf}: {exc}"
                errors.append(err)
                print(err)
        if interrupted:
            break

    if interrupted:
        msg = interrupt_message or "Execucao interrompida por solicitacao externa."
        errors.append(f"[STOP] {msg}")
        print(f"[INFO] {msg}")

    manifest = RunManifest(
        run_id=run_id,
        created_at_utc=pd.Timestamp.utcnow().isoformat(),
        symbol=args.symbol,
        start=start.isoformat(),
        end=end.isoformat(),
        timeframes=[str(tf) for tf in args.timeframes],
        args=vars(args),
        environment=environment_snapshot(),
        data_hashes=data_hashes,
        generated_files=sorted(set(generated_files)),
        errors=errors,
    )
    manifest_file = outputs_root / args.symbol / f"run_manifest_{run_id}.json"
    write_manifest(manifest, manifest_file)
    print(f"[INFO] Manifesto salvo em: {manifest_file}")


def run_for_timeframe(
    timeframe: str,
    loader_cfg: LoaderConfig,
    outputs_root: Path,
    base_bt_cfg: BacktestConfig,
    optimizer_cfg: OptimizerConfig,
    wf_cfg: WalkForwardConfig,
    strategies: dict[str, StrategySpec],
    verbose_progress: bool = False,
    progress_print_every: int = 25,
    skip_plots: bool = False,
    cycle_index: int = 0,
    cycle_seed_step: int = 1000,
    use_param_bank: bool = True,
    param_bank_top: int = 30,
    stop_file: Path | None = None,
    run_tag: str = "",
) -> dict[str, Any]:
    normalized_tf = normalize_timeframe_label(timeframe)
    df = load_timeframe_data(loader_cfg, normalized_tf)
    tf_output = outputs_root / loader_cfg.symbol / normalized_tf
    tf_output.mkdir(parents=True, exist_ok=True)
    generated_files: list[str] = []

    quality_report = build_data_quality_report(df, loader_cfg.symbol, normalized_tf)
    quality_file = tf_output / f"data_quality_{normalized_tf}.json"
    quality_file.write_text(json.dumps(quality_report.to_dict(), indent=2), encoding="utf-8")
    generated_files.append(str(quality_file))

    strategy_results: list[WalkForwardResult] = []
    summary_rows: list[dict[str, Any]] = []

    for strategy in strategies.values():
        if _has_stop_request(stop_file):
            raise StopRequested(
                f"Parada solicitada em {normalized_tf}/{strategy.name} antes de iniciar a estrategia."
            )
        seed_params: tuple[dict[str, float | int | bool], ...] = ()
        bank_file = tf_output / f"params_bank_{normalized_tf}_{strategy.name}.jsonl"
        trades_file = tf_output / f"trades_{normalized_tf}_{strategy.name}.csv"
        windows_file = tf_output / f"walkforward_windows_{normalized_tf}_{strategy.name}.csv"
        topk_file = tf_output / f"walkforward_topk_{normalized_tf}_{strategy.name}.csv"
        equity_curve_file = tf_output / f"equity_curve_{normalized_tf}_{strategy.name}.csv"
        checkpoint_file = tf_output / f"checkpoint_{normalized_tf}_{strategy.name}.json"
        if use_param_bank and param_bank_top > 0:
            seed_params = _load_param_bank(bank_file=bank_file, top_n=param_bank_top)
            if seed_params:
                print(
                    f"[INFO] {normalized_tf}/{strategy.name} carregou {len(seed_params)} parametros do historico.",
                    flush=True,
                )
        local_optimizer_cfg = replace(
            optimizer_cfg,
            random_seed=int(optimizer_cfg.random_seed + cycle_index * max(1, cycle_seed_step)),
            seed_params=seed_params,
        )
        try:
            wf_result = run_walkforward(
                df=df,
                strategy=strategy,
                base_config=base_bt_cfg,
                optimizer_config=local_optimizer_cfg,
                wf_config=wf_cfg,
                progress_callback=_build_cli_progress_callback(
                    timeframe=normalized_tf,
                    strategy_name=strategy.name,
                    print_every=progress_print_every,
                    verbose=bool(verbose_progress),
                    stop_file=stop_file,
                ),
                checkpoint_callback=_build_strategy_checkpoint_callback(
                    timeframe=normalized_tf,
                    strategy_name=strategy.name,
                    trades_file=trades_file,
                    windows_file=windows_file,
                    topk_file=topk_file,
                    equity_curve_file=equity_curve_file,
                    checkpoint_file=checkpoint_file,
                    stop_file=stop_file,
                ),
            )
        except StopRequested:
            print(f"[INFO] Parada solicitada durante {normalized_tf}/{strategy.name}.")
            raise
        except ValueError as exc:
            print(f"[WARN] {normalized_tf}/{strategy.name}: {exc}")
            continue
        strategy_results.append(wf_result)

        metrics = wf_result.consolidated_metrics
        summary_rows.append(
            {
                "timeframe": normalized_tf,
                "strategy": strategy.name,
                **metrics,
                "windows": int(len(wf_result.window_results)),
            }
        )

        _write_trades_csv(wf_result.oos_trades, trades_file)
        generated_files.append(str(trades_file))

        wf_result.window_results.to_csv(windows_file, index=False)
        generated_files.append(str(windows_file))

        wf_result.topk_test_results.to_csv(topk_file, index=False)
        generated_files.append(str(topk_file))
        if use_param_bank:
            _append_param_bank(bank_file=bank_file, topk_df=wf_result.topk_test_results)
            generated_files.append(str(bank_file))

        wf_result.oos_equity.to_csv(equity_curve_file, index=False)
        generated_files.append(str(equity_curve_file))
        generated_files.append(str(checkpoint_file))

        if not skip_plots:
            plot_file = tf_output / f"equity_{normalized_tf}_{strategy.name}.png"
            _plot_equity_curve(
                wf_result.oos_equity,
                title=f"{loader_cfg.symbol} {normalized_tf} - {strategy.name}",
                output_file=plot_file,
            )
            generated_files.append(str(plot_file))

        monthly_file = tf_output / f"monthly_{normalized_tf}_{strategy.name}.csv"
        monthly_df = build_monthly_report(
            trades=wf_result.oos_trades,
            equity_curve=wf_result.oos_equity,
            initial_capital=base_bt_cfg.initial_capital,
        )
        monthly_df.to_csv(monthly_file, index=False)
        generated_files.append(str(monthly_file))

        hourly_file = tf_output / f"hourly_{normalized_tf}_{strategy.name}.csv"
        hourly_df = build_hourly_report(wf_result.oos_trades)
        hourly_df.to_csv(hourly_file, index=False)
        generated_files.append(str(hourly_file))

        sensitivity_file = tf_output / f"sensitivity_{normalized_tf}_{strategy.name}.csv"
        sensitivity_df = build_parameter_sensitivity_report(wf_result.topk_test_results)
        sensitivity_df.to_csv(sensitivity_file, index=False)
        generated_files.append(str(sensitivity_file))

        robustness_file = tf_output / f"robustness_{normalized_tf}_{strategy.name}.json"
        robustness_payload = build_robustness_report(
            window_results=wf_result.window_results,
            topk_test_results=wf_result.topk_test_results,
            consolidated_metrics=wf_result.consolidated_metrics,
        )
        robustness_file.write_text(json.dumps(robustness_payload, indent=2), encoding="utf-8")
        generated_files.append(str(robustness_file))

        _write_timeframe_summary_snapshot(
            tf_output=tf_output,
            normalized_tf=normalized_tf,
            strategy_results=strategy_results,
            symbol=loader_cfg.symbol,
            skip_plots=skip_plots,
            run_tag=run_tag,
            append_history=False,
        )
        generated_files.extend(
            [
                str(tf_output / f"summary_{normalized_tf}.csv"),
                str(tf_output / f"best_params_{normalized_tf}.json"),
                str(tf_output / f"equity_curve_{normalized_tf}_best.csv"),
                str(tf_output / f"strategy_history_{normalized_tf}.csv"),
                str(tf_output / f"summary_{normalized_tf}_{(run_tag.strip() or 'snapshot')}.csv"),
                str(tf_output / f"best_params_{normalized_tf}_{(run_tag.strip() or 'snapshot')}.json"),
                str(tf_output / f"equity_curve_{normalized_tf}_best_{(run_tag.strip() or 'snapshot')}.csv"),
            ]
        )
        if not skip_plots:
            generated_files.append(str(tf_output / f"equity_{normalized_tf}_best.png"))
            generated_files.append(str(tf_output / f"equity_{normalized_tf}_best_{(run_tag.strip() or 'snapshot')}.png"))

    if not summary_rows:
        print(f"[WARN] Nenhum resultado valido em {loader_cfg.symbol} {normalized_tf}.")
        return {
            "timeframe": normalized_tf,
            "data_hash": dataframe_sha256(df),
            "generated_files": generated_files,
        }

    _write_timeframe_summary_snapshot(
        tf_output=tf_output,
        normalized_tf=normalized_tf,
        strategy_results=strategy_results,
        symbol=loader_cfg.symbol,
        skip_plots=skip_plots,
        run_tag=run_tag,
        append_history=True,
    )
    generated_files.extend(
        [
            str(tf_output / f"summary_{normalized_tf}.csv"),
            str(tf_output / f"best_params_{normalized_tf}.json"),
            str(tf_output / f"equity_curve_{normalized_tf}_best.csv"),
            str(tf_output / f"strategy_history_{normalized_tf}.csv"),
            str(tf_output / f"summary_{normalized_tf}_{(run_tag.strip() or 'snapshot')}.csv"),
            str(tf_output / f"best_params_{normalized_tf}_{(run_tag.strip() or 'snapshot')}.json"),
            str(tf_output / f"equity_curve_{normalized_tf}_best_{(run_tag.strip() or 'snapshot')}.csv"),
            str(tf_output / f"best_history_{normalized_tf}.csv"),
        ]
    )
    if not skip_plots:
        generated_files.append(str(tf_output / f"equity_{normalized_tf}_best.png"))
        generated_files.append(str(tf_output / f"equity_{normalized_tf}_best_{(run_tag.strip() or 'snapshot')}.png"))

    return {
        "timeframe": normalized_tf,
        "data_hash": dataframe_sha256(df),
        "generated_files": generated_files,
    }


def _build_strategy_checkpoint_callback(
    timeframe: str,
    strategy_name: str,
    trades_file: Path,
    windows_file: Path,
    topk_file: Path,
    equity_curve_file: Path,
    checkpoint_file: Path,
    stop_file: Path | None,
):
    def _checkpoint(checkpoint: WalkForwardCheckpoint) -> None:
        if _has_stop_request(stop_file):
            raise StopRequested(
                f"Parada solicitada durante checkpoint de {timeframe}/{strategy_name}."
            )
        _write_trades_csv(checkpoint.oos_trades, trades_file)
        checkpoint.window_results.to_csv(windows_file, index=False)
        checkpoint.topk_test_results.to_csv(topk_file, index=False)
        checkpoint.oos_equity.to_csv(equity_curve_file, index=False)
        checkpoint_file.write_text(
            json.dumps(
                {
                    "timeframe": timeframe,
                    "strategy": strategy_name,
                    "windows_completed": int(checkpoint.windows_completed),
                    "total_windows": int(checkpoint.total_windows),
                    "latest_oos_score": float(checkpoint.latest_oos_score),
                    "latest_oos_net_profit": float(checkpoint.latest_oos_net_profit),
                    "updated_at_utc": pd.Timestamp.utcnow().isoformat(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    return _checkpoint


def _write_timeframe_summary_snapshot(
    tf_output: Path,
    normalized_tf: str,
    strategy_results: list[WalkForwardResult],
    symbol: str,
    skip_plots: bool,
    run_tag: str = "",
    append_history: bool = True,
) -> None:
    if not strategy_results:
        return
    rows = [
        {
            "timeframe": normalized_tf,
            "strategy": item.strategy_name,
            **item.consolidated_metrics,
            "windows": int(len(item.window_results)),
        }
        for item in strategy_results
    ]
    summary_df = pd.DataFrame(rows).sort_values("score", ascending=False)
    summary_latest = tf_output / f"summary_{normalized_tf}.csv"
    summary_df.to_csv(summary_latest, index=False)
    run_token = run_tag.strip() or "snapshot"
    summary_snapshot = tf_output / f"summary_{normalized_tf}_{run_token}.csv"
    summary_df.to_csv(summary_snapshot, index=False)
    if summary_df.empty:
        return

    best_strategy_name = str(summary_df.iloc[0]["strategy"])
    best_strategy_result = next(x for x in strategy_results if x.strategy_name == best_strategy_name)
    best_params_payload = {
        "symbol": symbol,
        "timeframe": normalized_tf,
        "best_strategy": best_strategy_name,
        "best_score": float(summary_df.iloc[0]["score"]),
        "strategies": {
            result.strategy_name: {
                "best_params_from_tests": result.best_params_from_tests,
                "consolidated_metrics": result.consolidated_metrics,
            }
            for result in strategy_results
        },
    }
    best_params_file = tf_output / f"best_params_{normalized_tf}.json"
    best_params_file.write_text(
        json.dumps(best_params_payload, indent=2, default=_json_default),
        encoding="utf-8",
    )
    best_params_snapshot = tf_output / f"best_params_{normalized_tf}_{run_token}.json"
    best_params_snapshot.write_text(
        json.dumps(best_params_payload, indent=2, default=_json_default),
        encoding="utf-8",
    )

    _append_strategy_history_rows(
        history_file=tf_output / f"strategy_history_{normalized_tf}.csv",
        summary_df=summary_df,
        run_tag=run_token,
        created_at_utc=pd.Timestamp.utcnow().isoformat(),
        symbol=symbol,
        timeframe=normalized_tf,
        best_strategy=best_strategy_name,
        summary_latest=summary_latest,
        summary_snapshot=summary_snapshot,
        best_params_latest=best_params_file,
        best_params_snapshot=best_params_snapshot,
        tf_output=tf_output,
    )

    best_equity_curve = tf_output / f"equity_curve_{normalized_tf}_best.csv"
    best_strategy_result.oos_equity.to_csv(best_equity_curve, index=False)
    best_equity_snapshot = tf_output / f"equity_curve_{normalized_tf}_best_{run_token}.csv"
    best_strategy_result.oos_equity.to_csv(best_equity_snapshot, index=False)
    best_plot_latest: Path | None = None
    best_plot_snapshot: Path | None = None
    if not skip_plots:
        best_equity_plot = tf_output / f"equity_{normalized_tf}_best.png"
        _plot_equity_curve(
            best_strategy_result.oos_equity,
            title=f"{symbol} {normalized_tf} - best: {best_strategy_name}",
            output_file=best_equity_plot,
        )
        best_plot_latest = best_equity_plot
        best_equity_plot_snapshot = tf_output / f"equity_{normalized_tf}_best_{run_token}.png"
        _plot_equity_curve(
            best_strategy_result.oos_equity,
            title=f"{symbol} {normalized_tf} - best: {best_strategy_name} ({run_token})",
            output_file=best_equity_plot_snapshot,
        )
        best_plot_snapshot = best_equity_plot_snapshot

    if append_history:
        _append_best_history_row(
            history_file=tf_output / f"best_history_{normalized_tf}.csv",
            row={
                "run_tag": run_token,
                "created_at_utc": pd.Timestamp.utcnow().isoformat(),
                "symbol": symbol,
                "timeframe": normalized_tf,
                "best_strategy": best_strategy_name,
                "best_score": float(summary_df.iloc[0]["score"]),
                "best_net_profit": float(best_strategy_result.consolidated_metrics.get("net_profit", 0.0)),
                "best_trade_count": float(best_strategy_result.consolidated_metrics.get("trade_count", 0.0)),
                "best_win_rate": float(best_strategy_result.consolidated_metrics.get("win_rate", 0.0)),
                "best_max_drawdown": float(best_strategy_result.consolidated_metrics.get("max_drawdown", 0.0)),
                "best_max_drawdown_pct": float(best_strategy_result.consolidated_metrics.get("max_drawdown_pct", 0.0)),
                "windows": int(len(best_strategy_result.window_results)),
                "summary_latest": str(summary_latest),
                "summary_snapshot": str(summary_snapshot),
                "best_params_latest": str(best_params_file),
                "best_params_snapshot": str(best_params_snapshot),
                "best_equity_latest": str(best_equity_curve),
                "best_equity_snapshot": str(best_equity_snapshot),
                "best_plot_latest": str(best_plot_latest) if best_plot_latest else "",
                "best_plot_snapshot": str(best_plot_snapshot) if best_plot_snapshot else "",
            },
        )


def _plot_equity_curve(equity_df: pd.DataFrame, title: str, output_file: Path) -> None:
    if equity_df.empty:
        return
    plot_df = equity_df.copy()
    plot_df = plot_df.sort_values("datetime")
    plt.figure(figsize=(10, 5))
    plt.plot(pd.to_datetime(plot_df["datetime"]), plot_df["equity"], linewidth=1.3)
    plt.title(title)
    plt.xlabel("Data")
    plt.ylabel("Equity")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=120)
    plt.close()


def _write_trades_csv(trades: pd.DataFrame, output_file: Path) -> None:
    if trades.empty:
        pd.DataFrame(columns=default_trade_columns()).to_csv(output_file, index=False)
        return
    out = trades.copy()
    out["params"] = out["params"].apply(lambda p: json.dumps(p, sort_keys=True))
    out.to_csv(output_file, index=False)


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, date)):
        return value.isoformat()
    return str(value)


def _parse_start_timestamp(raw: str) -> pd.Timestamp:
    return pd.Timestamp(raw)


def _parse_end_timestamp(raw: str) -> pd.Timestamp:
    ts = pd.Timestamp(raw)
    if len(raw.strip()) <= 10:
        return ts + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return ts


def _select_strategies(raw: list[str] | None) -> dict[str, StrategySpec]:
    if raw is None:
        return STRATEGIES
    selected: dict[str, StrategySpec] = {}
    for name in raw:
        if name not in STRATEGIES:
            raise ValueError(
                f"Estrategia nao suportada: {name}. Disponiveis: {', '.join(sorted(STRATEGIES.keys()))}"
            )
        selected[name] = STRATEGIES[name]
    if not selected:
        raise ValueError("Nenhuma estrategia selecionada.")
    return selected


def _build_cli_progress_callback(
    timeframe: str,
    strategy_name: str,
    print_every: int,
    verbose: bool = True,
    stop_file: Path | None = None,
):
    def _callback(event: dict[str, Any]) -> None:
        if _has_stop_request(stop_file):
            raise StopRequested(
                f"Parada solicitada durante execucao de {timeframe}/{strategy_name}."
            )
        if not verbose:
            return
        stage = str(event.get("stage", ""))
        if stage == "window_start":
            print(
                f"[PROGRESS] {timeframe}/{strategy_name} "
                f"window {event.get('window_index')}/{event.get('total_windows')} "
                f"test {event.get('test_start')} -> {event.get('test_end')}",
                flush=True,
            )
            return
        if stage == "optimizer_sample":
            sample_idx = int(event.get("sample_index", 0))
            sample_total = int(event.get("samples_total", 0))
            if sample_idx == sample_total or sample_idx % max(print_every, 1) == 0:
                print(
                    f"[PROGRESS] {timeframe}/{strategy_name} "
                    f"sample {sample_idx}/{sample_total} score={float(event.get('score', 0.0)):.2f} "
                    f"net={float(event.get('net_profit', 0.0)):.2f} "
                    f"pf={float(event.get('profit_factor', 0.0)):.2f}",
                    flush=True,
                )
            return
        if stage == "window_complete":
            print(
                f"[PROGRESS] {timeframe}/{strategy_name} "
                f"window {event.get('window_index')}/{event.get('total_windows')} done "
                f"oos_score={float(event.get('oos_score', 0.0)):.2f} "
                f"oos_net={float(event.get('oos_net_profit', 0.0)):.2f} "
                f"oos_pf={float(event.get('oos_profit_factor', 0.0)):.2f}",
                flush=True,
            )
            return
        if stage == "walkforward_done":
            print(
                f"[PROGRESS] {timeframe}/{strategy_name} done "
                f"score={float(event.get('final_score', 0.0)):.2f} "
                f"net={float(event.get('net_profit', 0.0)):.2f} "
                f"pf={float(event.get('profit_factor', 0.0)):.2f}",
                flush=True,
            )

    return _callback


def _has_stop_request(stop_file: Path | None) -> bool:
    return bool(stop_file is not None and stop_file.exists())


def _append_best_history_row(history_file: Path, row: dict[str, Any]) -> None:
    history_file.parent.mkdir(parents=True, exist_ok=True)
    row_df = pd.DataFrame([row])
    if history_file.exists():
        prev = pd.read_csv(history_file)
        if not prev.empty and {"run_tag", "timeframe"}.issubset(set(prev.columns)):
            mask = (prev["run_tag"].astype(str) == str(row.get("run_tag", ""))) & (
                prev["timeframe"].astype(str) == str(row.get("timeframe", ""))
            )
            prev = prev.loc[~mask].copy()
        out = pd.concat([prev, row_df], ignore_index=True)
    else:
        out = row_df
    out = out.sort_values("created_at_utc", ascending=False)
    out.to_csv(history_file, index=False)


def _append_strategy_history_rows(
    history_file: Path,
    summary_df: pd.DataFrame,
    run_tag: str,
    created_at_utc: str,
    symbol: str,
    timeframe: str,
    best_strategy: str,
    summary_latest: Path,
    summary_snapshot: Path,
    best_params_latest: Path,
    best_params_snapshot: Path,
    tf_output: Path,
) -> None:
    if summary_df.empty:
        return
    history_file.parent.mkdir(parents=True, exist_ok=True)
    ranked = summary_df.sort_values("score", ascending=False).reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for rank_idx, (_, item) in enumerate(ranked.iterrows(), start=1):
        strategy = str(item.get("strategy", "")).strip()
        if not strategy:
            continue
        rows.append(
            {
                "run_tag": run_tag,
                "created_at_utc": created_at_utc,
                "symbol": symbol,
                "timeframe": timeframe,
                "strategy": strategy,
                "is_best": int(strategy == best_strategy),
                "rank_in_run": int(rank_idx),
                "score": float(item.get("score", 0.0)),
                "net_profit": float(item.get("net_profit", 0.0)),
                "trade_count": float(item.get("trade_count", 0.0)),
                "win_rate": float(item.get("win_rate", 0.0)),
                "max_drawdown": float(item.get("max_drawdown", 0.0)),
                "max_drawdown_pct": float(item.get("max_drawdown_pct", 0.0)),
                "windows": int(item.get("windows", 0)),
                "summary_latest": str(summary_latest),
                "summary_snapshot": str(summary_snapshot),
                "best_params_latest": str(best_params_latest),
                "best_params_snapshot": str(best_params_snapshot),
                "trades_latest": str(tf_output / f"trades_{timeframe}_{strategy}.csv"),
                "windows_latest": str(tf_output / f"walkforward_windows_{timeframe}_{strategy}.csv"),
                "topk_latest": str(tf_output / f"walkforward_topk_{timeframe}_{strategy}.csv"),
                "equity_latest": str(tf_output / f"equity_curve_{timeframe}_{strategy}.csv"),
                "equity_plot_latest": str(tf_output / f"equity_{timeframe}_{strategy}.png"),
            }
        )
    if not rows:
        return

    row_df = pd.DataFrame(rows)
    if history_file.exists():
        prev = pd.read_csv(history_file)
        if not prev.empty and {"run_tag", "timeframe", "strategy"}.issubset(set(prev.columns)):
            key_new = (
                (prev["run_tag"].astype(str) == str(run_tag))
                & (prev["timeframe"].astype(str) == str(timeframe))
                & (prev["strategy"].astype(str).isin(row_df["strategy"].astype(str)))
            )
            prev = prev.loc[~key_new].copy()
        out = pd.concat([prev, row_df], ignore_index=True)
    else:
        out = row_df
    out = out.sort_values(["created_at_utc", "rank_in_run"], ascending=[False, True])
    out.to_csv(history_file, index=False)


def _load_param_bank(
    bank_file: Path,
    top_n: int,
) -> tuple[dict[str, float | int | bool], ...]:
    if top_n <= 0 or not bank_file.exists():
        return ()
    best_by_params: dict[str, tuple[float, dict[str, float | int | bool]]] = {}
    with bank_file.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            params = payload.get("params")
            score_raw = payload.get("test_score", 0.0)
            if not isinstance(params, dict):
                continue
            try:
                score = float(score_raw)
            except (TypeError, ValueError):
                score = 0.0
            key = json.dumps(params, sort_keys=True)
            prev = best_by_params.get(key)
            if prev is None or score > prev[0]:
                best_by_params[key] = (score, params)
    ranked = sorted(best_by_params.values(), key=lambda x: x[0], reverse=True)
    return tuple(item[1] for item in ranked[:top_n])


def _append_param_bank(bank_file: Path, topk_df: pd.DataFrame) -> None:
    if topk_df.empty or "params_json" not in topk_df.columns:
        return
    rows = topk_df.copy()
    if "test_score" in rows.columns:
        rows = rows.sort_values("test_score", ascending=False)
    bank_file.parent.mkdir(parents=True, exist_ok=True)
    with bank_file.open("a", encoding="utf-8") as fh:
        for _, row in rows.iterrows():
            raw_params = row.get("params_json")
            if not isinstance(raw_params, str) or not raw_params.strip():
                continue
            try:
                params = json.loads(raw_params)
            except json.JSONDecodeError:
                continue
            if not isinstance(params, dict):
                continue
            payload = {
                "timestamp": pd.Timestamp.utcnow().isoformat(),
                "test_score": float(row.get("test_score", 0.0)),
                "train_score": float(row.get("train_score", 0.0)),
                "params": params,
            }
            fh.write(json.dumps(payload, sort_keys=True, default=_json_default))
            fh.write("\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[INFO] Execucao interrompida pelo usuario.")
