"""CLI for event-driven paper trading replay."""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from .backtest_engine import BacktestConfig
from .data_loader import LoaderConfig, load_timeframe_data, normalize_timeframe_label
from .paper_engine import PaperEngineConfig, run_paper_engine
from .reproducibility import (
    RunManifest,
    build_run_id,
    dataframe_sha256,
    environment_snapshot,
    write_manifest,
)
from .risk import RiskLimits
from .strategies import STRATEGIES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper trading event-driven com replay candle a candle.")
    parser.add_argument("--symbol", default="WINFUT")
    parser.add_argument("--timeframe", required=True, help="Ex.: 1m 5m 15m 30m daily weekly")
    parser.add_argument(
        "--strategy",
        default=None,
        help=f"Nome da estrategia ({', '.join(sorted(STRATEGIES.keys()))})",
    )
    parser.add_argument("--params-json", default=None, help="JSON de parametros da estrategia")
    parser.add_argument("--params-file", default=None, help="Arquivo JSON de parametros (best_params*.json ou dict)")
    parser.add_argument("--seed", type=int, default=42, help="Seed para sample de parametros quando nao informados")

    parser.add_argument("--start", required=True, help="Data inicial YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="Data final YYYY-MM-DD")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--outputs", default="outputs_paper")

    parser.add_argument("--initial-capital", type=float, default=100_000.0)
    parser.add_argument("--contracts", type=int, default=1)
    parser.add_argument("--point-value", type=float, default=0.2)
    parser.add_argument("--slippage", type=float, default=0.0)
    parser.add_argument("--slippage-model", choices=["fixed", "range_scaled"], default="fixed")
    parser.add_argument("--slippage-range-factor", type=float, default=0.0)
    parser.add_argument("--fixed-cost", type=float, default=0.0)
    parser.add_argument("--cost-per-contract", type=float, default=0.0)
    parser.add_argument("--cost-model", choices=["fixed", "range_scaled"], default="fixed")
    parser.add_argument("--cost-range-factor", type=float, default=0.0)
    parser.add_argument("--stop-points", type=float, default=300.0)
    parser.add_argument("--take-points", type=float, default=600.0)
    parser.add_argument("--entry-model", choices=["next_open", "close_slippage"], default="next_open")
    parser.add_argument("--session-start", default=None)
    parser.add_argument("--session-end", default="17:00")
    parser.add_argument("--close-on-session-end", dest="close_on_session_end", action="store_true", default=True)
    parser.add_argument("--no-close-on-session-end", dest="close_on_session_end", action="store_false")
    parser.add_argument("--open-auction-minutes", type=int, default=0)
    parser.add_argument("--open-auction-slippage-multiplier", type=float, default=1.0)
    parser.add_argument("--open-auction-cost-multiplier", type=float, default=1.0)

    parser.add_argument("--daily-loss-limit", type=float, default=0.0)
    parser.add_argument("--max-drawdown-pct", type=float, default=0.0)
    parser.add_argument("--max-consecutive-losses", type=int, default=0)
    parser.add_argument("--kill-switch-file", default=None)
    parser.add_argument("--no-halt-on-risk", action="store_true")

    parser.add_argument("--replay-delay-ms", type=int, default=0)
    parser.add_argument("--emit-every-bars", type=int, default=20)
    parser.add_argument("--print-events", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = _parse_start_timestamp(args.start)
    end = _parse_end_timestamp(args.end)
    if end < start:
        raise ValueError(f"Range invalido: end {args.end} < start {args.start}")

    timeframe = normalize_timeframe_label(args.timeframe)
    strategy_name, strategy_params = _resolve_strategy_and_params(args)
    if strategy_name not in STRATEGIES:
        raise ValueError(f"Estrategia invalida: {strategy_name}. Disponiveis: {sorted(STRATEGIES.keys())}")
    strategy = STRATEGIES[strategy_name]

    loader_cfg = LoaderConfig(
        data_root=Path(args.data_root),
        symbol=args.symbol,
        start=start,
        end=end,
    )
    df = load_timeframe_data(loader_cfg, timeframe)

    bt_cfg = BacktestConfig(
        initial_capital=float(args.initial_capital),
        contracts=int(args.contracts),
        point_value=float(args.point_value),
        slippage_points=float(args.slippage),
        slippage_model=str(args.slippage_model),
        slippage_range_factor=float(max(0.0, args.slippage_range_factor)),
        fixed_cost_per_trade=float(args.fixed_cost),
        cost_per_contract=float(args.cost_per_contract),
        cost_model=str(args.cost_model),
        cost_range_factor=float(max(0.0, args.cost_range_factor)),
        stop_points=float(args.stop_points),
        take_points=float(args.take_points),
        entry_mode=str(args.entry_model),
        session_start=args.session_start,
        session_end=args.session_end,
        close_on_session_end=bool(args.close_on_session_end),
        open_auction_minutes=int(max(0, args.open_auction_minutes)),
        open_auction_slippage_multiplier=float(max(0.0, args.open_auction_slippage_multiplier)),
        open_auction_cost_multiplier=float(max(0.0, args.open_auction_cost_multiplier)),
    )
    risk_limits = RiskLimits(
        daily_loss_limit=float(max(0.0, args.daily_loss_limit)),
        max_drawdown_pct=float(max(0.0, args.max_drawdown_pct)),
        max_consecutive_losses=int(max(0, args.max_consecutive_losses)),
        kill_switch_file=args.kill_switch_file,
    )
    paper_cfg = PaperEngineConfig(
        backtest_config=bt_cfg,
        risk_limits=risk_limits,
        halt_on_risk=not bool(args.no_halt_on_risk),
        replay_delay_ms=int(max(0, args.replay_delay_ms)),
        emit_every_bars=int(max(1, args.emit_every_bars)),
    )

    callback = _build_callback(enabled=bool(args.print_events))
    result = run_paper_engine(
        df=df,
        strategy=strategy,
        strategy_params=strategy_params,
        config=paper_cfg,
        callback=callback,
    )
    _save_outputs(
        args=args,
        timeframe=timeframe,
        strategy_name=strategy_name,
        strategy_params=strategy_params,
        df=df,
        result=result,
        start=start,
        end=end,
    )


def _resolve_strategy_and_params(args: argparse.Namespace) -> tuple[str, dict[str, float | int | bool]]:
    if args.params_json:
        params = json.loads(args.params_json)
        if not isinstance(params, dict):
            raise ValueError("--params-json deve ser um objeto JSON.")
        strategy_name = args.strategy or sorted(STRATEGIES.keys())[0]
        return strategy_name, _coerce_params(params)

    if args.params_file:
        payload = json.loads(Path(args.params_file).read_text(encoding="utf-8"))
        strategy_name, params = _extract_params_from_payload(payload, preferred_strategy=args.strategy)
        return strategy_name, _coerce_params(params)

    strategy_name = args.strategy or sorted(STRATEGIES.keys())[0]
    rng = np_random(args.seed)
    sampled = STRATEGIES[strategy_name].sample_parameters(rng)
    return strategy_name, _coerce_params(sampled)


def _extract_params_from_payload(
    payload: dict[str, Any],
    preferred_strategy: str | None,
) -> tuple[str, dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ValueError("Arquivo de parametros invalido: JSON deve ser objeto.")

    if "strategies" in payload and isinstance(payload["strategies"], dict):
        strategies = payload["strategies"]
        strategy_name = preferred_strategy or str(payload.get("best_strategy", ""))
        if not strategy_name:
            strategy_name = str(next(iter(strategies.keys())))
        node = strategies.get(strategy_name)
        if not isinstance(node, dict):
            raise ValueError(f"Estrategia '{strategy_name}' nao encontrada no arquivo.")
        params = node.get("best_params_from_tests", {})
        if not isinstance(params, dict):
            raise ValueError("best_params_from_tests invalido no arquivo.")
        return strategy_name, params

    if preferred_strategy:
        strategy_name = preferred_strategy
    else:
        strategy_name = str(payload.get("strategy", "")) or sorted(STRATEGIES.keys())[0]

    if "best_params_from_tests" in payload and isinstance(payload["best_params_from_tests"], dict):
        return strategy_name, payload["best_params_from_tests"]
    return strategy_name, payload


def _coerce_params(params: dict[str, Any]) -> dict[str, float | int | bool]:
    out: dict[str, float | int | bool] = {}
    for key, value in params.items():
        if isinstance(value, bool):
            out[str(key)] = value
        elif isinstance(value, int):
            out[str(key)] = int(value)
        elif isinstance(value, float):
            out[str(key)] = float(value)
        else:
            text = str(value).strip().lower()
            if text in {"true", "false"}:
                out[str(key)] = (text == "true")
            else:
                try:
                    if "." in text:
                        out[str(key)] = float(text)
                    else:
                        out[str(key)] = int(text)
                except ValueError:
                    pass
    return out


def _build_callback(enabled: bool) -> callable | None:
    if not enabled:
        return None

    def _cb(event: dict[str, Any]) -> None:
        stage = str(event.get("stage", ""))
        if stage == "bar":
            print(
                f"[BAR] {event.get('bar_index')}/{event.get('bars_total')} "
                f"{event.get('datetime')} cash={float(event.get('cash', 0.0)):.2f}"
            )
        elif stage == "trade_close":
            print(
                f"[TRADE] {event.get('datetime')} reason={event.get('reason')} "
                f"pnl={float(event.get('pnl_net', 0.0)):.2f} cash={float(event.get('cash', 0.0)):.2f}"
            )
        elif stage == "paper_done":
            print(
                f"[DONE] halted={event.get('halted')} halt_code={event.get('halt_code')} "
                f"net={float(event.get('net_profit', 0.0)):.2f} trades={float(event.get('trade_count', 0.0)):.0f}"
            )

    return _cb


def _save_outputs(
    args: argparse.Namespace,
    timeframe: str,
    strategy_name: str,
    strategy_params: dict[str, float | int | bool],
    df: pd.DataFrame,
    result: Any,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> None:
    out_dir = Path(args.outputs) / args.symbol / timeframe / strategy_name
    out_dir.mkdir(parents=True, exist_ok=True)

    trades_file = out_dir / f"paper_trades_{timeframe}_{strategy_name}.csv"
    equity_file = out_dir / f"paper_equity_{timeframe}_{strategy_name}.csv"
    alerts_file = out_dir / f"paper_alerts_{timeframe}_{strategy_name}.csv"
    summary_file = out_dir / f"paper_summary_{timeframe}_{strategy_name}.json"
    equity_png = out_dir / f"paper_equity_{timeframe}_{strategy_name}.png"

    trades_out = result.trades.copy()
    if not trades_out.empty and "params" in trades_out.columns:
        trades_out["params"] = trades_out["params"].apply(lambda p: json.dumps(p, sort_keys=True))
    trades_out.to_csv(trades_file, index=False)
    result.equity_curve.to_csv(equity_file, index=False)
    result.alerts.to_csv(alerts_file, index=False)

    payload = {
        "symbol": args.symbol,
        "timeframe": timeframe,
        "strategy": strategy_name,
        "strategy_params": strategy_params,
        "halted": bool(result.halted),
        "halt_code": str(result.halt_code),
        "halt_message": str(result.halt_message),
        "metrics": result.metrics,
    }
    summary_file.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    _plot_equity(result.equity_curve, title=f"{args.symbol} {timeframe} paper {strategy_name}", output_file=equity_png)

    run_id = build_run_id(prefix="paper")
    manifest = RunManifest(
        run_id=run_id,
        created_at_utc=pd.Timestamp.utcnow().isoformat(),
        symbol=args.symbol,
        start=start.isoformat(),
        end=end.isoformat(),
        timeframes=[timeframe],
        args=vars(args),
        environment=environment_snapshot(),
        data_hashes={timeframe: dataframe_sha256(df)},
        generated_files=[
            str(trades_file),
            str(equity_file),
            str(alerts_file),
            str(summary_file),
            str(equity_png),
        ],
        errors=[],
    )
    manifest_file = Path(args.outputs) / args.symbol / f"run_manifest_{run_id}.json"
    write_manifest(manifest, manifest_file)
    print(f"[INFO] Paper outputs: {out_dir}")
    print(f"[INFO] Manifesto: {manifest_file}")


def _plot_equity(equity_df: pd.DataFrame, title: str, output_file: Path) -> None:
    if equity_df.empty:
        return
    plot_df = equity_df.sort_values("datetime")
    plt.figure(figsize=(10, 5))
    plt.plot(pd.to_datetime(plot_df["datetime"]), plot_df["equity"], linewidth=1.2, color="#25f08a")
    plt.fill_between(
        pd.to_datetime(plot_df["datetime"]),
        plot_df["equity"].astype(float),
        y2=float(plot_df["equity"].astype(float).min()),
        color="#25f08a",
        alpha=0.15,
    )
    plt.title(title)
    plt.xlabel("Data")
    plt.ylabel("Equity")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=120)
    plt.close()


def np_random(seed: int) -> np.random.Generator:  # type: ignore[name-defined]
    import numpy as np

    return np.random.default_rng(seed)


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


if __name__ == "__main__":
    main()
