
"""Visual dashboard for running walk-forward backtests in real time."""

from __future__ import annotations

import json
import subprocess
import sys
import time as time_mod
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is in sys.path when Streamlit runs this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest_engine import BacktestConfig, default_trade_columns
from src.data_loader import (
    LoaderConfig,
    build_data_quality_report,
    load_timeframe_data,
    normalize_timeframe_label,
)
from src.metrics import ScoreConfig
from src.optimizer import OptimizerConfig
from src.reports import (
    build_monthly_report,
    build_parameter_sensitivity_report,
    build_robustness_report,
)
from src.reproducibility import (
    RunManifest,
    build_run_id,
    dataframe_sha256,
    environment_snapshot,
    write_manifest,
)
from src.strategies import STRATEGIES
from src.walkforward import WalkForwardConfig, WalkForwardResult, run_walkforward


@dataclass(slots=True)
class StrategyRun:
    timeframe: str
    strategy_name: str
    result: WalkForwardResult


@dataclass(slots=True)
class DashboardRunResult:
    summary: pd.DataFrame
    strategy_runs: list[StrategyRun]
    symbol: str
    start: pd.Timestamp
    end: pd.Timestamp
    base_cfg: BacktestConfig


@dataclass(slots=True)
class TurboJob:
    """Background CLI execution tracked by the dashboard."""

    process: Any
    command: list[str]
    log_file: str
    symbol: str
    timeframes: list[str]
    strategies: list[str]
    outputs_dir: str
    start_iso: str
    end_iso: str
    base_cfg_dict: dict[str, Any]


def main() -> None:
    st.set_page_config(
        page_title="Robo Backtest Pro",
        page_icon=":chart_with_upwards_trend:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_styles()
    _init_session_state()

    st.markdown("<h1 class='app-title'>ROBO BACKTEST PRO</h1>", unsafe_allow_html=True)
    st.caption("Walk-forward com visualizacao ao vivo de progresso, operacoes e patrimonio.")

    with st.sidebar:
        compact_sidebar = st.checkbox("Layout compacto", value=True)
        if compact_sidebar:
            _inject_compact_sidebar_css()

        with st.expander("Execucao", expanded=True):
            symbol = st.text_input("Ativo", value=st.session_state.symbol_default)
            timeframe_options = ["1m", "5m", "10m", "15m", "30m", "60m", "daily", "weekly"]
            selected_timeframes = st.multiselect(
                "Timeframes",
                options=timeframe_options,
                default=st.session_state.timeframes_default,
            )
            strategies = st.multiselect(
                "Estrategias",
                options=list(STRATEGIES.keys()),
                default=st.session_state.strategies_default,
            )
            execution_mode = st.radio("Modo", options=["OHLC", "Tick a Tick (desativado)"], index=0)
            if execution_mode.startswith("Tick"):
                st.warning("Tick a Tick ainda nao implementado. O modo OHLC sera utilizado.")
            start_date = st.date_input("Inicio", value=st.session_state.start_date_default)
            end_date = st.date_input("Fim", value=st.session_state.end_date_default)
            c1, c2 = st.columns(2)
            with c1:
                contracts = st.number_input("Qtd", min_value=1, max_value=100, value=1, step=1)
            with c2:
                max_positions = st.number_input("Qtd max", min_value=1, max_value=20, value=1, step=1)

        with st.expander("Parametros", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                train_days = st.number_input("Train", min_value=20, max_value=500, value=120, step=5)
                samples = st.number_input("Samples", min_value=5, max_value=3000, value=200, step=5)
                drawdown_weight = st.number_input("Peso DD", min_value=0.0, max_value=20.0, value=1.5, step=0.1)
            with c2:
                test_days = st.number_input("Test", min_value=5, max_value=250, value=30, step=5)
                top_k = st.number_input("Top K", min_value=1, max_value=50, value=5, step=1)
                min_trades = st.number_input("Min trades", min_value=0, max_value=1000, value=20, step=1)

        with st.expander("Risco", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                initial_capital = st.number_input("Capital", min_value=1000.0, max_value=50_000_000.0, value=100_000.0, step=1000.0)
                point_value = st.number_input("Valor ponto", min_value=0.01, max_value=100.0, value=0.2, step=0.01)
            with c2:
                penalty_missing = st.number_input("Penalidade", min_value=0.0, max_value=5000.0, value=150.0, step=10.0)
                seed = st.number_input("Seed", min_value=0, max_value=999_999, value=42, step=1)

        with st.expander("Custos", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                slippage = st.number_input("Slippage", min_value=0.0, max_value=200.0, value=0.0, step=1.0)
                slippage_model = st.selectbox("Modelo slip", options=["fixed", "range_scaled"], index=0)
                slippage_range_factor = st.number_input("Fator slip", min_value=0.0, max_value=2.0, value=0.0, step=0.01)
                fixed_cost = st.number_input("Custo fixo", min_value=0.0, max_value=5000.0, value=0.0, step=0.1)
            with c2:
                cost_per_contract = st.number_input("Custo contrato", min_value=0.0, max_value=1000.0, value=0.0, step=0.1)
                cost_model = st.selectbox("Modelo custo", options=["fixed", "range_scaled"], index=0)
                cost_range_factor = st.number_input("Fator custo", min_value=0.0, max_value=2.0, value=0.0, step=0.01)
                entry_model = st.selectbox("Entrada", options=["next_open", "close_slippage"], index=0)

        with st.expander("Sessao", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                session_start = st.text_input("Inicio", value="09:00")
                open_auction_minutes = st.number_input("Janela abertura", min_value=0, max_value=180, value=0, step=5)
            with c2:
                session_end = st.text_input("Fim", value="17:00")
                open_auction_slippage_multiplier = st.number_input("Mult slip ab.", min_value=0.0, max_value=5.0, value=1.0, step=0.05)
                open_auction_cost_multiplier = st.number_input("Mult custo ab.", min_value=0.0, max_value=5.0, value=1.0, step=0.05)
            close_on_session_end = st.checkbox("Fechar ao final da sessao", value=True)

        with st.expander("Visual", expanded=True):
            turbo_mode = st.checkbox("Modo turbo real (subprocesso)", value=True)
            live_updates = st.checkbox("Atualizacao ao vivo", value=True)
            fast_mode = st.checkbox("Modo rapido", value=True)
            update_every = st.slider("Atualizar a cada N amostras", min_value=1, max_value=50, value=5)
            turbo_train_step = st.slider("Turbo step treino", min_value=1, max_value=10, value=3)
            turbo_log_every = st.slider("Turbo log N samples", min_value=1, max_value=100, value=10)
            turbo_refresh_sec = st.slider("Turbo refresh (s)", min_value=1, max_value=10, value=2)
            turbo_skip_plots = st.checkbox("Turbo sem PNG", value=True)
            save_outputs = st.checkbox("Salvar CSV/JSON/PNG", value=True)
            outputs_dir = st.text_input("Pasta outputs", value="outputs")
            data_root = st.text_input("Pasta data", value="data")

        run_clicked = st.button("Executar", type="primary", use_container_width=True)
        stop_clicked = st.button("Parar turbo", use_container_width=True)

    if stop_clicked:
        _stop_turbo_job()

    if run_clicked:
        error = _validate_inputs(selected_timeframes=selected_timeframes, strategies=strategies, start_date=start_date, end_date=end_date)
        if error:
            st.error(error)
        else:
            base_bt_cfg = BacktestConfig(
                initial_capital=float(initial_capital),
                contracts=int(contracts),
                point_value=float(point_value),
                slippage_points=float(slippage),
                slippage_model=str(slippage_model),
                slippage_range_factor=float(slippage_range_factor),
                open_auction_minutes=int(open_auction_minutes),
                open_auction_slippage_multiplier=float(open_auction_slippage_multiplier),
                fixed_cost_per_trade=float(fixed_cost),
                cost_per_contract=float(cost_per_contract),
                cost_model=str(cost_model),
                cost_range_factor=float(cost_range_factor),
                open_auction_cost_multiplier=float(open_auction_cost_multiplier),
                max_positions=int(max_positions),
                entry_mode=entry_model,
                session_start=session_start.strip() or None,
                session_end=session_end.strip() or None,
                close_on_session_end=close_on_session_end,
            )
            optimizer_cfg = OptimizerConfig(
                n_samples=int(samples),
                top_k=int(top_k),
                random_seed=int(seed),
                train_bar_step=1,
                score_config=ScoreConfig(
                    drawdown_weight=float(drawdown_weight),
                    min_trade_count=int(min_trades),
                    penalty_per_missing_trade=float(penalty_missing),
                ),
            )
            wf_cfg = WalkForwardConfig(
                train_days=int(train_days),
                test_days=int(test_days),
                score_config=ScoreConfig(
                    drawdown_weight=float(drawdown_weight),
                    min_trade_count=int(min_trades),
                    penalty_per_missing_trade=float(penalty_missing),
                ),
            )
            if turbo_mode:
                if _turbo_job_running():
                    st.warning("Ja existe uma execucao turbo em andamento.")
                else:
                    if not save_outputs:
                        st.warning("Modo turbo requer salvar outputs; opcao foi forcada para ON.")
                    _start_turbo_job(
                        symbol=symbol.strip() or "WINFUT",
                        raw_timeframes=selected_timeframes,
                        strategy_names=strategies,
                        start_date=start_date,
                        end_date=end_date,
                        data_root=Path(data_root),
                        outputs_root=Path(outputs_dir),
                        base_bt_cfg=base_bt_cfg,
                        optimizer_cfg=optimizer_cfg,
                        wf_cfg=wf_cfg,
                        skip_plots=bool(turbo_skip_plots),
                        progress_print_every=max(1, int(turbo_log_every)),
                        train_bar_step=max(1, int(turbo_train_step)),
                    )
                    st.session_state.last_result = None
            else:
                result = _run_dashboard(
                    symbol=symbol.strip() or "WINFUT",
                    raw_timeframes=selected_timeframes,
                    strategy_names=strategies,
                    start_date=start_date,
                    end_date=end_date,
                    data_root=Path(data_root),
                    outputs_root=Path(outputs_dir),
                    save_outputs=save_outputs,
                    base_bt_cfg=base_bt_cfg,
                    optimizer_cfg=optimizer_cfg,
                    wf_cfg=wf_cfg,
                    live_updates=live_updates,
                    fast_mode=fast_mode,
                    update_every=max(1, int(update_every)),
                )
                st.session_state.last_result = result

    _render_turbo_status(refresh_seconds=int(turbo_refresh_sec))

    if st.session_state.last_result is None:
        st.info("Configure os parametros e clique em Executar.")
        return

    _render_result(st.session_state.last_result)


def _validate_inputs(
    selected_timeframes: list[str],
    strategies: list[str],
    start_date: date,
    end_date: date,
) -> str | None:
    if not selected_timeframes:
        return "Selecione ao menos um timeframe."
    if not strategies:
        return "Selecione ao menos uma estrategia."
    if end_date < start_date:
        return "Data final menor que data inicial."
    return None


def _turbo_job_running() -> bool:
    job = st.session_state.get("turbo_job")
    if job is None:
        return False
    return job.process.poll() is None


def _start_turbo_job(
    symbol: str,
    raw_timeframes: list[str],
    strategy_names: list[str],
    start_date: date,
    end_date: date,
    data_root: Path,
    outputs_root: Path,
    base_bt_cfg: BacktestConfig,
    optimizer_cfg: OptimizerConfig,
    wf_cfg: WalkForwardConfig,
    skip_plots: bool,
    progress_print_every: int,
    train_bar_step: int,
) -> None:
    outputs_root = outputs_root.resolve()
    data_root = data_root.resolve()
    run_id = build_run_id(prefix="turbo_ui")
    log_dir = outputs_root / symbol / "_turbo_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{run_id}.log"

    cmd = [
        sys.executable,
        "-m",
        "src.cli",
        "--symbol",
        symbol,
        "--timeframes",
        *raw_timeframes,
        "--strategies",
        *strategy_names,
        "--start",
        start_date.isoformat(),
        "--end",
        end_date.isoformat(),
        "--data-root",
        str(data_root),
        "--outputs",
        str(outputs_root),
        "--train-days",
        str(int(wf_cfg.train_days)),
        "--test-days",
        str(int(wf_cfg.test_days)),
        "--samples",
        str(int(optimizer_cfg.n_samples)),
        "--top-k",
        str(int(optimizer_cfg.top_k)),
        "--seed",
        str(int(optimizer_cfg.random_seed)),
        "--train-bar-step",
        str(int(max(1, train_bar_step))),
        "--initial-capital",
        str(float(base_bt_cfg.initial_capital)),
        "--contracts",
        str(int(base_bt_cfg.contracts)),
        "--point-value",
        str(float(base_bt_cfg.point_value)),
        "--slippage",
        str(float(base_bt_cfg.slippage_points)),
        "--slippage-model",
        str(base_bt_cfg.slippage_model),
        "--slippage-range-factor",
        str(float(base_bt_cfg.slippage_range_factor)),
        "--open-auction-minutes",
        str(int(base_bt_cfg.open_auction_minutes)),
        "--open-auction-slippage-multiplier",
        str(float(base_bt_cfg.open_auction_slippage_multiplier)),
        "--fixed-cost",
        str(float(base_bt_cfg.fixed_cost_per_trade)),
        "--cost-per-contract",
        str(float(base_bt_cfg.cost_per_contract)),
        "--cost-model",
        str(base_bt_cfg.cost_model),
        "--cost-range-factor",
        str(float(base_bt_cfg.cost_range_factor)),
        "--open-auction-cost-multiplier",
        str(float(base_bt_cfg.open_auction_cost_multiplier)),
        "--max-positions",
        str(int(base_bt_cfg.max_positions)),
        "--entry-model",
        str(base_bt_cfg.entry_mode),
        "--drawdown-weight",
        str(float(wf_cfg.score_config.drawdown_weight)),
        "--min-trades",
        str(int(wf_cfg.score_config.min_trade_count)),
        "--penalty-per-missing-trade",
        str(float(wf_cfg.score_config.penalty_per_missing_trade)),
        "--verbose-progress",
        "--progress-print-every",
        str(int(max(1, progress_print_every))),
    ]
    if base_bt_cfg.session_start:
        cmd.extend(["--session-start", str(base_bt_cfg.session_start)])
    if base_bt_cfg.session_end:
        cmd.extend(["--session-end", str(base_bt_cfg.session_end)])
    if base_bt_cfg.close_on_session_end:
        cmd.append("--close-on-session-end")
    else:
        cmd.append("--no-close-on-session-end")
    if skip_plots:
        cmd.append("--skip-plots")

    log_handle = open(log_file, "w", encoding="utf-8")
    process = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    log_handle.close()

    st.session_state.turbo_job = TurboJob(
        process=process,
        command=cmd,
        log_file=str(log_file),
        symbol=symbol,
        timeframes=list(raw_timeframes),
        strategies=list(strategy_names),
        outputs_dir=str(outputs_root),
        start_iso=pd.Timestamp(start_date).isoformat(),
        end_iso=(pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)).isoformat(),
        base_cfg_dict=asdict(base_bt_cfg),
    )
    st.success(f"Turbo iniciado (PID {process.pid}).")


def _stop_turbo_job() -> None:
    job = st.session_state.get("turbo_job")
    if job is None:
        return
    if job.process.poll() is None:
        job.process.terminate()
        st.warning("Sinal de parada enviado ao processo turbo.")
    else:
        st.info("Nao ha processo turbo em execucao.")


def _render_turbo_status(refresh_seconds: int) -> None:
    job = st.session_state.get("turbo_job")
    if job is None:
        return

    running = job.process.poll() is None
    st.markdown("### Modo Turbo")
    st.caption(f"PID {job.process.pid} | log: `{job.log_file}`")
    st.code(" ".join(job.command), language="bash")
    tail = _tail_text_file(Path(job.log_file), max_lines=40)
    if tail:
        st.code(tail, language="text")

    if running:
        st.info("Execucao turbo em andamento...")
        time_mod.sleep(max(1, int(refresh_seconds)))
        st.rerun()
        return

    exit_code = int(job.process.poll() or 0)
    if exit_code != 0:
        st.error(f"Turbo finalizou com erro (exit code {exit_code}).")
        st.session_state.turbo_job = None
        return

    try:
        result = _load_dashboard_result_from_outputs(job)
    except Exception as exc:  # pragma: no cover
        st.error(f"Turbo finalizado, mas falhou ao carregar resultados: {exc}")
        st.session_state.turbo_job = None
        return

    st.success("Turbo finalizado com sucesso.")
    st.session_state.last_result = result
    st.session_state.turbo_job = None


def _load_dashboard_result_from_outputs(job: TurboJob) -> DashboardRunResult:
    outputs_root = Path(job.outputs_dir)
    symbol = job.symbol
    base_cfg = BacktestConfig(**job.base_cfg_dict)
    start_ts = pd.Timestamp(job.start_iso)
    end_ts = pd.Timestamp(job.end_iso)

    summary_rows: list[dict[str, Any]] = []
    strategy_runs: list[StrategyRun] = []

    for raw_tf in job.timeframes:
        timeframe = normalize_timeframe_label(raw_tf)
        tf_output = outputs_root / symbol / timeframe
        summary_file = tf_output / f"summary_{timeframe}.csv"
        if not summary_file.exists():
            continue
        summary_df = pd.read_csv(summary_file)
        if summary_df.empty:
            continue
        summary_df = summary_df[summary_df["strategy"].isin(job.strategies)]
        if summary_df.empty:
            continue
        for _, row in summary_df.iterrows():
            strategy_name = str(row["strategy"])
            summary_rows.append(row.to_dict())
            trades = _read_trades_csv(tf_output / f"trades_{timeframe}_{strategy_name}.csv")
            windows = _read_csv_or_empty(tf_output / f"walkforward_windows_{timeframe}_{strategy_name}.csv")
            topk = _read_csv_or_empty(tf_output / f"walkforward_topk_{timeframe}_{strategy_name}.csv")
            equity = _read_csv_or_empty(tf_output / f"equity_curve_{timeframe}_{strategy_name}.csv")
            if not equity.empty and "datetime" in equity.columns:
                equity["datetime"] = pd.to_datetime(equity["datetime"], errors="coerce")

            best_params = {}
            best_file = tf_output / f"best_params_{timeframe}.json"
            if best_file.exists():
                payload = json.loads(best_file.read_text(encoding="utf-8"))
                best_params = (
                    payload.get("strategies", {})
                    .get(strategy_name, {})
                    .get("best_params_from_tests", {})
                )
            metrics = _extract_metrics_from_summary_row(row)
            wf_result = WalkForwardResult(
                strategy_name=strategy_name,
                window_results=windows,
                topk_test_results=topk,
                oos_trades=trades,
                oos_equity=equity,
                consolidated_metrics=metrics,
                best_params_from_tests=best_params if isinstance(best_params, dict) else {},
            )
            strategy_runs.append(
                StrategyRun(
                    timeframe=timeframe,
                    strategy_name=strategy_name,
                    result=wf_result,
                )
            )

    if not summary_rows or not strategy_runs:
        raise ValueError("Nenhum output valido encontrado para o job turbo.")

    summary = pd.DataFrame(summary_rows).sort_values("score", ascending=False).reset_index(drop=True)
    return DashboardRunResult(
        summary=summary,
        strategy_runs=strategy_runs,
        symbol=symbol,
        start=start_ts,
        end=end_ts,
        base_cfg=base_cfg,
    )


def _extract_metrics_from_summary_row(row: pd.Series) -> dict[str, float]:
    ignore = {"timeframe", "strategy", "windows"}
    metrics: dict[str, float] = {}
    for col in row.index:
        if col in ignore:
            continue
        val = row[col]
        if pd.isna(val):
            continue
        try:
            metrics[col] = float(val)
        except (TypeError, ValueError):
            continue
    return metrics


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_trades_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=default_trade_columns())
    trades = pd.read_csv(path)
    for col in ["entry_time", "exit_time"]:
        if col in trades.columns:
            trades[col] = pd.to_datetime(trades[col], errors="coerce")
    if "params" in trades.columns:
        trades["params"] = trades["params"].apply(_safe_json_loads)
    return trades


def _safe_json_loads(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _tail_text_file(path: Path, max_lines: int = 40) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return "\n".join(lines[-max_lines:])

def _run_dashboard(
    symbol: str,
    raw_timeframes: list[str],
    strategy_names: list[str],
    start_date: date,
    end_date: date,
    data_root: Path,
    outputs_root: Path,
    save_outputs: bool,
    base_bt_cfg: BacktestConfig,
    optimizer_cfg: OptimizerConfig,
    wf_cfg: WalkForwardConfig,
    live_updates: bool,
    fast_mode: bool,
    update_every: int,
) -> DashboardRunResult:
    status_box = st.empty()
    progress_box = st.progress(0)
    progress_text = st.empty()

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    loader_cfg = LoaderConfig(
        data_root=data_root,
        symbol=symbol,
        start=start_ts,
        end=end_ts,
    )
    outputs_root.mkdir(parents=True, exist_ok=True)

    normalized_timeframes = [normalize_timeframe_label(tf) for tf in raw_timeframes]
    selected_strategies = [STRATEGIES[name] for name in strategy_names]
    total_jobs = len(normalized_timeframes) * len(selected_strategies)
    run_id = build_run_id(prefix="ui_wf")

    strategy_runs: list[StrategyRun] = []
    summary_rows: list[dict[str, Any]] = []
    generated_files: list[str] = []
    data_hashes: dict[str, str] = {}
    errors: list[str] = []
    finished_jobs = 0

    status_box.markdown("### Executando backtests...")

    for timeframe in normalized_timeframes:
        try:
            df = load_timeframe_data(loader_cfg, timeframe)
        except (FileNotFoundError, ValueError) as exc:
            errors.append(f"{timeframe}: {exc}")
            continue
        tf_output = outputs_root / symbol / timeframe
        if save_outputs:
            tf_output.mkdir(parents=True, exist_ok=True)
            quality_report = build_data_quality_report(df=df, symbol=symbol, timeframe=timeframe)
            quality_file = tf_output / f"data_quality_{timeframe}.json"
            quality_file.write_text(json.dumps(quality_report.to_dict(), indent=2), encoding="utf-8")
            generated_files.append(str(quality_file))
        data_hashes[timeframe] = dataframe_sha256(df)

        for strategy in selected_strategies:
            latest_progress: dict[str, Any] = {}

            def _progress_callback(event: dict[str, Any]) -> None:
                if not live_updates:
                    return
                stage = str(event.get("stage", ""))
                if fast_mode and stage == "optimizer_sample":
                    sample_idx = int(event.get("sample_index", 0))
                    sample_total = int(event.get("samples_total", 0))
                    if sample_idx < sample_total and sample_idx % update_every != 0:
                        return
                latest_progress.clear()
                latest_progress.update(event)
                _render_progress(
                    status_box=status_box,
                    progress_text=progress_text,
                    event=latest_progress,
                    timeframe=timeframe,
                    strategy_name=strategy.name,
                )

            wf_result = run_walkforward(
                df=df,
                strategy=strategy,
                base_config=base_bt_cfg,
                optimizer_config=optimizer_cfg,
                wf_config=wf_cfg,
                progress_callback=_progress_callback if live_updates else None,
            )

            finished_jobs += 1
            progress_box.progress(min(1.0, finished_jobs / max(total_jobs, 1)))
            progress_text.caption(
                f"Concluido {finished_jobs}/{total_jobs}: {symbol} {timeframe} / {strategy.name}"
            )

            strategy_runs.append(StrategyRun(timeframe=timeframe, strategy_name=strategy.name, result=wf_result))
            summary_rows.append(
                {
                    "timeframe": timeframe,
                    "strategy": strategy.name,
                    **wf_result.consolidated_metrics,
                    "windows": int(len(wf_result.window_results)),
                }
            )

            if save_outputs:
                _save_strategy_outputs(
                    symbol=symbol,
                    timeframe=timeframe,
                    strategy_name=strategy.name,
                    wf_result=wf_result,
                    output_dir=tf_output,
                    initial_capital=base_bt_cfg.initial_capital,
                )
                generated_files.extend(
                    [
                        str(tf_output / f"trades_{timeframe}_{strategy.name}.csv"),
                        str(tf_output / f"walkforward_windows_{timeframe}_{strategy.name}.csv"),
                        str(tf_output / f"walkforward_topk_{timeframe}_{strategy.name}.csv"),
                        str(tf_output / f"equity_curve_{timeframe}_{strategy.name}.csv"),
                        str(tf_output / f"equity_{timeframe}_{strategy.name}.png"),
                        str(tf_output / f"monthly_{timeframe}_{strategy.name}.csv"),
                        str(tf_output / f"sensitivity_{timeframe}_{strategy.name}.csv"),
                        str(tf_output / f"robustness_{timeframe}_{strategy.name}.json"),
                    ]
                )

        if save_outputs:
            _save_best_outputs_for_timeframe(
                symbol=symbol,
                timeframe=timeframe,
                strategy_runs=[x for x in strategy_runs if x.timeframe == timeframe],
                output_dir=tf_output,
            )
            generated_files.extend(
                [
                    str(tf_output / f"summary_{timeframe}.csv"),
                    str(tf_output / f"best_params_{timeframe}.json"),
                    str(tf_output / f"equity_curve_{timeframe}_best.csv"),
                    str(tf_output / f"equity_{timeframe}_best.png"),
                ]
            )

    if not strategy_runs:
        raise ValueError("Nenhum resultado valido foi gerado para os filtros selecionados.")

    if save_outputs:
        manifest = RunManifest(
            run_id=run_id,
            created_at_utc=pd.Timestamp.utcnow().isoformat(),
            symbol=symbol,
            start=start_ts.isoformat(),
            end=end_ts.isoformat(),
            timeframes=normalized_timeframes,
            args={
                "raw_timeframes": raw_timeframes,
                "strategies": strategy_names,
                "live_updates": bool(live_updates),
                "fast_mode": bool(fast_mode),
                "update_every": int(update_every),
                "base_backtest_config": asdict(base_bt_cfg),
                "optimizer_config": asdict(optimizer_cfg),
                "walkforward_config": asdict(wf_cfg),
            },
            environment=environment_snapshot(),
            data_hashes=data_hashes,
            generated_files=sorted(set(generated_files)),
            errors=errors,
        )
        manifest_file = outputs_root / symbol / f"run_manifest_{run_id}.json"
        write_manifest(manifest, manifest_file)

    status_box.success("Execucao finalizada.")
    summary_df = pd.DataFrame(summary_rows).sort_values("score", ascending=False).reset_index(drop=True)
    return DashboardRunResult(
        summary=summary_df,
        strategy_runs=strategy_runs,
        symbol=symbol,
        start=start_ts,
        end=end_ts,
        base_cfg=base_bt_cfg,
    )


def _render_progress(
    status_box: Any,
    progress_text: Any,
    event: dict[str, Any],
    timeframe: str,
    strategy_name: str,
) -> None:
    stage = str(event.get("stage", ""))
    if stage == "walkforward_start":
        total = int(event.get("total_windows", 0))
        status_box.markdown(f"### {timeframe} / {strategy_name}: iniciando walk-forward ({total} janelas)")
        return
    if stage == "window_start":
        idx = int(event.get("window_index", 0))
        total = int(event.get("total_windows", 0))
        test_start = str(event.get("test_start", ""))
        test_end = str(event.get("test_end", ""))
        status_box.markdown(
            f"### {timeframe} / {strategy_name}: janela {idx}/{total} | teste {test_start} -> {test_end}"
        )
        return
    if stage == "optimizer_sample":
        idx = int(event.get("sample_index", 0))
        total = int(event.get("samples_total", 0))
        score = float(event.get("score", 0.0))
        progress_text.caption(f"Otimizacao: amostra {idx}/{total} | score treino {score:.2f}")
        return
    if stage == "window_complete":
        idx = int(event.get("window_index", 0))
        total = int(event.get("total_windows", 0))
        score = float(event.get("oos_score", 0.0))
        pnl = float(event.get("oos_net_profit", 0.0))
        progress_text.caption(f"Janela {idx}/{total} concluida | score OOS {score:.2f} | pnl OOS {format_brl(pnl)}")
        return
    if stage == "walkforward_done":
        score = float(event.get("final_score", 0.0))
        pnl = float(event.get("net_profit", 0.0))
        progress_text.caption(f"Consolidado: score {score:.2f} | lucro liquido {format_brl(pnl)}")


def _save_strategy_outputs(
    symbol: str,
    timeframe: str,
    strategy_name: str,
    wf_result: WalkForwardResult,
    output_dir: Path,
    initial_capital: float,
) -> None:
    trades_file = output_dir / f"trades_{timeframe}_{strategy_name}.csv"
    _write_trades_csv(wf_result.oos_trades, trades_file)

    windows_file = output_dir / f"walkforward_windows_{timeframe}_{strategy_name}.csv"
    wf_result.window_results.to_csv(windows_file, index=False)

    topk_file = output_dir / f"walkforward_topk_{timeframe}_{strategy_name}.csv"
    wf_result.topk_test_results.to_csv(topk_file, index=False)

    equity_curve_file = output_dir / f"equity_curve_{timeframe}_{strategy_name}.csv"
    wf_result.oos_equity.to_csv(equity_curve_file, index=False)

    plot_file = output_dir / f"equity_{timeframe}_{strategy_name}.png"
    _save_equity_png(
        wf_result.oos_equity,
        title=f"{symbol} {timeframe} - {strategy_name}",
        output_file=plot_file,
    )

    monthly_file = output_dir / f"monthly_{timeframe}_{strategy_name}.csv"
    monthly_df = build_monthly_report(
        trades=wf_result.oos_trades,
        equity_curve=wf_result.oos_equity,
        initial_capital=initial_capital,
    )
    monthly_df.to_csv(monthly_file, index=False)

    sensitivity_file = output_dir / f"sensitivity_{timeframe}_{strategy_name}.csv"
    sensitivity_df = build_parameter_sensitivity_report(wf_result.topk_test_results)
    sensitivity_df.to_csv(sensitivity_file, index=False)

    robustness_file = output_dir / f"robustness_{timeframe}_{strategy_name}.json"
    robustness_payload = build_robustness_report(
        window_results=wf_result.window_results,
        topk_test_results=wf_result.topk_test_results,
        consolidated_metrics=wf_result.consolidated_metrics,
    )
    robustness_file.write_text(json.dumps(robustness_payload, indent=2), encoding="utf-8")


def _save_best_outputs_for_timeframe(
    symbol: str,
    timeframe: str,
    strategy_runs: list[StrategyRun],
    output_dir: Path,
) -> None:
    if not strategy_runs:
        return

    rows = []
    for item in strategy_runs:
        rows.append(
            {
                "timeframe": timeframe,
                "strategy": item.strategy_name,
                **item.result.consolidated_metrics,
                "windows": int(len(item.result.window_results)),
            }
        )
    summary_df = pd.DataFrame(rows).sort_values("score", ascending=False)
    summary_df.to_csv(output_dir / f"summary_{timeframe}.csv", index=False)
    if summary_df.empty:
        return

    best_name = str(summary_df.iloc[0]["strategy"])
    best_run = next(x for x in strategy_runs if x.strategy_name == best_name)
    payload = {
        "symbol": symbol,
        "timeframe": timeframe,
        "best_strategy": best_name,
        "best_score": float(summary_df.iloc[0]["score"]),
        "strategies": {
            run.strategy_name: {
                "best_params_from_tests": run.result.best_params_from_tests,
                "consolidated_metrics": run.result.consolidated_metrics,
            }
            for run in strategy_runs
        },
    }
    (output_dir / f"best_params_{timeframe}.json").write_text(
        json.dumps(payload, indent=2, default=_json_default),
        encoding="utf-8",
    )
    best_run.result.oos_equity.to_csv(output_dir / f"equity_curve_{timeframe}_best.csv", index=False)
    _save_equity_png(
        best_run.result.oos_equity,
        title=f"{symbol} {timeframe} - best: {best_name}",
        output_file=output_dir / f"equity_{timeframe}_best.png",
    )


def _write_trades_csv(trades: pd.DataFrame, output_file: Path) -> None:
    if trades.empty:
        pd.DataFrame(columns=default_trade_columns()).to_csv(output_file, index=False)
        return
    out = trades.copy()
    out["params"] = out["params"].apply(lambda p: json.dumps(p, sort_keys=True))
    out.to_csv(output_file, index=False)


def _save_equity_png(equity_df: pd.DataFrame, title: str, output_file: Path) -> None:
    if equity_df.empty:
        return
    plot_df = equity_df.sort_values("datetime")
    x = pd.to_datetime(plot_df["datetime"])
    y = plot_df["equity"].astype(float)
    plt.figure(figsize=(12, 6))
    plt.plot(x, y, linewidth=1.6, color="#25f08a")
    plt.fill_between(x, y, y2=min(0.0, float(y.min())), color="#25f08a", alpha=0.18)
    plt.title(title)
    plt.xlabel("Data")
    plt.ylabel("Saldo Bruto (R$)")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_file, dpi=120)
    plt.close()

def _render_result(result: DashboardRunResult) -> None:
    st.subheader("Consolidado")
    st.dataframe(
        result.summary,
        use_container_width=True,
        hide_index=True,
    )

    timeframe_options = sorted(result.summary["timeframe"].unique().tolist())
    selected_tf = st.selectbox("Timeframe", options=timeframe_options, index=0)
    strategy_options = result.summary[result.summary["timeframe"] == selected_tf]["strategy"].tolist()
    selected_strategy = st.selectbox("Estrategia", options=strategy_options, index=0)

    selected_run = next(
        x
        for x in result.strategy_runs
        if x.timeframe == selected_tf and x.strategy_name == selected_strategy
    )
    trades = selected_run.result.oos_trades.copy()
    equity = selected_run.result.oos_equity.copy()

    monthly_df = build_monthly_report(
        trades=trades,
        equity_curve=equity,
        initial_capital=result.base_cfg.initial_capital,
    )
    robustness_payload = build_robustness_report(
        window_results=selected_run.result.window_results,
        topk_test_results=selected_run.result.topk_test_results,
        consolidated_metrics=selected_run.result.consolidated_metrics,
    )
    sensitivity_df = build_parameter_sensitivity_report(selected_run.result.topk_test_results)

    tabs = st.tabs(["Resumo", "Operacoes", "Grafico de Operacoes", "Patrimonio", "Mensal", "Robustez"])

    with tabs[0]:
        summary = _build_summary_snapshot(trades=trades, equity=equity, base_cfg=result.base_cfg)
        c1, c2 = st.columns(2)
        with c1:
            _render_kv("Saldo Liquido Total", format_brl(summary["net_profit"]))
            _render_kv("Lucro Bruto", format_brl(summary["gross_profit"]))
            _render_kv("Fator de Lucro", f"{summary['profit_factor']:.2f}")
            _render_kv("Numero de Operacoes", format_int(summary["trade_count"]))
            _render_kv("Operacoes Vencedoras", format_int(summary["wins"]))
            _render_kv("Operacoes Perdedoras", format_int(summary["losses"]))
            _render_kv("Media por Operacao", format_brl(summary["avg_trade"]))
            _render_kv("Maior Operacao Vencedora", format_brl(summary["max_win"]))
            _render_kv("Maior Sequencia Vencedora", format_int(summary["max_win_streak"]))
            _render_kv("Tempo Medio Op. Vencedoras", summary["avg_win_duration"])
            _render_kv("Patrimonio Maximo", format_brl(summary["equity_peak"]))
        with c2:
            _render_kv("Saldo Total", format_brl(summary["ending_capital"]))
            _render_kv("Prejuizo Bruto", format_brl(summary["gross_loss"]))
            _render_kv("Custos", format_brl(summary["total_costs"]))
            _render_kv("Percentual de Operacoes Vencedoras", format_pct(summary["win_rate"]))
            _render_kv("Operacoes Zeradas", format_int(summary["flat_trades"]))
            _render_kv("Razao Media Lucro/Prejuizo", f"{summary['avg_win_over_avg_loss']:.2f}")
            _render_kv("Media Op. Perdedoras", format_brl(summary["avg_loss"]))
            _render_kv("Maior Operacao Perdedora", format_brl(summary["max_loss"]))
            _render_kv("Maior Sequencia Perdedor", format_int(summary["max_loss_streak"]))
            _render_kv("Tempo Medio Op. Perdedoras", summary["avg_loss_duration"])
            _render_kv("Drawdown Maximo", format_brl(summary["max_drawdown"]))
        st.markdown("---")
        st.caption(
            f"Periodo {result.start.date()} a {result.end.date()} | "
            f"Ativo {result.symbol} | Timeframe {selected_tf} | Estrategia {selected_strategy}"
        )

    with tabs[1]:
        operations_df = _build_operations_table(
            trades=trades,
            initial_capital=result.base_cfg.initial_capital,
            contracts=result.base_cfg.contracts,
        )
        if operations_df.empty:
            st.warning("Sem operacoes para os filtros selecionados.")
        else:
            styled = operations_df.style.apply(
                lambda row: _row_result_style(float(row["Resultado_Num"])),
                axis=1,
            ).hide(axis="columns", subset=["Resultado_Num", "Total_Num"])
            st.dataframe(
                styled,
                use_container_width=True,
                hide_index=True,
            )

    with tabs[2]:
        op_fig = _build_operations_chart(trades)
        st.plotly_chart(op_fig, use_container_width=True)

    with tabs[3]:
        eq_fig = _build_equity_chart(
            equity=equity,
            symbol=result.symbol,
            timeframe=selected_tf,
            strategy_name=selected_strategy,
        )
        st.plotly_chart(eq_fig, use_container_width=True)

    with tabs[4]:
        if monthly_df.empty:
            st.info("Sem dados mensais para o periodo selecionado.")
        else:
            st.dataframe(monthly_df, use_container_width=True, hide_index=True)
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=monthly_df["month"],
                    y=monthly_df["net_profit"],
                    marker_color=["#22d481" if x >= 0 else "#ff4f5e" for x in monthly_df["net_profit"]],
                    name="PnL mensal",
                )
            )
            fig.update_layout(
                title="Lucro Liquido Mensal",
                template="plotly_dark",
                paper_bgcolor="#1e1f22",
                plot_bgcolor="#1e1f22",
                yaxis_title="R$",
            )
            st.plotly_chart(fig, use_container_width=True)

    with tabs[5]:
        alerts = robustness_payload.get("alerts", [])
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Janelas", int(robustness_payload.get("total_windows", 0)))
            st.metric("Win windows %", f"{float(robustness_payload.get('positive_oos_windows_pct', 0.0)):.2f}%")
        with c2:
            st.metric("Score OOS medio", f"{float(robustness_payload.get('oos_score_mean', 0.0)):.2f}")
            st.metric("Score OOS std", f"{float(robustness_payload.get('oos_score_std', 0.0)):.2f}")
        with c3:
            st.metric("Corr treino/teste", f"{float(robustness_payload.get('train_test_score_corr', 0.0)):.2f}")
            dom_ratio = float(
                robustness_payload.get("parameter_stability", {}).get("dominant_ratio", 0.0)
            )
            st.metric("Dominancia params", f"{100.0 * dom_ratio:.1f}%")

        if alerts:
            for item in alerts:
                sev = str(item.get("severity", "low")).upper()
                code = str(item.get("code", "ALERT"))
                msg = str(item.get("message", ""))
                if sev == "HIGH":
                    st.error(f"[{code}] {msg}")
                elif sev == "MEDIUM":
                    st.warning(f"[{code}] {msg}")
                else:
                    st.info(f"[{code}] {msg}")
        else:
            st.success("Sem alertas de robustez para o recorte selecionado.")

        if not sensitivity_df.empty:
            st.markdown("**Sensibilidade de Parametros (top-k em teste)**")
            st.dataframe(sensitivity_df, use_container_width=True, hide_index=True)


def _build_summary_snapshot(
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    base_cfg: BacktestConfig,
) -> dict[str, float | int | str]:
    if trades.empty:
        return {
            "trade_count": 0,
            "wins": 0,
            "losses": 0,
            "flat_trades": 0,
            "net_profit": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "profit_factor": 0.0,
            "avg_trade": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "avg_win_over_avg_loss": 0.0,
            "max_win": 0.0,
            "max_loss": 0.0,
            "max_win_streak": 0,
            "max_loss_streak": 0,
            "avg_win_duration": "0min",
            "avg_loss_duration": "0min",
            "total_costs": 0.0,
            "max_drawdown": _max_drawdown_from_equity(equity),
            "win_rate": 0.0,
            "ending_capital": base_cfg.initial_capital,
            "equity_peak": float(base_cfg.initial_capital),
        }

    pnl = trades["pnl_net"].astype(float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    flat = pnl[pnl == 0]

    durations = _trade_durations(trades)
    win_durations = durations[pnl > 0]
    loss_durations = durations[pnl < 0]

    avg_win = float(wins.mean()) if not wins.empty else 0.0
    avg_loss = float(losses.mean()) if not losses.empty else 0.0
    ratio = (avg_win / abs(avg_loss)) if avg_loss != 0 else 0.0

    ending_capital = float(base_cfg.initial_capital + pnl.sum())
    equity_peak = float(equity["equity"].max()) if not equity.empty else ending_capital
    gross_profit = float(wins.sum()) if not wins.empty else 0.0
    gross_loss = float(losses.sum()) if not losses.empty else 0.0
    profit_factor = abs(gross_profit / gross_loss) if gross_loss < 0 else (999.0 if gross_profit > 0 else 0.0)

    return {
        "trade_count": int(len(pnl)),
        "wins": int((pnl > 0).sum()),
        "losses": int((pnl < 0).sum()),
        "flat_trades": int(len(flat)),
        "net_profit": float(pnl.sum()),
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": float(profit_factor),
        "avg_trade": float(pnl.mean()),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "avg_win_over_avg_loss": float(ratio),
        "max_win": float(wins.max()) if not wins.empty else 0.0,
        "max_loss": float(losses.min()) if not losses.empty else 0.0,
        "max_win_streak": _max_streak(pnl, positive=True),
        "max_loss_streak": _max_streak(pnl, positive=False),
        "avg_win_duration": _format_timedelta_seconds(float(win_durations.mean())) if not win_durations.empty else "0min",
        "avg_loss_duration": _format_timedelta_seconds(float(loss_durations.mean())) if not loss_durations.empty else "0min",
        "total_costs": float(trades["costs"].astype(float).sum()) if "costs" in trades else 0.0,
        "max_drawdown": _max_drawdown_from_equity(equity),
        "win_rate": float((pnl > 0).mean()),
        "ending_capital": ending_capital,
        "equity_peak": equity_peak,
    }

def _build_operations_table(
    trades: pd.DataFrame,
    initial_capital: float,
    contracts: int,
) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()

    out = trades.copy()
    out["entry_time"] = pd.to_datetime(out["entry_time"])
    out["exit_time"] = pd.to_datetime(out["exit_time"])
    out = out.sort_values("exit_time").reset_index(drop=True)
    out["duration"] = out["exit_time"] - out["entry_time"]
    out["Lado"] = out["direction"].map({"long": "C", "short": "V"}).fillna("-")
    out["Qtd"] = contracts

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
    out["Resultado(%)"] = (100.0 * out["Resultado_Num"] / max(initial_capital, 1e-9)).map(lambda x: f"{x:.2f}%")
    out["Total_Num"] = initial_capital + out["Resultado_Num"].cumsum()
    out["Total"] = out["Total_Num"].apply(format_brl)
    out["Tempo Op"] = out["duration"].dt.total_seconds().apply(_format_timedelta_seconds)
    out["Resultado(pts)"] = out["pnl_points"].astype(float).map(lambda x: f"{x:.2f}")

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
    show = show.sort_values("exit_time", ascending=False).reset_index(drop=True)
    show.rename(
        columns={
            "entry_time": "Abertura",
            "exit_time": "Fechamento",
            "Preco Compra": "Preco Compra",
            "Preco Venda": "Preco Venda",
        },
        inplace=True,
    )
    show["Abertura"] = show["Abertura"].dt.strftime("%d/%m/%Y %H:%M:%S")
    show["Fechamento"] = show["Fechamento"].dt.strftime("%d/%m/%Y %H:%M:%S")
    show["Preco Compra"] = show["Preco Compra"].map(lambda x: f"{float(x):.3f}")
    show["Preco Venda"] = show["Preco Venda"].map(lambda x: f"{float(x):.3f}")
    return show


def _build_operations_chart(trades: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if trades.empty:
        fig.update_layout(
            title="Grafico de Operacoes",
            template="plotly_dark",
            paper_bgcolor="#1e1f22",
            plot_bgcolor="#1e1f22",
        )
        return fig

    plot_df = trades.copy()
    plot_df["exit_time"] = pd.to_datetime(plot_df["exit_time"])
    plot_df = plot_df.sort_values("exit_time")
    colors = ["#22d481" if x >= 0 else "#ff4f5e" for x in plot_df["pnl_net"].astype(float)]
    fig.add_trace(
        go.Bar(
            x=plot_df["exit_time"],
            y=plot_df["pnl_net"],
            marker_color=colors,
            name="Resultado",
        )
    )
    fig.update_layout(
        title="Grafico de Operacoes",
        template="plotly_dark",
        paper_bgcolor="#1e1f22",
        plot_bgcolor="#1e1f22",
        font=dict(family="Barlow Condensed, Segoe UI, sans-serif", color="#f5f5f5"),
        xaxis_title="Data/Hora",
        yaxis_title="Saldo Bruto (R$)",
        bargap=0.05,
    )
    return fig


def _build_equity_chart(
    equity: pd.DataFrame,
    symbol: str,
    timeframe: str,
    strategy_name: str,
) -> go.Figure:
    fig = go.Figure()
    if equity.empty:
        fig.update_layout(
            title=f"{symbol} {timeframe} - {strategy_name}",
            template="plotly_dark",
            paper_bgcolor="#1e1f22",
            plot_bgcolor="#1e1f22",
        )
        return fig

    plot_df = equity.copy()
    plot_df["datetime"] = pd.to_datetime(plot_df["datetime"])
    plot_df = plot_df.sort_values("datetime")
    fig.add_trace(
        go.Scatter(
            x=plot_df["datetime"],
            y=plot_df["equity"],
            mode="lines",
            line=dict(color="#22f08a", width=2),
            fill="tozeroy",
            fillcolor="rgba(34, 240, 138, 0.25)",
            name=symbol,
        )
    )
    fig.update_layout(
        title="Patrimonio",
        template="plotly_dark",
        paper_bgcolor="#1e1f22",
        plot_bgcolor="#1e1f22",
        font=dict(family="Barlow Condensed, Segoe UI, sans-serif", color="#f5f5f5"),
        xaxis_title=None,
        yaxis_title="Saldo Bruto (R$)",
        legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="left", x=0),
    )
    return fig


def _row_result_style(result_value: float) -> list[str]:
    if result_value > 0:
        return ["background-color: rgba(34, 212, 129, 0.16); color: #9bffd4;"] * 13
    if result_value < 0:
        return ["background-color: rgba(255, 79, 94, 0.16); color: #ffc2c8;"] * 13
    return [""] * 13


def _trade_durations(trades: pd.DataFrame) -> pd.Series:
    entry = pd.to_datetime(trades["entry_time"], errors="coerce")
    exit_ = pd.to_datetime(trades["exit_time"], errors="coerce")
    return (exit_ - entry).dt.total_seconds().fillna(0.0)


def _max_streak(pnl: pd.Series, positive: bool) -> int:
    best = 0
    current = 0
    for value in pnl.astype(float):
        cond = value > 0 if positive else value < 0
        if cond:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def _max_drawdown_from_equity(equity: pd.DataFrame) -> float:
    if equity.empty or "equity" not in equity:
        return 0.0
    series = equity["equity"].astype(float)
    if series.empty:
        return 0.0
    running_peak = series.cummax()
    drawdown = running_peak - series
    return float(drawdown.max()) if not drawdown.empty else 0.0


def _format_timedelta_seconds(seconds: float) -> str:
    total = int(max(seconds, 0))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    if hours > 0:
        return f"{hours}h{minutes:02d}min{secs:02d}s"
    if minutes > 0:
        return f"{minutes}min{secs:02d}s"
    return f"{secs}s"

def _render_kv(label: str, value: str) -> None:
    st.markdown(
        (
            "<div class='kv-row'>"
            f"<span class='kv-label'>{label}</span>"
            f"<span class='kv-value'>{value}</span>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, date)):
        return value.isoformat()
    return str(value)


def format_brl(value: float) -> str:
    sign = "-" if value < 0 else ""
    raw = f"{abs(value):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{sign}R$ {raw}"


def format_pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def format_int(value: float | int) -> str:
    return f"{int(value):,}".replace(",", ".")


def _init_session_state() -> None:
    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("turbo_job", None)
    st.session_state.setdefault("symbol_default", "WINFUT")
    st.session_state.setdefault("timeframes_default", ["5m"])
    st.session_state.setdefault("strategies_default", list(STRATEGIES.keys()))
    st.session_state.setdefault("start_date_default", date(2025, 1, 17))
    st.session_state.setdefault("end_date_default", date(2026, 1, 16))


def _inject_compact_sidebar_css() -> None:
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] .stExpander {
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 8px;
            margin-bottom: 0.35rem;
            background: rgba(255,255,255,0.02);
        }
        section[data-testid="stSidebar"] .stExpander details summary {
            font-size: 0.82rem !important;
            font-weight: 700 !important;
            padding-top: 0.15rem;
            padding-bottom: 0.15rem;
        }
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] .stMarkdown,
        section[data-testid="stSidebar"] .stRadio {
            font-size: 0.78rem !important;
        }
        section[data-testid="stSidebar"] .stNumberInput,
        section[data-testid="stSidebar"] .stTextInput,
        section[data-testid="stSidebar"] .stDateInput,
        section[data-testid="stSidebar"] .stSelectbox,
        section[data-testid="stSidebar"] .stMultiSelect {
            margin-bottom: 0.15rem !important;
        }
        section[data-testid="stSidebar"] .st-emotion-cache-1r6slb0,
        section[data-testid="stSidebar"] .st-emotion-cache-16txtl3 {
            padding-top: 0.4rem !important;
        }
        section[data-testid="stSidebar"] .stSlider {
            padding-top: 0.1rem;
            padding-bottom: 0.2rem;
        }
        section[data-testid="stSidebar"] .stButton > button {
            height: 2.1rem;
            font-size: 0.9rem;
            padding-top: 0.25rem;
            padding-bottom: 0.25rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at 20% 0%, rgba(36, 80, 62, 0.55), rgba(19, 19, 19, 0) 35%),
                radial-gradient(circle at 100% 100%, rgba(45, 45, 45, 0.7), rgba(15, 15, 15, 0) 40%),
                linear-gradient(135deg, #101114, #1a1b1f 35%, #101114 100%);
            color: #ececec;
            font-family: "Barlow Condensed", "Segoe UI", sans-serif;
        }
        .app-title {
            letter-spacing: 0.08em;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }
        .panel-title {
            margin-top: 0.8rem;
            margin-bottom: 0.3rem;
            padding: 0.2rem 0.5rem;
            border-left: 3px solid #22f08a;
            font-weight: 700;
            text-transform: uppercase;
            font-size: 0.85rem;
            background: rgba(255, 255, 255, 0.04);
        }
        .kv-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 6px;
            padding: 0.25rem 0.5rem;
            margin-bottom: 0.25rem;
        }
        .kv-label {
            color: #bec3cc;
            font-size: 0.95rem;
        }
        .kv-value {
            color: #22f08a;
            font-weight: 700;
            font-size: 0.95rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
