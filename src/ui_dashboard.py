
"""Visual dashboard for running walk-forward backtests in real time."""

from __future__ import annotations

import json
import re
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
    build_hourly_report,
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
    outputs_root: str


@dataclass(slots=True)
class TurboJob:
    """Background CLI execution tracked by the dashboard."""

    process: Any
    command: list[str]
    log_file: str
    stop_file: str
    symbol: str
    timeframes: list[str]
    strategies: list[str]
    outputs_dir: str
    start_iso: str
    end_iso: str
    base_cfg_dict: dict[str, Any]


SCORE_PRESETS: dict[str, dict[str, float | int]] = {
    "conservador": {
        "drawdown_weight": 2.4,
        "min_trades": 35,
        "penalty_missing": 220.0,
    },
    "equilibrado": {
        "drawdown_weight": 1.5,
        "min_trades": 20,
        "penalty_missing": 150.0,
    },
    "agressivo": {
        "drawdown_weight": 0.9,
        "min_trades": 8,
        "penalty_missing": 70.0,
    },
}


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
                key="ui_selected_timeframes",
            )
            strategies = st.multiselect(
                "Estrategias",
                options=list(STRATEGIES.keys()),
                default=st.session_state.strategies_default,
                key="ui_selected_strategies",
            )
            execution_mode = st.radio("Modo", options=["OHLC", "Tick a Tick"], index=0)
            tick_data_root = st.text_input(
                "Pasta Tick a Tick",
                value="assets/dados-historicos-winfut/tik-a-tik",
                disabled=not execution_mode.startswith("Tick"),
            )
            load_validated_positive = st.button("Usar validadas + positivas", use_container_width=True)
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
                max_stop_points = st.number_input("Stop max (pts)", min_value=0.0, max_value=5000.0, value=0.0, step=10.0)
                max_daily_loss = st.number_input("Perda diaria max (R$)", min_value=0.0, max_value=1_000_000.0, value=0.0, step=100.0)
            with c2:
                test_days = st.number_input("Test", min_value=5, max_value=250, value=30, step=5)
                top_k = st.number_input("Top K", min_value=1, max_value=50, value=5, step=1)
                max_drawdown_pct_hard = st.number_input("DD max hard (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.5)
                optimize_hours = st.checkbox("Otimizar horario", value=True)
                h1, h2 = st.columns(2)
                with h1:
                    hour_start_min = st.number_input("Hora ini min", min_value=0, max_value=23, value=9, step=1)
                    hour_end_min = st.number_input("Hora fim min", min_value=0, max_value=23, value=10, step=1)
                with h2:
                    hour_start_max = st.number_input("Hora ini max", min_value=0, max_value=23, value=16, step=1)
                    hour_end_max = st.number_input("Hora fim max", min_value=0, max_value=23, value=18, step=1)

        with st.expander("Score", expanded=False):
            p1, p2, p3 = st.columns(3)
            if p1.button("Conservador", use_container_width=True, key="score_preset_conservador_btn"):
                _apply_score_preset("conservador")
            if p2.button("Equilibrado", use_container_width=True, key="score_preset_equilibrado_btn"):
                _apply_score_preset("equilibrado")
            if p3.button("Agressivo", use_container_width=True, key="score_preset_agressivo_btn"):
                _apply_score_preset("agressivo")
            drawdown_weight = st.number_input(
                "Peso DD",
                min_value=0.0,
                max_value=20.0,
                value=1.5,
                step=0.1,
                key="score_drawdown_weight",
            )
            min_trades = st.number_input(
                "Min trades",
                min_value=0,
                max_value=1000,
                value=20,
                step=1,
                key="score_min_trades",
            )
            penalty_missing = st.number_input(
                "Penalidade trade faltante",
                min_value=0.0,
                max_value=5000.0,
                value=150.0,
                step=10.0,
                key="score_penalty_missing",
            )
            st.caption("Score = lucro liquido - (peso_dd * drawdown) - (trades_faltantes * penalidade).")

        with st.expander("Risco", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                initial_capital = st.number_input("Capital", min_value=1000.0, max_value=50_000_000.0, value=100_000.0, step=1000.0)
                point_value = st.number_input("Valor ponto", min_value=0.01, max_value=100.0, value=0.2, step=0.01)
            with c2:
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
            turbo_loop_enabled = st.checkbox("Loop turbo continuo", value=st.session_state.turbo_loop_enabled)
            turbo_loop_pause_sec = st.slider(
                "Loop pausa (s)",
                min_value=1,
                max_value=60,
                value=int(st.session_state.turbo_loop_pause_sec),
            )
            turbo_loop_max_cycles = st.number_input(
                "Loop max ciclos (0=infinito)",
                min_value=0,
                max_value=100_000,
                value=int(st.session_state.turbo_loop_max_cycles),
                step=1,
            )
            save_outputs = st.checkbox("Salvar CSV/JSON/PNG", value=True)
            outputs_dir = st.text_input("Pasta outputs", value="outputs_master")
            data_root = st.text_input("Pasta data", value="data")

        run_clicked = st.button("Executar", type="primary", use_container_width=True)
        run_max_clicked = st.button("Executar MAX 24h", use_container_width=True)
        stop_clicked = st.button("Parar turbo", use_container_width=True)

    selected_timeframes = list(st.session_state.get("ui_selected_timeframes", selected_timeframes))
    strategies = list(st.session_state.get("ui_selected_strategies", strategies))

    if load_validated_positive:
        positive_strategies = _load_validated_positive_strategies(
            outputs_root=Path(outputs_dir),
            symbol=symbol.strip() or "WINFUT",
            selected_timeframes=selected_timeframes,
        )
        if positive_strategies:
            st.session_state.ui_selected_strategies = positive_strategies
            strategies = positive_strategies
            st.success(f"Estrategias carregadas: {', '.join(positive_strategies)}")
            st.rerun()
        else:
            st.warning("Nenhuma estrategia validada+positiva encontrada para os timeframes selecionados.")

    if stop_clicked:
        _stop_turbo_job()

    if run_clicked or run_max_clicked:
        if run_max_clicked:
            turbo_mode = True
            live_updates = False
            fast_mode = True
            update_every = 50
            turbo_train_step = 2
            turbo_log_every = 25
            turbo_skip_plots = True
            turbo_loop_enabled = True
            turbo_loop_pause_sec = 15
            turbo_loop_max_cycles = 0
            save_outputs = True

            train_days = 120
            test_days = 30
            samples = 500
            top_k = 12
            drawdown_weight = 1.8
            min_trades = 25
            max_stop_points = 350.0
            max_daily_loss = 1500.0
            max_drawdown_pct_hard = 12.0
            optimize_hours = True
            hour_start_min = 9
            hour_start_max = 16
            hour_end_min = 10
            hour_end_max = 18

            if not selected_timeframes:
                selected_timeframes = ["5m"]
            if not strategies:
                strategies = list(STRATEGIES.keys())

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
                execution_mode="tick" if execution_mode.startswith("Tick") else "ohlc",
                tick_data_root=(tick_data_root.strip() or None) if execution_mode.startswith("Tick") else None,
                tick_symbol=symbol.strip() or "WINFUT",
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
                max_stop_points=float(max_stop_points),
                max_daily_loss=float(max_daily_loss),
                max_drawdown_pct_hard=float(max_drawdown_pct_hard),
                optimize_hours=bool(optimize_hours),
                hour_start_min=int(hour_start_min),
                hour_start_max=int(hour_start_max),
                hour_end_min=int(hour_end_min),
                hour_end_max=int(hour_end_max),
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
                    st.session_state.turbo_loop_enabled = bool(turbo_loop_enabled)
                    st.session_state.turbo_loop_pause_sec = int(turbo_loop_pause_sec)
                    st.session_state.turbo_loop_max_cycles = int(turbo_loop_max_cycles)
                    st.session_state.turbo_loop_cycles_done = 0
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


def _load_validated_positive_strategies(
    outputs_root: Path,
    symbol: str,
    selected_timeframes: list[str],
) -> list[str]:
    valid_names = set(STRATEGIES.keys())
    base = outputs_root / symbol
    if not base.exists():
        return []

    if selected_timeframes:
        normalized_tfs = [normalize_timeframe_label(tf) for tf in selected_timeframes]
    else:
        normalized_tfs = [p.name for p in base.iterdir() if p.is_dir()]

    picks: set[str] = set()
    for tf in normalized_tfs:
        tf_dir = base / tf
        if not tf_dir.exists():
            continue

        history_file = tf_dir / f"best_history_{tf}.csv"
        if history_file.exists():
            try:
                hist = pd.read_csv(history_file)
            except Exception:
                hist = pd.DataFrame()
            if not hist.empty and {"best_strategy", "best_net_profit"}.issubset(hist.columns):
                local = hist.copy()
                local["best_net_profit"] = pd.to_numeric(local["best_net_profit"], errors="coerce")
                local = local[local["best_net_profit"] > 0]
                for name in local["best_strategy"].dropna().astype(str):
                    if name in valid_names:
                        picks.add(name)

        summary_file = tf_dir / f"summary_{tf}.csv"
        if summary_file.exists():
            try:
                summary_df = pd.read_csv(summary_file)
            except Exception:
                summary_df = pd.DataFrame()
            if not summary_df.empty and {"strategy", "net_profit"}.issubset(summary_df.columns):
                local = summary_df.copy()
                local["net_profit"] = pd.to_numeric(local["net_profit"], errors="coerce")
                local = local[local["net_profit"] > 0]
                for name in local["strategy"].dropna().astype(str):
                    if name in valid_names:
                        picks.add(name)

    return sorted(picks)


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
    stop_file = log_dir / f"{run_id}.stop"
    if stop_file.exists():
        stop_file.unlink(missing_ok=True)

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
        "--stop-file",
        str(stop_file),
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
        "--max-stop-points",
        str(float(max(0.0, optimizer_cfg.max_stop_points))),
        "--max-daily-loss",
        str(float(max(0.0, optimizer_cfg.max_daily_loss))),
        "--max-drawdown-pct-hard",
        str(float(max(0.0, optimizer_cfg.max_drawdown_pct_hard))),
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
    if optimizer_cfg.optimize_hours:
        cmd.extend(
            [
                "--optimize-hours",
                "--hour-start-min",
                str(int(optimizer_cfg.hour_start_min)),
                "--hour-start-max",
                str(int(optimizer_cfg.hour_start_max)),
                "--hour-end-min",
                str(int(optimizer_cfg.hour_end_min)),
                "--hour-end-max",
                str(int(optimizer_cfg.hour_end_max)),
            ]
        )
    if base_bt_cfg.session_start:
        cmd.extend(["--session-start", str(base_bt_cfg.session_start)])
    if base_bt_cfg.session_end:
        cmd.extend(["--session-end", str(base_bt_cfg.session_end)])
    cmd.extend(["--execution-mode", str(base_bt_cfg.execution_mode)])
    if str(base_bt_cfg.execution_mode) == "tick" and base_bt_cfg.tick_data_root:
        cmd.extend(["--tick-data-root", str(base_bt_cfg.tick_data_root)])
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
        stop_file=str(stop_file),
        symbol=symbol,
        timeframes=list(raw_timeframes),
        strategies=list(strategy_names),
        outputs_dir=str(outputs_root),
        start_iso=pd.Timestamp(start_date).isoformat(),
        end_iso=(pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)).isoformat(),
        base_cfg_dict=asdict(base_bt_cfg),
    )
    st.success(f"Turbo iniciado (PID {process.pid}).")


def _restart_turbo_job(previous_job: TurboJob) -> None:
    outputs_root = Path(previous_job.outputs_dir).resolve()
    run_id = build_run_id(prefix="turbo_ui")
    log_dir = outputs_root / previous_job.symbol / "_turbo_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{run_id}.log"
    stop_file = log_dir / f"{run_id}.stop"
    if stop_file.exists():
        stop_file.unlink(missing_ok=True)
    cmd = list(previous_job.command)
    if "--stop-file" in cmd:
        idx = cmd.index("--stop-file")
        if idx + 1 < len(cmd):
            cmd[idx + 1] = str(stop_file)
    else:
        cmd.extend(["--stop-file", str(stop_file)])
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
        stop_file=str(stop_file),
        symbol=previous_job.symbol,
        timeframes=list(previous_job.timeframes),
        strategies=list(previous_job.strategies),
        outputs_dir=previous_job.outputs_dir,
        start_iso=previous_job.start_iso,
        end_iso=previous_job.end_iso,
        base_cfg_dict=dict(previous_job.base_cfg_dict),
    )
    st.success(f"Novo ciclo turbo iniciado (PID {process.pid}).")


def _stop_turbo_job() -> None:
    job = st.session_state.get("turbo_job")
    st.session_state.turbo_loop_enabled = False
    if job is None:
        return
    if job.process.poll() is None:
        stop_path = Path(job.stop_file)
        if not stop_path.exists():
            stop_path.write_text("stop", encoding="utf-8")
            st.warning("Parada graciosa solicitada. Clique novamente para forcar encerramento.")
            return
        job.process.terminate()
        st.warning("Processo turbo encerrado a forca.")
    else:
        st.info("Nao ha processo turbo em execucao.")


def _render_turbo_status(refresh_seconds: int) -> None:
    job = st.session_state.get("turbo_job")
    if job is None:
        return

    running = job.process.poll() is None
    st.markdown("### Modo Turbo")
    st.caption(f"PID {job.process.pid} | log: `{job.log_file}`")
    if Path(job.stop_file).exists() and running:
        st.caption("Parada solicitada: aguardando processo finalizar checkpoint atual.")
    if st.session_state.turbo_loop_enabled:
        max_cycles = int(st.session_state.turbo_loop_max_cycles)
        done = int(st.session_state.turbo_loop_cycles_done)
        target = "infinito" if max_cycles <= 0 else str(max_cycles)
        st.caption(f"Loop ativo: ciclo {done}/{target}")
    st.code(" ".join(job.command), language="bash")
    tail = _tail_text_file(Path(job.log_file), max_lines=40)
    if tail:
        st.code(tail, language="text")
    live_board = _build_turbo_live_board(job)
    _render_live_board(st, live_board, title="Ranking em Tempo Real (turbo)")

    if running:
        st.info("Execucao turbo em andamento...")
        time_mod.sleep(max(1, int(refresh_seconds)))
        st.rerun()
        return

    exit_code = int(job.process.poll() or 0)
    if exit_code != 0:
        if _schedule_next_turbo_cycle(
            job=job,
            info_message=f"Ciclo finalizou com erro (exit code {exit_code}). Tentando proximo ciclo...",
        ):
            return
        st.error(f"Turbo finalizou com erro (exit code {exit_code}).")
        st.session_state.turbo_job = None
        return

    try:
        result = _load_dashboard_result_from_outputs(job)
    except Exception as exc:  # pragma: no cover
        if _schedule_next_turbo_cycle(
            job=job,
            info_message=f"Turbo finalizado, mas sem output valido ({exc}). Reiniciando proximo ciclo...",
        ):
            return
        st.error(f"Turbo finalizado, mas falhou ao carregar resultados: {exc}")
        st.session_state.turbo_job = None
        return

    st.success("Turbo finalizado com sucesso.")
    st.session_state.last_result = result
    if _schedule_next_turbo_cycle(job=job, info_message=None):
        return
    st.session_state.turbo_job = None


def _schedule_next_turbo_cycle(job: TurboJob, info_message: str | None) -> bool:
    """Restart turbo job when loop mode is enabled.

    Returns True when flow was handled (either restarted or stopped on loop limit).
    """
    if not bool(st.session_state.turbo_loop_enabled):
        return False
    st.session_state.turbo_loop_cycles_done = int(st.session_state.turbo_loop_cycles_done) + 1
    max_cycles = int(st.session_state.turbo_loop_max_cycles)
    done = int(st.session_state.turbo_loop_cycles_done)
    if max_cycles > 0 and done >= max_cycles:
        st.info(f"Loop finalizado no limite configurado ({done}/{max_cycles}).")
        st.session_state.turbo_loop_enabled = False
        st.session_state.turbo_job = None
        return True
    pause_sec = max(1, int(st.session_state.turbo_loop_pause_sec))
    if info_message:
        st.warning(info_message)
    st.info(f"Iniciando proximo ciclo em {pause_sec}s...")
    time_mod.sleep(pause_sec)
    _restart_turbo_job(job)
    st.rerun()
    return True


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
        outputs_root=str(outputs_root),
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


def _init_live_board(timeframes: list[str], strategies: list[str]) -> dict[str, dict[str, Any]]:
    board: dict[str, dict[str, Any]] = {}
    for tf in timeframes:
        for strategy in strategies:
            key = f"{tf}|{strategy}"
            board[key] = {
                "Timeframe": tf,
                "Estrategia": strategy,
                "Status": "Pendente",
                "Score": None,
                "Lucro Liquido": None,
                "Fator Lucro": None,
                "Score Parcial OOS": None,
                "PnL Parcial OOS": None,
                "Score Treino Atual": None,
            }
    return board


def _upsert_live_board_row(
    board: dict[str, dict[str, Any]],
    timeframe: str,
    strategy_name: str,
    *,
    status: str | None = None,
    score: float | None = None,
    net_profit: float | None = None,
    profit_factor: float | None = None,
    partial_score: float | None = None,
    partial_net: float | None = None,
    train_score: float | None = None,
) -> None:
    key = f"{timeframe}|{strategy_name}"
    if key not in board:
        board[key] = {
            "Timeframe": timeframe,
            "Estrategia": strategy_name,
            "Status": "Pendente",
            "Score": None,
            "Lucro Liquido": None,
            "Fator Lucro": None,
            "Score Parcial OOS": None,
            "PnL Parcial OOS": None,
            "Score Treino Atual": None,
        }
    row = board[key]
    if status is not None:
        row["Status"] = status
    if score is not None:
        row["Score"] = float(score)
    if net_profit is not None:
        row["Lucro Liquido"] = float(net_profit)
    if profit_factor is not None:
        row["Fator Lucro"] = float(profit_factor)
    if partial_score is not None:
        row["Score Parcial OOS"] = float(partial_score)
    if partial_net is not None:
        row["PnL Parcial OOS"] = float(partial_net)
    if train_score is not None:
        row["Score Treino Atual"] = float(train_score)


def _update_live_board_from_event(
    board: dict[str, dict[str, Any]],
    timeframe: str,
    strategy_name: str,
    event: dict[str, Any],
) -> None:
    stage = str(event.get("stage", ""))
    sample_net = float(event.get("net_profit", 0.0)) if "net_profit" in event else None
    sample_pf = float(event.get("profit_factor", 0.0)) if "profit_factor" in event else None
    if stage in {"walkforward_start", "window_start"}:
        _upsert_live_board_row(board, timeframe, strategy_name, status="Em andamento")
        return
    if stage in {"optimizer_sample", "optimizer_seed"}:
        _upsert_live_board_row(
            board,
            timeframe,
            strategy_name,
            status="Em andamento",
            train_score=float(event.get("score", 0.0)),
            net_profit=sample_net,
            profit_factor=sample_pf,
        )
        return
    if stage == "window_complete":
        _upsert_live_board_row(
            board,
            timeframe,
            strategy_name,
            status="Em andamento",
            partial_score=float(event.get("oos_score", 0.0)),
            partial_net=float(event.get("oos_net_profit", 0.0)),
            net_profit=float(event.get("oos_net_profit", 0.0)),
            profit_factor=float(event.get("oos_profit_factor", 0.0)),
        )
        return
    if stage == "walkforward_done":
        _upsert_live_board_row(
            board,
            timeframe,
            strategy_name,
            status="Concluida",
            score=float(event.get("final_score", 0.0)),
            net_profit=float(event.get("net_profit", 0.0)),
            profit_factor=float(event.get("profit_factor", 0.0)),
        )


def _render_live_board(container: Any, board: dict[str, dict[str, Any]], title: str) -> None:
    if not board:
        return
    frame = pd.DataFrame(list(board.values()))
    if frame.empty:
        return
    frame["rank_score"] = pd.to_numeric(frame["Score"], errors="coerce")
    frame["rank_partial"] = pd.to_numeric(frame["Score Parcial OOS"], errors="coerce")
    frame["rank_key"] = frame["rank_score"].where(frame["rank_score"].notna(), frame["rank_partial"])
    frame = frame.sort_values(["rank_key", "Lucro Liquido"], ascending=[False, False], na_position="last")
    show = frame[
        [
            "Timeframe",
            "Estrategia",
            "Status",
            "Score",
            "Lucro Liquido",
            "Fator Lucro",
            "Score Parcial OOS",
            "PnL Parcial OOS",
            "Score Treino Atual",
        ]
    ].copy()

    styled = (
        show.style.format(
            {
                "Score": "{:.2f}",
                "Lucro Liquido": "{:.2f}",
                "Fator Lucro": "{:.2f}",
                "Score Parcial OOS": "{:.2f}",
                "PnL Parcial OOS": "{:.2f}",
                "Score Treino Atual": "{:.2f}",
            },
            na_rep="-",
        )
        .applymap(
            _color_pos_neg,
            subset=["Score", "Lucro Liquido", "Fator Lucro", "Score Parcial OOS", "PnL Parcial OOS", "Score Treino Atual"],
        )
    )

    with container.container():
        st.markdown(f"### {title}")
        st.dataframe(styled, use_container_width=True, hide_index=True)


def _color_pos_neg(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if number > 0:
        return "color: #22d481; font-weight: 700;"
    if number < 0:
        return "color: #ff4f5e; font-weight: 700;"
    return "color: #e9e9e9;"


def _build_turbo_live_board(job: TurboJob) -> dict[str, dict[str, Any]]:
    board = _init_live_board(job.timeframes, job.strategies)
    log_path = Path(job.log_file)
    if not log_path.exists():
        return board
    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    progress_re = re.compile(r"^\[PROGRESS\]\s+([^/]+)/([^\s]+)\s+(.*)$")
    done_re = re.compile(r"score=([-+]?\d+(?:\.\d+)?)\s+net=([-+]?\d+(?:\.\d+)?)\s+pf=([-+]?\d+(?:\.\d+)?)")
    window_score_re = re.compile(r"oos_score=([-+]?\d+(?:\.\d+)?)")
    window_net_re = re.compile(r"oos_net=([-+]?\d+(?:\.\d+)?)")
    window_pf_re = re.compile(r"oos_pf=([-+]?\d+(?:\.\d+)?)")
    sample_score_re = re.compile(r"score=([-+]?\d+(?:\.\d+)?)")
    sample_net_re = re.compile(r"net=([-+]?\d+(?:\.\d+)?)")
    sample_pf_re = re.compile(r"pf=([-+]?\d+(?:\.\d+)?)")
    legacy_done_re = re.compile(r"score=([-+]?\d+(?:\.\d+)?)\s+net=([-+]?\d+(?:\.\d+)?)")
    warn_re = re.compile(r"^\[WARN\]\s+([^/]+)/([^:]+):")

    for line in lines:
        progress_match = progress_re.match(line.strip())
        if progress_match:
            timeframe = progress_match.group(1)
            strategy_name = progress_match.group(2)
            rest = progress_match.group(3)
            if rest.startswith("done "):
                done_match = done_re.search(rest)
                if done_match:
                    _upsert_live_board_row(
                        board,
                        timeframe,
                        strategy_name,
                        status="Concluida",
                        score=float(done_match.group(1)),
                        net_profit=float(done_match.group(2)),
                        profit_factor=float(done_match.group(3)),
                    )
                else:
                    legacy_done_match = legacy_done_re.search(rest)
                    if legacy_done_match:
                        _upsert_live_board_row(
                            board,
                            timeframe,
                            strategy_name,
                            status="Concluida",
                            score=float(legacy_done_match.group(1)),
                            net_profit=float(legacy_done_match.group(2)),
                        )
                    else:
                        _upsert_live_board_row(board, timeframe, strategy_name, status="Concluida")
                continue
            if rest.startswith("window ") and "done" in rest:
                score_match = window_score_re.search(rest)
                net_match = window_net_re.search(rest)
                pf_match = window_pf_re.search(rest)
                if score_match:
                    _upsert_live_board_row(
                        board,
                        timeframe,
                        strategy_name,
                        status="Em andamento",
                        partial_score=float(score_match.group(1)),
                        partial_net=float(net_match.group(1)) if net_match else None,
                        net_profit=float(net_match.group(1)) if net_match else None,
                        profit_factor=float(pf_match.group(1)) if pf_match else None,
                    )
                else:
                    _upsert_live_board_row(board, timeframe, strategy_name, status="Em andamento")
                continue
            if rest.startswith("sample "):
                sample_match = sample_score_re.search(rest)
                sample_net_match = sample_net_re.search(rest)
                sample_pf_match = sample_pf_re.search(rest)
                _upsert_live_board_row(
                    board,
                    timeframe,
                    strategy_name,
                    status="Em andamento",
                    train_score=float(sample_match.group(1)) if sample_match else None,
                    net_profit=float(sample_net_match.group(1)) if sample_net_match else None,
                    profit_factor=float(sample_pf_match.group(1)) if sample_pf_match else None,
                )
                continue
            _upsert_live_board_row(board, timeframe, strategy_name, status="Em andamento")
            continue

        warn_match = warn_re.match(line.strip())
        if warn_match:
            _upsert_live_board_row(
                board,
                warn_match.group(1),
                warn_match.group(2).strip(),
                status="Erro",
            )

    return board

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
    ranking_box = st.empty()

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
    live_board = _init_live_board(
        normalized_timeframes,
        [x.name for x in selected_strategies],
    )
    _render_live_board(ranking_box, live_board, title="Ranking em Tempo Real")
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
            trades_file = tf_output / f"trades_{timeframe}_{strategy.name}.csv"
            windows_file = tf_output / f"walkforward_windows_{timeframe}_{strategy.name}.csv"
            topk_file = tf_output / f"walkforward_topk_{timeframe}_{strategy.name}.csv"
            equity_curve_file = tf_output / f"equity_curve_{timeframe}_{strategy.name}.csv"
            checkpoint_file = tf_output / f"checkpoint_{timeframe}_{strategy.name}.json"
            _upsert_live_board_row(live_board, timeframe, strategy.name, status="Em andamento")
            _render_live_board(ranking_box, live_board, title="Ranking em Tempo Real")
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
                _update_live_board_from_event(live_board, timeframe, strategy.name, latest_progress)
                _render_live_board(ranking_box, live_board, title="Ranking em Tempo Real")
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
                checkpoint_callback=(
                    _build_ui_checkpoint_callback(
                        timeframe=timeframe,
                        strategy_name=strategy.name,
                        trades_file=trades_file,
                        windows_file=windows_file,
                        topk_file=topk_file,
                        equity_curve_file=equity_curve_file,
                        checkpoint_file=checkpoint_file,
                    )
                    if save_outputs
                    else None
                ),
            )

            finished_jobs += 1
            progress_box.progress(min(1.0, finished_jobs / max(total_jobs, 1)))
            progress_text.caption(
                f"Concluido {finished_jobs}/{total_jobs}: {symbol} {timeframe} / {strategy.name}"
            )

            strategy_runs.append(StrategyRun(timeframe=timeframe, strategy_name=strategy.name, result=wf_result))
            _upsert_live_board_row(
                live_board,
                timeframe,
                strategy.name,
                status="Concluida",
                score=float(wf_result.consolidated_metrics.get("score", 0.0)),
                net_profit=float(wf_result.consolidated_metrics.get("net_profit", 0.0)),
                profit_factor=float(wf_result.consolidated_metrics.get("profit_factor", 0.0)),
            )
            _render_live_board(ranking_box, live_board, title="Ranking em Tempo Real")
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
                        str(trades_file),
                        str(windows_file),
                        str(topk_file),
                        str(equity_curve_file),
                        str(checkpoint_file),
                        str(tf_output / f"equity_{timeframe}_{strategy.name}.png"),
                        str(tf_output / f"monthly_{timeframe}_{strategy.name}.csv"),
                        str(tf_output / f"hourly_{timeframe}_{strategy.name}.csv"),
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
                run_tag=run_id,
            )
            generated_files.extend(
                [
                    str(tf_output / f"summary_{timeframe}.csv"),
                    str(tf_output / f"best_params_{timeframe}.json"),
                    str(tf_output / f"equity_curve_{timeframe}_best.csv"),
                    str(tf_output / f"strategy_history_{timeframe}.csv"),
                    str(tf_output / f"equity_{timeframe}_best.png"),
                    str(tf_output / f"summary_{timeframe}_{run_id}.csv"),
                    str(tf_output / f"best_params_{timeframe}_{run_id}.json"),
                    str(tf_output / f"equity_curve_{timeframe}_best_{run_id}.csv"),
                    str(tf_output / f"equity_{timeframe}_best_{run_id}.png"),
                    str(tf_output / f"best_history_{timeframe}.csv"),
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
        outputs_root=str(outputs_root),
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
        pnl = float(event.get("net_profit", 0.0))
        pf = float(event.get("profit_factor", 0.0))
        progress_text.caption(
            f"Otimizacao: amostra {idx}/{total} | score treino {score:.2f} "
            f"| lucro treino {format_brl(pnl)} | fator lucro {pf:.2f}"
        )
        return
    if stage == "window_complete":
        idx = int(event.get("window_index", 0))
        total = int(event.get("total_windows", 0))
        score = float(event.get("oos_score", 0.0))
        pnl = float(event.get("oos_net_profit", 0.0))
        pf = float(event.get("oos_profit_factor", 0.0))
        progress_text.caption(
            f"Janela {idx}/{total} concluida | score OOS {score:.2f} "
            f"| pnl OOS {format_brl(pnl)} | fator lucro OOS {pf:.2f}"
        )
        return
    if stage == "walkforward_done":
        score = float(event.get("final_score", 0.0))
        pnl = float(event.get("net_profit", 0.0))
        pf = float(event.get("profit_factor", 0.0))
        progress_text.caption(
            f"Consolidado: score {score:.2f} | lucro liquido {format_brl(pnl)} | fator lucro {pf:.2f}"
        )


def _build_ui_checkpoint_callback(
    timeframe: str,
    strategy_name: str,
    trades_file: Path,
    windows_file: Path,
    topk_file: Path,
    equity_curve_file: Path,
    checkpoint_file: Path,
):
    def _checkpoint(checkpoint: Any) -> None:
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

    hourly_file = output_dir / f"hourly_{timeframe}_{strategy_name}.csv"
    hourly_df = build_hourly_report(wf_result.oos_trades)
    hourly_df.to_csv(hourly_file, index=False)

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
    run_tag: str,
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
    summary_latest = output_dir / f"summary_{timeframe}.csv"
    summary_df.to_csv(summary_latest, index=False)
    run_token = run_tag.strip() or "snapshot"
    summary_snapshot = output_dir / f"summary_{timeframe}_{run_token}.csv"
    summary_df.to_csv(summary_snapshot, index=False)
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
    best_params_latest = output_dir / f"best_params_{timeframe}.json"
    best_params_latest.write_text(
        json.dumps(payload, indent=2, default=_json_default),
        encoding="utf-8",
    )
    best_params_snapshot = output_dir / f"best_params_{timeframe}_{run_token}.json"
    best_params_snapshot.write_text(
        json.dumps(payload, indent=2, default=_json_default),
        encoding="utf-8",
    )

    _append_strategy_history_rows(
        history_file=output_dir / f"strategy_history_{timeframe}.csv",
        summary_df=summary_df,
        run_tag=run_token,
        created_at_utc=pd.Timestamp.utcnow().isoformat(),
        symbol=symbol,
        timeframe=timeframe,
        best_strategy=best_name,
        summary_latest=summary_latest,
        summary_snapshot=summary_snapshot,
        best_params_latest=best_params_latest,
        best_params_snapshot=best_params_snapshot,
        output_dir=output_dir,
    )
    best_equity_latest = output_dir / f"equity_curve_{timeframe}_best.csv"
    best_run.result.oos_equity.to_csv(best_equity_latest, index=False)
    best_equity_snapshot = output_dir / f"equity_curve_{timeframe}_best_{run_token}.csv"
    best_run.result.oos_equity.to_csv(best_equity_snapshot, index=False)
    best_plot_latest = output_dir / f"equity_{timeframe}_best.png"
    _save_equity_png(
        best_run.result.oos_equity,
        title=f"{symbol} {timeframe} - best: {best_name}",
        output_file=best_plot_latest,
    )
    best_plot_snapshot = output_dir / f"equity_{timeframe}_best_{run_token}.png"
    _save_equity_png(
        best_run.result.oos_equity,
        title=f"{symbol} {timeframe} - best: {best_name} ({run_token})",
        output_file=best_plot_snapshot,
    )

    _append_best_history_row(
        history_file=output_dir / f"best_history_{timeframe}.csv",
        row={
            "run_tag": run_token,
            "created_at_utc": pd.Timestamp.utcnow().isoformat(),
            "symbol": symbol,
            "timeframe": timeframe,
            "best_strategy": best_name,
            "best_score": float(summary_df.iloc[0]["score"]),
            "best_net_profit": float(best_run.result.consolidated_metrics.get("net_profit", 0.0)),
            "best_trade_count": float(best_run.result.consolidated_metrics.get("trade_count", 0.0)),
            "best_win_rate": float(best_run.result.consolidated_metrics.get("win_rate", 0.0)),
            "best_max_drawdown": float(best_run.result.consolidated_metrics.get("max_drawdown", 0.0)),
            "best_max_drawdown_pct": float(best_run.result.consolidated_metrics.get("max_drawdown_pct", 0.0)),
            "windows": int(len(best_run.result.window_results)),
            "summary_latest": str(summary_latest),
            "summary_snapshot": str(summary_snapshot),
            "best_params_latest": str(best_params_latest),
            "best_params_snapshot": str(best_params_snapshot),
            "best_equity_latest": str(best_equity_latest),
            "best_equity_snapshot": str(best_equity_snapshot),
            "best_plot_latest": str(best_plot_latest),
            "best_plot_snapshot": str(best_plot_snapshot),
        },
    )


def _write_trades_csv(trades: pd.DataFrame, output_file: Path) -> None:
    if trades.empty:
        pd.DataFrame(columns=default_trade_columns()).to_csv(output_file, index=False)
        return
    out = trades.copy()
    out["params"] = out["params"].apply(lambda p: json.dumps(p, sort_keys=True))
    out.to_csv(output_file, index=False)


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
    output_dir: Path,
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
                "trades_latest": str(output_dir / f"trades_{timeframe}_{strategy}.csv"),
                "windows_latest": str(output_dir / f"walkforward_windows_{timeframe}_{strategy}.csv"),
                "topk_latest": str(output_dir / f"walkforward_topk_{timeframe}_{strategy}.csv"),
                "equity_latest": str(output_dir / f"equity_curve_{timeframe}_{strategy}.csv"),
                "equity_plot_latest": str(output_dir / f"equity_{timeframe}_{strategy}.png"),
            }
        )
    if not rows:
        return

    row_df = pd.DataFrame(rows)
    if history_file.exists():
        prev = pd.read_csv(history_file)
        if not prev.empty and {"run_tag", "timeframe", "strategy"}.issubset(set(prev.columns)):
            mask = (
                (prev["run_tag"].astype(str) == str(run_tag))
                & (prev["timeframe"].astype(str) == str(timeframe))
                & (prev["strategy"].astype(str).isin(row_df["strategy"].astype(str)))
            )
            prev = prev.loc[~mask].copy()
        out = pd.concat([prev, row_df], ignore_index=True)
    else:
        out = row_df
    out = out.sort_values(["created_at_utc", "rank_in_run"], ascending=[False, True])
    out.to_csv(history_file, index=False)


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
    menu_options = [
        "Consolidado",
        "Resumo",
        "Operacoes",
        "Grafico de Operacoes",
        "Patrimonio",
        "Mensal",
        "Melhores Horas",
        "Robustez",
    ]
    selected_view = st.radio(
        "Menu",
        options=menu_options,
        horizontal=True,
        label_visibility="collapsed",
        key="dashboard_top_menu",
    )

    if selected_view == "Consolidado":
        st.subheader("Consolidado")
        st.dataframe(
            result.summary,
            use_container_width=True,
            hide_index=True,
        )
        history_df = _load_best_history(outputs_root=Path(result.outputs_root), symbol=result.symbol)
        if history_df.empty:
            st.info("Historico de melhores ainda vazio para este ativo.")
        else:
            st.markdown("### Historico de Melhores")
            _render_best_history_table(history_df)
        st.caption(
            f"Periodo {result.start.date()} a {result.end.date()} | "
            f"Ativo {result.symbol} | Estrategias validadas: {len(result.strategy_runs)}"
        )
        return

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
    hourly_df = build_hourly_report(trades)
    robustness_payload = build_robustness_report(
        window_results=selected_run.result.window_results,
        topk_test_results=selected_run.result.topk_test_results,
        consolidated_metrics=selected_run.result.consolidated_metrics,
    )
    sensitivity_df = build_parameter_sensitivity_report(selected_run.result.topk_test_results)

    if selected_view == "Resumo":
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

    if selected_view == "Operacoes":
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

    if selected_view == "Grafico de Operacoes":
        op_fig = _build_operations_chart(trades)
        st.plotly_chart(op_fig, use_container_width=True)

    if selected_view == "Patrimonio":
        eq_fig = _build_equity_chart(
            equity=equity,
            symbol=result.symbol,
            timeframe=selected_tf,
            strategy_name=selected_strategy,
        )
        st.plotly_chart(eq_fig, use_container_width=True)

    if selected_view == "Mensal":
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

    if selected_view == "Melhores Horas":
        if hourly_df.empty:
            st.info("Sem operacoes suficientes para analise por hora.")
        else:
            show = hourly_df.copy()
            show["hour"] = show["hour"].astype(int).map(lambda x: f"{x:02d}:00")
            st.dataframe(show, use_container_width=True, hide_index=True)

            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=show["hour"],
                    y=hourly_df["net_profit"],
                    marker_color=["#22d481" if x >= 0 else "#ff4f5e" for x in hourly_df["net_profit"]],
                    name="PnL por hora",
                )
            )
            fig.update_layout(
                title="Melhores Horas para Operar (PnL)",
                template="plotly_dark",
                paper_bgcolor="#1e1f22",
                plot_bgcolor="#1e1f22",
                yaxis_title="R$",
                xaxis_title="Hora",
            )
            st.plotly_chart(fig, use_container_width=True)

    if selected_view == "Robustez":
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


def _load_best_history(outputs_root: Path, symbol: str) -> pd.DataFrame:
    base = outputs_root / symbol
    if not base.exists():
        return pd.DataFrame()
    files = sorted(base.glob("*/best_history_*.csv"))
    if not files:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for path in files:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty:
            continue
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    if "created_at_utc" in out.columns:
        out["created_at_utc"] = pd.to_datetime(out["created_at_utc"], errors="coerce")
    sort_cols = [c for c in ["created_at_utc", "best_score", "best_net_profit"] if c in out.columns]
    if sort_cols:
        ascending = [False] * len(sort_cols)
        out = out.sort_values(sort_cols, ascending=ascending, na_position="last")
    return out.reset_index(drop=True)


def _render_best_history_table(history_df: pd.DataFrame) -> None:
    local = history_df.copy()
    def _series(col: str, default: Any = "") -> pd.Series:
        if col in local.columns:
            return local[col]
        return pd.Series([default] * len(local))

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        tf_options = ["Todos"] + sorted(_series("timeframe").dropna().astype(str).unique().tolist())
        tf_filter = st.selectbox("Timeframe (historico)", options=tf_options, index=0, key="hist_tf_filter")
    with c2:
        strat_options = ["Todas"] + sorted(_series("best_strategy").dropna().astype(str).unique().tolist())
        strat_filter = st.selectbox("Estrategia (historico)", options=strat_options, index=0, key="hist_strat_filter")
    with c3:
        max_rows = int(st.slider("Linhas", min_value=10, max_value=500, value=120, step=10, key="hist_rows"))

    if tf_filter != "Todos":
        local = local.loc[_series("timeframe").astype(str) == tf_filter].copy()
    if strat_filter != "Todas":
        local = local.loc[_series("best_strategy").astype(str) == strat_filter].copy()

    if local.empty:
        st.warning("Nenhuma linha no historico com os filtros selecionados.")
        return

    work = local.copy()
    work["created_at_utc"] = pd.to_datetime(_series("created_at_utc"), errors="coerce")
    work["best_score"] = pd.to_numeric(_series("best_score"), errors="coerce")
    work["best_net_profit"] = pd.to_numeric(_series("best_net_profit"), errors="coerce")
    work["best_trade_count"] = pd.to_numeric(_series("best_trade_count"), errors="coerce")
    work["best_win_rate"] = pd.to_numeric(_series("best_win_rate"), errors="coerce")
    work["best_max_drawdown"] = pd.to_numeric(_series("best_max_drawdown"), errors="coerce")
    work["timeframe"] = _series("timeframe").astype(str)
    work["best_strategy"] = _series("best_strategy").astype(str)
    work["run_tag"] = _series("run_tag").astype(str)

    st.markdown("#### Resumo das Melhores Estrategias Achadas")
    rollup = (
        work.groupby(["timeframe", "best_strategy"], as_index=False)
        .agg(
            runs=("run_tag", "count"),
            melhor_score=("best_score", "max"),
            score_medio=("best_score", "mean"),
            melhor_pnl=("best_net_profit", "max"),
            pnl_medio=("best_net_profit", "mean"),
            win_rate_medio=("best_win_rate", "mean"),
            dd_medio=("best_max_drawdown", "mean"),
            trades_medios=("best_trade_count", "mean"),
        )
        .sort_values(["melhor_pnl", "melhor_score"], ascending=[False, False], na_position="last")
    )

    latest = (
        work.sort_values("created_at_utc", ascending=False)
        .drop_duplicates(subset=["timeframe", "best_strategy"], keep="first")[
            ["timeframe", "best_strategy", "created_at_utc", "best_net_profit", "best_score"]
        ]
        .rename(
            columns={
                "created_at_utc": "ultima_execucao",
                "best_net_profit": "ultimo_pnl",
                "best_score": "ultimo_score",
            }
        )
    )
    rollup = rollup.merge(latest, on=["timeframe", "best_strategy"], how="left")
    rollup_view = pd.DataFrame(
        {
            "TF": rollup["timeframe"],
            "Estrategia": rollup["best_strategy"],
            "Runs": rollup["runs"].fillna(0).astype(int),
            "Melhor PnL (R$)": rollup["melhor_pnl"].round(2),
            "PnL Medio (R$)": rollup["pnl_medio"].round(2),
            "Melhor Score": rollup["melhor_score"].round(2),
            "Score Medio": rollup["score_medio"].round(2),
            "Win Medio %": (100.0 * rollup["win_rate_medio"]).round(2),
            "DD Medio (R$)": rollup["dd_medio"].round(2),
            "Trades Medios": rollup["trades_medios"].round(1),
            "Ultimo PnL (R$)": rollup["ultimo_pnl"].round(2),
            "Ultimo Score": rollup["ultimo_score"].round(2),
            "Ultima Execucao": pd.to_datetime(rollup["ultima_execucao"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )

    def _color_value(v: Any) -> str:
        try:
            return "color: #22d481;" if float(v) >= 0 else "color: #ff4f5e;"
        except Exception:
            return ""

    rollup_styled = (
        rollup_view.style
        .map(_color_value, subset=["Melhor PnL (R$)", "PnL Medio (R$)", "Ultimo PnL (R$)"])
        .map(_color_value, subset=["Melhor Score", "Score Medio", "Ultimo Score"])
    )
    st.dataframe(rollup_styled, use_container_width=True, hide_index=True)

    if not rollup.empty:
        best_row = rollup.iloc[0]
        st.caption(
            "Top atual: "
            f"{best_row['timeframe']}/{best_row['best_strategy']} | "
            f"melhor pnl {format_brl(float(best_row['melhor_pnl']))} | "
            f"melhor score {float(best_row['melhor_score']):.2f}"
        )

    view = pd.DataFrame(
        {
            "Data UTC": pd.to_datetime(_series("created_at_utc"), errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S"),
            "Run": _series("run_tag").astype(str),
            "TF": _series("timeframe").astype(str),
            "Estrategia": _series("best_strategy").astype(str),
            "Score": pd.to_numeric(_series("best_score"), errors="coerce").round(2),
            "PnL (R$)": pd.to_numeric(_series("best_net_profit"), errors="coerce").round(2),
            "Trades": pd.to_numeric(_series("best_trade_count"), errors="coerce").fillna(0).astype(int),
            "Win %": (100.0 * pd.to_numeric(_series("best_win_rate"), errors="coerce")).round(2),
            "DD (R$)": pd.to_numeric(_series("best_max_drawdown"), errors="coerce").round(2),
            "DD %": pd.to_numeric(_series("best_max_drawdown_pct"), errors="coerce").round(2),
            "Janelas": pd.to_numeric(_series("windows"), errors="coerce").fillna(0).astype(int),
        }
    ).head(max_rows)

    def _color_pnl(v: Any) -> str:
        try:
            return "color: #22d481;" if float(v) >= 0 else "color: #ff4f5e;"
        except Exception:
            return ""

    styled = view.style.map(_color_pnl, subset=["PnL (R$)"]).map(_color_pnl, subset=["Score"])
    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.download_button(
        "Baixar historico filtrado (CSV)",
        data=local.to_csv(index=False).encode("utf-8"),
        file_name="best_history_filtrado.csv",
        mime="text/csv",
    )

    run_options = _series("run_tag").astype(str).tolist()
    selected_run = st.selectbox("Detalhe da execucao", options=run_options, index=0, key="hist_run_detail")
    row = local.loc[_series("run_tag").astype(str) == selected_run].head(1)
    if row.empty:
        return
    item = row.iloc[0]
    d1, d2, d3 = st.columns(3)
    with d1:
        st.metric("Estrategia", str(item.get("best_strategy", "-")))
        st.metric("PnL", format_brl(float(item.get("best_net_profit", 0.0))))
    with d2:
        st.metric("Score", f"{float(item.get('best_score', 0.0)):.2f}")
        st.metric("Trades", format_int(float(item.get("best_trade_count", 0))))
    with d3:
        st.metric("Win rate", f"{100.0 * float(item.get('best_win_rate', 0.0)):.2f}%")
        st.metric("DD", format_brl(float(item.get("best_max_drawdown", 0.0))))

    st.caption("Arquivos da execucao selecionada")
    artifact_cols = [
        "summary_snapshot",
        "best_params_snapshot",
        "best_equity_snapshot",
        "best_plot_snapshot",
    ]
    for col in artifact_cols:
        raw = str(item.get(col, "")).strip()
        if raw:
            st.code(raw, language="text")


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
    plot_df["datetime"] = pd.to_datetime(plot_df["datetime"], errors="coerce")
    plot_df["equity"] = pd.to_numeric(plot_df["equity"], errors="coerce")
    plot_df = plot_df.dropna(subset=["datetime", "equity"]).sort_values("datetime")
    if plot_df.empty:
        return fig

    initial_equity = float(plot_df["equity"].iloc[0])
    plot_df["pnl_curve"] = plot_df["equity"] - initial_equity
    plot_df["pnl_pos"] = plot_df["pnl_curve"].where(plot_df["pnl_curve"] >= 0)
    plot_df["pnl_neg"] = plot_df["pnl_curve"].where(plot_df["pnl_curve"] < 0)

    fig.add_trace(
        go.Scatter(
            x=plot_df["datetime"],
            y=plot_df["pnl_pos"],
            mode="lines",
            line=dict(color="#22f08a", width=2.2),
            fill="tozeroy",
            fillcolor="rgba(34, 240, 138, 0.22)",
            name=symbol,
            hovertemplate="%{x|%d/%m/%Y %H:%M}<br>Valor: R$ %{y:,.2f}<extra>Positivo</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df["datetime"],
            y=plot_df["pnl_neg"],
            mode="lines",
            line=dict(color="#ff4f5e", width=2.2),
            fill="tozeroy",
            fillcolor="rgba(255, 79, 94, 0.20)",
            name=symbol,
            hovertemplate="%{x|%d/%m/%Y %H:%M}<br>Valor: R$ %{y:,.2f}<extra>Negativo</extra>",
            showlegend=False,
        )
    )
    last_x = plot_df["datetime"].iloc[-1]
    last_y = float(plot_df["pnl_curve"].iloc[-1])
    first_y = 0.0
    fig.add_hline(
        y=first_y,
        line_width=1,
        line_dash="dot",
        line_color="rgba(255,255,255,0.22)",
    )
    fig.add_annotation(
        x=last_x,
        y=last_y,
        text=_format_brl_compact(last_y),
        showarrow=False,
        xanchor="left",
        yanchor="middle",
        xshift=8,
        font=dict(color="#0f1f16", size=11, family="Barlow Condensed, Segoe UI, sans-serif"),
        bgcolor="#25f08a" if last_y >= 0 else "#ff4f5e",
        bordercolor="#25f08a" if last_y >= 0 else "#ff4f5e",
        borderwidth=1,
        opacity=0.95,
    )
    fig.update_layout(
        title="Patrimonio",
        template="plotly_dark",
        paper_bgcolor="#1b1d20",
        plot_bgcolor="#1a1c1f",
        font=dict(family="Barlow Condensed, Segoe UI, sans-serif", color="#f5f5f5"),
        xaxis_title=None,
        yaxis_title="Resultado Acumulado (R$)",
        legend=dict(orientation="h", yanchor="top", y=-0.11, xanchor="left", x=0),
        margin=dict(l=16, r=72, t=52, b=48),
        hovermode="x unified",
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.08)",
        tickformat="%d/%m/%Y",
        zeroline=False,
    )
    fig.update_yaxes(
        side="right",
        showgrid=True,
        gridcolor="rgba(255,255,255,0.08)",
        zeroline=True,
        zerolinecolor="rgba(255,255,255,0.22)",
        tickformat="~s",
        rangemode="normal",
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


def _format_brl_compact(value: float) -> str:
    abs_value = abs(float(value))
    sign = "-" if value < 0 else ""
    if abs_value >= 1_000_000:
        raw = f"{abs_value / 1_000_000:.2f}".replace(".", ",")
        return f"{sign}{raw}M"
    if abs_value >= 1_000:
        raw = f"{abs_value / 1_000:.2f}".replace(".", ",")
        return f"{sign}{raw}k"
    raw = f"{abs_value:.2f}".replace(".", ",")
    return f"{sign}{raw}"


def format_pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def format_int(value: float | int) -> str:
    return f"{int(value):,}".replace(",", ".")


def _apply_score_preset(name: str) -> None:
    preset = SCORE_PRESETS.get(str(name).strip().lower())
    if not preset:
        return
    st.session_state["score_drawdown_weight"] = float(preset["drawdown_weight"])
    st.session_state["score_min_trades"] = int(preset["min_trades"])
    st.session_state["score_penalty_missing"] = float(preset["penalty_missing"])


def _init_session_state() -> None:
    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("turbo_job", None)
    st.session_state.setdefault("turbo_loop_enabled", False)
    st.session_state.setdefault("turbo_loop_pause_sec", 3)
    st.session_state.setdefault("turbo_loop_max_cycles", 0)
    st.session_state.setdefault("turbo_loop_cycles_done", 0)
    st.session_state.setdefault("symbol_default", "WINFUT")
    st.session_state.setdefault("timeframes_default", ["5m"])
    st.session_state.setdefault("strategies_default", list(STRATEGIES.keys()))
    st.session_state.setdefault("start_date_default", date(2025, 1, 17))
    st.session_state.setdefault("end_date_default", date(2026, 1, 16))
    st.session_state.setdefault("score_drawdown_weight", float(SCORE_PRESETS["equilibrado"]["drawdown_weight"]))
    st.session_state.setdefault("score_min_trades", int(SCORE_PRESETS["equilibrado"]["min_trades"]))
    st.session_state.setdefault("score_penalty_missing", float(SCORE_PRESETS["equilibrado"]["penalty_missing"]))


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
