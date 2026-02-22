"""Standalone viewer for all outputs folders (read-only)."""

from __future__ import annotations

import io
import json
import re
import sys
from dataclasses import replace
from datetime import date as date_cls
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest_engine import BacktestConfig, run_backtest
from src.data_loader import LoaderConfig, load_timeframe_data, normalize_timeframe_label
from src.metrics import ScoreConfig, compute_metrics
from src.optimizer import build_runtime_config_for_params, generate_signals_with_time_filter
from src.strategies import STRATEGIES
from src.study_setup_cli import consolidate_outputs
from src.tick_loader import load_ticks_between


_TICK_FILE_DATE_RE = re.compile(r".*_(\d{2})-(\d{2})-(\d{4})\.csv$", re.IGNORECASE)


def main() -> None:
    st.set_page_config(
        page_title="Visualizador de Resultados",
        page_icon=":bar_chart:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_dark_theme_css()
    st.title("VISUALIZADOR DE RESULTADOS")
    st.caption("Painel unico para todos os outputs* (somente leitura).")

    roots = _discover_output_roots(PROJECT_ROOT)
    if not roots:
        st.warning("Nenhuma pasta outputs* encontrada no projeto.")
        return

    selected_roots: list[Path] = []
    with st.sidebar:
        st.markdown("### Fonte de dados")
        single_folder_mode = st.checkbox("Modo pasta unica (outputs_master)", value=True)
        sync_now = False
        sync_report: dict[str, Any] | None = None
        if single_folder_mode:
            auto_sync = st.checkbox("Consolidar automaticamente", value=True)
            auto_sync_seconds = int(st.slider("Intervalo de consolidacao (s)", min_value=10, max_value=300, value=45, step=5))
            if st.button("Consolidar agora", use_container_width=True):
                sync_now = True
            last_sync_raw = st.session_state.get("viewer_last_sync_utc")
            now_utc = pd.Timestamp.now(tz="UTC")
            last_sync = pd.to_datetime(last_sync_raw, errors="coerce", utc=True) if last_sync_raw is not None else pd.NaT
            need_auto_sync = bool(
                auto_sync
                and (pd.isna(last_sync) or (now_utc - last_sync) >= pd.Timedelta(seconds=auto_sync_seconds))
            )
            if sync_now or need_auto_sync:
                sync_report = consolidate_outputs(
                    target=(PROJECT_ROOT / "outputs_master").resolve(),
                    sources=None,
                    copy_manifests=False,
                )
                st.session_state.viewer_last_sync_utc = now_utc.isoformat()
                st.session_state.viewer_last_sync_report = sync_report
            else:
                cached = st.session_state.get("viewer_last_sync_report")
                if isinstance(cached, dict):
                    sync_report = cached

            if isinstance(sync_report, dict):
                merged_best = int(sync_report.get("merged_best_history_files", 0))
                merged_strategy = int(sync_report.get("merged_strategy_history_files", 0))
                merged_bank = int(sync_report.get("merged_params_bank_files", 0))
                st.caption(
                    "Consolidacao: "
                    f"best_history={merged_best} | strategy_history={merged_strategy} | params_bank={merged_bank}"
                )

        root_names = [p.name for p in roots]
        if single_folder_mode:
            selected_names = ["outputs_master"] if "outputs_master" in root_names else root_names[:1]
            st.caption(f"Pasta ativa: {selected_names[0] if selected_names else '-'}")
        else:
            default_names = ["outputs_master"] if "outputs_master" in root_names else root_names
            selected_names = st.multiselect("Pastas de resultados", options=root_names, default=default_names)
        show_only_positive = st.checkbox("Mostrar apenas lucro positivo", value=False)
        max_rows = int(st.slider("Maximo de linhas por tabela", min_value=20, max_value=2000, value=300, step=20))

    roots = _discover_output_roots(PROJECT_ROOT)
    if not roots:
        st.warning("Nenhuma pasta outputs* encontrada no projeto.")
        return
    if single_folder_mode and any(p.name == "outputs_master" for p in roots):
        selected_names = ["outputs_master"]
    selected_roots = [p for p in roots if p.name in selected_names]
    if not selected_roots:
        st.info("Selecione ao menos uma pasta outputs.")
        return

    history_df = _load_best_history(selected_roots)
    summary_df = _load_latest_summaries(selected_roots)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Pastas lidas", len(selected_roots))
    with c2:
        st.metric("Execucoes no historico", int(len(history_df)))
    with c3:
        st.metric("Estrategias encontradas", int(history_df["best_strategy"].nunique()) if not history_df.empty else 0)
    with c4:
        st.metric("Ativos encontrados", int(history_df["symbol"].nunique()) if not history_df.empty else 0)

    with st.expander("Glossario rapido", expanded=False):
        st.markdown(
            "- `Pontuacao`: nota de qualidade da estrategia (quanto maior, melhor).\n"
            "- `Lucro (PnL)`: resultado liquido em reais.\n"
            "- `Win %`: percentual de operacoes vencedoras.\n"
            "- `DD (Drawdown)`: pior queda acumulada.\n"
            "- `Execucao`: um ciclo/rodada completo de teste salvo em historico."
        )

    tab_rank, tab_hist, tab_sum, tab_art, tab_tick = st.tabs(
        ["Ranking global", "Historico de execucoes", "Resumos atuais", "Resumo e graficos", "Tick a Tick diario"]
    )

    selected_row = _render_history_table(tab_hist, history_df, show_only_positive=show_only_positive, max_rows=max_rows)
    _render_global_ranking(tab_rank, history_df, show_only_positive=show_only_positive, max_rows=max_rows)
    _render_top_runs(tab_rank, history_df, show_only_positive=show_only_positive)
    _render_latest_summaries(tab_sum, summary_df, show_only_positive=show_only_positive, max_rows=max_rows)
    _render_artifacts(tab_art, selected_row)
    _render_tick_daily_validation(tab_tick, selected_row)


def _discover_output_roots(project_root: Path) -> list[Path]:
    out = [p for p in project_root.iterdir() if p.is_dir() and p.name.startswith("outputs")]
    return sorted(out, key=lambda p: p.name.lower())


def _read_csv_safe(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _load_best_history(roots: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for root in roots:
        for history_file in root.glob("**/best_history_*.csv"):
            df = _read_csv_safe(history_file)
            if df.empty:
                continue
            rel = history_file.relative_to(root)
            symbol = rel.parts[0] if len(rel.parts) >= 3 else ""
            timeframe = rel.parts[1] if len(rel.parts) >= 3 else ""
            if "symbol" not in df.columns:
                df["symbol"] = symbol
            if "timeframe" not in df.columns:
                df["timeframe"] = timeframe
            df["source_root"] = root.name
            df["history_file"] = str(history_file.relative_to(PROJECT_ROOT))
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out["created_at_utc"] = pd.to_datetime(out.get("created_at_utc"), errors="coerce")
    for col in [
        "best_score",
        "best_net_profit",
        "best_trade_count",
        "best_win_rate",
        "best_max_drawdown",
        "best_max_drawdown_pct",
        "windows",
    ]:
        out[col] = pd.to_numeric(out.get(col), errors="coerce")
    out = out.sort_values(["created_at_utc", "best_net_profit"], ascending=[False, False], na_position="last")
    return out.reset_index(drop=True)


def _load_latest_summaries(roots: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    latest_name = re.compile(r"summary_(?P<tf>[^_]+)\.csv$")
    for root in roots:
        for file in root.glob("**/summary_*.csv"):
            if not latest_name.fullmatch(file.name):
                continue
            df = _read_csv_safe(file)
            if df.empty:
                continue
            if "strategy" not in df.columns:
                continue
            rel = file.relative_to(root)
            symbol = rel.parts[0] if len(rel.parts) >= 3 else ""
            timeframe = rel.parts[1] if len(rel.parts) >= 3 else latest_name.fullmatch(file.name).group("tf")
            cur = df.copy()
            cur["symbol"] = symbol
            cur["timeframe"] = timeframe
            cur["source_root"] = root.name
            cur["summary_file"] = str(file.relative_to(PROJECT_ROOT))
            for col in ["score", "net_profit", "win_rate", "max_drawdown", "trade_count"]:
                if col in cur.columns:
                    cur[col] = pd.to_numeric(cur[col], errors="coerce")
            frames.append(cur)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    sort_cols = [c for c in ["net_profit", "score"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, ascending=False, na_position="last")
    return out.reset_index(drop=True)


def _render_global_ranking(container: Any, history_df: pd.DataFrame, show_only_positive: bool, max_rows: int) -> None:
    with container:
        st.subheader("Ranking global (agregado por estrategia)")
        if history_df.empty:
            st.info("Sem historico para montar ranking.")
            return
        work = history_df.copy()
        if show_only_positive:
            work = work.loc[work["best_net_profit"] > 0].copy()
        if work.empty:
            st.info("Sem linhas apos filtro de PnL positivo.")
            return

        grouped = (
            work.groupby(["source_root", "symbol", "timeframe", "best_strategy"], as_index=False)
            .agg(
                runs=("run_tag", "count"),
                best_pnl=("best_net_profit", "max"),
                avg_pnl=("best_net_profit", "mean"),
                best_score=("best_score", "max"),
                avg_score=("best_score", "mean"),
                avg_win_rate=("best_win_rate", "mean"),
                avg_dd=("best_max_drawdown", "mean"),
                last_run=("created_at_utc", "max"),
            )
            .sort_values(["best_pnl", "best_score"], ascending=[False, False], na_position="last")
        )
        show = pd.DataFrame(
            {
                "Pasta": grouped["source_root"],
                "Ativo": grouped["symbol"],
                "TF": grouped["timeframe"],
                "Estrategia": grouped["best_strategy"],
                "Execucoes": grouped["runs"].astype(int),
                "Melhor PnL": grouped["best_pnl"].round(2),
                "PnL medio": grouped["avg_pnl"].round(2),
                "Melhor pontuacao": grouped["best_score"].round(2),
                "Pontuacao media": grouped["avg_score"].round(2),
                "Win medio %": (100.0 * grouped["avg_win_rate"]).round(2),
                "DD medio": grouped["avg_dd"].round(2),
                "Ultima execucao": pd.to_datetime(grouped["last_run"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S"),
            }
        ).head(max_rows)
        st.dataframe(
            _style_pos_neg(
                show,
                pnl_cols=["Melhor PnL", "PnL medio"],
                score_cols=["Melhor pontuacao", "Pontuacao media"],
            ),
            width="stretch",
            hide_index=True,
        )


def _render_top_runs(container: Any, history_df: pd.DataFrame, show_only_positive: bool) -> None:
    with container:
        st.markdown("#### Top 10 melhores execucoes (todas estrategias)")
        if history_df.empty:
            st.info("Sem historico para montar ranking.")
            return

        work = history_df.copy()
        if show_only_positive:
            work = work.loc[work["best_net_profit"] > 0].copy()
        # Deduplica por hash de arquivo de parametros para evitar entradas repetidas.
        work["params_hash"] = work.apply(
            lambda r: _hash_file(_resolve_project_path(str(r.get("best_params_latest") or "")) or _resolve_project_path(str(r.get("best_params_snapshot") or ""))),
            axis=1,
        )
        work = (
            work.sort_values(["best_net_profit", "best_score"], ascending=False, na_position="last")
            .drop_duplicates(subset=["params_hash"], keep="first")
            .head(10)
        )
        if work.empty:
            st.info("Sem linhas para os filtros atuais.")
            return

        table = pd.DataFrame(
            {
                "Rank": range(1, len(work) + 1),
                "Pasta": work["source_root"],
                "Ativo": work["symbol"],
                "TF": work["timeframe"],
                "Estrategia": work["best_strategy"],
                "PnL": work["best_net_profit"].round(2),
                "Score": work["best_score"].round(2),
                "Win %": (100.0 * work["best_win_rate"]).round(2),
                "DD": work["best_max_drawdown"].round(2),
                "Run": work["run_tag"],
            }
        )
        st.dataframe(_style_pos_neg(table, pnl_cols=["PnL"], score_cols=["Score"]), width="stretch", hide_index=True)

        st.markdown("**Downloads das top 10**")
        for idx, (_, row) in enumerate(work.iterrows()):
            params_path = _resolve_project_path(str(row.get("best_params_latest") or "")) or _resolve_project_path(
                str(row.get("best_params_snapshot") or "")
            )
            summary_path = _resolve_project_path(str(row.get("summary_snapshot") or ""))
            plot_path = _resolve_project_path(str(row.get("best_plot_snapshot") or ""))
            label = f"{row.get('run_tag', '')} | {row.get('best_strategy', '')} | PnL {row.get('best_net_profit', 0):.2f}"

            st.markdown(f"**{idx+1}. {label}**")
            c1, c2, c3 = st.columns(3)
            with c1:
                if params_path and params_path.exists():
                    st.download_button(
                        "Parametros (JSON)",
                        data=params_path.read_bytes(),
                        file_name=f"params_{row.get('run_tag', 'top')}.json",
                        mime="application/json",
                        use_container_width=True,
                        key=f"dl_top_params_{idx}_{row.get('run_tag', '')}",
                    )
                else:
                    st.button("Parametros (JSON)", disabled=True, use_container_width=True, key=f"dl_top_params_{idx}_{row.get('run_tag', '')}_disabled")
            with c2:
                if summary_path and summary_path.exists():
                    st.download_button(
                        "Resumo (CSV)",
                        data=summary_path.read_bytes(),
                        file_name=f"resumo_{row.get('run_tag', 'top')}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key=f"dl_top_summary_{idx}_{row.get('run_tag', '')}",
                    )
                else:
                    st.button("Resumo (CSV)", disabled=True, use_container_width=True, key=f"dl_top_summary_{idx}_{row.get('run_tag', '')}_disabled")
            with c3:
                if plot_path and plot_path.exists():
                    st.download_button(
                        "Resumo (PNG)",
                        data=plot_path.read_bytes(),
                        file_name=f"resumo_{row.get('run_tag', 'top')}.png",
                        mime="image/png",
                        use_container_width=True,
                        key=f"dl_top_png_{idx}_{row.get('run_tag', '')}",
                    )
                else:
                    st.button("Resumo (PNG)", disabled=True, use_container_width=True, key=f"dl_top_png_{idx}_{row.get('run_tag', '')}_disabled")


def _render_history_table(
    container: Any,
    history_df: pd.DataFrame,
    show_only_positive: bool,
    max_rows: int,
) -> dict[str, Any] | None:
    with container:
        st.subheader("Historico de execucoes")
        if history_df.empty:
            st.info("Sem best_history carregado.")
            return None

        work = history_df.copy()
        c1, c2, c3 = st.columns(3)
        with c1:
            symbol_opts = ["Todos"] + sorted(work["symbol"].dropna().astype(str).unique().tolist())
            symbol = st.selectbox("Ativo", options=symbol_opts, index=0, key="viewer_hist_symbol")
        with c2:
            tf_opts = ["Todos"] + sorted(work["timeframe"].dropna().astype(str).unique().tolist())
            timeframe = st.selectbox("Timeframe", options=tf_opts, index=0, key="viewer_hist_tf")
        with c3:
            strat_opts = ["Todas"] + sorted(work["best_strategy"].dropna().astype(str).unique().tolist())
            strategy = st.selectbox("Estrategia", options=strat_opts, index=0, key="viewer_hist_strategy")

        if symbol != "Todos":
            work = work.loc[work["symbol"].astype(str) == symbol].copy()
        if timeframe != "Todos":
            work = work.loc[work["timeframe"].astype(str) == timeframe].copy()
        if strategy != "Todas":
            work = work.loc[work["best_strategy"].astype(str) == strategy].copy()
        if show_only_positive:
            work = work.loc[work["best_net_profit"] > 0].copy()

        if work.empty:
            st.info("Sem runs para os filtros selecionados.")
            return None

        show = pd.DataFrame(
            {
                "Data UTC": pd.to_datetime(work["created_at_utc"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S"),
                "Pasta": work["source_root"].astype(str),
                "Execucao": work["run_tag"].astype(str),
                "Ativo": work["symbol"].astype(str),
                "TF": work["timeframe"].astype(str),
                "Estrategia": work["best_strategy"].astype(str),
                "Lucro (R$)": work["best_net_profit"].round(2),
                "Pontuacao": work["best_score"].round(2),
                "Trades": work["best_trade_count"].fillna(0).astype(int),
                "Win %": (100.0 * work["best_win_rate"]).round(2),
                "Drawdown (R$)": work["best_max_drawdown"].round(2),
            }
        ).head(max_rows)
        st.dataframe(_style_pos_neg(show, pnl_cols=["Lucro (R$)"], score_cols=["Pontuacao"]), width="stretch", hide_index=True)

        options = work["run_tag"].astype(str) + " | " + work["source_root"].astype(str) + " | " + work["best_strategy"].astype(str)
        selected_label = st.selectbox("Selecionar execucao para ver detalhes", options=options.tolist(), index=0)
        idx = int(options[options == selected_label].index[0])
        row = work.loc[idx].to_dict()
        return row


def _render_latest_summaries(container: Any, summary_df: pd.DataFrame, show_only_positive: bool, max_rows: int) -> None:
    with container:
        st.subheader("Resumos atuais (summary_<tf>.csv)")
        if summary_df.empty:
            st.info("Sem summary latest encontrado.")
            return
        work = summary_df.copy()
        if show_only_positive and "net_profit" in work.columns:
            work = work.loc[pd.to_numeric(work["net_profit"], errors="coerce") > 0].copy()
        if work.empty:
            st.info("Sem linhas apos filtro.")
            return

        cols = [c for c in ["source_root", "symbol", "timeframe", "strategy", "score", "net_profit", "trade_count", "win_rate", "max_drawdown", "summary_file"] if c in work.columns]
        show = work[cols].head(max_rows).copy()
        rename = {
            "source_root": "Pasta",
            "symbol": "Ativo",
            "timeframe": "TF",
            "strategy": "Estrategia",
            "score": "Pontuacao",
            "net_profit": "Lucro (R$)",
            "trade_count": "Trades",
            "win_rate": "Win %",
            "max_drawdown": "Drawdown (R$)",
            "summary_file": "Arquivo",
        }
        show.rename(columns=rename, inplace=True)
        if "Win %" in show.columns:
            show["Win %"] = (100.0 * pd.to_numeric(show["Win %"], errors="coerce")).round(2)
        for col in ["Pontuacao", "Lucro (R$)", "Drawdown (R$)"]:
            if col in show.columns:
                show[col] = pd.to_numeric(show[col], errors="coerce").round(2)
        if "Trades" in show.columns:
            show["Trades"] = pd.to_numeric(show["Trades"], errors="coerce").fillna(0).astype(int)

        st.dataframe(_style_pos_neg(show, pnl_cols=["Lucro (R$)"], score_cols=["Pontuacao"]), width="stretch", hide_index=True)


def _render_artifacts(container: Any, selected_row: dict[str, Any] | None) -> None:
    with container:
        st.subheader("Resumo e artefatos")
        if not selected_row:
            st.info("Selecione uma execucao na aba Historico de execucoes.")
            return

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Ativo/TF", f"{selected_row.get('symbol', '-')}/{selected_row.get('timeframe', '-')}")
            st.metric("Estrategia", str(selected_row.get("best_strategy", "-")))
        with c2:
            st.metric("Lucro", _fmt_brl(selected_row.get("best_net_profit", 0.0)))
            st.metric("Pontuacao", f"{float(pd.to_numeric(selected_row.get('best_score', 0.0), errors='coerce')):.2f}")
        with c3:
            st.metric("Trades", f"{int(pd.to_numeric(selected_row.get('best_trade_count', 0), errors='coerce') or 0)}")
            st.metric("Drawdown", _fmt_brl(selected_row.get("best_max_drawdown", 0.0)))

        eq_csv_best = _resolve_project_path(str(selected_row.get("best_equity_snapshot", "")))
        eq_png_best = _resolve_project_path(str(selected_row.get("best_plot_snapshot", "")))
        summary_csv = _resolve_project_path(str(selected_row.get("summary_snapshot", "")))
        params_json = _resolve_project_path(str(selected_row.get("best_params_snapshot", "")))
        tf_dir = summary_csv.parent if summary_csv and summary_csv.exists() else None
        timeframe_raw = str(selected_row.get("timeframe", "5m"))
        try:
            timeframe = normalize_timeframe_label(timeframe_raw)
        except Exception:
            timeframe = str(timeframe_raw).strip() or "5m"

        params_payload: dict[str, Any] = {}
        if params_json and params_json.exists():
            try:
                params_payload = json.loads(params_json.read_text(encoding="utf-8"))
            except Exception:
                params_payload = {}

        default_strategy = str(selected_row.get("best_strategy", "")).strip() or "-"
        if tf_dir is not None:
            strategy_options = _merge_strategy_options(
                summary_dir=tf_dir,
                timeframe=timeframe,
                params_payload=params_payload,
                fallback_strategy=default_strategy,
            )
        else:
            strategy_options = _extract_strategy_options(params_payload, fallback_strategy=default_strategy)
        selected_strategy = st.selectbox(
            "Estrategia para visualizar/exportar",
            options=strategy_options,
            index=max(0, strategy_options.index(default_strategy)) if default_strategy in strategy_options else 0,
            key="viewer_artifacts_strategy",
        )

        top_candidates: list[dict[str, Any]] = []
        if tf_dir is not None:
            top_candidates = _load_top_param_candidates(
                tf_dir=tf_dir,
                timeframe=timeframe,
                strategy_name=selected_strategy,
                limit=10,
            )
        if not top_candidates:
            top_candidates = _load_top_param_candidates_from_history(
                selected_row=selected_row,
                timeframe=timeframe,
                strategy_name=selected_strategy,
                limit=10,
            )
        param_options = ["Validado (best_params)"]
        for item in top_candidates:
            param_options.append(
                f"Top {item['rank']} | score {float(item['avg_test_score']):.2f} | "
                f"pnl {float(item['avg_test_net_profit']):.2f} | janelas {int(item['windows'])}"
            )
        selected_param_label = st.selectbox(
            "Parametro validado para exportar",
            options=param_options,
            index=0,
            key="viewer_artifacts_param_source",
        )
        if top_candidates:
            with st.expander("Top candidatos (opcional)", expanded=False):
                top_view = pd.DataFrame(
                    [
                        {
                            "Rank": int(item["rank"]),
                            "Score medio": round(float(item["avg_test_score"]), 2),
                            "PnL medio": round(float(item["avg_test_net_profit"]), 2),
                            "Janelas": int(item["windows"]),
                        }
                        for item in top_candidates
                    ]
                )
                st.dataframe(
                    _style_pos_neg(top_view, pnl_cols=["PnL medio"], score_cols=["Score medio"]),
                    width="stretch",
                    hide_index=True,
                )
        strategy_params = _extract_params_for_strategy(params_payload, selected_strategy)
        if selected_param_label != "Validado (best_params)" and top_candidates:
            top_idx = max(0, min(len(top_candidates) - 1, param_options.index(selected_param_label) - 1))
            strategy_params = top_candidates[top_idx].get("params", {})

        eq_csv = _resolve_equity_csv_for_strategy(
            summary_csv=summary_csv,
            selected_row=selected_row,
            strategy_name=selected_strategy,
            fallback_best_csv=eq_csv_best,
        )
        eq_png = _resolve_equity_png_for_strategy(
            summary_csv=summary_csv,
            selected_row=selected_row,
            strategy_name=selected_strategy,
            fallback_best_png=eq_png_best,
        )

        summary_payload = _build_profit_like_summary_payload(
            selected_row=selected_row,
            summary_csv=summary_csv,
            equity_csv=eq_csv,
            strategy_name=selected_strategy,
        )
        if summary_payload:
            _render_profit_like_summary(summary_payload)
            st.caption(
                f"Mostrando apenas a estrategia selecionada: {selected_strategy} | Parametro: {selected_param_label}"
            )

        st.markdown("### Rodar parametro selecionado")
        run_key = f"{selected_row.get('run_tag', 'snapshot')}"

        run_sel_c1, run_sel_c2 = st.columns(2)
        with run_sel_c1:
            run_strategy = st.selectbox(
                "Estrategia (simulacao)",
                options=strategy_options,
                index=max(0, strategy_options.index(selected_strategy)) if selected_strategy in strategy_options else 0,
                key=f"viewer_run_strategy_{run_key}",
            )
        run_top_candidates: list[dict[str, Any]] = []
        if tf_dir is not None:
            run_top_candidates = _load_top_param_candidates(
                tf_dir=tf_dir,
                timeframe=timeframe,
                strategy_name=run_strategy,
                limit=10,
            )
        if not run_top_candidates:
            run_top_candidates = _load_top_param_candidates_from_history(
                selected_row=selected_row,
                timeframe=timeframe,
                strategy_name=run_strategy,
                limit=10,
            )
        run_param_options = ["Validado (best_params)"]
        for item in run_top_candidates:
            run_param_options.append(
                f"Top {item['rank']} | score {float(item['avg_test_score']):.2f} | "
                f"pnl {float(item['avg_test_net_profit']):.2f} | janelas {int(item['windows'])}"
            )
        with run_sel_c2:
            run_selected_param_label = st.selectbox(
                "Parametros (simulacao)",
                options=run_param_options,
                index=0,
                key=f"viewer_run_param_source_{run_key}",
            )

        run_strategy_params = _extract_params_for_strategy(params_payload, run_strategy)
        if run_selected_param_label != "Validado (best_params)" and run_top_candidates:
            run_top_idx = max(0, min(len(run_top_candidates) - 1, run_param_options.index(run_selected_param_label) - 1))
            run_strategy_params = run_top_candidates[run_top_idx].get("params", {})
        st.caption("Parametros que serao usados na simulacao:")
        st.code(
            json.dumps(run_strategy_params, indent=2, ensure_ascii=False, sort_keys=True)
            if run_strategy_params
            else "{}",
            language="json",
        )

        default_start = date_cls(2025, 1, 1)
        default_end = pd.Timestamp.now(tz="UTC").date()
        run_eq_csv = _resolve_equity_csv_for_strategy(
            summary_csv=summary_csv,
            selected_row=selected_row,
            strategy_name=run_strategy,
            fallback_best_csv=eq_csv_best,
        )
        if run_eq_csv and run_eq_csv.exists():
            eq_seed = _read_csv_safe(run_eq_csv)
            if not eq_seed.empty and "datetime" in eq_seed.columns:
                dt_seed = pd.to_datetime(eq_seed["datetime"], errors="coerce").dropna()
                if not dt_seed.empty:
                    default_start = dt_seed.min().date()
                    default_end = dt_seed.max().date()

        sim_c1, sim_c2, sim_c3 = st.columns(3)
        with sim_c1:
            sim_data_root = st.text_input(
                "Pasta data (simulacao)",
                value="data",
                key=f"viewer_sim_data_root_{run_key}",
            )
            sim_mode = st.radio(
                "Modo simulacao",
                options=["OHLC", "Tick a Tick"],
                horizontal=True,
                key=f"viewer_sim_mode_{run_key}",
            )
            sim_tick_root = st.text_input(
                "Pasta tick-a-tick (simulacao)",
                value="assets/dados-historicos-winfut/tik-a-tik",
                disabled=not sim_mode.startswith("Tick"),
                key=f"viewer_sim_tick_root_{run_key}",
            )
        with sim_c2:
            sim_start_date = st.date_input(
                "Inicio simulacao",
                value=default_start,
                key=f"viewer_sim_start_{run_key}",
            )
            sim_end_date = st.date_input(
                "Fim simulacao",
                value=default_end,
                key=f"viewer_sim_end_{run_key}",
            )
            sim_initial_capital = st.number_input(
                "Capital inicial simulacao",
                min_value=1000.0,
                max_value=50_000_000.0,
                value=100_000.0,
                step=1000.0,
                key=f"viewer_sim_capital_{run_key}",
            )
        with sim_c3:
            sim_contracts = st.number_input(
                "Contratos simulacao",
                min_value=1,
                max_value=100,
                value=1,
                step=1,
                key=f"viewer_sim_contracts_{run_key}",
            )
            sim_point_value = st.number_input(
                "Valor ponto simulacao",
                min_value=0.01,
                max_value=100.0,
                value=0.2,
                step=0.01,
                key=f"viewer_sim_point_{run_key}",
            )
            sim_session_start = st.text_input(
                "Inicio sessao simulacao",
                value="09:00",
                key=f"viewer_sim_session_start_{run_key}",
            )
            sim_session_end = st.text_input(
                "Fim sessao simulacao",
                value="17:40",
                key=f"viewer_sim_session_end_{run_key}",
            )
            sim_close_on_end = st.checkbox(
                "Fechar no fim da sessao",
                value=True,
                key=f"viewer_sim_close_end_{run_key}",
            )

        run_param_btn = st.button(
            "Rodar parametro selecionado e mostrar resultados",
            type="primary",
            use_container_width=True,
            key=f"viewer_run_selected_param_{run_key}",
        )
        if run_param_btn:
            if sim_end_date < sim_start_date:
                st.error("Periodo invalido: fim menor que inicio.")
            elif not run_strategy_params:
                st.error("Parametros vazios para executar simulacao.")
            else:
                strategy_spec = STRATEGIES.get(run_strategy)
                if strategy_spec is None:
                    st.error(f"Estrategia nao registrada: {run_strategy}")
                else:
                    with st.spinner("Rodando simulacao com parametro selecionado..."):
                        try:
                            sim_df = load_timeframe_data(
                                LoaderConfig(
                                    data_root=Path(sim_data_root),
                                    symbol=str(selected_row.get("symbol", "WINFUT")),
                                    start=pd.Timestamp(sim_start_date),
                                    end=pd.Timestamp(sim_end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1),
                                ),
                                timeframe,
                            )
                        except Exception as exc:
                            st.error(f"Falha ao carregar dados da simulacao: {exc}")
                            sim_df = pd.DataFrame()

                        if not sim_df.empty:
                            base_cfg = BacktestConfig(
                                initial_capital=float(sim_initial_capital),
                                contracts=int(sim_contracts),
                                point_value=float(sim_point_value),
                                execution_mode="tick" if sim_mode.startswith("Tick") else "ohlc",
                                tick_data_root=(sim_tick_root.strip() or None) if sim_mode.startswith("Tick") else None,
                                tick_symbol=str(selected_row.get("symbol", "WINFUT")),
                                session_start=sim_session_start.strip() or None,
                                session_end=sim_session_end.strip() or None,
                                close_on_session_end=bool(sim_close_on_end),
                            )
                            runtime_cfg = build_runtime_config_for_params(base_cfg, run_strategy_params)
                            runtime_cfg = replace(
                                runtime_cfg,
                                initial_capital=float(sim_initial_capital),
                                contracts=int(sim_contracts),
                                point_value=float(sim_point_value),
                                execution_mode="tick" if sim_mode.startswith("Tick") else "ohlc",
                                tick_data_root=(sim_tick_root.strip() or None) if sim_mode.startswith("Tick") else None,
                                tick_symbol=str(selected_row.get("symbol", "WINFUT")),
                                session_start=sim_session_start.strip() or None,
                                session_end=sim_session_end.strip() or None,
                                close_on_session_end=bool(sim_close_on_end),
                            )
                            sim_signals = generate_signals_with_time_filter(sim_df, strategy_spec, run_strategy_params)
                            sim_result = run_backtest(
                                df=sim_df,
                                signals=sim_signals,
                                config=runtime_cfg,
                                strategy_name=run_strategy,
                                strategy_params=run_strategy_params,
                            )
                            sim_metrics = compute_metrics(
                                trades=sim_result.trades,
                                equity_curve=sim_result.equity_curve,
                                initial_capital=float(runtime_cfg.initial_capital),
                                score_config=ScoreConfig(),
                            )
                            sim_payload = _build_profit_like_summary_payload_from_runtime(
                                selected_row=selected_row,
                                strategy_name=run_strategy,
                                timeframe=timeframe,
                                metrics=sim_metrics,
                                trades_df=sim_result.trades,
                                equity_df=sim_result.equity_curve,
                                initial_capital=float(runtime_cfg.initial_capital),
                                contracts=int(runtime_cfg.contracts),
                            )
                            st.markdown("#### Resultado da simulacao")
                            _render_profit_like_summary(sim_payload)
                            if not sim_result.equity_curve.empty and {"datetime", "equity"}.issubset(sim_result.equity_curve.columns):
                                sim_fig = _build_pnl_curve_figure(
                                    sim_result.equity_curve,
                                    title=f"Curva de patrimonio (simulacao: {run_strategy})",
                                )
                                st.plotly_chart(sim_fig, width="stretch")
                            if not sim_result.trades.empty:
                                st.markdown("**Operacoes da simulacao**")
                                tview = sim_result.trades.copy()
                                tview["entry_time"] = pd.to_datetime(tview["entry_time"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
                                tview["exit_time"] = pd.to_datetime(tview["exit_time"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
                                for col in ["entry_price", "exit_price", "pnl_gross", "costs", "pnl_net"]:
                                    if col in tview.columns:
                                        tview[col] = pd.to_numeric(tview[col], errors="coerce").round(2)
                                cols = [c for c in ["entry_time", "exit_time", "direction", "entry_price", "exit_price", "reason", "pnl_net"] if c in tview.columns]
                                tshow = tview[cols].rename(
                                    columns={
                                        "entry_time": "Abertura",
                                        "exit_time": "Fechamento",
                                        "direction": "Lado",
                                        "entry_price": "Preco Entrada",
                                        "exit_price": "Preco Saida",
                                        "reason": "Motivo",
                                        "pnl_net": "Resultado (R$)",
                                    }
                                )
                                st.dataframe(
                                    _style_pos_neg(tshow, pnl_cols=["Resultado (R$)"], score_cols=[]),
                                    width="stretch",
                                    hide_index=True,
                                )

        st.markdown("### Exportar")
        p1, p2, p3 = st.columns(3)
        params_text = json.dumps(strategy_params, indent=2, ensure_ascii=False, sort_keys=True)
        with p1:
            st.download_button(
                "Baixar parametros (JSON)",
                data=params_text.encode("utf-8"),
                file_name=f"parametros_{selected_strategy}.json",
                mime="application/json",
                use_container_width=True,
            )

        curve_png_bytes: bytes | None = None
        if eq_png and eq_png.exists():
            curve_png_bytes = eq_png.read_bytes()
        elif eq_csv and eq_csv.exists():
            eq_tmp = _read_csv_safe(eq_csv)
            curve_png_bytes = _build_curve_png_bytes(eq_tmp, f"Curva - {selected_strategy}")
        with p2:
            if curve_png_bytes:
                st.download_button(
                    "Baixar foto da curva (PNG)",
                    data=curve_png_bytes,
                    file_name=f"curva_capital_{selected_strategy}.png",
                    mime="image/png",
                    use_container_width=True,
                )
            else:
                st.button("Baixar foto da curva (PNG)", disabled=True, use_container_width=True)

        summary_png_bytes = _build_summary_png_bytes(summary_payload) if summary_payload else None
        with p3:
            if summary_png_bytes:
                st.download_button(
                    "Baixar foto do resumo (PNG)",
                    data=summary_png_bytes,
                    file_name=f"resumo_{selected_strategy}.png",
                    mime="image/png",
                    use_container_width=True,
                )
            else:
                st.button("Baixar foto do resumo (PNG)", disabled=True, use_container_width=True)

        if eq_csv and eq_csv.exists():
            eq = _read_csv_safe(eq_csv)
            if not eq.empty and {"datetime", "equity"}.issubset(eq.columns):
                fig = _build_pnl_curve_figure(eq, title=f"Curva de patrimonio ({selected_strategy})")
                st.plotly_chart(fig, width="stretch")
        if eq_png and eq_png.exists():
            st.image(str(eq_png), caption=str(eq_png.relative_to(PROJECT_ROOT)), width="stretch")

        with st.expander("Arquivos e tabelas (opcional)", expanded=False):
            st.markdown("**Arquivos**")
            for p in [summary_csv, params_json, eq_csv, eq_png]:
                if p and p.exists():
                    st.code(str(p.relative_to(PROJECT_ROOT)), language="text")

            if summary_csv and summary_csv.exists():
                df = _read_csv_safe(summary_csv)
                if not df.empty:
                    st.markdown("**Resumo da execucao (snapshot)**")
                    st.dataframe(df.head(30), width="stretch", hide_index=True)

            st.markdown("**Parametros da estrategia selecionada**")
            st.code(params_text if params_text.strip() else "{}", language="json")


def _render_tick_daily_validation(container: Any, selected_row: dict[str, Any] | None) -> None:
    with container:
        st.subheader("Validacao Tick a Tick por dia")
        if not selected_row:
            st.info("Selecione uma execucao na aba Historico de execucoes para carregar estrategia/parametros.")
            return

        summary_csv = _resolve_project_path(str(selected_row.get("summary_snapshot", "")))
        params_json = _resolve_project_path(str(selected_row.get("best_params_snapshot", "")))
        if not summary_csv or not summary_csv.exists():
            st.warning("Summary snapshot nao encontrado para a execucao selecionada.")
            return
        if not params_json or not params_json.exists():
            st.warning("Best params snapshot nao encontrado para a execucao selecionada.")
            return

        try:
            params_payload = json.loads(params_json.read_text(encoding="utf-8"))
        except Exception:
            st.warning("Nao foi possivel ler parametros da execucao selecionada.")
            return

        default_strategy = str(selected_row.get("best_strategy", "")).strip() or "-"
        timeframe_raw = str(selected_row.get("timeframe", "5m"))
        try:
            timeframe = normalize_timeframe_label(timeframe_raw)
        except Exception as exc:
            st.error(f"Timeframe invalido: {timeframe_raw} ({exc})")
            return

        tf_dir = summary_csv.parent
        strategy_options = _merge_strategy_options(
            summary_dir=tf_dir,
            timeframe=timeframe,
            params_payload=params_payload,
            fallback_strategy=default_strategy,
        )

        c1, c2 = st.columns(2)
        with c1:
            selected_strategy = st.selectbox(
                "Estrategia validada",
                options=strategy_options,
                index=max(0, strategy_options.index(default_strategy)) if default_strategy in strategy_options else 0,
                key="tick_daily_strategy",
            )
            data_root = st.text_input("Pasta candles (data root)", value="data", key="tick_daily_data_root")
            tick_root = st.text_input(
                "Pasta tick-a-tick",
                value="assets/dados-historicos-winfut/tik-a-tik",
                key="tick_daily_tick_root",
            )
            top_candidates = _load_top_param_candidates(tf_dir=tf_dir, timeframe=timeframe, strategy_name=selected_strategy, limit=10)
            param_options = ["Validado (best_params)"]
            for item in top_candidates:
                label = (
                    f"Top {item['rank']} | score {float(item['avg_test_score']):.2f} | "
                    f"pnl {float(item['avg_test_net_profit']):.2f} | janelas {int(item['windows'])}"
                )
                param_options.append(label)
            selected_param_label = st.selectbox(
                "Parametros para Tick a Tick",
                options=param_options,
                index=0,
                key="tick_daily_param_source",
            )
            if top_candidates:
                top_view = pd.DataFrame(
                    [
                        {
                            "Rank": int(item["rank"]),
                            "Score medio": round(float(item["avg_test_score"]), 2),
                            "PnL medio": round(float(item["avg_test_net_profit"]), 2),
                            "Janelas": int(item["windows"]),
                        }
                        for item in top_candidates
                    ]
                )
                st.dataframe(
                    _style_pos_neg(top_view, pnl_cols=["PnL medio"], score_cols=["Score medio"]),
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.caption("Sem top10 no walkforward_topk para esta estrategia neste output.")
        with c2:
            initial_capital = st.number_input(
                "Capital inicial por dia (R$)",
                min_value=1000.0,
                max_value=50_000_000.0,
                value=100_000.0,
                step=1000.0,
                key="tick_daily_initial_capital",
            )
            point_value = st.number_input(
                "Valor do ponto",
                min_value=0.01,
                max_value=100.0,
                value=0.2,
                step=0.01,
                key="tick_daily_point_value",
            )
            contracts = st.number_input(
                "Contratos",
                min_value=1,
                max_value=100,
                value=1,
                step=1,
                key="tick_daily_contracts",
            )

        tick_days = _discover_tick_days(Path(tick_root), str(selected_row.get("symbol", "")))
        st.caption(f"Dias detectados em tick-a-tick: {len(tick_days)}")
        if not tick_days:
            st.warning("Nenhum dia de tick detectado na pasta informada.")
            return

        run_clicked = st.button("Rodar Tick a Tick (todos os dias)", type="primary", key="tick_daily_run_btn")
        if not run_clicked:
            return

        strategy_spec = STRATEGIES.get(selected_strategy)
        if strategy_spec is None:
            st.error(f"Estrategia nao registrada: {selected_strategy}")
            return

        params = _extract_params_for_strategy(params_payload, selected_strategy)
        if selected_param_label != "Validado (best_params)" and top_candidates:
            chosen_idx = max(0, min(len(top_candidates) - 1, param_options.index(selected_param_label) - 1))
            params = top_candidates[chosen_idx].get("params", {})
        if not params:
            st.error(f"Parametros validados nao encontrados para {selected_strategy}.")
            return

        day_min = pd.Timestamp(min(tick_days))
        day_max = pd.Timestamp(max(tick_days)) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        ohlc_df = pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])
        try:
            ohlc_df = load_timeframe_data(
                LoaderConfig(
                    data_root=Path(data_root),
                    symbol=str(selected_row.get("symbol", "WINFUT")),
                    start=day_min,
                    end=day_max,
                ),
                timeframe,
            )
        except Exception as exc:
            try:
                ohlc_df = load_timeframe_data(
                    LoaderConfig(
                        data_root=Path(data_root),
                        symbol=str(selected_row.get("symbol", "WINFUT")),
                        start=None,
                        end=None,
                    ),
                    timeframe,
                )
                st.warning(
                    "Nao encontrei candles no range dos ticks. "
                    "Vou usar candles do arquivo inteiro e/ou gerar candles pelos ticks por dia."
                )
            except Exception as exc2:
                st.warning(
                    "Nao foi possivel carregar candles da pasta data. "
                    f"Vou tentar gerar candles diretamente dos ticks. Detalhe: {exc2}"
                )

        base_cfg = BacktestConfig(
            initial_capital=float(initial_capital),
            contracts=int(contracts),
            point_value=float(point_value),
            execution_mode="tick",
            tick_data_root=str(Path(tick_root)),
            tick_symbol=str(selected_row.get("symbol", "WINFUT")),
            entry_mode="next_open",
            session_start="09:00",
            session_end="17:40",
            close_on_session_end=True,
        )
        runtime_cfg = build_runtime_config_for_params(base_cfg, params)
        runtime_cfg = replace(
            runtime_cfg,
            execution_mode="tick",
            tick_data_root=str(Path(tick_root)),
            tick_symbol=str(selected_row.get("symbol", "WINFUT")),
            initial_capital=float(initial_capital),
            point_value=float(point_value),
            contracts=int(contracts),
        )

        status = st.empty()
        progress = st.progress(0)
        rows: list[dict[str, Any]] = []
        all_trades: list[pd.DataFrame] = []
        all_equity: list[pd.DataFrame] = []

        ohlc_dt = pd.to_datetime(ohlc_df["datetime"], errors="coerce") if not ohlc_df.empty else pd.Series(dtype="datetime64[ns]")
        for idx, d in enumerate(tick_days, start=1):
            status.info(f"Rodando {idx}/{len(tick_days)} - {d.isoformat()}")
            progress.progress(int(100 * idx / max(1, len(tick_days))))

            ticks_df = load_ticks_between(
                tick_root=Path(tick_root),
                symbol=str(selected_row.get("symbol", "WINFUT")),
                start=pd.Timestamp(d),
                end=pd.Timestamp(d) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1),
            )
            if ticks_df.empty:
                rows.append(
                    {
                        "dia": d.isoformat(),
                        "status": "sem_tick",
                        "ticks": 0,
                        "trades": 0,
                        "net_profit": 0.0,
                        "win_rate": 0.0,
                        "max_drawdown": 0.0,
                        "score": 0.0,
                    }
                )
                continue

            day_df = ohlc_df[ohlc_dt.dt.date == d].copy() if not ohlc_df.empty else pd.DataFrame()
            if day_df.empty:
                day_df = _build_ohlc_from_ticks(ticks_df, timeframe=timeframe)
            if day_df.empty:
                rows.append(
                    {
                        "dia": d.isoformat(),
                        "status": "sem_candle",
                        "ticks": int(len(ticks_df)),
                        "trades": 0,
                        "net_profit": 0.0,
                        "win_rate": 0.0,
                        "max_drawdown": 0.0,
                        "score": 0.0,
                    }
                )
                continue

            signals = generate_signals_with_time_filter(day_df, strategy_spec, params)
            result = run_backtest(
                df=day_df,
                signals=signals,
                config=runtime_cfg,
                strategy_name=selected_strategy,
                strategy_params=params,
            )
            metrics = compute_metrics(
                trades=result.trades,
                equity_curve=result.equity_curve,
                initial_capital=float(runtime_cfg.initial_capital),
                score_config=ScoreConfig(),
            )
            rows.append(
                {
                    "dia": d.isoformat(),
                    "status": "ok",
                    "ticks": int(len(ticks_df)),
                    "trades": int(metrics.get("trade_count", 0.0)),
                    "net_profit": float(metrics.get("net_profit", 0.0)),
                    "win_rate": float(metrics.get("win_rate", 0.0)),
                    "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
                    "score": float(metrics.get("score", 0.0)),
                    "profit_factor": float(metrics.get("profit_factor", 0.0)),
                }
            )
            if not result.trades.empty:
                t = result.trades.copy()
                t["day"] = d.isoformat()
                all_trades.append(t)
            if not result.equity_curve.empty:
                e = result.equity_curve.copy()
                e["day"] = d.isoformat()
                all_equity.append(e)

        progress.empty()
        status.success("Tick a Tick diario concluido.")
        daily_df = pd.DataFrame(rows)
        if daily_df.empty:
            st.warning("Nenhum resultado gerado.")
            return

        ok_df = daily_df[daily_df["status"] == "ok"].copy()
        total_days = int(len(daily_df))
        ok_days = int(len(ok_df))
        net_total = float(ok_df["net_profit"].sum()) if not ok_df.empty else 0.0
        win_days_pct = float((ok_df["net_profit"] > 0).mean() * 100.0) if not ok_df.empty else 0.0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Dias com tick", f"{ok_days}/{total_days}")
        m2.metric("Lucro total", _fmt_brl(net_total))
        m3.metric("Dia medio", _fmt_brl(float(ok_df["net_profit"].mean()) if not ok_df.empty else 0.0))
        m4.metric("Dias positivos", f"{win_days_pct:.2f}%")

        view = daily_df.copy()
        view["win_rate"] = (100.0 * pd.to_numeric(view["win_rate"], errors="coerce")).round(2)
        view["net_profit"] = pd.to_numeric(view["net_profit"], errors="coerce").round(2)
        view["max_drawdown"] = pd.to_numeric(view["max_drawdown"], errors="coerce").round(2)
        view["score"] = pd.to_numeric(view["score"], errors="coerce").round(2)
        show = view.rename(
            columns={
                "dia": "Dia",
                "status": "Status",
                "ticks": "Ticks",
                "trades": "Trades",
                "net_profit": "Lucro (R$)",
                "win_rate": "Win %",
                "max_drawdown": "Drawdown (R$)",
                "score": "Pontuacao",
                "profit_factor": "Fator Lucro",
            }
        )
        st.dataframe(
            _style_pos_neg(show, pnl_cols=["Lucro (R$)"], score_cols=["Pontuacao"]),
            width="stretch",
            hide_index=True,
        )

        st.download_button(
            "Baixar resumo diario Tick a Tick (CSV)",
            data=daily_df.to_csv(index=False).encode("utf-8"),
            file_name=f"tick_daily_{selected_strategy}_{timeframe}.csv",
            mime="text/csv",
            use_container_width=True,
        )
        if all_trades:
            trades_df = pd.concat(all_trades, ignore_index=True)
            st.download_button(
                "Baixar trades diarios Tick a Tick (CSV)",
                data=trades_df.to_csv(index=False).encode("utf-8"),
                file_name=f"tick_trades_{selected_strategy}_{timeframe}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        if all_equity:
            equity_df = pd.concat(all_equity, ignore_index=True)
            st.download_button(
                "Baixar equity diaria Tick a Tick (CSV)",
                data=equity_df.to_csv(index=False).encode("utf-8"),
                file_name=f"tick_equity_{selected_strategy}_{timeframe}.csv",
                mime="text/csv",
                use_container_width=True,
            )


def _discover_tick_days(tick_root: Path, symbol: str) -> list[date_cls]:
    root = tick_root.resolve()
    if not root.exists():
        return []
    symbol_key = str(symbol).strip().upper()
    out: list[date_cls] = []
    for file_path in sorted(root.glob("*.csv")):
        name_upper = file_path.name.upper()
        if symbol_key and not name_upper.startswith(symbol_key):
            continue
        m = _TICK_FILE_DATE_RE.match(file_path.name)
        if not m:
            continue
        dd, mm, yyyy = m.groups()
        try:
            out.append(date_cls(int(yyyy), int(mm), int(dd)))
        except ValueError:
            continue
    return sorted(set(out))


def _merge_strategy_options(
    summary_dir: Path,
    timeframe: str,
    params_payload: dict[str, Any],
    fallback_strategy: str,
) -> list[str]:
    by_params = _extract_strategy_options(params_payload, fallback_strategy=fallback_strategy)
    by_topk = _discover_strategies_from_topk(summary_dir=summary_dir, timeframe=timeframe)
    merged = sorted(set(by_params + by_topk))
    if fallback_strategy and fallback_strategy not in merged:
        merged.insert(0, fallback_strategy)
    return merged or [fallback_strategy or "estrategia"]


def _discover_strategies_from_topk(summary_dir: Path, timeframe: str) -> list[str]:
    out: list[str] = []
    prefix = f"walkforward_topk_{timeframe}_"
    for file_path in summary_dir.glob(f"{prefix}*.csv"):
        name = file_path.name
        if not name.startswith(prefix):
            continue
        strategy = name[len(prefix) : -4]
        if strategy:
            out.append(strategy)
    return sorted(set(out))


def _load_top_param_candidates(
    tf_dir: Path,
    timeframe: str,
    strategy_name: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    topk_file = tf_dir / f"walkforward_topk_{timeframe}_{strategy_name}.csv"
    if not topk_file.exists():
        return []
    df = _read_csv_safe(topk_file)
    if df.empty or "params_json" not in df.columns:
        return []
    work = df.copy()
    work["test_score_num"] = pd.to_numeric(work.get("test_score"), errors="coerce")
    work["test_net_profit_num"] = pd.to_numeric(work.get("test_net_profit"), errors="coerce")
    grouped = (
        work.groupby("params_json", as_index=False)
        .agg(
            avg_test_score=("test_score_num", "mean"),
            avg_test_net_profit=("test_net_profit_num", "mean"),
            windows=("params_json", "count"),
        )
        .sort_values(["avg_test_score", "avg_test_net_profit", "windows"], ascending=[False, False, False])
        .head(max(1, int(limit)))
        .reset_index(drop=True)
    )
    out: list[dict[str, Any]] = []
    for idx, (_, row) in enumerate(grouped.iterrows(), start=1):
        try:
            params = json.loads(str(row["params_json"]))
        except Exception:
            continue
        if not isinstance(params, dict):
            continue
        out.append(
            {
                "rank": idx,
                "params": params,
                "avg_test_score": float(pd.to_numeric(row.get("avg_test_score"), errors="coerce") or 0.0),
                "avg_test_net_profit": float(pd.to_numeric(row.get("avg_test_net_profit"), errors="coerce") or 0.0),
                "windows": int(pd.to_numeric(row.get("windows"), errors="coerce") or 0),
            }
        )
    return out


def _load_top_param_candidates_from_history(
    selected_row: dict[str, Any],
    timeframe: str,
    strategy_name: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    history_file = _resolve_project_path(str(selected_row.get("history_file", "")))
    if history_file is None or not history_file.exists():
        summary_snapshot = _resolve_project_path(str(selected_row.get("summary_snapshot", "")))
        if summary_snapshot is None:
            return []
        history_file = summary_snapshot.parent / f"best_history_{timeframe}.csv"
        if not history_file.exists():
            return []

    df = _read_csv_safe(history_file)
    if df.empty:
        return []
    work = df.copy()
    if "timeframe" in work.columns:
        work = work[work["timeframe"].astype(str) == str(timeframe)]
    if "best_strategy" in work.columns:
        work = work[work["best_strategy"].astype(str) == str(strategy_name)]
    if work.empty:
        return []

    for col in ["best_score", "best_net_profit", "windows"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    sort_cols = [c for c in ["best_score", "best_net_profit"] if c in work.columns]
    if sort_cols:
        work = work.sort_values(sort_cols, ascending=[False] * len(sort_cols), na_position="last")

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _, row in work.iterrows():
        params_path = _resolve_project_path(str(row.get("best_params_snapshot", ""))) or _resolve_project_path(
            str(row.get("best_params_latest", ""))
        )
        if params_path is None or not params_path.exists():
            continue
        try:
            payload = json.loads(params_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        params = _extract_params_for_strategy(payload, strategy_name)
        if not params:
            continue
        key = json.dumps(params, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "rank": len(out) + 1,
                "params": params,
                "avg_test_score": float(pd.to_numeric(row.get("best_score"), errors="coerce") or 0.0),
                "avg_test_net_profit": float(pd.to_numeric(row.get("best_net_profit"), errors="coerce") or 0.0),
                "windows": int(pd.to_numeric(row.get("windows"), errors="coerce") or 0),
            }
        )
        if len(out) >= max(1, int(limit)):
            break
    return out


def _build_ohlc_from_ticks(ticks_df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if ticks_df.empty or not {"datetime", "price"}.issubset(ticks_df.columns):
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])
    freq = _timeframe_to_pandas_freq(timeframe)
    if not freq:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])
    work = ticks_df.copy()
    work["datetime"] = pd.to_datetime(work["datetime"], errors="coerce")
    work["price"] = pd.to_numeric(work["price"], errors="coerce")
    work["volume"] = pd.to_numeric(work.get("volume", 0.0), errors="coerce").fillna(0.0)
    work = work.dropna(subset=["datetime", "price"]).sort_values("datetime")
    if work.empty:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])
    resampled = (
        work.set_index("datetime")
        .resample(freq, label="right", closed="right")
        .agg(open=("price", "first"), high=("price", "max"), low=("price", "min"), close=("price", "last"), volume=("volume", "sum"))
        .dropna(subset=["open", "high", "low", "close"])
        .reset_index()
    )
    if resampled.empty:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])
    return resampled[["datetime", "open", "high", "low", "close", "volume"]]


def _timeframe_to_pandas_freq(timeframe: str) -> str | None:
    tf = str(timeframe).strip().lower()
    if tf == "daily":
        return "1D"
    if tf == "weekly":
        return "1W"
    m = re.match(r"^(\d+)m$", tf)
    if not m:
        return None
    mins = int(m.group(1))
    if mins <= 0:
        return None
    return f"{mins}min"


def _extract_strategy_options(params_payload: dict[str, Any], fallback_strategy: str) -> list[str]:
    strategies = params_payload.get("strategies", {}) if isinstance(params_payload, dict) else {}
    names = sorted(str(k) for k in strategies.keys()) if isinstance(strategies, dict) else []
    if fallback_strategy and fallback_strategy not in names:
        names.insert(0, fallback_strategy)
    return names or [fallback_strategy or "estrategia"]


def _extract_params_for_strategy(params_payload: dict[str, Any], strategy_name: str) -> dict[str, Any]:
    if not isinstance(params_payload, dict):
        return {}
    strategies = params_payload.get("strategies", {})
    if not isinstance(strategies, dict):
        return {}
    strategy_payload = strategies.get(strategy_name, {})
    if not isinstance(strategy_payload, dict):
        return {}
    out = strategy_payload.get("best_params_from_tests", {})
    return out if isinstance(out, dict) else {}


def _resolve_equity_csv_for_strategy(
    summary_csv: Path | None,
    selected_row: dict[str, Any],
    strategy_name: str,
    fallback_best_csv: Path | None,
) -> Path | None:
    if summary_csv and summary_csv.exists():
        tf_folder = summary_csv.parent
        timeframe = str(selected_row.get("timeframe", "")).strip()
        candidate = tf_folder / f"equity_curve_{timeframe}_{strategy_name}.csv"
        if candidate.exists():
            return candidate
        latest_best = tf_folder / f"equity_curve_{timeframe}_best.csv"
        if strategy_name == str(selected_row.get("best_strategy", "")).strip() and latest_best.exists():
            return latest_best
    if fallback_best_csv and fallback_best_csv.exists():
        return fallback_best_csv
    return None


def _resolve_equity_png_for_strategy(
    summary_csv: Path | None,
    selected_row: dict[str, Any],
    strategy_name: str,
    fallback_best_png: Path | None,
) -> Path | None:
    if summary_csv and summary_csv.exists():
        tf_folder = summary_csv.parent
        timeframe = str(selected_row.get("timeframe", "")).strip()
        candidate = tf_folder / f"equity_{timeframe}_{strategy_name}.png"
        if candidate.exists():
            return candidate
        latest_best = tf_folder / f"equity_{timeframe}_best.png"
        if strategy_name == str(selected_row.get("best_strategy", "")).strip() and latest_best.exists():
            return latest_best
    if fallback_best_png and fallback_best_png.exists():
        return fallback_best_png
    return None


def _build_pnl_curve_figure(eq_raw: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    eq = eq_raw.copy()
    eq["datetime"] = pd.to_datetime(eq["datetime"], errors="coerce")
    eq["equity"] = pd.to_numeric(eq["equity"], errors="coerce")
    eq = eq.dropna(subset=["datetime", "equity"]).sort_values("datetime")
    if eq.empty:
        return fig

    base = float(eq["equity"].iloc[0])
    eq["pnl_curve"] = eq["equity"] - base
    eq["pnl_pos"] = eq["pnl_curve"].where(eq["pnl_curve"] >= 0)
    eq["pnl_neg"] = eq["pnl_curve"].where(eq["pnl_curve"] < 0)

    fig.add_trace(
        go.Scatter(
            x=eq["datetime"],
            y=eq["pnl_pos"],
            mode="lines",
            line=dict(color="#22f08a", width=2.2),
            fill="tozeroy",
            fillcolor="rgba(34, 240, 138, 0.22)",
            name="Positivo",
            hovertemplate="%{x|%d/%m/%Y %H:%M}<br>Valor: R$ %{y:,.2f}<extra>Positivo</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=eq["datetime"],
            y=eq["pnl_neg"],
            mode="lines",
            line=dict(color="#ff4f5e", width=2.2),
            fill="tozeroy",
            fillcolor="rgba(255, 79, 94, 0.20)",
            name="Negativo",
            hovertemplate="%{x|%d/%m/%Y %H:%M}<br>Valor: R$ %{y:,.2f}<extra>Negativo</extra>",
        )
    )

    last_x = eq["datetime"].iloc[-1]
    last_y = float(eq["pnl_curve"].iloc[-1])
    fig.add_hline(y=0.0, line_width=1, line_dash="dot", line_color="rgba(255,255,255,0.25)")
    fig.add_annotation(
        x=last_x,
        y=last_y,
        text=_fmt_brl_short(last_y),
        showarrow=False,
        xanchor="left",
        yanchor="middle",
        xshift=8,
        font=dict(color="#10151c", size=11, family="Segoe UI, Arial, sans-serif"),
        bgcolor="#24e286" if last_y >= 0 else "#ff4f5e",
        bordercolor="#24e286" if last_y >= 0 else "#ff4f5e",
        borderwidth=1,
    )

    fig.update_layout(
        template="plotly_dark",
        title=title,
        paper_bgcolor="#1a1c1f",
        plot_bgcolor="#1a1c1f",
        yaxis_title="Resultado Acumulado (R$)",
        xaxis_title=None,
        hovermode="x unified",
        margin=dict(l=12, r=70, t=48, b=38),
    )
    fig.update_yaxes(side="right", zeroline=True, zerolinecolor="rgba(255,255,255,0.25)")
    fig.update_xaxes(showspikes=True, spikethickness=1, spikecolor="rgba(255,255,255,0.35)")
    return fig


def _build_curve_png_bytes(eq_raw: pd.DataFrame, title: str) -> bytes | None:
    eq = eq_raw.copy()
    if eq.empty or not {"datetime", "equity"}.issubset(eq.columns):
        return None
    eq["datetime"] = pd.to_datetime(eq["datetime"], errors="coerce")
    eq["equity"] = pd.to_numeric(eq["equity"], errors="coerce")
    eq = eq.dropna(subset=["datetime", "equity"]).sort_values("datetime")
    if eq.empty:
        return None
    base = float(eq["equity"].iloc[0])
    pnl = eq["equity"] - base
    pos = pnl.where(pnl >= 0)
    neg = pnl.where(pnl < 0)

    fig, ax = plt.subplots(figsize=(12, 5), dpi=120)
    fig.patch.set_facecolor("#12161d")
    ax.set_facecolor("#12161d")
    ax.plot(eq["datetime"], pos, color="#22f08a", linewidth=1.6)
    ax.plot(eq["datetime"], neg, color="#ff4f5e", linewidth=1.6)
    ax.fill_between(eq["datetime"], 0, pos, where=pos.notna(), color="#22f08a", alpha=0.20)
    ax.fill_between(eq["datetime"], 0, neg, where=neg.notna(), color="#ff4f5e", alpha=0.20)
    ax.axhline(0, color="#cfd7e3", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_title(title, color="#f5f8fb")
    ax.tick_params(colors="#d5dbe6")
    ax.set_ylabel("Resultado Acumulado (R$)", color="#d5dbe6")
    ax.grid(alpha=0.18)
    for spine in ax.spines.values():
        spine.set_color("#2a3342")
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def _build_summary_png_bytes(payload: dict[str, Any]) -> bytes | None:
    if not payload:
        return None
    left = payload.get("left", [])
    right = payload.get("right", [])
    if not isinstance(left, list) or not isinstance(right, list):
        return None

    rows = max(len(left), len(right)) + 3
    fig_h = max(6.0, min(16.0, rows * 0.32))
    fig, ax = plt.subplots(figsize=(14, fig_h), dpi=120)
    fig.patch.set_facecolor("#11161f")
    ax.set_facecolor("#11161f")
    ax.axis("off")
    ax.text(0.01, 0.98, str(payload.get("header", "")), color="#dbe2ee", fontsize=9, va="top", ha="left")

    y = 0.92
    step = 0.045
    for i in range(max(len(left), len(right))):
        if i < len(left):
            l_label, l_val = left[i]
            ax.text(0.01, y, str(l_label), color="#d0d7e4", fontsize=10, va="top", ha="left")
            ax.text(0.36, y, str(l_val), color=_summary_value_color(str(l_val)), fontsize=10, va="top", ha="right", fontweight="bold")
        if i < len(right):
            r_label, r_val = right[i]
            ax.text(0.50, y, str(r_label), color="#d0d7e4", fontsize=10, va="top", ha="left")
            ax.text(0.98, y, str(r_val), color=_summary_value_color(str(r_val)), fontsize=10, va="top", ha="right", fontweight="bold")
        ax.plot([0.01, 0.99], [y - 0.006, y - 0.006], color="#2a3342", linewidth=0.6, alpha=0.7)
        y -= step

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()


def _summary_value_color(text: str) -> str:
    clean = text.strip()
    if clean.startswith("-"):
        return "#ff6b76"
    if clean.startswith("R$") and "R$ 0,00" not in clean:
        return "#24e286"
    return "#eaf0fa"


def _build_profit_like_summary_payload(
    selected_row: dict[str, Any],
    summary_csv: Path | None,
    equity_csv: Path | None,
    strategy_name: str | None = None,
) -> dict[str, Any]:
    if summary_csv is None or not summary_csv.exists():
        return {}
    summary_df = _read_csv_safe(summary_csv)
    if summary_df.empty:
        return {}

    strategy = str(strategy_name or selected_row.get("best_strategy", "")).strip()
    timeframe = str(selected_row.get("timeframe", "")).strip()
    symbol = str(selected_row.get("symbol", "")).strip()
    run_tag = str(selected_row.get("run_tag", "")).strip()

    row: pd.Series
    if strategy and "strategy" in summary_df.columns:
        matches = summary_df.loc[summary_df["strategy"].astype(str) == strategy]
        row = matches.iloc[0] if not matches.empty else summary_df.iloc[0]
    else:
        row = summary_df.iloc[0]

    equity_df = pd.DataFrame()
    if equity_csv is not None and equity_csv.exists():
        equity_df = _read_csv_safe(equity_csv)
    initial_capital = 100_000.0
    period_start = "-"
    period_end = "-"
    if not equity_df.empty and {"datetime", "equity"}.issubset(equity_df.columns):
        eq = equity_df.copy()
        eq["datetime"] = pd.to_datetime(eq["datetime"], errors="coerce")
        eq["equity"] = pd.to_numeric(eq["equity"], errors="coerce")
        eq = eq.dropna(subset=["datetime", "equity"]).sort_values("datetime")
        if not eq.empty:
            initial_capital = float(eq["equity"].iloc[0])
            period_start = eq["datetime"].iloc[0].strftime("%d/%m/%Y")
            period_end = eq["datetime"].iloc[-1].strftime("%d/%m/%Y")

    output_root = str(selected_row.get("source_root", "")).strip()
    tf_folder = summary_csv.parent
    trades_csv = tf_folder / f"trades_{timeframe}_{strategy}.csv"
    trades_df = _read_csv_safe(trades_csv) if trades_csv.exists() else pd.DataFrame()

    net_profit = float(pd.to_numeric(row.get("net_profit", 0.0), errors="coerce") or 0.0)
    gross_profit = float(pd.to_numeric(row.get("gross_profit", 0.0), errors="coerce") or 0.0)
    gross_loss = float(pd.to_numeric(row.get("gross_loss", 0.0), errors="coerce") or 0.0)
    profit_factor = float(pd.to_numeric(row.get("profit_factor", 0.0), errors="coerce") or 0.0)
    trade_count = int(pd.to_numeric(row.get("trade_count", 0.0), errors="coerce") or 0)
    win_rate = float(pd.to_numeric(row.get("win_rate", 0.0), errors="coerce") or 0.0)
    avg_trade = float(pd.to_numeric(row.get("avg_trade", 0.0), errors="coerce") or 0.0)
    payoff_ratio = float(pd.to_numeric(row.get("payoff_ratio", 0.0), errors="coerce") or 0.0)
    max_drawdown = float(pd.to_numeric(row.get("max_drawdown", 0.0), errors="coerce") or 0.0)
    max_drawdown_pct = float(pd.to_numeric(row.get("max_drawdown_pct", 0.0), errors="coerce") or 0.0)
    max_win_streak = int(pd.to_numeric(row.get("max_consecutive_wins", 0.0), errors="coerce") or 0)
    max_loss_streak = int(pd.to_numeric(row.get("max_consecutive_losses", 0.0), errors="coerce") or 0)

    winners = int(round(trade_count * win_rate))
    losers = max(trade_count - winners, 0)
    flats = 0
    avg_win = (gross_profit / winners) if winners > 0 else 0.0
    avg_loss = (gross_loss / losers) if losers > 0 else 0.0
    max_win = 0.0
    max_loss = 0.0
    avg_win_duration = "-"
    avg_loss_duration = "-"
    avg_total_duration = "-"
    dd_trade_abs = 0.0
    dd_trade_pct = 0.0

    if not trades_df.empty and {"entry_time", "exit_time", "pnl_net"}.issubset(trades_df.columns):
        trades = trades_df.copy()
        trades["entry_time"] = pd.to_datetime(trades["entry_time"], errors="coerce")
        trades["exit_time"] = pd.to_datetime(trades["exit_time"], errors="coerce")
        trades["pnl_net"] = pd.to_numeric(trades["pnl_net"], errors="coerce")
        trades = trades.dropna(subset=["entry_time", "exit_time", "pnl_net"]).sort_values("exit_time")
        if not trades.empty:
            pnl = trades["pnl_net"]
            winners = int((pnl > 0).sum())
            losers = int((pnl < 0).sum())
            flats = int((pnl == 0).sum())
            avg_win = float(pnl[pnl > 0].mean()) if winners > 0 else 0.0
            avg_loss = float(pnl[pnl < 0].mean()) if losers > 0 else 0.0
            max_win = float(pnl.max())
            max_loss = float(pnl.min())
            dur_sec = (trades["exit_time"] - trades["entry_time"]).dt.total_seconds().fillna(0.0)
            avg_total_duration = _fmt_duration(float(dur_sec.mean())) if not dur_sec.empty else "-"
            avg_win_duration = _fmt_duration(float(dur_sec[pnl > 0].mean())) if winners > 0 else "-"
            avg_loss_duration = _fmt_duration(float(dur_sec[pnl < 0].mean())) if losers > 0 else "-"

            eq_after_trade = initial_capital + pnl.cumsum()
            peak = eq_after_trade.cummax()
            dd_series = peak - eq_after_trade
            if not dd_series.empty:
                dd_trade_abs = float(dd_series.max())
                peak_base = float(max(peak.max(), 1e-9))
                dd_trade_pct = 100.0 * dd_trade_abs / peak_base

    ending_capital = initial_capital + net_profit
    return_pct = (100.0 * net_profit / initial_capital) if abs(initial_capital) > 1e-9 else 0.0
    equity_peak = float(pd.to_numeric(equity_df.get("equity"), errors="coerce").max()) if not equity_df.empty else ending_capital

    return {
        "header": (
            f"Estrategia: {strategy} | Periodo: {timeframe} | "
            f"Periodo: {period_start} - {period_end} | Ajustes: Todos | Execucao: {run_tag} | Pasta: {output_root}"
        ),
        "left": [
            ("Saldo Liquido Total", _fmt_brl(net_profit)),
            ("Lucro Bruto", _fmt_brl(gross_profit)),
            ("Fator de Lucro", f"{profit_factor:.2f}"),
            ("Numero Total de Operacoes", f"{trade_count:,}".replace(",", ".")),
            ("Operacoes Vencedoras", f"{winners:,}".replace(",", ".")),
            ("Operacoes Zeradas", f"{flats:,}".replace(",", ".")),
            ("Media de Lucro/Prejuizo", _fmt_brl(avg_trade)),
            ("Media de Operacoes Vencedoras", _fmt_brl(avg_win)),
            ("Maior Operacao Vencedora", _fmt_brl(max_win)),
            ("Maior Sequencia Vencedora", f"{max_win_streak:,}".replace(",", ".")),
            ("Media de Tempo em Op. Vencedoras", avg_win_duration),
            ("Tempo Medio de Operacao Total", avg_total_duration),
            ("Maximo Acoes/Contratos", "1"),
            ("Retorno no Capital Inicial", f"{return_pct:.2f}%"),
            ("Patrimonio Maximo", _fmt_brl(equity_peak)),
            ("Declinio Maximo (Topo ao Fundo)", _fmt_brl(max_drawdown)),
            ("Drawdown como % do Saldo Total", f"{max_drawdown_pct:.2f}%"),
        ],
        "right": [
            ("Saldo Total", _fmt_brl(net_profit)),
            ("Prejuizo Bruto", _fmt_brl(gross_loss)),
            ("Custos", _fmt_brl(0.0)),
            ("Percentual de Operacoes Vencedoras", f"{100.0 * win_rate:.2f}%"),
            ("Operacoes Perdedoras", f"{losers:,}".replace(",", ".")),
            ("Razao Media Lucro:Media Prejuizo", f"{payoff_ratio:.2f}"),
            ("Media de Operacoes Perdedoras", _fmt_brl(avg_loss)),
            ("Maior Operacao Perdedora", _fmt_brl(max_loss)),
            ("Maior Sequencia Perdedora", f"{max_loss_streak:,}".replace(",", ".")),
            ("Media de Tempo em Op. Perdedoras", avg_loss_duration),
            ("Tempo Medio de Operacao no Intervalo", avg_total_duration),
            ("Patrimonio Necessario(Maior Operacao)", "-"),
            ("Percentual de Tempo no Mercado", "-"),
            ("Declinio Maximo (Trade a Trade)", _fmt_brl(dd_trade_abs)),
            ("Drawdown como % do Saldo Total", f"{dd_trade_pct:.2f}%"),
            ("Saldo Final (capital)", _fmt_brl(ending_capital)),
        ],
    }


def _build_profit_like_summary_payload_from_runtime(
    selected_row: dict[str, Any],
    strategy_name: str,
    timeframe: str,
    metrics: dict[str, Any],
    trades_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    initial_capital: float,
    contracts: int,
) -> dict[str, Any]:
    strategy = str(strategy_name).strip()
    run_tag = str(selected_row.get("run_tag", "")).strip()
    output_root = str(selected_row.get("source_root", "")).strip()

    period_start = "-"
    period_end = "-"
    eq = pd.DataFrame()
    if not equity_df.empty and {"datetime", "equity"}.issubset(equity_df.columns):
        eq = equity_df.copy()
        eq["datetime"] = pd.to_datetime(eq["datetime"], errors="coerce")
        eq["equity"] = pd.to_numeric(eq["equity"], errors="coerce")
        eq = eq.dropna(subset=["datetime", "equity"]).sort_values("datetime")
        if not eq.empty:
            period_start = eq["datetime"].iloc[0].strftime("%d/%m/%Y")
            period_end = eq["datetime"].iloc[-1].strftime("%d/%m/%Y")

    net_profit = float(pd.to_numeric(metrics.get("net_profit", 0.0), errors="coerce") or 0.0)
    gross_profit = float(pd.to_numeric(metrics.get("gross_profit", 0.0), errors="coerce") or 0.0)
    gross_loss = float(pd.to_numeric(metrics.get("gross_loss", 0.0), errors="coerce") or 0.0)
    profit_factor = float(pd.to_numeric(metrics.get("profit_factor", 0.0), errors="coerce") or 0.0)
    trade_count = int(pd.to_numeric(metrics.get("trade_count", 0.0), errors="coerce") or 0)
    win_rate = float(pd.to_numeric(metrics.get("win_rate", 0.0), errors="coerce") or 0.0)
    avg_trade = float(pd.to_numeric(metrics.get("avg_trade", 0.0), errors="coerce") or 0.0)
    payoff_ratio = float(pd.to_numeric(metrics.get("payoff_ratio", 0.0), errors="coerce") or 0.0)
    max_drawdown = float(pd.to_numeric(metrics.get("max_drawdown", 0.0), errors="coerce") or 0.0)
    max_drawdown_pct = float(pd.to_numeric(metrics.get("max_drawdown_pct", 0.0), errors="coerce") or 0.0)
    max_win_streak = int(pd.to_numeric(metrics.get("max_consecutive_wins", 0.0), errors="coerce") or 0)
    max_loss_streak = int(pd.to_numeric(metrics.get("max_consecutive_losses", 0.0), errors="coerce") or 0)

    winners = int(round(trade_count * win_rate))
    losers = max(trade_count - winners, 0)
    flats = 0
    avg_win = (gross_profit / winners) if winners > 0 else 0.0
    avg_loss = (gross_loss / losers) if losers > 0 else 0.0
    max_win = 0.0
    max_loss = 0.0
    avg_win_duration = "-"
    avg_loss_duration = "-"
    avg_total_duration = "-"
    dd_trade_abs = 0.0
    dd_trade_pct = 0.0

    if not trades_df.empty and {"entry_time", "exit_time", "pnl_net"}.issubset(trades_df.columns):
        trades = trades_df.copy()
        trades["entry_time"] = pd.to_datetime(trades["entry_time"], errors="coerce")
        trades["exit_time"] = pd.to_datetime(trades["exit_time"], errors="coerce")
        trades["pnl_net"] = pd.to_numeric(trades["pnl_net"], errors="coerce")
        trades = trades.dropna(subset=["entry_time", "exit_time", "pnl_net"]).sort_values("exit_time")
        if not trades.empty:
            pnl = trades["pnl_net"]
            winners = int((pnl > 0).sum())
            losers = int((pnl < 0).sum())
            flats = int((pnl == 0).sum())
            avg_win = float(pnl[pnl > 0].mean()) if winners > 0 else 0.0
            avg_loss = float(pnl[pnl < 0].mean()) if losers > 0 else 0.0
            max_win = float(pnl.max())
            max_loss = float(pnl.min())
            dur_sec = (trades["exit_time"] - trades["entry_time"]).dt.total_seconds().fillna(0.0)
            avg_total_duration = _fmt_duration(float(dur_sec.mean())) if not dur_sec.empty else "-"
            avg_win_duration = _fmt_duration(float(dur_sec[pnl > 0].mean())) if winners > 0 else "-"
            avg_loss_duration = _fmt_duration(float(dur_sec[pnl < 0].mean())) if losers > 0 else "-"

            eq_after_trade = float(initial_capital) + pnl.cumsum()
            peak = eq_after_trade.cummax()
            dd_series = peak - eq_after_trade
            if not dd_series.empty:
                dd_trade_abs = float(dd_series.max())
                peak_base = float(max(peak.max(), 1e-9))
                dd_trade_pct = 100.0 * dd_trade_abs / peak_base

    ending_capital = float(initial_capital) + net_profit
    return_pct = (100.0 * net_profit / float(initial_capital)) if abs(float(initial_capital)) > 1e-9 else 0.0
    equity_peak = float(eq["equity"].max()) if not eq.empty else ending_capital

    return {
        "header": (
            f"Estrategia: {strategy} | Periodo: {timeframe} | "
            f"Periodo: {period_start} - {period_end} | Ajustes: Parametro Selecionado | "
            f"Execucao: {run_tag} | Pasta: {output_root}"
        ),
        "left": [
            ("Saldo Liquido Total", _fmt_brl(net_profit)),
            ("Lucro Bruto", _fmt_brl(gross_profit)),
            ("Fator de Lucro", f"{profit_factor:.2f}"),
            ("Numero Total de Operacoes", f"{trade_count:,}".replace(",", ".")),
            ("Operacoes Vencedoras", f"{winners:,}".replace(",", ".")),
            ("Operacoes Zeradas", f"{flats:,}".replace(",", ".")),
            ("Media de Lucro/Prejuizo", _fmt_brl(avg_trade)),
            ("Media de Operacoes Vencedoras", _fmt_brl(avg_win)),
            ("Maior Operacao Vencedora", _fmt_brl(max_win)),
            ("Maior Sequencia Vencedora", f"{max_win_streak:,}".replace(",", ".")),
            ("Media de Tempo em Op. Vencedoras", avg_win_duration),
            ("Tempo Medio de Operacao Total", avg_total_duration),
            ("Maximo Acoes/Contratos", f"{int(max(1, contracts))}"),
            ("Retorno no Capital Inicial", f"{return_pct:.2f}%"),
            ("Patrimonio Maximo", _fmt_brl(equity_peak)),
            ("Declinio Maximo (Topo ao Fundo)", _fmt_brl(max_drawdown)),
            ("Drawdown como % do Saldo Total", f"{max_drawdown_pct:.2f}%"),
        ],
        "right": [
            ("Saldo Total", _fmt_brl(net_profit)),
            ("Prejuizo Bruto", _fmt_brl(gross_loss)),
            ("Custos", _fmt_brl(0.0)),
            ("Percentual de Operacoes Vencedoras", f"{100.0 * win_rate:.2f}%"),
            ("Operacoes Perdedoras", f"{losers:,}".replace(",", ".")),
            ("Razao Media Lucro:Media Prejuizo", f"{payoff_ratio:.2f}"),
            ("Media de Operacoes Perdedoras", _fmt_brl(avg_loss)),
            ("Maior Operacao Perdedora", _fmt_brl(max_loss)),
            ("Maior Sequencia Perdedora", f"{max_loss_streak:,}".replace(",", ".")),
            ("Media de Tempo em Op. Perdedoras", avg_loss_duration),
            ("Tempo Medio de Operacao no Intervalo", avg_total_duration),
            ("Patrimonio Necessario(Maior Operacao)", "-"),
            ("Percentual de Tempo no Mercado", "-"),
            ("Declinio Maximo (Trade a Trade)", _fmt_brl(dd_trade_abs)),
            ("Drawdown como % do Saldo Total", f"{dd_trade_pct:.2f}%"),
            ("Saldo Final (capital)", _fmt_brl(ending_capital)),
        ],
    }


def _render_profit_like_summary(payload: dict[str, Any]) -> None:
    st.markdown("### Resumo estilo Profit")
    st.caption(str(payload.get("header", "")))
    c1, c2 = st.columns(2)
    for label, value in payload.get("left", []):
        _render_profit_kv(c1, str(label), str(value))
    for label, value in payload.get("right", []):
        _render_profit_kv(c2, str(label), str(value))


def _render_profit_kv(container: Any, label: str, value: str) -> None:
    color = "#e8ebf1"
    if value.startswith("-R$") or value.startswith("-"):
        color = "#ff6b76"
    elif value.startswith("R$") and not value.startswith("R$ 0"):
        color = "#24e286"
    container.markdown(
        (
            "<div style='display:flex;justify-content:space-between;"
            "padding:2px 0;border-bottom:1px solid rgba(255,255,255,0.06);'>"
            f"<span style='color:#d6dce7'>{label}</span>"
            f"<span style='font-weight:700;color:{color}'>{value}</span>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _fmt_duration(seconds: float) -> str:
    total = int(max(seconds, 0))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    if h > 0:
        return f"{h}h{m:02d}min{s:02d}s"
    return f"{m}min{s:02d}s"


def _resolve_project_path(raw: str) -> Path | None:
    raw = raw.strip()
    if not raw:
        return None
    p = Path(raw)
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


def _hash_file(path: Path | None) -> str:
    if not path or not path.exists():
        return ""
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _style_pos_neg(df: pd.DataFrame, pnl_cols: list[str], score_cols: list[str]) -> Any:
    def _color(v: Any) -> str:
        try:
            return "color: #22d481;" if float(v) >= 0 else "color: #ff4f5e;"
        except Exception:
            return ""

    styled = df.style.set_table_styles(
        [
            {"selector": "th", "props": [("background-color", "#171b23"), ("color", "#f3f5f8"), ("font-weight", "700")]},
            {"selector": "td", "props": [("background-color", "#0f131a"), ("color", "#eef2f7")]},
            {"selector": "tr:hover td", "props": [("background-color", "#1b2230")]},
        ]
    )
    for col in pnl_cols + score_cols:
        if col in df.columns:
            styled = styled.map(_color, subset=[col])
    return styled


def _fmt_brl(value: Any) -> str:
    num = float(pd.to_numeric(value, errors="coerce"))
    sign = "-" if num < 0 else ""
    raw = f"{abs(num):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{sign}R$ {raw}"


def _fmt_brl_short(value: Any) -> str:
    num = float(pd.to_numeric(value, errors="coerce"))
    sign = "-" if num < 0 else ""
    abs_num = abs(num)
    if abs_num >= 1_000_000:
        return f"{sign}{(abs_num / 1_000_000):.2f}M".replace(".", ",")
    if abs_num >= 1_000:
        return f"{sign}{(abs_num / 1_000):.2f}k".replace(".", ",")
    return f"{sign}{abs_num:.2f}".replace(".", ",")


def _inject_dark_theme_css() -> None:
    st.markdown(
        """
        <style>
        :root { color-scheme: dark; }
        .stApp {
            background: radial-gradient(140% 90% at 10% 5%, #162233 0%, #0f1117 55%, #0a0c11 100%);
            color: #f2f5fb;
        }
        [data-testid="stHeader"] {
            background: rgba(10, 12, 17, 0.82);
            border-bottom: 1px solid rgba(255,255,255,0.08);
        }
        [data-testid="stSidebar"] {
            background: #11161f;
            border-right: 1px solid rgba(255,255,255,0.10);
        }
        [data-testid="stMetric"] {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.09);
            border-radius: 10px;
            padding: 8px 10px;
        }
        h1, h2, h3, h4, h5, h6, p, label {
            color: #f2f5fb;
        }
        [data-testid="stTabs"] button[role="tab"] {
            color: #d8deea !important;
        }
        [data-testid="stTabs"] button[aria-selected="true"] {
            color: #ffffff !important;
            border-bottom: 2px solid #26e58a !important;
        }
        .stDataFrame, [data-testid="stDataFrame"] {
            background: transparent !important;
        }
        [data-testid="stCodeBlock"] pre, [data-testid="stCode"] pre {
            background: #f2f4f8 !important;
            color: #101722 !important;
            border-radius: 8px !important;
        }
        [data-testid="stCodeBlock"] code, [data-testid="stCode"] code {
            color: #101722 !important;
        }
        [data-baseweb="popover"] * {
            color: #101722 !important;
        }
        [data-baseweb="select"] [role="option"] {
            color: #101722 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
