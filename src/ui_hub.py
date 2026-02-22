"""Pagina inicial para navegar entre Dashboard e Visualizador."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    st.set_page_config(
        page_title="Hub Robo Backtest",
        page_icon=":compass:",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _inject_css()

    st.title("ROBO BACKTEST HUB")
    st.caption("Escolha para onde deseja ir.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Robo Backtest Pro")
        st.write("Executar testes, otimizacao e acompanhamento em tempo real.")
        if st.button("Abrir Dashboard", use_container_width=True, type="primary"):
            st.switch_page("pages/1_Robo_Backtest_Pro.py")
    with c2:
        st.markdown("### Visualizador de Resultados")
        st.write("Ler e comparar todos os outputs salvos em um painel unico.")
        if st.button("Abrir Visualizador", use_container_width=True):
            st.switch_page("pages/2_Visualizador_Resultados.py")

    st.info("Execute este hub com: `streamlit run src/ui_hub.py`")


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: radial-gradient(120% 80% at 10% 10%, #15243a 0%, #0f1117 55%, #0a0c12 100%);
            color: #f3f6fb;
        }
        [data-testid="stHeader"] {
            background: rgba(12, 14, 20, 0.8);
            border-bottom: 1px solid rgba(255,255,255,0.08);
        }
        [data-testid="stButton"] button {
            border-radius: 10px;
            height: 3rem;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
