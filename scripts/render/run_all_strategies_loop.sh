#!/usr/bin/env bash
set -euo pipefail

SYMBOL="${SYMBOL:-WINFUT}"
TIMEFRAMES="${TIMEFRAMES:-5m}"
START_DATE="${START_DATE:-2025-02-19}"
END_DATE="${END_DATE:-2026-02-19}"
DATA_ROOT="${DATA_ROOT:-data}"
OUTPUTS_DIR="${OUTPUTS_DIR:-/var/data/outputs_master}"

TRAIN_DAYS="${TRAIN_DAYS:-120}"
TEST_DAYS="${TEST_DAYS:-30}"
SAMPLES="${SAMPLES:-100}"
TOP_K="${TOP_K:-10}"
PARAM_BANK_TOP="${PARAM_BANK_TOP:-70}"
INITIAL_CAPITAL="${INITIAL_CAPITAL:-100000}"
CONTRACTS="${CONTRACTS:-1}"
POINT_VALUE="${POINT_VALUE:-0.2}"
SESSION_START="${SESSION_START:-09:00}"
SESSION_END="${SESSION_END:-17:40}"
PROGRESS_PRINT_EVERY="${PROGRESS_PRINT_EVERY:-3}"
LOOP_SLEEP_SEC="${LOOP_SLEEP_SEC:-10}"
STOP_FILE="${STOP_FILE:-/var/data/outputs_master/STOP_RENDER_ALL}"

STRATEGIES=(
  trx_htsl
  trx_melhor_20_02
  ganhador_80
  ema_pullback
  breakout_range
  scalp_break_even
)

mkdir -p "${OUTPUTS_DIR}"

echo "[START] Worker de loop infinito iniciado."
echo "[START] outputs=${OUTPUTS_DIR} data_root=${DATA_ROOT}"
echo "[START] strategies=${STRATEGIES[*]}"

while true; do
  if [ -f "${STOP_FILE}" ]; then
    echo "[STOP] Arquivo de parada encontrado em ${STOP_FILE}. Encerrando loop."
    break
  fi

  python -m src.cli \
    --symbol "${SYMBOL}" \
    --timeframes "${TIMEFRAMES}" \
    --strategies "${STRATEGIES[@]}" \
    --start "${START_DATE}" \
    --end "${END_DATE}" \
    --data-root "${DATA_ROOT}" \
    --outputs "${OUTPUTS_DIR}" \
    --execution-mode ohlc \
    --train-days "${TRAIN_DAYS}" \
    --test-days "${TEST_DAYS}" \
    --samples "${SAMPLES}" \
    --top-k "${TOP_K}" \
    --param-bank-top "${PARAM_BANK_TOP}" \
    --initial-capital "${INITIAL_CAPITAL}" \
    --contracts "${CONTRACTS}" \
    --point-value "${POINT_VALUE}" \
    --session-start "${SESSION_START}" \
    --session-end "${SESSION_END}" \
    --close-on-session-end \
    --stop-file "${STOP_FILE}" \
    --verbose-progress \
    --progress-print-every "${PROGRESS_PRINT_EVERY}" \
    --skip-plots || true

  if [ -f "${STOP_FILE}" ]; then
    echo "[STOP] Parada solicitada apos o ciclo atual."
    break
  fi

  echo "[INFO] Ciclo concluido. Proximo ciclo em ${LOOP_SLEEP_SEC}s..."
  sleep "${LOOP_SLEEP_SEC}"
done

echo "[END] Worker encerrado."

