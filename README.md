# robo

Backtesting local para WINFUT com:
- leitura de CSV do Profit (multi-timeframe),
- tres estrategias (`ema_pullback`, `breakout_range`, `scalp_break_even`),
- custos/slippage,
- fechamento forcado de posicoes no fim da sessao (padrao `17:00`),
- slippage/custo dinamicos por volatilidade da barra (opcional),
- otimizacao em treino + validacao walk-forward em teste (sem leak),
- relatorio de qualidade dos dados por timeframe,
- manifesto de reprodutibilidade com hash dos dados e configs da execucao,
- relatorios CSV/JSON e graficos de equity.

## Estrutura

```text
data/
  WINFUT/
    1m/ 5m/ 10m/ 15m/ 30m/ 60m/ daily/ weekly/
src/
  data_loader.py
  backtest_engine.py
  optimizer.py
  walkforward.py
  metrics.py
  cli.py
  ui_dashboard.py
  strategies/
    __init__.py
    ema_pullback.py
    breakout_range.py
outputs/
```

## Requisitos

- Windows 10
- Python 3.11+
- `pandas`, `numpy`, `matplotlib`, `streamlit`, `plotly`

Instalacao:

```bash
pip install -r requirements.txt
```

## Como rodar

Exemplo principal:

```bash
python -m src.cli --symbol WINFUT --timeframes 1m 5m 15m 30m --start 2020-01-01 --end 2026-02-19
```

Exemplo com configuracoes de execucao/custo:

```bash
python -m src.cli ^
  --symbol WINFUT ^
  --timeframes 5m 15m ^
  --start 2020-01-01 --end 2026-02-19 ^
  --train-days 120 --test-days 30 ^
  --samples 200 --top-k 5 ^
  --entry-model next_open ^
  --slippage 5 --fixed-cost 1.5 --cost-per-contract 0.2 ^
  --session-start 09:10 --session-end 17:00 --close-on-session-end
```

Exemplo com slippage/custo dinamicos:

```bash
python -m src.cli ^
  --symbol WINFUT ^
  --timeframes 5m ^
  --start 2025-01-17 --end 2026-01-16 ^
  --slippage 2 --slippage-model range_scaled --slippage-range-factor 0.02 ^
  --cost-model range_scaled --cost-range-factor 0.01 ^
  --open-auction-minutes 20 ^
  --open-auction-slippage-multiplier 1.3 ^
  --open-auction-cost-multiplier 1.2
```

Exemplo acelerado (otimizacao com downsample no treino):

```bash
python -m src.cli ^
  --symbol WINFUT ^
  --timeframes 5m ^
  --strategies ema_pullback ^
  --start 2025-12-01 --end 2026-01-16 ^
  --samples 80 --top-k 3 ^
  --train-bar-step 3 ^
  --skip-plots
```

Exemplo evolutivo (reaproveita historico e roda multiplos ciclos):

```bash
python -m src.cli ^
  --symbol WINFUT ^
  --timeframes 5m ^
  --strategies scalp_break_even ^
  --start 2025-01-17 --end 2026-02-19 ^
  --samples 300 --top-k 10 ^
  --evolution-cycles 5 ^
  --param-bank-top 50 ^
  --verbose-progress
```

### Paper Trading (Fase 2)

Replay event-driven candle a candle com risco operacional e kill switch:

```bash
python -m src.paper_cli ^
  --symbol WINFUT ^
  --timeframe 5m ^
  --strategy breakout_range ^
  --params-file outputs/WINFUT/5m/best_params_5m.json ^
  --start 2025-12-01 --end 2026-01-16 ^
  --daily-loss-limit 1500 ^
  --max-drawdown-pct 8 ^
  --max-consecutive-losses 6 ^
  --kill-switch-file KILL_SWITCH ^
  --emit-every-bars 30 --print-events
```

Criar arquivo `KILL_SWITCH` na pasta atual para parar execucao por risco.

Observacao:
- por padrao, `cli` e `paper_cli` rodam com `--session-end 17:00 --close-on-session-end`.
- para desativar, use `--no-close-on-session-end`.
- a estrategia `scalp_break_even` usa `stop_points`, `take_points` e `break_even_trigger_points` (trava no 0x0).
- o modo evolutivo usa banco de parametros em `outputs/<SIMBOLO>/<TF>/params_bank_<TF>_<estrategia>.jsonl`.

### Modo visual (tempo real)

Para abrir o painel visual com resumo, operacoes, grafico de operacoes e patrimonio:

```bash
streamlit run src/ui_dashboard.py
```

No painel lateral:
- marque `Atualizacao ao vivo` para acompanhar janelas/samples em tempo real;
- ative `Modo rapido (menos repintura)` para reduzir overhead visual;
- ajuste `Atualizar a cada N amostras` para controlar frequencia de refresh.
- ative `Modo turbo real (subprocesso)` para executar em processo separado sem travar a UI;
- use `Parar turbo` para interromper o processo em execucao.
- ajuste `Turbo step treino (N barras)` para acelerar a otimizacao (N>1 usa downsample no treino).

Abas disponiveis no painel:
- `Resumo`
- `Operacoes`
- `Grafico de Operacoes`
- `Patrimonio`
- `Mensal` (PnL por mes)
- `Robustez` (alertas e sensibilidade de parametros)

No modo turbo, logs de progresso aparecem no painel e o resultado e carregado automaticamente ao fim.

## Saida

Para cada timeframe em `outputs/WINFUT/<timeframe>/`:
- `summary_<timeframe>.csv`
- `best_params_<timeframe>.json`
- `data_quality_<timeframe>.json`
- `trades_<timeframe>_<strategy>.csv`
- `walkforward_windows_<timeframe>_<strategy>.csv`
- `walkforward_topk_<timeframe>_<strategy>.csv`
- `equity_curve_<timeframe>_<strategy>.csv`
- `monthly_<timeframe>_<strategy>.csv`
- `sensitivity_<timeframe>_<strategy>.csv`
- `robustness_<timeframe>_<strategy>.json`
- `equity_<timeframe>_<strategy>.png`
- `equity_curve_<timeframe>_best.csv`
- `equity_<timeframe>_best.png`

Na pasta `outputs/WINFUT/`:
- `run_manifest_<run_id>.json` (reprodutibilidade)

Para paper trading (`outputs_paper/WINFUT/<timeframe>/<strategy>/`):
- `paper_trades_<timeframe>_<strategy>.csv`
- `paper_equity_<timeframe>_<strategy>.csv`
- `paper_alerts_<timeframe>_<strategy>.csv`
- `paper_summary_<timeframe>_<strategy>.json`
- `paper_equity_<timeframe>_<strategy>.png`

## Testes automatizados

```bash
python -m pytest tests -q
```
