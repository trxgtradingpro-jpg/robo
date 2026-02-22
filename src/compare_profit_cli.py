"""CLI to compare robot trades against Profit operation report."""

from __future__ import annotations

import argparse
from pathlib import Path

from .profit_compare import (
    compare_tradebooks,
    load_profit_operations,
    load_robot_trades,
    save_compare_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compara tradebook do robo com relatorio de operacoes do Profit.")
    parser.add_argument("--robot-trades", required=True, help="CSV trades_<timeframe>_<strategy>.csv do robo")
    parser.add_argument("--profit-ops", required=True, help="CSV exportado da aba Operacoes no Profit")
    parser.add_argument("--output-dir", default="outputs_compare", help="Pasta para salvar resumo e divergencias")
    parser.add_argument("--time-tolerance-sec", type=int, default=300, help="Tolerancia de horario por trade (seg)")
    parser.add_argument("--pnl-tolerance", type=float, default=5.0, help="Tolerancia de PnL por trade")
    parser.add_argument("--price-tolerance", type=float, default=5.0, help="Tolerancia de preco entrada/saida")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    robot = load_robot_trades(Path(args.robot_trades))
    profit = load_profit_operations(Path(args.profit_ops))
    result = compare_tradebooks(
        robot_trades=robot,
        profit_trades=profit,
        time_tolerance_seconds=int(max(0, args.time_tolerance_sec)),
        pnl_tolerance=float(max(0.0, args.pnl_tolerance)),
        price_tolerance=float(max(0.0, args.price_tolerance)),
    )
    files = save_compare_outputs(result=result, output_dir=Path(args.output_dir))

    s = result.summary
    print(
        "[COMPARE]",
        f"robot={s.robot_trade_count}",
        f"profit={s.profit_trade_count}",
        f"matched={s.matched_trade_count}",
        f"only_robot={s.only_robot_count}",
        f"only_profit={s.only_profit_count}",
    )
    print(
        "[COMPARE]",
        f"net_robot={s.robot_net:.2f}",
        f"net_profit={s.profit_net:.2f}",
        f"net_diff={s.net_diff:.2f}",
        f"equiv_matches={s.equivalent_matches_pct:.2f}%",
    )
    if s.strict_parity:
        print("[OK] Paridade estrita atingida (dentro das tolerancias).")
    else:
        print("[WARN] Divergencias detectadas. Veja arquivos em:", args.output_dir)
    print("[INFO] summary:", files["summary"])
    print("[INFO] matched:", files["matched"])
    print("[INFO] only_robot:", files["only_robot"])
    print("[INFO] only_profit:", files["only_profit"])


if __name__ == "__main__":
    main()

