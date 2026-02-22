from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.profit_compare import compare_tradebooks, load_profit_operations, load_robot_trades


def test_load_profit_operations_parses_ptbr_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "profit_ops.csv"
    csv_path.write_text(
        "\n".join(
            [
                "Abertura;Fechamento;Qtd;Lado;Preco Compra;Preco Venda;Resultado",
                "19/02/2026 11:20:00;19/02/2026 11:25:00;1;C;190.200;189.900;R$ -60,00",
                "19/02/2026 10:30:00;19/02/2026 10:40:00;-1;V;190.355;190.055;R$ 60,00",
            ]
        ),
        encoding="utf-8",
    )

    df = load_profit_operations(csv_path)
    assert len(df) == 2
    assert df["direction"].tolist() == ["short", "long"]
    assert float(df.iloc[0]["qty"]) == 1.0
    assert float(df.iloc[0]["entry_price"]) == 190355.0
    assert float(df.iloc[0]["pnl_net"]) == 60.0


def test_compare_tradebooks_detects_mismatch_and_net_diff(tmp_path: Path) -> None:
    robot_file = tmp_path / "robot.csv"
    profit_file = tmp_path / "profit.csv"
    robot_file.write_text(
        "\n".join(
            [
                "strategy,entry_time,exit_time,direction,entry_price,exit_price,pnl_points,pnl_gross,costs,pnl_net,duration_bars,exit_reason,params",
                "s1,2026-01-02 09:00:00,2026-01-02 09:10:00,long,1000,1020,20,20,0,20,3,take_profit,{}",
                "s1,2026-01-02 10:00:00,2026-01-02 10:15:00,short,1030,1010,20,20,0,20,4,take_profit,{}",
            ]
        ),
        encoding="utf-8",
    )
    profit_file.write_text(
        "\n".join(
            [
                "Abertura;Fechamento;Qtd;Lado;Preco Compra;Preco Venda;Resultado",
                "02/01/2026 09:00:00;02/01/2026 09:10:00;1;C;1.000;1.020;R$ 20,00",
                "02/01/2026 11:00:00;02/01/2026 11:10:00;1;C;1.100;1.090;R$ -10,00",
            ]
        ),
        encoding="utf-8",
    )

    robot = load_robot_trades(robot_file)
    profit = load_profit_operations(profit_file)
    result = compare_tradebooks(
        robot_trades=robot,
        profit_trades=profit,
        time_tolerance_seconds=60,
        pnl_tolerance=1.0,
        price_tolerance=1.0,
    )

    assert result.summary.robot_trade_count == 2
    assert result.summary.profit_trade_count == 2
    assert result.summary.matched_trade_count == 1
    assert result.summary.only_robot_count == 1
    assert result.summary.only_profit_count == 1
    assert result.summary.net_diff == 30.0
    assert not result.summary.strict_parity
    assert isinstance(result.matched, pd.DataFrame)

