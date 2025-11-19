from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .backtester import Backtester
from .config import load_backtest_config, load_config
from .data_loader import load_price_data
from .forward_simulator import ForwardSimulator
from .report import (
    render_backtest_summary,
    render_forward_summary,
    save_backtest_chart,
    save_forward_chart,
)


def parse_quantiles(raw: str) -> Sequence[float]:
    values: list[float] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            value = float(item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"无法解析分位数：{item}") from exc
        if not 0 < value < 1:
            raise argparse.ArgumentTypeError("分位数必须在 (0, 1) 范围内")
        values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("至少需要一个分位数")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="invest-sim", description="投资组合模拟和回测命令行工具"
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="显示版本号后退出",
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # 前瞻性模拟命令
    forward_parser = subparsers.add_parser(
        "forward", help="运行前瞻性模拟（基于假设参数预测未来收益）"
    )
    forward_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="配置文件路径（JSON/YAML）",
    )
    forward_parser.add_argument(
        "--seed", type=int, default=None, help="随机种子，方便重现实验"
    )
    forward_parser.add_argument(
        "--quantiles",
        type=parse_quantiles,
        default=(0.1, 0.5, 0.9),
        help="输出的分位数，逗号分隔，例如 0.1,0.5,0.9",
    )
    forward_parser.add_argument(
        "--chart",
        type=Path,
        default=None,
        help="保存分位数走势图的路径（可选）",
    )
    forward_parser.add_argument(
        "--show-chart",
        action="store_true",
        help="保存图表后同步在窗口中展示（需要 GUI 支持）",
    )

    # 历史回测命令
    backtest_parser = subparsers.add_parser(
        "backtest", help="运行历史回测（基于历史数据回测过去表现）"
    )
    backtest_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="回测配置文件路径（JSON/YAML）",
    )
    backtest_parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="历史价格数据文件路径（CSV/Excel/Parquet）",
    )
    backtest_parser.add_argument(
        "--date-column",
        type=str,
        default="date",
        help="日期列名（默认：date）",
    )
    backtest_parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=0.0,
        help="无风险利率，用于计算夏普比率（默认：0.0）",
    )
    backtest_parser.add_argument(
        "--chart",
        type=Path,
        default=None,
        help="保存回测结果图表的路径（可选）",
    )
    backtest_parser.add_argument(
        "--show-chart",
        action="store_true",
        help="保存图表后同步在窗口中展示（需要 GUI 支持）",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    console = Console()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        from . import __version__

        console.print(__version__)
        return

    if args.command == "forward":
        _run_forward(console, args)
    elif args.command == "backtest":
        _run_backtest(console, args)
    else:
        parser.print_help()


def _run_forward(console: Console, args: argparse.Namespace) -> None:
    """运行前瞻性模拟。"""
    console.print(Panel(Text("前瞻性模拟（预测未来收益）", justify="center"), expand=False))

    try:
        config = load_config(args.config)
        simulator = ForwardSimulator(config, seed=args.seed)
        result = simulator.run()
    except Exception as exc:
        console.print(f"[red]运行失败：{exc}[/red]")
        if console.is_terminal:
            sys.exit(1)
        raise

    render_forward_summary(result, console=console, quantiles=args.quantiles)

    if args.chart:
        path = save_forward_chart(
            result,
            output=args.chart,
            quantiles=args.quantiles,
            show=args.show_chart,
        )
        console.print(f"[green]图表已保存至[/green] {path}")


def _run_backtest(console: Console, args: argparse.Namespace) -> None:
    """运行历史回测。"""
    console.print(Panel(Text("历史回测（回测过去表现）", justify="center"), expand=False))

    try:
        # 加载配置
        config = load_backtest_config(args.config)

        # 加载历史数据
        console.print(f"[cyan]正在加载历史数据：{args.data}[/cyan]")
        price_data = load_price_data(
            args.data,
            date_column=args.date_column,
        )

        # 验证资产名称匹配
        missing_assets = set(config.asset_names) - set(price_data.columns)
        if missing_assets:
            raise ValueError(
                f"历史数据中缺少以下资产：{missing_assets}。"
                f"可用资产：{list(price_data.columns)}"
            )

        # 运行回测
        console.print("[cyan]正在运行回测...[/cyan]")
        backtester = Backtester(config)
        result = backtester.run(price_data)

    except Exception as exc:
        console.print(f"[red]运行失败：{exc}[/red]")
        if console.is_terminal:
            sys.exit(1)
        raise

    render_backtest_summary(result, console=console, risk_free_rate=args.risk_free_rate)

    if args.chart:
        path = save_backtest_chart(
            result,
            output=args.chart,
            show=args.show_chart,
        )
        console.print(f"[green]图表已保存至[/green] {path}")


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
