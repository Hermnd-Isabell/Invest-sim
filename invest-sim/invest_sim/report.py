from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .backtester import BacktestResult
from .forward_simulator import ForwardSimulationResult


def render_forward_summary(
    result: ForwardSimulationResult,
    *,
    console: Console,
    quantiles: Sequence[float] = (0.1, 0.5, 0.9),
    risk_level: float = 0.05,
) -> None:
    """渲染前瞻性模拟结果的统计摘要。"""

    summary = _build_forward_summary_table(result, risk_level=risk_level)
    console.print(summary)

    quantile_df = result.quantiles(quantiles)
    trends = Table(title="投资组合价值区间（单位：年）", show_lines=False)
    trends.add_column("年份")
    for prob in quantiles:
        trends.add_column(f"P{int(prob*100):02d}")
    for year, row in quantile_df.iterrows():
        trends.add_row(f"{year:.1f}", *[f"{val:,.0f}" for val in row.values])

    console.print(Panel(trends, title="Monte Carlo 结果", expand=False))


def _build_forward_summary_table(result: ForwardSimulationResult, *, risk_level: float) -> Table:
    final_values = result.final_distribution()
    stats = {
        "均值": final_values.mean(),
        "中位数": final_values.median(),
        "标准差": final_values.std(),
        "最小值": final_values.min(),
        "最大值": final_values.max(),
    }
    risk = result.risk_metrics(level=risk_level)
    risk_rows = {
        f"VaR({int((1 - risk_level) * 100)}%)": risk["value_at_risk"],
        f"CVaR({int((1 - risk_level) * 100)}%)": risk["conditional_value_at_risk"],
        "中位最大回撤": risk["max_drawdown"],
    }

    table = Table(title="投资组合终值统计（前瞻性模拟）", show_header=False, expand=False)
    for key, value in stats.items():
        table.add_row(key, f"{value:,.0f}")
    table.add_row("—", "—")
    for key, value in risk_rows.items():
        table.add_row(key, f"{value:,.0f}")
    return table


def render_backtest_summary(
    result: BacktestResult,
    *,
    console: Console,
    risk_free_rate: float = 0.0,
) -> None:
    """渲染历史回测结果的统计摘要。"""

    metrics = result.risk_metrics(risk_free_rate=risk_free_rate)

    table = Table(title="历史回测结果", show_header=False, expand=False)
    table.add_row("总收益率", f"{metrics['total_return']:.2%}")
    table.add_row("年化收益率", f"{metrics['annualized_return']:.2%}")
    table.add_row("年化波动率", f"{metrics['volatility']:.2%}")
    table.add_row("夏普比率", f"{metrics['sharpe_ratio']:.2f}")
    table.add_row("最大回撤", f"{metrics['max_drawdown']:.2%}")
    table.add_row("—", "—")
    table.add_row("初始资金", f"{result.config.initial_balance:,.0f}")
    table.add_row("最终资金", f"{result.portfolio_values.iloc[-1]:,.0f}")

    console.print(table)

    # 显示时间范围
    start_date = result.dates[0].strftime("%Y-%m-%d")
    end_date = result.dates[-1].strftime("%Y-%m-%d")
    console.print(f"\n回测期间：{start_date} 至 {end_date}")


def save_forward_chart(
    result: ForwardSimulationResult,
    *,
    output: Path,
    quantiles: Sequence[float] = (0.1, 0.5, 0.9),
    show: bool = False,
) -> Path:
    """保存前瞻性模拟的分位数图表。"""

    df = result.quantiles(quantiles)
    plt.figure(figsize=(10, 6))
    for column in df.columns:
        plt.plot(df.index, df[column], label=column.upper())
    plt.xlabel("Year")
    plt.ylabel("Portfolio Value")
    plt.title("Forward Simulation: Portfolio Value Quantiles")
    plt.grid(alpha=0.3)
    plt.legend()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    if show:
        plt.show()
    plt.close()
    return output


def save_backtest_chart(
    result: BacktestResult,
    *,
    output: Path,
    show: bool = False,
) -> Path:
    """保存历史回测的价值走势图。"""

    plt.figure(figsize=(12, 6))

    # 绘制组合价值
    plt.subplot(2, 1, 1)
    plt.plot(result.dates, result.portfolio_values, label="Portfolio Value", linewidth=2)
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.title("Backtest: Portfolio Value Over Time")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.xticks(rotation=45)

    # 绘制权重变化
    plt.subplot(2, 1, 2)
    for asset in result.asset_names:
        plt.plot(result.dates, result.weights_history[asset], label=asset, alpha=0.7)
    plt.xlabel("Date")
    plt.ylabel("Weight")
    plt.title("Asset Weights Over Time")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.xticks(rotation=45)

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    if show:
        plt.show()
    plt.close()
    return output


def save_backtest_charts(
    result: BacktestResult,
    output_dir: Path,
    *,
    show: bool = False,
) -> list[Path]:
    """保存历史回测的多个图表到指定目录。

    Args:
        result: 回测结果
        output_dir: 输出目录
        show: 是否显示图表

    Returns:
        保存的图表文件路径列表
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存主图表（价值走势和权重变化）
    main_chart = output_dir / "portfolio_overview.png"
    save_backtest_chart(result, output=main_chart, show=show)

    # 保存收益率分布图
    returns_chart = output_dir / "returns_distribution.png"
    plt.figure(figsize=(10, 6))
    plt.hist(result.returns.dropna(), bins=50, alpha=0.7, edgecolor="black")
    plt.xlabel("Daily Return")
    plt.ylabel("Frequency")
    plt.title("Daily Returns Distribution")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(returns_chart, dpi=200)
    if show:
        plt.show()
    plt.close()

    # 保存回撤图
    drawdown_chart = output_dir / "drawdown.png"
    cumulative_peaks = result.portfolio_values.expanding().max()
    drawdowns = (result.portfolio_values - cumulative_peaks) / cumulative_peaks
    plt.figure(figsize=(12, 6))
    plt.fill_between(result.dates, drawdowns, 0, alpha=0.3, color="red")
    plt.plot(result.dates, drawdowns, color="red", linewidth=1)
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.title("Portfolio Drawdown Over Time")
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(drawdown_chart, dpi=200)
    if show:
        plt.show()
    plt.close()

    return [main_chart, returns_chart, drawdown_chart]

