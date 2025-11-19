"""
回测演示脚本：运行多个策略并生成对比报告。

展示常见回测框架的设计思路和功能。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from invest_sim.backtester import Backtester
from invest_sim.config import load_backtest_config
from invest_sim.data_loader import load_price_data
from invest_sim.data_models import BacktestConfig
from invest_sim.report import render_backtest_summary, save_backtest_charts
from rich.console import Console


def render_backtest_summary_to_file(result, *, file) -> None:
    """渲染历史回测结果的统计摘要到文件。"""
    console = Console(file=file)
    render_backtest_summary(result, console=console)


def run_backtest_demo(
    data_path: Path,
    config_dir: Path,
    output_dir: Path,
) -> None:
    """运行多个回测策略并生成对比报告。

    Args:
        data_path: 历史价格数据文件路径
        config_dir: 回测配置目录
        output_dir: 输出目录
    """
    print("=" * 80)
    print("📊 回测框架演示")
    print("=" * 80)
    print()

    # 1. 加载历史数据
    print("📂 步骤 1: 加载历史价格数据")
    print(f"   数据文件: {data_path}")
    price_data = load_price_data(data_path)
    print(f"   ✅ 加载成功: {len(price_data)} 个交易日")
    print(f"   📅 日期范围: {price_data.index[0].date()} 至 {price_data.index[-1].date()}")
    print(f"   💰 资产列表: {', '.join(price_data.columns)}")
    print()

    # 2. 加载所有回测配置
    print("📋 步骤 2: 加载回测策略配置")
    config_files = sorted(config_dir.glob("backtest_*.json"))
    if not config_files:
        print(f"   ⚠️  未找到回测配置文件（backtest_*.json）")
        return

    configs = {}
    for config_file in config_files:
        config_name = config_file.stem.replace("backtest_", "")
        config = load_backtest_config(config_file)
        configs[config_name] = (config_file, config)
        print(f"   ✅ {config_name}: {config_file.name}")
    print()

    # 3. 运行所有回测
    print("🔄 步骤 3: 运行回测")
    print("-" * 80)
    results = {}
    for config_name, (config_file, config) in configs.items():
        print(f"\n📈 运行策略: {config_name}")
        print(f"   配置: {config_file.name}")
        print(f"   初始资金: ¥{config.initial_balance:,.0f}")
        print(f"   资产配置: {dict(config.asset_weights)}")
        print(f"   再平衡频率: 每 {config.rebalance_frequency} 个交易日")
        print(f"   策略类型: {config.strategy.name}")

        try:
            backtester = Backtester(config)
            result = backtester.run(price_data)
            results[config_name] = result

            # 显示简要结果
            metrics = result.risk_metrics()
            print(f"   ✅ 回测完成")
            print(f"      总收益率: {metrics['total_return']*100:.2f}%")
            print(f"      年化收益率: {metrics['annualized_return']*100:.2f}%")
            print(f"      年化波动率: {metrics['volatility']*100:.2f}%")
            print(f"      夏普比率: {metrics['sharpe_ratio']:.2f}")
            print(f"      最大回撤: {metrics['max_drawdown']*100:.2f}%")
        except Exception as e:
            print(f"   ❌ 回测失败: {e}")
            import traceback
            traceback.print_exc()

    print()
    print("=" * 80)

    # 4. 生成对比报告
    if results:
        print("\n📊 步骤 4: 生成对比报告")
        output_dir.mkdir(parents=True, exist_ok=True)

        # 生成各策略的详细报告
        for config_name, result in results.items():
            print(f"\n   生成 {config_name} 策略报告...")
            report_path = output_dir / f"report_{config_name}.txt"
            with open(report_path, "w", encoding="utf-8") as f:
                render_backtest_summary_to_file(result, file=f)
            print(f"   ✅ 报告已保存: {report_path}")

            # 生成图表
            charts_dir = output_dir / f"charts_{config_name}"
            charts_dir.mkdir(exist_ok=True)
            save_backtest_charts(result, charts_dir)
            print(f"   ✅ 图表已保存: {charts_dir}")

        # 生成对比表格
        print("\n   生成策略对比表...")
        comparison = compare_strategies(results)
        comparison_path = output_dir / "strategy_comparison.csv"
        comparison.to_csv(comparison_path, index=True, encoding="utf-8-sig")
        print(f"   ✅ 对比表已保存: {comparison_path}")
        print("\n" + "=" * 80)
        print("\n📊 策略对比摘要:")
        print(comparison.to_string())
        print()

    print("✅ 演示完成！")


def compare_strategies(results: dict[str, any]) -> pd.DataFrame:
    """对比多个策略的表现。

    Args:
        results: 策略名称到回测结果的映射

    Returns:
        对比表格（DataFrame）
    """
    rows = []
    for name, result in results.items():
        metrics = result.risk_metrics()
        rows.append(
            {
                "策略": name,
                "总收益率 (%)": metrics["total_return"] * 100,
                "年化收益率 (%)": metrics["annualized_return"] * 100,
                "年化波动率 (%)": metrics["volatility"] * 100,
                "夏普比率": metrics["sharpe_ratio"],
                "最大回撤 (%)": metrics["max_drawdown"] * 100,
                "最终价值": result.portfolio_values.iloc[-1],
            }
        )

    df = pd.DataFrame(rows)
    df = df.set_index("策略")
    return df.round(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行回测演示")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "sample_prices.csv",
        help="历史价格数据文件路径",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path(__file__).parent.parent / "examples",
        help="回测配置目录",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / "output",
        help="输出目录",
    )

    args = parser.parse_args()

    run_backtest_demo(
        data_path=args.data,
        config_dir=args.config_dir,
        output_dir=args.output,
    )

