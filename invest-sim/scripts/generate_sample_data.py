"""
生成模拟的历史价格数据，用于回测演示。

生成的数据包括：
- 股票（高波动、高收益）
- 基金（中等波动、中等收益）
- 债券（低波动、低收益）
- 现金（无波动、无收益）
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from invest_sim.backend.input_modeling.distributions import generate_returns


def generate_asset_prices(
    name: str,
    start_date: str,
    end_date: str,
    initial_price: float,
    annual_return: float,
    annual_volatility: float,
    seed: int | None = None,
) -> pd.Series:
    """生成单个资产的价格时间序列。

    Args:
        name: 资产名称
        start_date: 开始日期（YYYY-MM-DD）
        end_date: 结束日期（YYYY-MM-DD）
        initial_price: 初始价格
        annual_return: 年化收益率（小数形式，如 0.08 表示 8%）
        annual_volatility: 年化波动率（小数形式，如 0.15 表示 15%）
        seed: 随机种子

    Returns:
        价格时间序列（pd.Series），索引为日期
    """
    if seed is not None:
        np.random.seed(seed)

    # 生成交易日序列（排除周末）
    dates = pd.bdate_range(start=start_date, end=end_date)
    n_days = len(dates)

    # 计算每日收益率参数
    # 假设一年有 252 个交易日
    daily_return = annual_return / 252
    daily_volatility = annual_volatility / np.sqrt(252)

    # 生成随机收益率（几何布朗运动）
    # 使用对数收益率，然后转换为价格
    log_returns = generate_returns(
        dist_name="normal",
        size=n_days,
        params={
            "mean": daily_return - 0.5 * daily_volatility**2,  # 调整漂移项
            "vol": daily_volatility,
        },
    )

    # 转换为价格序列
    log_prices = np.log(initial_price) + np.cumsum(log_returns)
    prices = np.exp(log_prices)

    return pd.Series(prices, index=dates, name=name)


def generate_portfolio_data(
    output_path: Path,
    start_date: str = "2020-01-01",
    end_date: str = "2024-12-31",
    seed: int = 42,
) -> None:
    """生成完整的投资组合历史数据。

    Args:
        output_path: 输出 CSV 文件路径
        start_date: 开始日期
        end_date: 结束日期
        seed: 随机种子
    """
    np.random.seed(seed)

    # 定义资产特征
    assets = [
        {
            "name": "股票_大盘",
            "initial_price": 100.0,
            "annual_return": 0.12,  # 12% 年化收益
            "annual_volatility": 0.20,  # 20% 年化波动
            "seed_offset": 1,
        },
        {
            "name": "股票_成长",
            "initial_price": 50.0,
            "annual_return": 0.15,  # 15% 年化收益
            "annual_volatility": 0.30,  # 30% 年化波动（高波动）
            "seed_offset": 2,
        },
        {
            "name": "基金_混合",
            "initial_price": 10.0,
            "annual_return": 0.08,  # 8% 年化收益
            "annual_volatility": 0.12,  # 12% 年化波动
            "seed_offset": 3,
        },
        {
            "name": "基金_债券",
            "initial_price": 1.0,
            "annual_return": 0.04,  # 4% 年化收益
            "annual_volatility": 0.05,  # 5% 年化波动（低波动）
            "seed_offset": 4,
        },
        {
            "name": "债券_国债",
            "initial_price": 100.0,
            "annual_return": 0.03,  # 3% 年化收益
            "annual_volatility": 0.03,  # 3% 年化波动（极低波动）
            "seed_offset": 5,
        },
        {
            "name": "现金",
            "initial_price": 1.0,
            "annual_return": 0.02,  # 2% 年化收益（近似无风险利率）
            "annual_volatility": 0.001,  # 几乎无波动
            "seed_offset": 6,
        },
    ]

    # 生成各资产价格序列
    price_series = []
    for asset in assets:
        series = generate_asset_prices(
            name=asset["name"],
            start_date=start_date,
            end_date=end_date,
            initial_price=asset["initial_price"],
            annual_return=asset["annual_return"],
            annual_volatility=asset["annual_volatility"],
            seed=seed + asset["seed_offset"],
        )
        price_series.append(series)

    # 合并为 DataFrame
    price_df = pd.concat(price_series, axis=1)

    # 确保所有资产都有相同的日期索引
    price_df = price_df.reindex(
        pd.bdate_range(start=start_date, end=end_date), method="ffill"
    )

    # 保存为 CSV
    price_df.to_csv(output_path, index=True, index_label="date")
    print(f"✅ 已生成 {len(price_df)} 个交易日的数据")
    print(f"📅 日期范围: {price_df.index[0].date()} 至 {price_df.index[-1].date()}")
    print(f"📊 资产数量: {len(price_df.columns)}")
    print(f"💾 保存路径: {output_path}")
    print("\n资产列表:")
    for col in price_df.columns:
        total_return = (price_df[col].iloc[-1] / price_df[col].iloc[0] - 1) * 100
        print(f"  - {col}: 总收益率 {total_return:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成模拟历史价格数据")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "sample_prices.csv",
        help="输出文件路径",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2020-01-01",
        help="开始日期 (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2024-12-31",
        help="结束日期 (YYYY-MM-DD)",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    args = parser.parse_args()

    # 确保输出目录存在
    args.output.parent.mkdir(parents=True, exist_ok=True)

    generate_portfolio_data(
        output_path=args.output,
        start_date=args.start_date,
        end_date=args.end_date,
        seed=args.seed,
    )

