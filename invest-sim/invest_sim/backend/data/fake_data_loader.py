"""假数据生成模块。

提供基于收益分布的价格数据生成功能。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..input_modeling.distributions import generate_returns


def generate_fake_price_data(
    n_days: int = 300,
    start_price: float = 100.0,
    dist_name: str = "normal",
    dist_params: dict | None = None,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """生成假价格数据。

    先生成收益序列，然后将收益累乘成价格。

    Args:
        n_days: 生成的天数（交易日）
        start_price: 起始价格
        dist_name: 收益分布类型，当前支持 "normal"
        dist_params: 分布参数字典。如果为 None，使用默认值
            {"mean": 0.0005, "vol": 0.02}
        rng: 可选的随机数生成器

    Returns:
        包含日期和价格的 DataFrame，列包括：
        - Date: 日期索引
        - Price: 价格序列
        - Returns: 收益序列（可选）

    Example:
        >>> df = generate_fake_price_data(
        ...     n_days=100,
        ...     start_price=100.0,
        ...     dist_name="normal",
        ...     dist_params={"mean": 0.001, "vol": 0.02}
        ... )
        >>> print(df.head())
    """
    # 设置默认参数
    if dist_params is None:
        dist_params = {"mean": 0.0005, "vol": 0.02}

    # 生成日期序列（工作日）
    dates = pd.date_range(
        start="2020-01-01",
        periods=n_days,
        freq='B'  # 工作日频率
    )

    # 生成收益序列
    returns = generate_returns(
        dist_name=dist_name,
        size=n_days,
        params=dist_params,
        rng=rng,
    )

    # 将收益累乘成价格
    # price = start_price * (1 + returns).cumprod()
    price = start_price * np.cumprod(1 + returns)

    # 构建 DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Price': price,
        'Returns': returns,
    }).set_index('Date')

    return df

