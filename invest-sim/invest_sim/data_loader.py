from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def load_price_data(
    path: str | Path,
    *,
    date_column: str = "date",
    asset_columns: Optional[list[str]] = None,
    date_format: Optional[str] = None,
) -> pd.DataFrame:
    """加载历史价格数据。

    参数
    ----
    path:
        数据文件路径（支持 CSV、Excel、Parquet 等 pandas 支持的格式）
    date_column:
        日期列名，默认为 "date"
    asset_columns:
        资产列名列表。如果为 None，则使用除日期列外的所有列
    date_format:
        日期格式字符串，用于解析日期列

    返回
    ----
    DataFrame:
        索引为日期（DatetimeIndex），列为各资产的价格数据

    示例
    ----
    >>> df = load_price_data("prices.csv", asset_columns=["SPY", "AGG", "CASH"])
    >>> df.head()
                    SPY    AGG   CASH
    date
    2020-01-01  100.00  50.00   1.00
    2020-02-01  102.50  50.25   1.00
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在：{path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"不支持的文件格式：{suffix}，支持 CSV、Excel、Parquet")

    if date_column not in df.columns:
        raise ValueError(f"日期列 '{date_column}' 不存在于数据中")

    # 解析日期列
    df[date_column] = pd.to_datetime(df[date_column], format=date_format)
    df = df.set_index(date_column)
    df = df.sort_index()

    # 选择资产列
    if asset_columns is not None:
        missing = set(asset_columns) - set(df.columns)
        if missing:
            raise ValueError(f"资产列不存在：{missing}")
        df = df[asset_columns]

    # 确保数值列
    df = df.select_dtypes(include=["number"])

    if df.empty:
        raise ValueError("数据为空或没有有效的数值列")

    return df


def calculate_returns(prices: pd.DataFrame, method: str = "simple") -> pd.DataFrame:
    """计算收益率。

    参数
    ----
    prices:
        价格数据，索引为日期，列为各资产
    method:
        计算方法，"simple" 为简单收益率，(P_t - P_{t-1}) / P_{t-1}，
        "log" 为对数收益率，log(P_t / P_{t-1})

    返回
    ----
    DataFrame:
        收益率数据，第一行为 NaN（因为没有前一期数据）
    """
    if method == "simple":
        returns = prices.pct_change()
    elif method == "log":
        returns = np.log(prices / prices.shift(1))
    else:
        raise ValueError(f"不支持的计算方法：{method}，支持 'simple' 或 'log'")

    return returns

