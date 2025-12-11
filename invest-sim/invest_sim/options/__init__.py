"""
期权相关模块

包含期权数据存储、期权定价和模拟等功能。
"""

from .data import OptionDataStore
from .simulator import (
    OptionLeg,
    OptionMarginSimulator,
    bs_delta,
    bs_gamma,
    bs_price,
    bs_vega,
)

__all__ = [
    # 数据存储
    "OptionDataStore",
    # 期权定价函数
    "bs_price",
    "bs_delta",
    "bs_gamma",
    "bs_vega",
    # 期权类和模拟器
    "OptionLeg",
    "OptionMarginSimulator",
]

