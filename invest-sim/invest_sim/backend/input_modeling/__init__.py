"""Input Modeling 模块：收益分布生成与拟合。"""

from .distributions import generate_returns
from .fitting import fit_normal

__all__ = ["generate_returns", "fit_normal"]

