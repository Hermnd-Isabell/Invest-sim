"""
Invest Sim - 投资组合模拟和回测工具包。

支持两种模式：
1. 前瞻性模拟（Forward Simulation）：基于假设参数预测未来收益
2. 历史回测（Backtest）：基于历史数据回测过去表现
"""

from .backtester import Backtester, BacktestResult
from .config import load_backtest_config, load_config
from .data_loader import load_price_data
from .forward_simulator import ForwardSimulator, ForwardSimulationResult

__all__ = [
    # 前瞻性模拟
    "ForwardSimulator",
    "ForwardSimulationResult",
    "load_config",
    # 历史回测
    "Backtester",
    "BacktestResult",
    "load_backtest_config",
    "load_price_data",
]

__version__ = "0.1.0"

