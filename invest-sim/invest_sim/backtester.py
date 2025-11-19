from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .data_models import BacktestConfig
from .strategies import Strategy, build_strategy


@dataclass(frozen=True)
class BacktestResult:
    """历史回测结果封装。"""

    dates: pd.DatetimeIndex
    portfolio_values: pd.Series  # 组合价值时间序列
    weights_history: pd.DataFrame  # 权重历史，索引为日期，列为各资产
    returns: pd.Series  # 组合收益率时间序列
    config: BacktestConfig

    @property
    def asset_names(self) -> list[str]:
        """返回资产名称列表。"""
        return list(self.weights_history.columns)

    def total_return(self) -> float:
        """计算总收益率。"""
        initial = self.config.initial_balance
        final = float(self.portfolio_values.iloc[-1])
        return (final - initial) / initial

    def annualized_return(self) -> float:
        """计算年化收益率。"""
        total_ret = self.total_return()
        years = (self.dates[-1] - self.dates[0]).days / 365.25
        if years <= 0:
            return 0.0
        return (1 + total_ret) ** (1 / years) - 1

    def volatility(self) -> float:
        """计算年化波动率。"""
        if len(self.returns) < 2:
            return 0.0
        # 假设每日数据，转换为年化
        periods_per_year = 252  # 交易日数
        return float(self.returns.std() * np.sqrt(periods_per_year))

    def sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """计算夏普比率。"""
        ann_ret = self.annualized_return()
        ann_vol = self.volatility()
        if ann_vol == 0:
            return 0.0
        return (ann_ret - risk_free_rate) / ann_vol

    def max_drawdown(self) -> float:
        """计算最大回撤。"""
        cumulative_peaks = self.portfolio_values.expanding().max()
        drawdowns = (self.portfolio_values - cumulative_peaks) / cumulative_peaks
        return float(drawdowns.min())

    def risk_metrics(self, *, risk_free_rate: float = 0.0) -> dict[str, float]:
        """返回风险指标。"""
        return {
            "total_return": self.total_return(),
            "annualized_return": self.annualized_return(),
            "volatility": self.volatility(),
            "sharpe_ratio": self.sharpe_ratio(risk_free_rate),
            "max_drawdown": self.max_drawdown(),
        }


class Backtester:
    """历史回测器（基于历史数据回测过去表现）。"""

    def __init__(
        self,
        config: BacktestConfig,
        *,
        strategy: Optional[Strategy] = None,
    ) -> None:
        self.config = config
        # 注意：历史回测需要适配策略接口
        # 这里我们需要一个适配器，因为策略接口是为前瞻性模拟设计的
        self.strategy = strategy or self._build_strategy_adapter()
        self.asset_names = list(config.asset_weights.keys())
        self.num_assets = len(self.asset_names)

    def _build_strategy_adapter(self) -> Strategy:
        """构建策略适配器。

        注意：由于策略接口是为前瞻性模拟设计的，我们需要创建一个适配的配置。
        这里我们创建一个临时的 SimulationConfig 来适配策略接口。
        """
        from .data_models import Asset, SimulationConfig

        # 创建临时资产配置（历史回测不需要这些参数，但策略接口需要）
        # 为历史回测提供合理的默认值（仅用于策略接口，实际计算使用历史数据）
        assets = [
            Asset(
                name=name,
                expected_return=0.08,  # 默认 8% 年化收益（仅占位符）
                volatility=0.15,  # 默认 15% 年化波动（仅占位符，历史回测中不使用）
                weight=weight,
            )
            for name, weight in self.config.normalized_weights.items()
        ]

        temp_config = SimulationConfig(
            years=1,  # 占位符
            initial_balance=self.config.initial_balance,
            num_trials=1,
            rebalance_frequency=self.config.rebalance_frequency,
            assets=assets,
            contribution_plan=self.config.contribution_plan,
            strategy=self.config.strategy,
        )

        from .strategies import build_strategy

        return build_strategy(temp_config)

    def run(self, price_data: pd.DataFrame) -> BacktestResult:
        """运行历史回测。

        参数
        ----
        price_data:
            历史价格数据，索引为日期（DatetimeIndex），列为各资产的价格

        返回
        ----
        BacktestResult:
            回测结果
        """
        # 验证数据
        missing_assets = set(self.asset_names) - set(price_data.columns)
        if missing_assets:
            raise ValueError(f"价格数据中缺少资产：{missing_assets}")

        # 选择需要的资产并按顺序排列
        prices = price_data[self.asset_names].copy()
        prices = prices.sort_index()

        if len(prices) < 2:
            raise ValueError("历史数据至少需要 2 个时间点")

        # 计算收益率
        returns = prices.pct_change().dropna()

        # 初始化
        initial_balance = self.config.initial_balance
        portfolio_value = initial_balance
        weights = np.array(
            [self.config.normalized_weights[name] for name in self.asset_names],
            dtype=float,
        )
        asset_values = portfolio_value * weights

        # 存储历史
        portfolio_values = [initial_balance]
        weights_history = [weights.copy()]
        portfolio_returns = [0.0]  # 第一期为 0

        periodic_contribution = self.config.contribution_plan.periodic_contribution
        periods_per_year = self._estimate_periods_per_year(returns.index)

        # 如果定投频率与数据频率不匹配，需要调整
        # 这里简化处理：假设数据频率与定投频率一致
        contribution_per_period = periodic_contribution
        if self.config.contribution_plan.frequency != periods_per_year:
            # 调整每期投入
            contribution_per_period = (
                self.config.contribution_plan.annual_contribution / periods_per_year
            )

        # 逐期回测
        for i, (date, period_returns) in enumerate(returns.iterrows()):
            # 注入定期投入
            if contribution_per_period > 0:
                asset_values += contribution_per_period * weights

            # 应用收益率
            asset_values *= 1.0 + period_returns.values

            # 计算新的组合价值
            portfolio_value = asset_values.sum()

            # 再平衡
            if (i + 1) % self.config.rebalance_frequency == 0:
                current_weights = asset_values / portfolio_value if portfolio_value > 0 else weights
                # 计算协方差（使用最近的数据窗口）
                window_size = min(20, i + 1)  # 使用最近 20 期或所有可用数据
                if window_size > 1:
                    recent_returns = returns.iloc[max(0, i - window_size + 1) : i + 1]
                    covariance = np.cov(recent_returns.values.T)
                else:
                    covariance = None

                weights = self.strategy.rebalance(current_weights, covariance=covariance)
                asset_values = portfolio_value * weights

            # 记录历史
            portfolio_values.append(portfolio_value)
            weights_history.append(weights.copy())
            if i > 0:
                prev_value = portfolio_values[-2]
                portfolio_returns.append((portfolio_value - prev_value) / prev_value)
            else:
                portfolio_returns.append(0.0)

        # 构建结果
        dates = pd.DatetimeIndex([prices.index[0]] + list(returns.index))
        portfolio_series = pd.Series(portfolio_values, index=dates, name="portfolio_value")
        returns_series = pd.Series(portfolio_returns, index=dates, name="returns")

        weights_df = pd.DataFrame(
            weights_history,
            index=dates,
            columns=self.asset_names,
        )

        return BacktestResult(
            dates=dates,
            portfolio_values=portfolio_series,
            weights_history=weights_df,
            returns=returns_series,
            config=self.config,
        )

    def _estimate_periods_per_year(self, dates: pd.DatetimeIndex) -> int:
        """估算每年有多少个交易期。"""
        if len(dates) < 2:
            return 252  # 默认交易日数

        total_days = (dates[-1] - dates[0]).days
        if total_days == 0:
            return 252

        periods = len(dates) - 1
        years = total_days / 365.25
        if years <= 0:
            return 252

        return int(round(periods / years))

