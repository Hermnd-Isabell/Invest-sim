from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class Asset(BaseModel):
    """单一资产配置假设。"""

    name: str = Field(..., min_length=1, description="资产名称")
    expected_return: float = Field(
        ..., gt=-1.0, description="年化期望收益率，使用小数形式，例如 0.07"
    )
    volatility: float = Field(
        ..., gt=0, description="年化波动率（标准差），使用小数形式，例如 0.15"
    )
    weight: float = Field(..., ge=0, description="目标权重，0-1 之间")


class ContributionPlan(BaseModel):
    """定投/定额投入计划。"""

    annual_contribution: float = Field(
        0.0, ge=0.0, description="每年投入金额，默认为 0"
    )
    frequency: int = Field(
        12, gt=0, description="一年投入次数，例如 12 表示每月一次"
    )

    @property
    def periodic_contribution(self) -> float:
        return self.annual_contribution / self.frequency if self.frequency else 0.0


class StrategyConfig(BaseModel):
    """策略配置，name 用于工厂模式创建实例。"""

    name: Literal["fixed", "target_risk", "adaptive"] = "fixed"
    target_volatility: Optional[float] = Field(
        None, gt=0, description="目标风险策略的年化波动率目标"
    )
    rebalance_threshold: float = Field(
        0.05,
        ge=0.0,
        description="自适应再平衡触发阈值，权重偏离超过该值会触发调仓",
    )


class SimulationConfig(BaseModel):
    """前瞻性模拟配置（基于假设参数预测未来）。"""

    years: int = Field(..., gt=0, description="模拟年数")
    initial_balance: float = Field(..., gt=0, description="初始组合市值")
    num_trials: int = Field(500, gt=0, description="Monte Carlo 路径数量")
    rebalance_frequency: int = Field(
        12, gt=0, description="再平衡频率，以期数计（默认每月）"
    )
    assets: list[Asset]
    contribution_plan: ContributionPlan = ContributionPlan()
    strategy: StrategyConfig = StrategyConfig()

    @field_validator("assets")
    @classmethod
    def _check_assets(cls, assets: list[Asset]) -> list[Asset]:
        if not assets:
            raise ValueError("至少需要一个资产配置")
        total_weight = sum(asset.weight for asset in assets)
        if total_weight <= 0:
            raise ValueError("资产权重总和必须大于 0")
        return assets

    @property
    def normalized_weights(self) -> list[float]:
        total = sum(asset.weight for asset in self.assets)
        return [asset.weight / total for asset in self.assets]

    @property
    def asset_names(self) -> list[str]:
        return [asset.name for asset in self.assets]


class BacktestConfig(BaseModel):
    """历史回测配置（基于历史数据回测过去表现）。"""

    initial_balance: float = Field(..., gt=0, description="初始组合市值")
    rebalance_frequency: int = Field(
        1, gt=0, description="再平衡频率，以交易日数计（默认每1个交易日，即每日）"
    )
    asset_weights: dict[str, float] = Field(
        ..., description="资产名称到权重的映射，例如 {'SPY': 0.6, 'AGG': 0.3, 'CASH': 0.1}"
    )
    contribution_plan: ContributionPlan = ContributionPlan()
    strategy: StrategyConfig = StrategyConfig()

    @field_validator("asset_weights")
    @classmethod
    def _check_weights(cls, weights: dict[str, float]) -> dict[str, float]:
        if not weights:
            raise ValueError("至少需要一个资产配置")
        total_weight = sum(weights.values())
        if total_weight <= 0:
            raise ValueError("资产权重总和必须大于 0")
        # 归一化权重
        if abs(total_weight - 1.0) > 1e-6:
            weights = {k: v / total_weight for k, v in weights.items()}
        return weights

    @property
    def normalized_weights(self) -> dict[str, float]:
        total = sum(self.asset_weights.values())
        if abs(total - 1.0) < 1e-6:
            return self.asset_weights.copy()
        return {k: v / total for k, v in self.asset_weights.items()}

    @property
    def asset_names(self) -> list[str]:
        return list(self.asset_weights.keys())

