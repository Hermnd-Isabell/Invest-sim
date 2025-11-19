from __future__ import annotations

import abc
from typing import Optional

import numpy as np

from .data_models import SimulationConfig


class Strategy(abc.ABC):
    """策略抽象基类。"""

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.base_weights = np.array(config.normalized_weights, dtype=float)

    def initialize(self) -> np.ndarray:
        return self.base_weights.copy()

    @abc.abstractmethod
    def rebalance(
        self,
        current_weights: np.ndarray,
        *,
        covariance: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """返回新的权重分配。"""

    @staticmethod
    def _normalize(weights: np.ndarray) -> np.ndarray:
        total = float(weights.sum())
        if total <= 0:
            raise ValueError("权重和必须大于 0")
        normalized = weights / total
        return np.clip(normalized, 0.0, 1.0)


class FixedAllocationStrategy(Strategy):
    """固定权重策略，始终回归目标权重。"""

    def rebalance(
        self,
        current_weights: np.ndarray,
        *,
        covariance: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return self.base_weights.copy()


class TargetRiskStrategy(Strategy):
    """目标风险策略，若当前风险高于阈值，则向低波动资产倾斜。"""

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        self.target_volatility = config.strategy.target_volatility or (
            np.sqrt(np.sum((self.base_weights * self._asset_vols) ** 2))
        )
        self._cash_index = int(np.argmin(self._asset_vols))

    @property
    def _asset_vols(self) -> np.ndarray:
        return np.array([asset.volatility for asset in self.config.assets], dtype=float)

    def rebalance(
        self,
        current_weights: np.ndarray,
        *,
        covariance: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        weights = self.base_weights.copy()
        current_vol = float(np.sqrt(np.sum((weights * self._asset_vols) ** 2)))
        if current_vol <= self.target_volatility:
            return weights

        scale = self.target_volatility / current_vol
        weights *= scale
        residual = 1.0 - weights.sum()
        if residual > 0:
            weights[self._cash_index] += residual
        return self._normalize(weights)


class AdaptiveRebalanceStrategy(Strategy):
    """自适应再平衡策略，只有偏离阈值时才恢复目标权重。"""

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        self.threshold = config.strategy.rebalance_threshold

    def rebalance(
        self,
        current_weights: np.ndarray,
        *,
        covariance: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        deviation = np.abs(current_weights - self.base_weights)
        if np.any(deviation > self.threshold):
            return self.base_weights.copy()
        return current_weights


def build_strategy(config: SimulationConfig) -> Strategy:
    """按名称构建策略实例。"""

    name = config.strategy.name
    if name == "fixed":
        return FixedAllocationStrategy(config)
    if name == "target_risk":
        return TargetRiskStrategy(config)
    if name == "adaptive":
        return AdaptiveRebalanceStrategy(config)
    raise ValueError(f"不支持的策略类型：{name}")

