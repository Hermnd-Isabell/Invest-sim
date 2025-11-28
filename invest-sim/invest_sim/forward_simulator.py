from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from .backend.input_modeling.distributions import generate_returns
from .backend.risk.risk_metrics import summarize_tail_risk
from .data_models import SimulationConfig
from .strategies import Strategy, build_strategy


@dataclass(frozen=True)
class ForwardSimulationResult:
    """前瞻性模拟结果封装。"""

    timeline_years: np.ndarray
    trajectories: np.ndarray  # shape: (num_trials, num_periods + 1)
    weights_history: np.ndarray  # shape: (num_periods + 1, num_assets)
    config: SimulationConfig
    input_model: dict[str, Any] | None = None

    def quantiles(self, probs: Sequence[float] = (0.1, 0.5, 0.9)) -> pd.DataFrame:
        probs_arr = np.array(probs)
        percentiles = np.clip(probs_arr * 100, 0, 100)
        quantile_matrix = np.percentile(self.trajectories, percentiles, axis=0)
        data = {
            f"p{int(prob*100):02d}": quantile_matrix[idx]
            for idx, prob in enumerate(probs_arr)
        }
        data["year"] = self.timeline_years
        df = pd.DataFrame(data)
        return df.set_index("year")

    def final_distribution(self) -> pd.Series:
        final_values = self.trajectories[:, -1]
        return pd.Series(final_values, name="final_value")

    def max_drawdown_series(self) -> pd.Series:
        """计算每条路径的最大回撤比率。"""

        cumulative_peaks = np.maximum.accumulate(self.trajectories, axis=1)
        ratios = np.divide(
            self.trajectories,
            cumulative_peaks,
            out=np.ones_like(self.trajectories),
            where=cumulative_peaks > 0,
        )
        drawdowns = 1.0 - ratios
        max_drawdown = drawdowns.max(axis=1)
        return pd.Series(max_drawdown, name="max_drawdown")

    def risk_metrics(self, *, level: float = 0.05) -> dict[str, float]:
        """返回组合终值的常见风险指标。

        参数
        ----
        level:
            VaR/CVaR 的左尾置信水平，默认 5%（VaR95）。
        """

        if not 0 < level < 1:
            raise ValueError("风险水平必须在 (0, 1) 之间")

        final_values = self.final_distribution().to_numpy()
        initial = float(self.config.initial_balance)
        tail_risk = summarize_tail_risk(final_values, initial_balance=initial, level=level)
        max_drawdown = float(self.max_drawdown_series().median())

        return {
            **tail_risk,
            "max_drawdown": max_drawdown,
            "input_model": self.input_model,
        }


class ForwardSimulator:
    """Monte Carlo 投资组合前瞻性模拟器（基于假设参数预测未来收益）。"""

    PERIODS_PER_YEAR = 12

    def __init__(
        self,
        config: SimulationConfig,
        *,
        strategy: Optional[Strategy] = None,
        seed: Optional[int] = None,
        input_model: dict[str, Any] | None = None,
    ) -> None:
        self.config = config
        self.strategy = strategy or build_strategy(config)
        self.rng = np.random.default_rng(seed)
        self.assets = config.assets
        self.num_assets = len(self.assets)
        self.input_model = input_model
        self.monthly_return_mean = self._annual_to_periodic(
            np.array([asset.expected_return for asset in self.assets], dtype=float)
        )
        self.monthly_volatility = np.array(
            [asset.volatility for asset in self.assets], dtype=float
        ) / np.sqrt(self.PERIODS_PER_YEAR)

    def run(self) -> ForwardSimulationResult:
        periods = self.config.years * self.PERIODS_PER_YEAR
        timeline = np.linspace(
            0, self.config.years, periods + 1, dtype=float
        )  # 以年为单位
        weights = self.strategy.initialize()
        weights_history = np.zeros((periods + 1, self.num_assets), dtype=float)
        weights_history[0] = weights

        trajectories = np.zeros((self.config.num_trials, periods + 1), dtype=float)
        trajectories[:, 0] = self.config.initial_balance
        asset_values = trajectories[:, [0]] * weights  # (trials, assets)

        periodic_contribution = self.config.contribution_plan.periodic_contribution

        for step in range(1, periods + 1):
            # 注入定期投入
            if periodic_contribution > 0:
                asset_values += (
                    periodic_contribution * weights
                )  # 假设按目标权重分摊投入

            # 为每个资产生成收益
            asset_returns = np.zeros((self.config.num_trials, self.num_assets))
            for i in range(self.num_assets):
                dist_name, dist_params = self._distribution_for_asset(i)
                asset_returns[:, i] = generate_returns(
                    dist_name=dist_name,
                    size=self.config.num_trials,
                    params=dist_params,
                    rng=self.rng,
                )
            asset_values *= 1.0 + asset_returns

            portfolio_values = asset_values.sum(axis=1)
            trajectories[:, step] = portfolio_values

            if step % self.config.rebalance_frequency == 0:
                current_weights = np.divide(
                    asset_values,
                    portfolio_values[:, None],
                    out=np.zeros_like(asset_values),
                    where=portfolio_values[:, None] > 0,
                )
                average_weight = current_weights.mean(axis=0)
                covariance = None
                if self.config.num_trials > 1:
                    covariance = np.cov(asset_returns, rowvar=False)
                weights = self.strategy.rebalance(average_weight, covariance=covariance)
                asset_values = portfolio_values[:, None] * weights

            weights_history[step] = weights

        sanitized_model = self._sanitize_input_model(self.input_model)
        return ForwardSimulationResult(
            timeline, trajectories, weights_history, self.config, sanitized_model
        )

    def _annual_to_periodic(self, annual_returns: np.ndarray) -> np.ndarray:
        """将年化收益转为月收益。"""
        return np.power(1.0 + annual_returns, 1 / self.PERIODS_PER_YEAR) - 1.0

    def _sanitize_input_model(self, model: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        if not model:
            return None
        dist_name = model.get("dist_name", "normal")
        raw_params = model.get("params") or {}
        cleaned_params: dict[str, Any] = {}
        for key, value in raw_params.items():
            if isinstance(value, np.ndarray):
                cleaned_params[key] = value.tolist()
            elif isinstance(value, (np.generic,)):
                cleaned_params[key] = value.item()
            else:
                cleaned_params[key] = value
        return {"dist_name": dist_name, "params": cleaned_params}

    def _distribution_for_asset(self, asset_index: int) -> tuple[str, dict[str, float]]:
        """根据输入模型解析单个资产的分布配置。"""
        base_params: dict[str, float] = {
            "mean": float(self.monthly_return_mean[asset_index]),
            "vol": float(self.monthly_volatility[asset_index]),
        }
        if not self.input_model:
            return "normal", base_params

        dist_name = self.input_model.get("dist_name", "normal")
        user_params = self.input_model.get("params") or {}
        resolved_params = base_params.copy()
        for key, value in user_params.items():
            resolved_params[key] = self._select_param_value(value, asset_index)
        return dist_name, resolved_params

    def _select_param_value(self, value: Any, asset_index: int) -> float:
        """支持列表/字典形式的参数选择，方便针对资产自定义。"""
        if isinstance(value, dict):
            asset_name = self.assets[asset_index].name
            if asset_name not in value:
                raise ValueError(
                    f"输入模型缺少资产 {asset_name} 的参数配置"
                )
            return float(value[asset_name])
        if isinstance(value, (list, tuple, np.ndarray)):
            if len(value) != self.num_assets:
                raise ValueError(
                    "输入模型的序列参数长度必须等于资产数量"
                )
            return float(value[asset_index])
        return float(value)

