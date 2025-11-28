from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Ensure we can import invest_sim when running via Streamlit
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from invest_sim.backtester import Backtester
from invest_sim.data_models import (
    Asset,
    BacktestConfig,
    ContributionPlan,
    SimulationConfig,
    StrategyConfig,
)
from invest_sim.forward_simulator import ForwardSimulator


@dataclass
class BacktestBridgeResult:
    df: pd.DataFrame
    metrics: Dict[str, float]


class InvestSimBridge:
    """
    Facade layer to decouple Streamlit UI from backend logic.
    Provides helpers to translate lightweight UI dictionaries into
    full Pydantic configs consumed by the simulation and backtest engines.
    """

    _STRATEGY_NAME_MAP = {
        "Fixed Weights": "fixed",
        "Target Risk": "target_risk",
        "Adaptive Rebalance": "adaptive",
    }
    _DEFAULT_ASSET_TEMPLATES: List[Dict[str, float]] = [
        {"name": "Global Equity", "expected_return": 0.07, "volatility": 0.15, "weight": 0.6},
        {"name": "Global Bonds", "expected_return": 0.03, "volatility": 0.06, "weight": 0.3},
        {"name": "Real Assets", "expected_return": 0.05, "volatility": 0.10, "weight": 0.1},
    ]
    _DEFAULT_ANNUAL_CONTRIBUTION = 0.0
    _DEFAULT_CONTRIBUTION_FREQUENCY = 12
    _DEFAULT_INITIAL_BALANCE = 100_000.0
    _DEFAULT_NUM_TRIALS = 500
    _DEFAULT_SIM_YEARS = 10
    _DEFAULT_SIM_REBAL_FREQ = 12
    _DEFAULT_BT_REBAL_FREQ = 21  # about monthly on daily data

    @classmethod
    def get_available_strategies(cls) -> List[str]:
        return list(cls._STRATEGY_NAME_MAP.keys())

    @staticmethod
    def load_market_data(uploaded_file=None):
        """Loads CSV data provided by the user or fabricates a simple demo series."""
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
            return df.select_dtypes(include=[np.number]).sort_index()

        dates = pd.date_range(start="2020-01-01", periods=1000, freq="B")
        data = pd.DataFrame(
            np.random.normal(0.0005, 0.01, (1000, 3)),
            index=dates,
            columns=["Stock", "Bond", "Gold"],
        )
        return (1 + data).cumprod() * 100

    @classmethod
    def run_forward_simulation(cls, params: Dict[str, Any]):
        config = cls._build_simulation_config(params)
        simulator = ForwardSimulator(
            config,
            seed=params.get("seed"),
            input_model=params.get("input_model"),
        )
        result = simulator.run()
        return cls._format_forward_result(result)

    @classmethod
    def run_backtest(cls, params: Dict[str, Any], market_data: pd.DataFrame) -> BacktestBridgeResult:
        config = cls._build_backtest_config(params, market_data)
        backtester = Backtester(config)
        result = backtester.run(market_data)
        return cls._format_backtest_result(result, risk_free=params.get("risk_free", 0.0))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @classmethod
    def _build_simulation_config(cls, params: Dict[str, Any]) -> SimulationConfig:
        years = int(params.get("duration", cls._DEFAULT_SIM_YEARS))
        initial_balance = float(params.get("capital", cls._DEFAULT_INITIAL_BALANCE))
        num_trials = int(params.get("num_trials", cls._DEFAULT_NUM_TRIALS))
        rebalance_frequency = int(
            params.get("rebalance_frequency", cls._DEFAULT_SIM_REBAL_FREQ)
        )

        assets = cls._build_assets(params)
        contribution = cls._build_contribution_plan(params)
        strategy = cls._build_strategy_config(params)

        return SimulationConfig(
            years=years,
            initial_balance=initial_balance,
            num_trials=num_trials,
            rebalance_frequency=rebalance_frequency,
            assets=assets,
            contribution_plan=contribution,
            strategy=strategy,
        )

    @classmethod
    def _build_backtest_config(
        cls, params: Dict[str, Any], market_data: pd.DataFrame
    ) -> BacktestConfig:
        initial_balance = float(params.get("capital", cls._DEFAULT_INITIAL_BALANCE))
        rebalance_frequency = int(
            params.get("rebalance_frequency", cls._DEFAULT_BT_REBAL_FREQ)
        )

        asset_weights = cls._extract_asset_weights(params, market_data)
        contribution = cls._build_contribution_plan(params)
        strategy = cls._build_strategy_config(params)

        return BacktestConfig(
            initial_balance=initial_balance,
            rebalance_frequency=rebalance_frequency,
            asset_weights=asset_weights,
            contribution_plan=contribution,
            strategy=strategy,
        )

    @classmethod
    def _build_assets(cls, params: Dict[str, Any]) -> List[Asset]:
        if "assets" in params and params["assets"]:
            return [Asset.model_validate(asset_payload) for asset_payload in params["assets"]]

        leverage = float(params.get("leverage", 1.0))
        assets = []
        for template in cls._DEFAULT_ASSET_TEMPLATES:
            payload = {
                **template,
                "expected_return": template["expected_return"] * leverage,
                "volatility": template["volatility"] * leverage,
            }
            assets.append(Asset.model_validate(payload))
        return assets

    @classmethod
    def _build_contribution_plan(cls, params: Dict[str, Any]) -> ContributionPlan:
        annual = float(params.get("annual_contribution", cls._DEFAULT_ANNUAL_CONTRIBUTION))
        frequency = int(params.get("contribution_frequency", cls._DEFAULT_CONTRIBUTION_FREQUENCY))
        return ContributionPlan(annual_contribution=annual, frequency=frequency)

    @classmethod
    def _build_strategy_config(cls, params: Dict[str, Any]) -> StrategyConfig:
        display_name = params.get("strategy", "Fixed Weights")
        internal_name = cls._STRATEGY_NAME_MAP.get(display_name, "fixed")
        kwargs: Dict[str, Any] = {"name": internal_name}

        if internal_name == "target_risk" and params.get("target_vol") is not None:
            kwargs["target_volatility"] = float(params["target_vol"])
        if internal_name == "adaptive" and params.get("threshold") is not None:
            kwargs["rebalance_threshold"] = float(params["threshold"])

        return StrategyConfig(**kwargs)

    @staticmethod
    def _extract_asset_weights(params: Dict[str, Any], market_data: pd.DataFrame) -> Dict[str, float]:
        if params.get("asset_weights"):
            return {k: float(v) for k, v in params["asset_weights"].items()}

        columns = [col for col in market_data.columns if pd.api.types.is_numeric_dtype(market_data[col])]
        if not columns:
            raise ValueError("Market data must contain at least one numeric asset column.")

        weight = 1.0 / len(columns)
        return {col: weight for col in columns}

    @staticmethod
    def _format_forward_result(result) -> Dict[str, Any]:
        dates = InvestSimBridge._projection_dates(result.timeline_years)
        paths = result.trajectories.T  # shape: (periods, trials)
        median_path = np.median(result.trajectories, axis=0)
        risk_metrics = result.risk_metrics()
        return {
            "dates": dates,
            "paths": paths,
            "median": median_path,
            "quantiles": result.quantiles(),
            "risk_metrics": risk_metrics,
            "input_model": result.input_model,
            "risk_summary": InvestSimBridge._build_risk_sentence(result.input_model),
        }

    @staticmethod
    def _projection_dates(timeline_years: np.ndarray) -> pd.DatetimeIndex:
        start = pd.Timestamp(datetime.utcnow().date())
        offsets = pd.to_timedelta(timeline_years * 365.25, unit="D")
        return start + offsets

    @staticmethod
    def _build_risk_sentence(input_model: Dict[str, Any] | None) -> str:
        if not input_model:
            return "本次 Monte Carlo 模拟基于默认正态分布输入模型，得到以下 VaR/CVaR 结果。"
        params = input_model.get("params", {})
        params_str = ", ".join(f"{k}={v}" for k, v in params.items())
        return (
            f"本次 Monte Carlo 模拟基于 {input_model.get('dist_name', 'normal')} 分布"
            f"（参数：{params_str}）构建输入模型，得到以下 VaR/CVaR 结果。"
        )

    @staticmethod
    def _format_backtest_result(result, risk_free: float) -> BacktestBridgeResult:
        portfolio = result.portfolio_values.rename("Portfolio")
        drawdown = ((portfolio - portfolio.cummax()) / portfolio.cummax()).rename("Drawdown")
        df = pd.concat([portfolio, drawdown], axis=1)

        metrics = {
            "total_return": result.total_return(),
            "sharpe": result.sharpe_ratio(risk_free),
            "max_dd": result.max_drawdown(),
            "volatility": result.volatility(),
            "annualized_return": result.annualized_return(),
        }
        return BacktestBridgeResult(df=df, metrics=metrics)