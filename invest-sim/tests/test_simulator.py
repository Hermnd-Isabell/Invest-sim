from pathlib import Path

import numpy as np

from invest_sim.data_models import Asset, SimulationConfig
from invest_sim.config import load_config
from invest_sim.forward_simulator import ForwardSimulator


def test_forward_simulator_runs(tmp_path: Path) -> None:
    config_path = Path(__file__).resolve().parents[1] / "examples" / "balanced.json"
    config = load_config(config_path)
    simulator = ForwardSimulator(config, seed=123)
    result = simulator.run()

    expected_steps = config.years * simulator.PERIODS_PER_YEAR + 1
    assert result.trajectories.shape == (config.num_trials, expected_steps)
    assert result.weights_history.shape == (expected_steps, len(config.assets))

    quantiles = result.quantiles((0.1, 0.5, 0.9))
    assert not quantiles.empty
    assert quantiles.iloc[-1]["p10"] < quantiles.iloc[-1]["p90"]

    max_drawdowns = result.max_drawdown_series()
    assert len(max_drawdowns) == config.num_trials
    assert max_drawdowns.between(0, 1).all()

    risk_metrics = result.risk_metrics(level=0.05)
    assert set(risk_metrics) == {
        "value_at_risk",
        "conditional_value_at_risk",
        "max_drawdown",
        "input_model",
    }
    assert risk_metrics["value_at_risk"] >= 0
    assert risk_metrics["conditional_value_at_risk"] >= 0
    assert 0 <= risk_metrics["max_drawdown"] <= 1
    assert risk_metrics["input_model"] is None


def test_forward_simulator_accepts_custom_input_model() -> None:
    config = SimulationConfig(
        years=1,
        initial_balance=10_000,
        num_trials=4,
        rebalance_frequency=12,
        assets=[
            Asset(
                name="Single",
                expected_return=0.0,
                volatility=0.01,
                weight=1.0,
            )
        ],
    )
    input_model = {
        "dist_name": "normal",
        "params": {
            "mean": 0.01,  # 每期固定 1% 收益
            "vol": 0.0,
        },
    }
    simulator = ForwardSimulator(config, seed=123, input_model=input_model)
    result = simulator.run()

    periods = config.years * simulator.PERIODS_PER_YEAR
    expected = config.initial_balance * (1.01 ** periods)
    assert np.allclose(result.trajectories[:, -1], expected)
    assert result.input_model is not None
    assert result.input_model["dist_name"] == "normal"
