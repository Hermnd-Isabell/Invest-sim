from pathlib import Path

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
    }
    assert risk_metrics["value_at_risk"] >= 0
    assert risk_metrics["conditional_value_at_risk"] >= 0
    assert 0 <= risk_metrics["max_drawdown"] <= 1

