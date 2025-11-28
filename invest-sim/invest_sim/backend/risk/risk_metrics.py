"""Centralized risk metric helpers."""

from __future__ import annotations

from typing import Any

import numpy as np


def _ensure_array(final_values: Any) -> np.ndarray:
    """Convert inputs to a flattened float array."""
    values = np.asarray(final_values, dtype=float)
    if values.ndim == 0:
        return values.reshape(1)
    return values.ravel()


def compute_var(final_values: Any, *, initial_balance: float, level: float) -> float:
    """Compute Value at Risk (loss) at the given confidence level."""
    samples = _ensure_array(final_values)
    if not 0 < level < 1:
        raise ValueError("VaR level must be between 0 and 1.")
    threshold = float(np.quantile(samples, level))
    return max(0.0, initial_balance - threshold)


def compute_cvar(final_values: Any, *, initial_balance: float, level: float) -> float:
    """Compute Conditional VaR (Expected Shortfall) for the left tail."""
    samples = _ensure_array(final_values)
    if not 0 < level < 1:
        raise ValueError("CVaR level must be between 0 and 1.")
    threshold = float(np.quantile(samples, level))
    tail = samples[samples <= threshold]
    expected_tail = float(tail.mean()) if tail.size else threshold
    return max(0.0, initial_balance - expected_tail)


def summarize_tail_risk(
    final_values: Any, *, initial_balance: float, level: float
) -> dict[str, float]:
    """Return a dict containing both VaR and CVaR."""
    return {
        "value_at_risk": compute_var(final_values, initial_balance=initial_balance, level=level),
        "conditional_value_at_risk": compute_cvar(
            final_values, initial_balance=initial_balance, level=level
        ),
    }

