"""Distribution fitting helpers for input modeling."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

try:  # Optional SciPy support
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover - SciPy is optional
    scipy_stats = None


def fit_normal(returns: np.ndarray) -> Dict[str, float]:
    """Fit a normal distribution to returns and return its parameters."""
    values = np.asarray(returns, dtype=float).ravel()
    if values.size == 0:
        raise ValueError("returns array cannot be empty.")
    mean = float(np.mean(values))
    vol = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    return {"mean": mean, "vol": vol}


def fit_student_t(returns: np.ndarray) -> Dict[str, float]:
    """Fit a Student-t distribution if SciPy is available."""
    if scipy_stats is None:
        raise ImportError("scipy is required to fit a student-t distribution.")
    values = np.asarray(returns, dtype=float).ravel()
    if values.size == 0:
        raise ValueError("returns array cannot be empty.")
    df, loc, scale = scipy_stats.t.fit(values)
    return {"df": float(df), "mean": float(loc), "scale": float(scale)}

