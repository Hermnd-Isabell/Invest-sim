"""
Implied Volatility Utilities

Pure numpy-based Black-Scholes pricing and implied volatility solver.
No external dependencies beyond numpy.
"""

import math
import numpy as np


def _norm_cdf(x):
    """Standard normal CDF using math.erf."""
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


def _norm_pdf(x):
    """Standard normal PDF."""
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * float(x) ** 2)


def bs_price_scalar(S, K, T, r, sigma, option_type: str) -> float:
    """
    Simple scalar BS price for calls/puts.
    
    Parameters
    ----------
    S : float
        Spot price
    K : float
        Strike price
    T : float
        Time to maturity (years)
    r : float
        Risk-free rate
    sigma : float
        Volatility (annualized)
    option_type : str
        "call" or "put" (case-insensitive)
    
    Returns
    -------
    float
        Option price
        
    Notes
    -----
    If T <= 0, falls back to intrinsic value:
    - Call: max(S - K, 0)
    - Put: max(K - S, 0)
    """
    option_type = option_type.lower()
    
    # Handle expired options
    if T <= 0:
        if option_type == "call":
            return max(S - K, 0.0)
        else:
            return max(K - S, 0.0)
    
    # Safety checks
    S_safe = max(S, 1e-12)
    K_safe = max(K, 1e-12)
    sigma_safe = max(sigma, 1e-12)
    T_safe = max(T, 1e-12)
    
    sqrt_T = math.sqrt(T_safe)
    d1 = (math.log(S_safe / K_safe) + (r + 0.5 * sigma_safe**2) * T_safe) / (sigma_safe * sqrt_T)
    d2 = d1 - sigma_safe * sqrt_T
    
    discount = math.exp(-r * T_safe)
    
    if option_type == "call":
        return S_safe * _norm_cdf(d1) - K_safe * discount * _norm_cdf(d2)
    else:
        return K_safe * discount * _norm_cdf(-d2) - S_safe * _norm_cdf(-d1)


def bs_vega_scalar(S, K, T, r, sigma) -> float:
    """
    Scalar vega for Black-Scholes model.
    
    Parameters
    ----------
    S : float
        Spot price
    K : float
        Strike price
    T : float
        Time to maturity (years)
    r : float
        Risk-free rate
    sigma : float
        Volatility (annualized)
    
    Returns
    -------
    float
        Vega (sensitivity to volatility change)
        
    Notes
    -----
    If T <= 0, returns 0.0.
    """
    if T <= 0:
        return 0.0
    
    # Safety checks
    S_safe = max(S, 1e-12)
    K_safe = max(K, 1e-12)
    sigma_safe = max(sigma, 1e-12)
    T_safe = max(T, 1e-12)
    
    sqrt_T = math.sqrt(T_safe)
    d1 = (math.log(S_safe / K_safe) + (r + 0.5 * sigma_safe**2) * T_safe) / (sigma_safe * sqrt_T)
    
    return S_safe * _norm_pdf(d1) * sqrt_T


def implied_vol_newton(
    S: float,
    K: float,
    T: float,
    r: float,
    market_price: float,
    option_type: str,
    initial_sigma: float = 0.2,
    tol: float = 1e-6,
    max_iter: int = 100,
    sigma_min: float = 1e-4,
    sigma_max: float = 5.0,
) -> float | None:
    """
    Solve for implied volatility using a damped Newton-Raphson method.
    
    Parameters
    ----------
    S : float
        Spot price
    K : float
        Strike price
    T : float
        Time to maturity (years)
    r : float
        Risk-free rate
    market_price : float
        Observed market price of the option
    option_type : str
        "call" or "put" (case-insensitive)
    initial_sigma : float, default=0.2
        Starting guess for volatility
    tol : float, default=1e-6
        Convergence tolerance
    max_iter : int, default=100
        Maximum number of iterations
    sigma_min : float, default=1e-4
        Minimum allowed volatility
    sigma_max : float, default=5.0
        Maximum allowed volatility
    
    Returns
    -------
    float or None
        Implied volatility if convergence successful and within bounds,
        None if:
        - market_price is non-sensical (<= 0)
        - T <= 0
        - Solver cannot converge within max_iter
        - Vega becomes too small (< 1e-8)
    """
    option_type = option_type.lower()
    
    # Validation checks
    if market_price <= 0:
        return None
    
    if T <= 0:
        return None
    
    # Initialize
    sigma = initial_sigma
    sigma = max(sigma_min, min(sigma, sigma_max))  # Clamp initial guess
    
    # Newton-Raphson iteration
    for _ in range(max_iter):
        # Calculate current price and vega
        price = bs_price_scalar(S, K, T, r, sigma, option_type)
        diff = price - market_price
        
        # Check convergence
        if abs(diff) < tol:
            return sigma
        
        # Calculate vega
        vega = bs_vega_scalar(S, K, T, r, sigma)
        
        # Check if vega is too small (numerical instability)
        if abs(vega) < 1e-8:
            return None
        
        # Newton-Raphson update: sigma_new = sigma - f(sigma) / f'(sigma)
        sigma_next = sigma - diff / vega
        
        # Clamp to bounds
        sigma_next = max(sigma_min, min(sigma_next, sigma_max))
        
        # Check if we're stuck (no progress)
        if abs(sigma_next - sigma) < 1e-10:
            return None
        
        sigma = sigma_next
    
    # Failed to converge
    return None

