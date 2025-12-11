import math
import numpy as np


def norm_cdf(x):
    """Standard normal CDF using math.erf to avoid numpy.erf dependency."""
    x_arr = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + np.vectorize(math.erf)(x_arr / math.sqrt(2.0)))


def norm_pdf(x):
    return (1.0 / math.sqrt(2.0 * math.pi)) * np.exp(-0.5 * np.asarray(x, dtype=float) ** 2)


def _bs_terms(S, K, T, r, sigma):
    S_arr = np.asarray(S, dtype=float)
    S_safe = np.maximum(S_arr, 1e-12)
    K_safe = max(K, 1e-12)
    sigma_safe = max(sigma, 1e-12)
    T_safe = max(T, 1e-12)
    sqrt_T = math.sqrt(T_safe)
    d1 = (np.log(S_safe / K_safe) + (r + 0.5 * sigma_safe**2) * T_safe) / (sigma_safe * sqrt_T)
    d2 = d1 - sigma_safe * sqrt_T
    return S_safe, K_safe, sigma_safe, T_safe, d1, d2


def bs_price(S, K, T, r, sigma, option_type):
    option_type = option_type.lower()
    if T <= 0:
        S_arr = np.asarray(S, dtype=float)
        if option_type == "call":
            return np.maximum(S_arr - K, 0.0)
        return np.maximum(K - S_arr, 0.0)
    S_safe, K_safe, _, T_safe, d1, d2 = _bs_terms(S, K, T, r, sigma)
    discount = math.exp(-r * T_safe)
    if option_type == "call":
        return S_safe * norm_cdf(d1) - K_safe * discount * norm_cdf(d2)
    return K_safe * discount * norm_cdf(-d2) - S_safe * norm_cdf(-d1)


def bs_delta(S, K, T, r, sigma, option_type):
    option_type = option_type.lower()
    S_arr = np.asarray(S, dtype=float)
    if T <= 0:
        if option_type == "call":
            return np.where(S_arr > K, 1.0, 0.0)
        return np.where(S_arr > K, 0.0, -1.0)
    _, _, _, T_safe, d1, _ = _bs_terms(S, K, T, r, sigma)
    if option_type == "call":
        return norm_cdf(d1)
    return norm_cdf(d1) - 1.0


def bs_gamma(S, K, T, r, sigma):
    """Black-Scholes gamma for a single vanilla option."""
    if T <= 0:
        return np.zeros_like(np.asarray(S, dtype=float))
    S_safe, _, sigma_safe, T_safe, d1, _ = _bs_terms(S, K, T, r, sigma)
    return norm_pdf(d1) / (S_safe * sigma_safe * math.sqrt(T_safe))


def bs_vega(S, K, T, r, sigma):
    """Black-Scholes vega (per 1.0 volatility, not per 1%)."""
    if T <= 0:
        return np.zeros_like(np.asarray(S, dtype=float))
    S_safe, _, _, T_safe, d1, _ = _bs_terms(S, K, T, r, sigma)
    return S_safe * norm_pdf(d1) * math.sqrt(T_safe)


class OptionLeg:
    def __init__(self, option_type, side, strike, contract_size=100, symbol=None):
        self.option_type = option_type.lower()  # "call" / "put"
        self.side = side.lower()                # "long" / "short"
        self.strike = float(strike)
        self.contract_size = float(contract_size)
        self.symbol = symbol                    # Real contract symbol (optional)

    @property
    def multiplier(self) -> float:
        return 1.0 if self.side == "long" else -1.0


class OptionMarginSimulator:
    """
    Multi-leg option margin and path simulator with optional delta hedging and dynamic volatility.
    """

    def __init__(
        self,
        option_type,
        position_side,
        strike,
        contract_size,
        spot0,
        implied_vol,
        r,
        days_to_maturity,
        scan_risk_factor,
        min_margin_factor,
        maintenance_margin_rate,
        daily_return_mean,
        daily_return_vol,
        reference_equity,
        seed: int | None = None,
        enable_hedge: bool = False,
        hedge_frequency: int = 1,
        hedge_threshold: float | None = None,
        dynamic_vol: bool = False,
        vol_sensitivity: float = 5.0,
        legs: list[OptionLeg] | None = None,
    ):
        # Build legs (backward compatible single-leg mode)
        if legs is None:
            self.legs = [OptionLeg(option_type, position_side, strike, contract_size)]
        else:
            self.legs = legs

        primary = self.legs[0]
        self.option_type = primary.option_type
        self.position_side = primary.side
        self.strike = primary.strike
        self.contract_size = primary.contract_size

        self.spot0 = float(max(spot0, 1e-8))
        self.implied_vol = float(implied_vol)
        self.r = float(r)
        self.days_to_maturity = int(days_to_maturity)
        self.scan_risk_factor = float(scan_risk_factor)
        self.min_margin_factor = float(min_margin_factor)
        self.maintenance_margin_rate = float(maintenance_margin_rate)
        self.daily_return_mean = float(daily_return_mean)
        self.daily_return_vol = float(daily_return_vol)
        self.reference_equity = float(reference_equity)
        self.seed = None if seed is None else int(seed)

        self.enable_hedge = bool(enable_hedge)
        self.hedge_frequency = max(1, int(hedge_frequency))
        self.hedge_threshold = hedge_threshold
        self.dynamic_vol = bool(dynamic_vol)
        self.vol_sensitivity = float(vol_sensitivity)

        self.has_short_legs = any(leg.side == "short" for leg in self.legs)
        self.total_contract_exposure = max(1e-8, sum(abs(leg.contract_size) for leg in self.legs))

    def _rng(self):
        """
        Use a non-deterministic RNG unless a seed is explicitly provided.
        This avoids a fixed path when the caller does not request reproducibility.
        """
        if self.seed is None:
            return np.random.default_rng()
        return np.random.default_rng(self.seed)

    def _margin_requirements_for_leg(self, premium: float, spot: float, leg: OptionLeg) -> float:
        if leg.option_type == "call":
            otm = max(leg.strike - spot, 0.0)
        else:
            otm = max(spot - leg.strike, 0.0)
        scan_part = premium + self.scan_risk_factor * spot - otm
        min_part = premium + self.min_margin_factor * spot
        margin_per_unit = max(max(scan_part, min_part), 0.0)
        return margin_per_unit * leg.contract_size

    def _leg_price(self, leg: OptionLeg, spot: float, days_remaining: int, sigma: float | None = None) -> float:
        if sigma is None:
            sigma = self.implied_vol
        T_years = max(days_remaining / 365.0, 0.0)
        return float(bs_price(spot, leg.strike, T_years, self.r, sigma, leg.option_type))

    def _leg_delta(self, leg: OptionLeg, spot: float, days_remaining: int, sigma: float) -> float:
        T_years = max(days_remaining / 365.0, 0.0)
        return float(bs_delta(spot, leg.strike, T_years, self.r, sigma, leg.option_type))

    # ------------------------------------------------------------------
    # Single-path simulation
    # ------------------------------------------------------------------
    def run_single_path(self, n_days: int, spot_series=None):
        """
        Run a single-path simulation.
        If spot_series is provided (>=2 points), use it directly as the spot path.
        Otherwise simulate a path using the configured drift/vol and a non-fixed RNG.
        """
        steps = int(n_days)
        use_real_path = spot_series is not None and len(spot_series) >= 2

        if use_real_path:
            steps = min(steps, len(spot_series) - 1)
            spot_path = np.asarray(spot_series, dtype=float)[: steps + 1]
        else:
            rng = self._rng()
            spot_path = np.zeros(steps + 1)
            spot_path[0] = self.spot0
            for t in range(1, steps + 1):
                rtn = rng.normal(self.daily_return_mean, self.daily_return_vol)
                spot_path[t] = max(spot_path[t - 1] * (1.0 + rtn), 1e-8)

        option_price_path = np.zeros(steps + 1)
        margin_path = np.zeros(steps + 1)
        equity_path = np.zeros(steps + 1)
        equity_path[0] = self.reference_equity
        margin_ratio_path = np.full(steps + 1, np.inf)
        hedge_units = np.zeros(steps + 1)
        hedge_pnl = np.zeros(steps + 1)

        # Price & margin per day
        for t in range(steps + 1):
            days_left = self.days_to_maturity - t
            if self.dynamic_vol and t > 0:
                S_prev = spot_path[t - 1]
                S_t = spot_path[t]
                dd = max(0.0, (S_prev - S_t) / max(S_prev, 1e-9))
                sigma_t = self.implied_vol * (1.0 + self.vol_sensitivity * dd)
            else:
                sigma_t = self.implied_vol

            total_price = 0.0
            total_margin = 0.0
            for leg in self.legs:
                leg_price = self._leg_price(leg, spot_path[t], days_left, sigma_t)
                total_price += leg.multiplier * leg_price * leg.contract_size
                if leg.side == "short":
                    total_margin += self._margin_requirements_for_leg(leg_price, spot_path[t], leg)
            option_price_path[t] = total_price
            margin_path[t] = total_margin

        # Hedging, equity, margin ratio
        liquidation_day = None
        for t in range(1, steps + 1):
            if self.dynamic_vol:
                S_prev = spot_path[t - 1]
                S_t = spot_path[t]
                dd = max(0.0, (S_prev - S_t) / max(S_prev, 1e-9))
                sigma_t = self.implied_vol * (1.0 + self.vol_sensitivity * dd)
            else:
                sigma_t = self.implied_vol

            days_left = self.days_to_maturity - t
            total_delta = 0.0
            for leg in self.legs:
                leg_delta = self._leg_delta(leg, spot_path[t], days_left, sigma_t)
                total_delta += leg.multiplier * leg_delta * leg.contract_size

            if not self.enable_hedge:
                hedge_units[t] = hedge_units[t - 1]
            else:
                should_hedge = (t % self.hedge_frequency == 0)
                if self.hedge_threshold is not None:
                    delta_per_unit = total_delta / self.total_contract_exposure
                    should_hedge = should_hedge or (abs(delta_per_unit) > self.hedge_threshold)
                if should_hedge:
                    hedge_units[t] = -total_delta
                else:
                    hedge_units[t] = hedge_units[t - 1]

            hedge_pnl[t] = hedge_units[t - 1] * (spot_path[t] - spot_path[t - 1])
            pnl_option = option_price_path[t] - option_price_path[t - 1]
            equity_path[t] = equity_path[t - 1] + pnl_option + hedge_pnl[t]

            if self.has_short_legs and margin_path[t] > 0:
                margin_ratio_path[t] = equity_path[t] / max(margin_path[t], 1e-8)
                if liquidation_day is None and margin_ratio_path[t] < self.maintenance_margin_rate:
                    liquidation_day = t
                    equity_path[t:] = equity_path[t]
                    margin_path[t:] = margin_path[t]
                    margin_ratio_path[t:] = margin_ratio_path[t]
                    hedge_units[t:] = hedge_units[t]
                    hedge_pnl[t:] = hedge_pnl[t]
                    break
            else:
                margin_ratio_path[t] = np.inf

        return {
            "spot_path": spot_path,
            "option_price_path": option_price_path,
            "equity_path": equity_path,
            "margin_path": margin_path,
            "margin_ratio_path": margin_ratio_path,
            "liquidation_day": liquidation_day,
            "hedge_units_path": hedge_units,
            "hedge_pnl_path": hedge_pnl,
        }

    # ------------------------------------------------------------------
    # Monte Carlo simulation
    # ------------------------------------------------------------------
    def run_monte_carlo(self, num_paths: int, n_days: int):
        T = int(n_days)
        n = int(num_paths)
        rng = self._rng()

        spot_paths = np.zeros((n, T + 1))
        option_price_paths = np.zeros((n, T + 1))
        equity_paths = np.zeros((n, T + 1))
        margin_paths = np.zeros((n, T + 1))
        margin_ratio_paths = np.full((n, T + 1), np.inf)
        liquidation_days = np.full(n, T, dtype=int)
        hedge_units = np.zeros((n, T + 1))
        hedge_pnl = np.zeros((n, T + 1))

        spot_paths[:, 0] = self.spot0
        equity_paths[:, 0] = self.reference_equity

        for j in range(n):
            # Generate spot path
            for t in range(1, T + 1):
                rtn = rng.normal(self.daily_return_mean, self.daily_return_vol)
                spot_paths[j, t] = max(spot_paths[j, t - 1] * (1.0 + rtn), 1e-8)

            # Price & margin per day
            for t in range(T + 1):
                days_left = self.days_to_maturity - t
                if self.dynamic_vol and t > 0:
                    S_prev = spot_paths[j, t - 1]
                    S_t = spot_paths[j, t]
                    dd = max(0.0, (S_prev - S_t) / max(S_prev, 1e-9))
                    sigma_t = self.implied_vol * (1.0 + self.vol_sensitivity * dd)
                else:
                    sigma_t = self.implied_vol

                total_price = 0.0
                total_margin = 0.0
                for leg in self.legs:
                    leg_price = self._leg_price(leg, spot_paths[j, t], days_left, sigma_t)
                    total_price += leg.multiplier * leg_price * leg.contract_size
                    if leg.side == "short":
                        total_margin += self._margin_requirements_for_leg(leg_price, spot_paths[j, t], leg)
                option_price_paths[j, t] = total_price
                margin_paths[j, t] = total_margin

            liquidation_day = T
            # Hedging, equity, margin ratio
            for t in range(1, T + 1):
                if self.dynamic_vol:
                    S_prev = spot_paths[j, t - 1]
                    S_t = spot_paths[j, t]
                    dd = max(0.0, (S_prev - S_t) / max(S_prev, 1e-9))
                    sigma_t = self.implied_vol * (1.0 + self.vol_sensitivity * dd)
                else:
                    sigma_t = self.implied_vol

                days_left = self.days_to_maturity - t
                total_delta = 0.0
                for leg in self.legs:
                    leg_delta = self._leg_delta(leg, spot_paths[j, t], days_left, sigma_t)
                    total_delta += leg.multiplier * leg_delta * leg.contract_size

                if not self.enable_hedge:
                    hedge_units[j, t] = hedge_units[j, t - 1]
                else:
                    should_hedge = (t % self.hedge_frequency == 0)
                    if self.hedge_threshold is not None:
                        delta_per_unit = total_delta / self.total_contract_exposure
                        should_hedge = should_hedge or (abs(delta_per_unit) > self.hedge_threshold)
                    if should_hedge:
                        hedge_units[j, t] = -total_delta
                    else:
                        hedge_units[j, t] = hedge_units[j, t - 1]

                hedge_pnl[j, t] = hedge_units[j, t - 1] * (spot_paths[j, t] - spot_paths[j, t - 1])
                pnl_option = option_price_paths[j, t] - option_price_paths[j, t - 1]
                equity_paths[j, t] = equity_paths[j, t - 1] + pnl_option + hedge_pnl[j, t]

                margin_t = margin_paths[j, t]
                if self.has_short_legs and margin_t > 0:
                    margin_ratio_paths[j, t] = equity_paths[j, t] / max(margin_t, 1e-8)
                    if (
                        liquidation_day == T
                        and margin_ratio_paths[j, t] < self.maintenance_margin_rate
                    ):
                        liquidation_day = t
                        if t < T:
                            equity_paths[j, t + 1 :] = equity_paths[j, t]
                            margin_paths[j, t + 1 :] = margin_paths[j, t]
                            margin_ratio_paths[j, t + 1 :] = margin_ratio_paths[j, t]
                            hedge_units[j, t + 1 :] = hedge_units[j, t]
                            hedge_pnl[j, t + 1 :] = hedge_pnl[j, t]
                        break
                else:
                    margin_ratio_paths[j, t] = np.inf

            liquidation_days[j] = liquidation_day

        return {
            "spot_paths": spot_paths,
            "option_price_paths": option_price_paths,
            "equity_paths": equity_paths,
            "margin_paths": margin_paths,
            "margin_ratio_paths": margin_ratio_paths,
            "liquidation_days": liquidation_days,
            "hedge_units_paths": hedge_units,
            "hedge_pnl_paths": hedge_pnl,
        }

    # ------------------------------------------------------------------
    # Historical replay simulation
    # ------------------------------------------------------------------
    def run_historical_replay(
        self,
        spot_series,
        option_price_series=None,
        use_market_option_price: bool = False,
    ) -> dict:
        """
        Run a single-path margin & P&L simulation using a REAL historical
        spot price path (e.g., 50ETF close prices).

        Parameters
        ----------
        spot_series : 1D array-like
            Sequence of spot prices over time (e.g., daily close).
        option_price_series : 1D array-like, optional
            If provided and use_market_option_price=True, use this as the
            option price at each step instead of BS re-pricing.
        use_market_option_price : bool, default False
            If True and option_price_series is not None, use the historical
            option prices directly; otherwise, use BS to reprice options.

        Returns
        -------
        dict
            Dictionary with the same keys as run_single_path:
            - spot_path: array of spot prices
            - option_price_path: array of total option values
            - equity_path: array of equity values
            - margin_path: array of margin requirements
            - margin_ratio_path: array of margin ratios
            - liquidation_day: int or None
            - hedge_units_path: array of hedge units
            - hedge_pnl_path: array of hedge P&L
        """
        # Convert input to numpy arrays
        spot_path = np.asarray(spot_series, dtype=float)
        T = len(spot_path) - 1  # T is the number of steps (0 to T inclusive)
        
        # Ensure spot prices are positive
        spot_path = np.maximum(spot_path, 1e-8)
        
        # Handle option_price_series if provided
        if option_price_series is not None:
            option_price_series = np.asarray(option_price_series, dtype=float)
            # Truncate if longer, pad with NaN if shorter (will handle gracefully)
            if len(option_price_series) < len(spot_path):
                padded = np.full(len(spot_path), np.nan)
                padded[:len(option_price_series)] = option_price_series
                option_price_series = padded
            elif len(option_price_series) > len(spot_path):
                option_price_series = option_price_series[:len(spot_path)]
        else:
            option_price_series = None

        # Initialize arrays (same structure as run_single_path)
        option_price_path = np.zeros(T + 1)
        margin_path = np.zeros(T + 1)
        equity_path = np.zeros(T + 1)
        equity_path[0] = self.reference_equity
        margin_ratio_path = np.full(T + 1, np.inf)
        hedge_units = np.zeros(T + 1)
        hedge_pnl = np.zeros(T + 1)

        # Price & margin per day
        for t in range(T + 1):
            days_left = max(self.days_to_maturity - t, 0)
            
            # Determine volatility (for now, use constant implied_vol)
            # Dynamic vol logic can be added later if needed
            sigma_t = self.implied_vol

            # Determine if we should use market option price
            use_market_price = (
                use_market_option_price 
                and option_price_series is not None
                and not np.isnan(option_price_series[t])
                and option_price_series[t] >= 0
            )
            
            # For single-leg strategies, market price can be used directly
            # For multi-leg, market price represents total portfolio value,
            # but we need per-leg prices for margin calculation, so we fall back to BS
            can_use_market_price = use_market_price and len(self.legs) == 1

            # Calculate total option value and margin
            total_price = 0.0
            total_margin = 0.0
            
            for leg in self.legs:
                if can_use_market_price:
                    # Single leg: market price is total value, convert to per-unit
                    leg_price_val = option_price_series[t] / abs(leg.contract_size)
                else:
                    # Use BS pricing (default, or fallback for multi-leg)
                        leg_price_val = self._leg_price(leg, spot_path[t], days_left, sigma_t)
                
                total_price += leg.multiplier * leg_price_val * leg.contract_size
                if leg.side == "short":
                    total_margin += self._margin_requirements_for_leg(leg_price_val, spot_path[t], leg)
            
            option_price_path[t] = total_price
            margin_path[t] = total_margin

        # Hedging, equity, margin ratio
        liquidation_day = None
        for t in range(1, T + 1):
            # Use constant volatility for now (can add dynamic_vol later)
            sigma_t = self.implied_vol
            
            days_left = max(self.days_to_maturity - t, 0)
            
            # Calculate total delta
            total_delta = 0.0
            for leg in self.legs:
                leg_delta = self._leg_delta(leg, spot_path[t], days_left, sigma_t)
                total_delta += leg.multiplier * leg_delta * leg.contract_size

            # Update hedge units
            if not self.enable_hedge:
                hedge_units[t] = hedge_units[t - 1]
            else:
                should_hedge = (t % self.hedge_frequency == 0)
                if self.hedge_threshold is not None:
                    delta_per_unit = total_delta / self.total_contract_exposure
                    should_hedge = should_hedge or (abs(delta_per_unit) > self.hedge_threshold)
                if should_hedge:
                    hedge_units[t] = -total_delta
                else:
                    hedge_units[t] = hedge_units[t - 1]

            # Calculate hedge P&L
            hedge_pnl[t] = hedge_units[t - 1] * (spot_path[t] - spot_path[t - 1])
            
            # Calculate option P&L
            pnl_option = option_price_path[t] - option_price_path[t - 1]
            
            # Update equity
            equity_path[t] = equity_path[t - 1] + pnl_option + hedge_pnl[t]

            # Check liquidation
            if self.has_short_legs and margin_path[t] > 0:
                margin_ratio_path[t] = equity_path[t] / max(margin_path[t], 1e-8)
                if liquidation_day is None and margin_ratio_path[t] < self.maintenance_margin_rate:
                    liquidation_day = t
                    # Freeze values after liquidation
                    equity_path[t:] = equity_path[t]
                    margin_path[t:] = margin_path[t]
                    margin_ratio_path[t:] = margin_ratio_path[t]
                    hedge_units[t:] = hedge_units[t]
                    hedge_pnl[t:] = hedge_pnl[t]
                    break
            else:
                margin_ratio_path[t] = np.inf

        return {
            "spot_path": spot_path,
            "option_price_path": option_price_path,
            "equity_path": equity_path,
            "margin_path": margin_path,
            "margin_ratio_path": margin_ratio_path,
            "liquidation_day": liquidation_day,
            "hedge_units_path": hedge_units,
            "hedge_pnl_path": hedge_pnl,
        }

