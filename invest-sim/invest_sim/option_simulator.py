import numpy as np


def norm_cdf(x):
    return 0.5 * (1 + np.erf(x / np.sqrt(2)))


def norm_pdf(x):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)


def _bs_terms(S, K, T, r, sigma):
    S_arr = np.asarray(S, dtype=float)
    S_safe = np.maximum(S_arr, 1e-9)
    K_safe = max(K, 1e-9)
    sigma_safe = max(sigma, 1e-9)
    T_safe = max(T, 1e-9)
    sqrt_T = np.sqrt(T_safe)
    d1 = (np.log(S_safe / K_safe) + (r + 0.5 * sigma_safe**2) * T_safe) / (sigma_safe * sqrt_T)
    d2 = d1 - sigma_safe * sqrt_T
    return S_safe, K_safe, sigma_safe, T_safe, d1, d2


def bs_price(S, K, T, r, sigma, option_type):
    option_type = option_type.lower()
    if T <= 0:
        if option_type == "call":
            return np.maximum(np.asarray(S, dtype=float) - K, 0)
        return np.maximum(K - np.asarray(S, dtype=float), 0)
    S_safe, K_safe, _, T_safe, d1, d2 = _bs_terms(S, K, T, r, sigma)
    discount = np.exp(-r * T_safe)
    if option_type == "call":
        return S_safe * norm_cdf(d1) - K_safe * discount * norm_cdf(d2)
    return K_safe * discount * norm_cdf(-d2) - S_safe * norm_cdf(-d1)


def bs_delta(S, K, T, r, sigma, option_type):
    option_type = option_type.lower()
    if T <= 0:
        if option_type == "call":
            return np.where(np.asarray(S, dtype=float) > K, 1.0, 0.0)
        return np.where(np.asarray(S, dtype=float) > K, 0.0, -1.0)
    _, _, _, _, d1, _ = _bs_terms(S, K, T, r, sigma)
    if option_type == "call":
        return norm_cdf(d1)
    return norm_cdf(d1) - 1


def bs_gamma(S, K, T, r, sigma):
    if T <= 0:
        return np.zeros_like(np.asarray(S, dtype=float))
    S_safe, _, sigma_safe, T_safe, d1, _ = _bs_terms(S, K, T, r, sigma)
    return norm_pdf(d1) / (S_safe * sigma_safe * np.sqrt(T_safe))


def bs_vega(S, K, T, r, sigma):
    if T <= 0:
        return np.zeros_like(np.asarray(S, dtype=float))
    S_safe, _, _, T_safe, d1, _ = _bs_terms(S, K, T, r, sigma)
    return S_safe * norm_pdf(d1) * np.sqrt(T_safe)


class OptionLeg:
    def __init__(self, option_type, side, strike, contract_size=100):
        self.option_type = option_type.lower()
        self.side = side.lower()
        self.strike = strike
        self.contract_size = contract_size

    @property
    def multiplier(self):
        return 1 if self.side == "long" else -1


class OptionMarginSimulator:
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
        seed=12345,
        enable_hedge: bool = False,
        hedge_frequency: int = 1,
        hedge_threshold: float | None = None,
        dynamic_vol: bool = False,
        vol_sensitivity: float = 5.0,
        legs: list[OptionLeg] | None = None,
    ):
        if legs is None:
            self.legs = [OptionLeg(option_type, position_side, strike, contract_size)]
        else:
            self.legs = legs
        primary_leg = self.legs[0]
        self.option_type = primary_leg.option_type  # backward compatibility only
        self.position_side = primary_leg.side
        self.strike = primary_leg.strike
        self.contract_size = primary_leg.contract_size
        self.spot0 = max(spot0, 1e-6)
        self.implied_vol = implied_vol
        self.r = r
        self.days_to_maturity = days_to_maturity
        self.scan_risk_factor = scan_risk_factor
        self.min_margin_factor = min_margin_factor
        self.maintenance_margin_rate = maintenance_margin_rate
        self.daily_return_mean = daily_return_mean
        self.daily_return_vol = daily_return_vol
        self.reference_equity = reference_equity
        self.seed = seed
        self.enable_hedge = enable_hedge
        self.hedge_frequency = max(1, int(hedge_frequency))
        self.hedge_threshold = hedge_threshold
        self.dynamic_vol = dynamic_vol
        self.vol_sensitivity = vol_sensitivity
        self.has_short_legs = any(leg.side == "short" for leg in self.legs)
        self.total_contract_exposure = max(
            1e-8, sum(leg.contract_size for leg in self.legs)
        )

    def _rng(self):
        return np.random.default_rng(self.seed)

    def _margin_requirements_for_leg(self, premium, spot, leg: OptionLeg):
        if leg.option_type == "call":
            otm = max(leg.strike - spot, 0.0)
        else:
            otm = max(spot - leg.strike, 0.0)
        scan_part = premium + self.scan_risk_factor * spot - otm
        min_part = premium + self.min_margin_factor * spot
        margin_per_unit = max(max(scan_part, min_part), 0.0)
        return margin_per_unit * leg.contract_size

    def _leg_price(self, leg: OptionLeg, spot, days_remaining, sigma=None):
        if sigma is None:
            sigma = self.implied_vol
        T_remaining = max(days_remaining / 365.0, 0.0)
        return float(
            np.squeeze(
                bs_price(
                    spot,
                    leg.strike,
                    max(T_remaining, 1e-9),
                    self.r,
                    max(sigma, 1e-6),
                    leg.option_type,
                )
            )
        )

    def _leg_delta(self, leg: OptionLeg, spot, days_remaining, sigma):
        T_remaining = max(days_remaining / 365.0, 0.0)
        return float(
            np.squeeze(
                bs_delta(
                    spot,
                    leg.strike,
                    T_remaining,
                    self.r,
                    max(sigma, 1e-6),
                    leg.option_type,
                )
            )
        )

    def run_single_path(self, n_days):
        steps = int(n_days)
        rng = self._rng()

        spot_path = np.zeros(steps + 1)
        spot_path[0] = self.spot0
        option_price_path = np.zeros(steps + 1)
        margin_path = np.zeros(steps + 1)
        equity_path = np.zeros(steps + 1)
        equity_path[0] = self.reference_equity
        margin_ratio_path = np.full(steps + 1, np.inf)
        hedge_units = np.zeros(steps + 1, dtype=float)
        hedge_pnl = np.zeros(steps + 1, dtype=float)

        for t in range(1, steps + 1):
            rtn = rng.normal(self.daily_return_mean, self.daily_return_vol)
            spot_path[t] = max(spot_path[t - 1] * (1 + rtn), 1e-6)

        for t in range(steps + 1):
            days_left = self.days_to_maturity - t
            if self.dynamic_vol and t > 0:
                S_prev = spot_path[t - 1]
                S_t = spot_path[t]
                dd = max(0.0, (S_prev - S_t) / max(S_prev, 1e-9))
                sigma_t = self.implied_vol * (1 + self.vol_sensitivity * dd)
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

        liquidation_day = None
        for t in range(1, steps + 1):
            if self.dynamic_vol:
                S_prev = spot_path[t - 1]
                S_t = spot_path[t]
                dd = max(0.0, (S_prev - S_t) / max(S_prev, 1e-9))
                sigma_t = self.implied_vol * (1 + self.vol_sensitivity * dd)
            else:
                sigma_t = self.implied_vol

            T_remaining = self.days_to_maturity - t
            total_delta = 0.0
            for leg in self.legs:
                leg_delta = self._leg_delta(leg, spot_path[t], T_remaining, sigma_t)
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

    def run_monte_carlo(self, num_paths, n_days):
        T = int(n_days)
        n = int(num_paths)
        rng = self._rng()

        spot_paths = np.zeros((n, T + 1))
        option_price_paths = np.zeros((n, T + 1))
        equity_paths = np.zeros((n, T + 1))
        margin_paths = np.zeros((n, T + 1))
        margin_ratio_paths = np.full((n, T + 1), np.inf)
        liquidation_days = np.full(n, T, dtype=int)
        hedge_units = np.zeros((n, T + 1), dtype=float)
        hedge_pnl = np.zeros((n, T + 1), dtype=float)

        spot_paths[:, 0] = self.spot0
        equity_paths[:, 0] = self.reference_equity

        for j in range(n):
            for t in range(1, T + 1):
                rtn = rng.normal(self.daily_return_mean, self.daily_return_vol)
                spot_paths[j, t] = max(spot_paths[j, t - 1] * (1 + rtn), 1e-6)

            for t in range(T + 1):
                days_left = self.days_to_maturity - t
                if self.dynamic_vol and t > 0:
                    S_prev = spot_paths[j, t - 1]
                    S_t = spot_paths[j, t]
                    dd = max(0.0, (S_prev - S_t) / max(S_prev, 1e-9))
                    sigma_t = self.implied_vol * (1 + self.vol_sensitivity * dd)
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
            for t in range(1, T + 1):
                if self.dynamic_vol:
                    S_prev = spot_paths[j, t - 1]
                    S_t = spot_paths[j, t]
                    dd = max(0.0, (S_prev - S_t) / max(S_prev, 1e-9))
                    sigma_t = self.implied_vol * (1 + self.vol_sensitivity * dd)
                else:
                    sigma_t = self.implied_vol

                T_remaining = self.days_to_maturity - t
                total_delta = 0.0
                for leg in self.legs:
                    leg_delta = self._leg_delta(leg, spot_paths[j, t], T_remaining, sigma_t)
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
import numpy as np


def norm_cdf(x):
    return 0.5 * (1 + np.erf(x / np.sqrt(2)))


def norm_pdf(x):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)


def _bs_terms(S, K, T, r, sigma):
    S_arr = np.asarray(S, dtype=float)
    S_safe = np.maximum(S_arr, 1e-9)
    K_safe = max(K, 1e-9)
    sigma_safe = max(sigma, 1e-9)
    T_safe = max(T, 1e-9)
    sqrt_T = np.sqrt(T_safe)
    d1 = (np.log(S_safe / K_safe) + (r + 0.5 * sigma_safe**2) * T_safe) / (sigma_safe * sqrt_T)
    d2 = d1 - sigma_safe * sqrt_T
    return S_safe, K_safe, sigma_safe, T_safe, d1, d2


def bs_price(S, K, T, r, sigma, option_type):
    option_type = option_type.lower()
    if T <= 0:
        if option_type == "call":
            return np.maximum(np.asarray(S, dtype=float) - K, 0)
        return np.maximum(K - np.asarray(S, dtype=float), 0)
    S_safe, K_safe, _, T_safe, d1, d2 = _bs_terms(S, K, T, r, sigma)
    discount = np.exp(-r * T_safe)
    if option_type == "call":
        return S_safe * norm_cdf(d1) - K_safe * discount * norm_cdf(d2)
    return K_safe * discount * norm_cdf(-d2) - S_safe * norm_cdf(-d1)


def bs_delta(S, K, T, r, sigma, option_type):
    option_type = option_type.lower()
    if T <= 0:
        if option_type == "call":
            return np.where(np.asarray(S, dtype=float) > K, 1.0, 0.0)
        return np.where(np.asarray(S, dtype=float) > K, 0.0, -1.0)
    _, _, _, _, d1, _ = _bs_terms(S, K, T, r, sigma)
    if option_type == "call":
        return norm_cdf(d1)
    return norm_cdf(d1) - 1


def bs_gamma(S, K, T, r, sigma):
    if T <= 0:
        return np.zeros_like(np.asarray(S, dtype=float))
    S_safe, _, sigma_safe, T_safe, d1, _ = _bs_terms(S, K, T, r, sigma)
    return norm_pdf(d1) / (S_safe * sigma_safe * np.sqrt(T_safe))


def bs_vega(S, K, T, r, sigma):
    if T <= 0:
        return np.zeros_like(np.asarray(S, dtype=float))
    S_safe, _, _, T_safe, d1, _ = _bs_terms(S, K, T, r, sigma)
    return S_safe * norm_pdf(d1) * np.sqrt(T_safe)


class OptionLeg:
    def __init__(self, option_type, side, strike, contract_size=100):
        self.option_type = option_type.lower()
        self.side = side.lower()
        self.strike = strike
        self.contract_size = contract_size

    @property
    def multiplier(self):
        return 1 if self.side == "long" else -1


class OptionMarginSimulator:
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
        seed=12345,
        enable_hedge: bool = False,
        hedge_frequency: int = 1,
        hedge_threshold: float | None = None,
        dynamic_vol: bool = False,
        vol_sensitivity: float = 5.0,
    ):
        self.option_type = option_type.lower()
        self.position_side = position_side
        self.strike = strike
        self.contract_size = contract_size
        self.spot0 = max(spot0, 1e-6)
        self.implied_vol = implied_vol
        self.r = r
        self.days_to_maturity = days_to_maturity
        self.scan_risk_factor = scan_risk_factor
        self.min_margin_factor = min_margin_factor
        self.maintenance_margin_rate = maintenance_margin_rate
        self.daily_return_mean = daily_return_mean
        self.daily_return_vol = daily_return_vol
        self.reference_equity = reference_equity
        self.seed = seed
        self.enable_hedge = enable_hedge
        self.hedge_frequency = max(1, int(hedge_frequency))
        self.hedge_threshold = hedge_threshold
        self.dynamic_vol = dynamic_vol
        self.vol_sensitivity = vol_sensitivity

    def _rng(self):
        return np.random.default_rng(self.seed)

    def _margin_requirements(self, premium, spot):
        if self.option_type == "call":
            otm = max(self.strike - spot, 0.0)
        else:
            otm = max(spot - self.strike, 0.0)
        scan_part = premium + self.scan_risk_factor * spot - otm
        min_part = premium + self.min_margin_factor * spot
        margin_per_unit = max(max(scan_part, min_part), 0.0)
        return margin_per_unit * self.contract_size

    def _option_price(self, spot, days_remaining, sigma=None):
        if sigma is None:
            sigma = self.implied_vol
        T_remaining = max(days_remaining / 365.0, 0.0)
        return float(
            np.squeeze(
                bs_price(
                    spot,
                    self.strike,
                    max(T_remaining, 1e-9),
                    self.r,
                    max(sigma, 1e-6),
                    self.option_type,
                )
            )
        )

    def run_single_path(self, n_days):
        steps = int(n_days)
        rng = self._rng()

        spot_path = np.zeros(steps + 1)
        spot_path[0] = self.spot0
        option_price_path = np.zeros(steps + 1)
        margin_path = np.zeros(steps + 1)
        equity_path = np.zeros(steps + 1)
        equity_path[0] = self.reference_equity
        margin_ratio_path = np.full(steps + 1, np.nan)
        hedge_units = np.zeros(steps + 1, dtype=float)
        hedge_pnl = np.zeros(steps + 1, dtype=float)
        multiplier = 1 if self.position_side == "Long" else -1

        for t in range(1, steps + 1):
            rtn = rng.normal(self.daily_return_mean, self.daily_return_vol)
            spot_path[t] = max(spot_path[t - 1] * (1 + rtn), 1e-6)

        # Compute option prices with dynamic volatility
        for t in range(steps + 1):
            days_left = self.days_to_maturity - t
            # Compute dynamic volatility
            if self.dynamic_vol and t > 0:
                S_prev = spot_path[t - 1]
                S_t = spot_path[t]
                dd = max(0.0, (S_prev - S_t) / max(S_prev, 1e-9))
                sigma_t = self.implied_vol * (1 + self.vol_sensitivity * dd)
            else:
                sigma_t = self.implied_vol
            
            option_price_path[t] = self._option_price(spot_path[t], days_left, sigma_t)
            if self.position_side == "Short":
                margin_path[t] = self._margin_requirements(option_price_path[t], spot_path[t])
            else:
                margin_path[t] = 0.0

        liquidation_day = None
        for t in range(1, steps + 1):
            # Compute dynamic volatility for delta calculation
            if self.dynamic_vol:
                S_prev = spot_path[t - 1]
                S_t = spot_path[t]
                dd = max(0.0, (S_prev - S_t) / max(S_prev, 1e-9))
                sigma_t = self.implied_vol * (1 + self.vol_sensitivity * dd)
            else:
                sigma_t = self.implied_vol
            
            # Compute delta for hedging decision
            T_remaining = max(self.days_to_maturity - t, 0) / 365.0
            delta_t = float(
                np.squeeze(
                    bs_delta(
                        spot_path[t],
                        self.strike,
                        T_remaining,
                        self.r,
                        sigma_t,
                        self.option_type,
                    )
                )
            )

            # Hedge decision logic
            if not self.enable_hedge:
                hedge_units[t] = hedge_units[t - 1]
            else:
                should_hedge = (t % self.hedge_frequency == 0)
                if self.hedge_threshold is not None:
                    should_hedge = should_hedge or (abs(delta_t) > self.hedge_threshold)

                if should_hedge:
                    hedge_units[t] = -delta_t * self.contract_size
                else:
                    hedge_units[t] = hedge_units[t - 1]

            # Hedge P&L
            hedge_pnl[t] = hedge_units[t - 1] * (spot_path[t] - spot_path[t - 1])

            # Option P&L
            pnl_option = (option_price_path[t] - option_price_path[t - 1]) * self.contract_size * multiplier

            # Equity update (option P&L + hedge P&L)
            equity_path[t] = equity_path[t - 1] + pnl_option + hedge_pnl[t]

            if self.position_side == "Short":
                margin_denominator = max(margin_path[t], 1e-8)
                margin_ratio_path[t] = equity_path[t] / margin_denominator if margin_path[t] > 0 else np.inf
                if liquidation_day is None and margin_path[t] > 0 and margin_ratio_path[t] < self.maintenance_margin_rate:
                    liquidation_day = t
                    equity_path[t:] = equity_path[t]
                    margin_path[t:] = margin_path[t]
                    margin_ratio_path[t:] = margin_ratio_path[t]
                    hedge_units[t:] = hedge_units[t]
                    hedge_pnl[t:] = hedge_pnl[t]
                    break
            else:
                margin_ratio_path[t] = np.nan

        if self.position_side == "Short" and np.isnan(margin_ratio_path[0]) and margin_path[0] > 0:
            margin_ratio_path[0] = equity_path[0] / max(margin_path[0], 1e-8)

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

    def run_monte_carlo(self, num_paths, n_days):
        T = int(n_days)
        n = int(num_paths)
        rng = self._rng()

        spot_paths = np.zeros((n, T + 1))
        option_price_paths = np.zeros((n, T + 1))
        equity_paths = np.zeros((n, T + 1))
        margin_paths = np.zeros((n, T + 1))
        margin_ratio_paths = np.full((n, T + 1), np.inf)
        liquidation_days = np.full(n, T, dtype=int)
        hedge_units = np.zeros((n, T + 1), dtype=float)
        hedge_pnl = np.zeros((n, T + 1), dtype=float)

        spot_paths[:, 0] = self.spot0
        equity_paths[:, 0] = self.reference_equity
        multiplier = 1 if self.position_side == "Long" else -1

        for j in range(n):
            for t in range(1, T + 1):
                rtn = rng.normal(self.daily_return_mean, self.daily_return_vol)
                spot_paths[j, t] = max(spot_paths[j, t - 1] * (1 + rtn), 1e-6)

            # Compute option prices with dynamic volatility
            for t in range(T + 1):
                days_left = self.days_to_maturity - t
                # Compute dynamic volatility
                if self.dynamic_vol and t > 0:
                    S_prev = spot_paths[j, t - 1]
                    S_t = spot_paths[j, t]
                    dd = max(0.0, (S_prev - S_t) / max(S_prev, 1e-9))
                    sigma_t = self.implied_vol * (1 + self.vol_sensitivity * dd)
                else:
                    sigma_t = self.implied_vol
                
                option_price_paths[j, t] = self._option_price(spot_paths[j, t], days_left, sigma_t)

            liquidation_day = T
            for t in range(1, T + 1):
                # Compute dynamic volatility for delta calculation
                if self.dynamic_vol:
                    S_prev = spot_paths[j, t - 1]
                    S_t = spot_paths[j, t]
                    dd = max(0.0, (S_prev - S_t) / max(S_prev, 1e-9))
                    sigma_t = self.implied_vol * (1 + self.vol_sensitivity * dd)
                else:
                    sigma_t = self.implied_vol
                
                # Compute delta for hedging decision
                T_remaining = max(self.days_to_maturity - t, 0) / 365.0
                delta_t = float(
                    np.squeeze(
                        bs_delta(
                            spot_paths[j, t],
                            self.strike,
                            T_remaining,
                            self.r,
                            sigma_t,
                            self.option_type,
                        )
                    )
                )

                # Hedge decision logic
                if not self.enable_hedge:
                    hedge_units[j, t] = hedge_units[j, t - 1]
                else:
                    should_hedge = (t % self.hedge_frequency == 0)
                    if self.hedge_threshold is not None:
                        should_hedge = should_hedge or (abs(delta_t) > self.hedge_threshold)

                    if should_hedge:
                        hedge_units[j, t] = -delta_t * self.contract_size
                    else:
                        hedge_units[j, t] = hedge_units[j, t - 1]

                # Hedge P&L
                hedge_pnl[j, t] = hedge_units[j, t - 1] * (spot_paths[j, t] - spot_paths[j, t - 1])

                # Option P&L
                pnl_option = (option_price_paths[j, t] - option_price_paths[j, t - 1]) * self.contract_size * multiplier

                # Equity update (option P&L + hedge P&L)
                equity_paths[j, t] = equity_paths[j, t - 1] + pnl_option + hedge_pnl[j, t]

                if self.position_side == "Short":
                    premium = option_price_paths[j, t]
                    spot_t = spot_paths[j, t]
                    margin_t = self._margin_requirements(premium, spot_t)
                    margin_paths[j, t] = margin_t

                    if margin_t > 0:
                        margin_ratio_paths[j, t] = equity_paths[j, t] / max(margin_t, 1e-8)
                    else:
                        margin_ratio_paths[j, t] = np.inf

                    if (
                        liquidation_day == T
                        and margin_t > 0
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
                    margin_paths[j, t] = 0.0
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

