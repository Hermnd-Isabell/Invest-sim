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

    def _option_price(self, spot, days_remaining):
        T_remaining = max(days_remaining / 365.0, 0.0)
        return float(
            np.squeeze(
                bs_price(
                    spot,
                    self.strike,
                    max(T_remaining, 1e-9),
                    self.r,
                    max(self.implied_vol, 1e-6),
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
        multiplier = 1 if self.position_side == "Long" else -1

        for t in range(1, steps + 1):
            rtn = rng.normal(self.daily_return_mean, self.daily_return_vol)
            spot_path[t] = max(spot_path[t - 1] * (1 + rtn), 1e-6)

        for t in range(steps + 1):
            days_left = self.days_to_maturity - t
            option_price_path[t] = self._option_price(spot_path[t], days_left)
            if self.position_side == "Short":
                margin_path[t] = self._margin_requirements(option_price_path[t], spot_path[t])
            else:
                margin_path[t] = 0.0

        liquidation_day = None
        for t in range(1, steps + 1):
            pnl_option = (option_price_path[t] - option_price_path[t - 1]) * self.contract_size * multiplier
            equity_path[t] = equity_path[t - 1] + pnl_option

            if self.position_side == "Short":
                margin_denominator = max(margin_path[t], 1e-8)
                margin_ratio_path[t] = equity_path[t] / margin_denominator if margin_path[t] > 0 else np.inf
                if liquidation_day is None and margin_path[t] > 0 and margin_ratio_path[t] < self.maintenance_margin_rate:
                    liquidation_day = t
                    equity_path[t:] = equity_path[t]
                    margin_path[t:] = margin_path[t]
                    margin_ratio_path[t:] = margin_ratio_path[t]
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

        spot_paths[:, 0] = self.spot0
        equity_paths[:, 0] = self.reference_equity
        multiplier = 1 if self.position_side == "Long" else -1

        for j in range(n):
            for t in range(1, T + 1):
                rtn = rng.normal(self.daily_return_mean, self.daily_return_vol)
                spot_paths[j, t] = max(spot_paths[j, t - 1] * (1 + rtn), 1e-6)

            for t in range(T + 1):
                days_left = self.days_to_maturity - t
                option_price_paths[j, t] = self._option_price(spot_paths[j, t], days_left)

            liquidation_day = T
            for t in range(1, T + 1):
                pnl_option = (option_price_paths[j, t] - option_price_paths[j, t - 1]) * self.contract_size * multiplier
                equity_paths[j, t] = equity_paths[j, t - 1] + pnl_option

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
        }

