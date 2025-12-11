"""
期权数据存储和管理模块

提供 OptionDataStore 类用于加载和管理 50ETF 期权数据。
"""

import pandas as pd
import numpy as np
from pathlib import Path
from invest_sim.iv_utils import implied_vol_newton


class OptionDataStore:
    """
    期权数据存储类
    
    用于加载和管理期权合约信息和价格历史数据。
    支持按到期日、行权价、期权类型查询合约，并提供价格历史和波动率估算。
    """
    
    def __init__(
        self,
        instruments_path: str = "data/50ETF/Filtered_OptionInstruments_FIXED.pkl",
        prices_path: str = "data/50ETF/Filtered_OptionPrice_2020_2022.feather",
    ):
        """
        初始化 OptionDataStore
        
        Parameters
        ----------
        instruments_path : str
            期权合约信息文件路径（pickle 格式）
        prices_path : str
            期权价格历史数据文件路径（feather 格式）
        """
        self.instruments_path = Path(instruments_path)
        self.prices_path = Path(prices_path)
        self._instruments = None
        self._prices = None
        self._replayable_ids = None
        self._replayable_instruments = None

    def load(self):
        """
        加载合约信息和价格数据到内存（如果尚未加载）。
        
        使用懒加载模式，只有在首次访问时才会加载数据。
        """
        if hasattr(self, "_already_loaded") and self._already_loaded is True:
            return self
        self._already_loaded = True
        # print("Loaded instruments_path =", self.instruments_path)
        # print("Loaded prices_path =", self.prices_path)
        if self._instruments is None:
            self._instruments = pd.read_pickle(self.instruments_path)
        # Fix maturity_date filtering by converting it to datetime
        # Ensure this conversion happens every time load() is called
        if self._instruments is not None and "maturity_date" in self._instruments.columns:
            self._instruments["maturity_date"] = pd.to_datetime(self._instruments["maturity_date"])
        if self._prices is None:
            self._prices = pd.read_feather(self.prices_path)
        # 防止旧实例缺失属性，并在每次 load 时刷新 replayable 集合
        if not hasattr(self, "_replayable_ids"):
            self._replayable_ids = None
        if not hasattr(self, "_replayable_instruments"):
            self._replayable_instruments = None

        # 构建“真实有价格行”的可回放集合
        price_counts = self._prices["order_book_id"].astype(str).value_counts()
        non_empty_price_ids = set(price_counts[price_counts > 0].index)
        inst_ids = set(self._instruments["order_book_id"].astype(str))
        valid_ids = inst_ids & non_empty_price_ids
        self._replayable_ids = sorted(valid_ids)

        # Additional strict filter: require at least 1 non-NaN close value
        final_ids = []
        for obid in self._replayable_ids:
            df_price = self._prices[self._prices["order_book_id"].astype(str) == str(obid)]
            # require at least 1 non-NaN close value
            if "close" in df_price.columns and df_price["close"].dropna().shape[0] >= 1:
                final_ids.append(obid)

        self._replayable_ids = final_ids

        # Rebuild replayable instruments
        self._replayable_instruments = self._instruments[
            self._instruments["order_book_id"].astype(str).isin(self._replayable_ids)
        ]

        # --- NEW: build fast price lookup cache ---
        if not hasattr(self, "_price_cache"):
            # groupby once → O(N)，only happens on load()
            self._price_cache = {
                str(obid): df.sort_values("date")
                for obid, df in self._prices.groupby(self._prices["order_book_id"].astype(str))
            }

    # Replayable helpers
    def get_replayable_ids(self):
        self.load()
        return self._replayable_ids

    def is_replayable(self, order_book_id) -> bool:
        self.load()
        return str(order_book_id) in self._replayable_ids

    @property
    def instruments(self) -> pd.DataFrame:
        """
        获取期权合约信息 DataFrame
        
        Returns
        -------
        pd.DataFrame
            包含合约信息的 DataFrame，预期列包括：
            symbol, underlying, strike, expiry, option_type 等
        """
        self.load()
        return self._instruments

    @property
    def prices(self) -> pd.DataFrame:
        """
        获取期权价格历史数据 DataFrame
        
        Returns
        -------
        pd.DataFrame
            包含价格历史的 DataFrame，预期列包括：
            symbol, date, open, high, low, close, volume 等
        """
        self.load()
        return self._prices

    def list_expiries(self, replayable_only: bool = False):
        """
        返回所有唯一的到期日列表（已排序）
        只返回有价格数据的合约的到期日
        
        Returns
        -------
        list
            排序后的到期日列表
        """
        # 确保最新数据及可回放集合
        self.load()
        if not hasattr(self, "_replayable_instruments") or self._replayable_instruments is None:
            inst_ids = set(self.instruments["order_book_id"].astype(str))
            price_ids = set(self.prices["order_book_id"].astype(str))
            self._replayable_ids = sorted(inst_ids & price_ids)
            self._replayable_instruments = self.instruments[
                self.instruments["order_book_id"].astype(str).isin(self._replayable_ids)
            ]

        inst = self._replayable_instruments if replayable_only else self.instruments
        return sorted(inst["maturity_date"].unique())

    def list_strikes(self, expiry, option_type=None, replayable_only: bool = False):
        """
        返回指定到期日的行权价列表
        只返回有价格数据的合约的行权价
        
        Parameters
        ----------
        expiry
            到期日
        option_type : str, optional
            期权类型（"call" 或 "put"），如果为 None 则返回所有类型
            
        Returns
        -------
        list
            排序后的行权价列表
        """
        # 确保最新数据及可回放集合
        self.load()
        if not hasattr(self, "_replayable_instruments") or self._replayable_instruments is None:
            inst_ids = set(self.instruments["order_book_id"].astype(str))
            price_ids = set(self.prices["order_book_id"].astype(str))
            self._replayable_ids = sorted(inst_ids & price_ids)
            self._replayable_instruments = self.instruments[
                self.instruments["order_book_id"].astype(str).isin(self._replayable_ids)
            ]

        inst = self._replayable_instruments if replayable_only else self.instruments
        df = inst[(inst["maturity_date"] == expiry)]
        if option_type is not None:
            # option_type 在数据中是 'C' 或 'P'，需要转换
            opt_map = {"call": "C", "put": "P"}
            opt_code = opt_map.get(option_type.lower(), option_type.upper())
            df = df[df["option_type"] == opt_code]
        return sorted(df["strike_price"].unique())

    def find_symbol(self, expiry, option_type, strike, replayable_only: bool = False):
        """
        根据到期日、期权类型和行权价查找合约代码
        
        Parameters
        ----------
        expiry : any
            到期日
        option_type : str
            "call" / "put"
        strike : float
            行权价
        replayable_only : bool
            若为 True，仅在 replayable_instruments 中查找
            
        Returns
        -------
        tuple (symbol, order_book_id)
            
        Raises
        ------
        KeyError 如果找不到匹配合约
        """
        # 确保加载并具备必要属性（兼容旧实例）
        self.load()
        if not hasattr(self, "_replayable_instruments") or self._replayable_instruments is None:
            replayable_ids = set(self.get_replayable_ids()) if hasattr(self, "get_replayable_ids") else set()
            self._replayable_instruments = self._instruments[
                self._instruments["order_book_id"].astype(str).isin(replayable_ids)
            ]

        df_base = self._replayable_instruments if replayable_only else self.instruments

        # option_type 兼容大小写 / C/P
        opt_lower = option_type.lower()
        opt_map = {"call": "c", "put": "p"}
        target_opt = opt_map.get(opt_lower, opt_lower)

        mask = (
            (df_base["maturity_date"] == expiry)
            & (df_base["option_type"].str.lower().isin([target_opt, opt_lower]))
            & (df_base["strike_price"].astype(float) == float(strike))
        )
        subset = df_base.loc[mask]
        if subset.empty:
            raise KeyError(
                f"No instrument found for expiry={expiry}, type={option_type}, strike={strike}, replayable_only={replayable_only}"
            )
        row = subset.iloc[0]
        return row["symbol"], row["order_book_id"]

    def get_contract_row(self, symbol):
        """
        根据合约代码获取合约信息行
        
        Parameters
        ----------
        symbol : str
            合约代码
            
        Returns
        -------
        pd.Series
            包含合约信息的 Series
            
        Raises
        ------
        KeyError
            如果找不到对应的合约
        """
        inst = self.instruments
        row = inst[inst["symbol"] == symbol]
        if row.empty:
            raise KeyError(f"No instrument for symbol={symbol}")
        return row.iloc[0]

    def get_price_history(self, symbol):
        """
        获取指定合约的价格历史数据
        
        如果同一个 symbol 对应多个 order_book_id，优先返回有价格数据的那个。
        
        Parameters
        ----------
        symbol : str
            合约代码（symbol）
            
        Returns
        -------
        pd.DataFrame
            价格历史 DataFrame，按日期排序
            预期列至少包括：
            date, open, high, low, close, volume, 可能还有 bid1-5, ask1-5
        """
        prices = self.prices
        inst_df = self.instruments

        def _debug(msg):
            # print(f"[debug|get_price_history] {msg}")
            pass

        symbol_str = str(symbol)
        # _debug(f"Searching prices for symbol={symbol_str}")

        # --- NEW: direct cache lookup ---
        if hasattr(self, "_price_cache"):
            if symbol_str in self._price_cache:
                df_cached = self._price_cache[symbol_str]
                if df_cached is not None and not df_cached.empty:
                    return df_cached

        # Step 1: direct match on prices.order_book_id
        df_direct = prices[prices["order_book_id"].astype(str) == symbol_str]
        if not df_direct.empty:
            # _debug(f"✅ matched prices by order_book_id=={symbol_str} rows={len(df_direct)}")
            if "close" not in df_direct.columns:
                # _debug("❌ matched rows missing 'close' column")
                return df_direct
            return df_direct.sort_values("date")

        # Step 2: match on prices.symbol if exists
        if "symbol" in prices.columns:
            df_sym = prices[prices["symbol"].astype(str) == symbol_str]
            if not df_sym.empty:
                # _debug(f"✅ matched prices by prices.symbol=={symbol_str} rows={len(df_sym)}")
                if "close" not in df_sym.columns:
                    # _debug("❌ matched rows missing 'close' column")
                    return df_sym
                return df_sym.sort_values("date")

        # Step 3: look up instrument row by symbol OR by order_book_id
        contract_rows = inst_df[(inst_df["symbol"] == symbol_str) | (inst_df["order_book_id"].astype(str) == symbol_str)]
        if contract_rows.empty:
            # _debug(f"❌ symbol not found in instruments | symbol={symbol_str}")
            raise KeyError(f"No instrument found for symbol={symbol_str}")

        order_book_id = contract_rows.iloc[0].get("order_book_id")
        trading_code = contract_rows.iloc[0].get("trading_code") if "trading_code" in contract_rows.columns else None

        # _debug(f"instrument.order_book_id={order_book_id}, trading_code={trading_code}")

        # Step 4: match by instrument.order_book_id (string and digit-only)
        ob_str = str(order_book_id) if order_book_id is not None else ""
        candidates = []
        if ob_str:
            candidates.append(ob_str)
            digits = "".join(ch for ch in ob_str if ch.isdigit())
            if digits and digits != ob_str:
                candidates.append(digits)

        # include trading_code as candidate
        if trading_code:
            tc_str = str(trading_code)
            candidates.append(tc_str)

        df = prices[prices["order_book_id"].astype(str).isin(candidates)].copy()

        # Step 5: fuzzy match by symbol substring if still empty
        if df.empty and "symbol" in prices.columns:
            base_sym = symbol_str.replace(" ", "")
            df = prices[prices["symbol"].astype(str).str.contains(base_sym, regex=False, na=False)]

        if df.empty:
            # _debug(f"❌ no price rows for symbol={symbol_str}, candidates={candidates}")
            return df

        if "close" not in df.columns:
            # _debug(f"❌ 'close' column missing for symbol={symbol_str}")
            return df

        # _debug(f"✅ matched prices rows={len(df)} for symbol={symbol_str}")
        return df.sort_values("date")

    def debug_symbol(self, symbol: str):
        """
        调试辅助：打印符号与价格映射情况
        """
        # print(f"[debug_symbol] Symbol: {symbol}")
        inst = self.instruments[self.instruments["symbol"] == symbol]
        # print(f"[debug_symbol] Instrument row:\n{inst}")
        if inst.empty:
            # print("[debug_symbol] ❌ Symbol not found in instruments.pkl")
            return
        ob = inst.iloc[0].get("order_book_id")
        # print(f"[debug_symbol] order_book_id: {ob}")
        df = self.prices[self.prices["order_book_id"] == ob]
        # print(f"[debug_symbol] Matched price rows (head):\n{df.head()}")
        if df.empty:
            # print("[debug_symbol] ❌ No matching price rows for order_book_id in prices.feather")
            return
        if "close" not in df.columns:
            # print("[debug_symbol] ❌ Price history missing 'close' column")
            return

    def get_underlying_price_history(self, underlying_symbol: str = None, underlying_order_book_id: str = None):
        """
        获取标的资产的价格历史数据
        
        优先使用 underlying_order_book_id，如果没有提供则从 underlying_symbol 查找。
        
        Parameters
        ----------
        underlying_symbol : str, optional
            标的资产代码（如 "510050" 或 "510050.XSHG"）
        underlying_order_book_id : str, optional
            标的资产的 order_book_id
            
        Returns
        -------
        pd.DataFrame
            价格历史 DataFrame，按日期排序
            如果找不到数据，返回 None
        """
        prices = self.prices
        
        # 如果提供了 underlying_order_book_id，直接使用
        if underlying_order_book_id:
            # 确保类型一致（可能都是字符串或都是数字）
            underlying_order_book_id_str = str(underlying_order_book_id)
            # 尝试字符串匹配
            df = prices[prices["order_book_id"].astype(str) == underlying_order_book_id_str].copy()
            if not df.empty:
                df = df.sort_values("date")
                return df
            # 如果字符串匹配失败，尝试原始类型匹配
            df = prices[prices["order_book_id"] == underlying_order_book_id].copy()
            if not df.empty:
                df = df.sort_values("date")
                return df
        
        # 如果没有提供 order_book_id，尝试从 instruments 中查找
        if underlying_symbol:
            inst = self.instruments
            
            # 首先尝试精确匹配 underlying_symbol
            underlying_rows = inst[inst["underlying_symbol"] == underlying_symbol]
            if not underlying_rows.empty:
                # 获取第一个合约的 underlying_order_book_id
                underlying_ob_id = underlying_rows.iloc[0].get("underlying_order_book_id")
                if underlying_ob_id and pd.notna(underlying_ob_id):
                    df = prices[prices["order_book_id"] == underlying_ob_id].copy()
                    if not df.empty:
                        df = df.sort_values("date")
                        return df
            
            # 如果精确匹配失败，尝试部分匹配（处理 "510050" vs "510050.XSHG" 的情况）
            # 提取基础代码（去掉后缀）
            base_symbol = underlying_symbol.split('.')[0] if '.' in underlying_symbol else underlying_symbol
            underlying_rows_partial = inst[inst["underlying_symbol"].str.startswith(base_symbol, na=False)]
            if not underlying_rows_partial.empty:
                # 获取第一个合约的 underlying_order_book_id
                underlying_ob_id = underlying_rows_partial.iloc[0].get("underlying_order_book_id")
                if underlying_ob_id and pd.notna(underlying_ob_id):
                    df = prices[prices["order_book_id"] == underlying_ob_id].copy()
                    if not df.empty:
                        df = df.sort_values("date")
                        return df
            
            # 如果找不到，尝试直接查找 symbol 等于 underlying_symbol 的合约
            symbol_rows = inst[inst["symbol"] == underlying_symbol]
            if not symbol_rows.empty:
                order_book_id = symbol_rows.iloc[0].get("order_book_id")
                if order_book_id and pd.notna(order_book_id):
                    df = prices[prices["order_book_id"] == order_book_id].copy()
                    if not df.empty:
                        df = df.sort_values("date")
                        return df
            
            # 最后尝试部分匹配 symbol
            symbol_rows_partial = inst[inst["symbol"].str.startswith(base_symbol, na=False)]
            if not symbol_rows_partial.empty:
                order_book_id = symbol_rows_partial.iloc[0].get("order_book_id")
                if order_book_id and pd.notna(order_book_id):
                    df = prices[prices["order_book_id"] == order_book_id].copy()
                    if not df.empty:
                        df = df.sort_values("date")
                        return df
        
        return None

    def estimate_hv_iv(self, symbol, window=60):
        """
        估算历史波动率（基于收盘价到收盘价的收益率）
        
        使用滚动窗口计算对数收益率的年化波动率。
        可用于作为 Black-Scholes 模型中的 sigma 参数。
        
        Parameters
        ----------
        symbol : str
            合约代码
        window : int, default=60
            滚动窗口大小（交易日数）。如果数据不足，会自动调整窗口大小。
            
        Returns
        -------
        float or None
            年化波动率，如果数据不足或计算失败则返回 None
        """
        df = self.get_price_history(symbol)
        
        # 检查数据是否足够
        if df is None or df.empty or len(df) < 2:
            return None
        
        df = df.sort_values("date")
        
        # 确保有 close 列
        if "close" not in df.columns:
            return None
        
        # 计算收益率
        df["ret"] = np.log(df["close"]).diff()
        
        # 如果数据不足窗口大小，使用所有可用数据（至少需要2个数据点）
        available_data_points = len(df["ret"].dropna())
        if available_data_points < window:
            # 使用至少2个数据点，但不超过可用数据点
            actual_window = max(2, min(window, available_data_points))
        else:
            actual_window = window
        
        # 计算滚动波动率
        rolling_vol = df["ret"].rolling(window=actual_window).std()
        
        # 获取最后一个有效值（跳过 NaN）
        valid_vol = rolling_vol.dropna()
        if len(valid_vol) == 0:
            return None
        
        vol_daily = valid_vol.iloc[-1]
        
        if pd.isna(vol_daily) or vol_daily <= 0:
            return None
        
        vol_annual = vol_daily * np.sqrt(252)
        return float(vol_annual)

    def estimate_implied_vol(
        self,
        symbol: str,
        risk_free_rate: float = 0.02,
    ) -> float | None:
        """
        Attempt to estimate an implied volatility for the given option symbol
        using the most recent available price snapshot.
        
        Parameters
        ----------
        symbol : str
            合约代码
        risk_free_rate : float, default=0.02
            无风险利率（年化）
            
        Returns
        -------
        float or None
            Implied volatility (annualized) if solvable, otherwise None.
        """
        try:
            # 1) Fetch instrument row
            inst = self.get_contract_row(symbol)
            
            # 2) Fetch option price history
            df_price = self.get_price_history(symbol)
            if df_price is None or df_price.empty:
                return None
            
            # 3) Take the latest row
            latest = df_price.iloc[-1]
            
            # Market price - prefer "close"
            market_price = latest.get("close", np.nan)
            if pd.isna(market_price) or market_price <= 0:
                return None
            
            # 4) Underlying spot S
            S = np.nan
            
            # Try to get underlying symbol from instrument
            underlying_symbol = None
            if "underlying_symbol" in inst.index:
                underlying_symbol = inst["underlying_symbol"]
            
            if underlying_symbol:
                try:
                    df_under = self.get_price_history(underlying_symbol)
                    if df_under is not None and not df_under.empty:
                        S = df_under.iloc[-1].get("close", np.nan)
                except:
                    pass
            
            # Fallback: try underlying_close from latest row
            if pd.isna(S) or S <= 0:
                S = latest.get("underlying_close", np.nan)
            
            if pd.isna(S) or S <= 0:
                return None
            
            # 5) Strike K
            if "strike_price" not in inst.index:
                return None
            K = inst["strike_price"]
            try:
                K = float(K)
                if K <= 0:
                    return None
            except (ValueError, TypeError):
                return None
            
            # 6) Time to maturity T (in years)
            if "maturity_date" not in inst.index:
                return None
            
            expiry_date = inst["maturity_date"]
            price_date = latest.get("date")
            
            if pd.isna(expiry_date) or pd.isna(price_date):
                return None
            
            # Convert to datetime if needed
            if not isinstance(expiry_date, pd.Timestamp):
                try:
                    expiry_date = pd.to_datetime(expiry_date)
                except:
                    return None
            
            if not isinstance(price_date, pd.Timestamp):
                try:
                    price_date = pd.to_datetime(price_date)
                except:
                    return None
            
            days_to_maturity = max((expiry_date - price_date).days, 0)
            if days_to_maturity <= 0:
                return None
            
            T_years = days_to_maturity / 365.0
            
            # 7) Option type
            if "option_type" not in inst.index:
                return None
            
            opt_type_code = inst["option_type"]
            # Normalize: "C"/"P" -> "call"/"put"
            if opt_type_code.upper() == "C":
                option_type = "call"
            elif opt_type_code.upper() == "P":
                option_type = "put"
            else:
                # Try direct match
                opt_type_lower = str(opt_type_code).lower()
                if opt_type_lower in ["call", "put"]:
                    option_type = opt_type_lower
                else:
                    return None
            
            # 8) Call implied_vol_newton
            sigma = implied_vol_newton(
                S=float(S),
                K=float(K),
                T=T_years,
                r=risk_free_rate,
                market_price=float(market_price),
                option_type=option_type,
            )
            
            return sigma
            
        except Exception:
            # Conservative: return None on any error
            return None

