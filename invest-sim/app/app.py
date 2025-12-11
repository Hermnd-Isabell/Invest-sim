import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from statistics import NormalDist

# 引入后端桥接 (保持原有引用)
from bridge import InvestSimBridge
from invest_sim.backend.input_modeling.fitting import fit_normal
from invest_sim.options import (
    OptionDataStore,
    OptionLeg,
    OptionMarginSimulator,
    bs_delta,
    bs_gamma,
    bs_price,
    bs_vega,
)

# ==========================================
# 1. 核心配置 & 视觉系统 (Visual Identity)
# ==========================================
st.set_page_config(
    page_title="QUANT | TERMINAL",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 调色板：黑金流光 (Professional Dark Mode)
COLORS = {
    "bg": "#0E1117",
    "card_bg": "#161B22",
    "border": "#30363D",
    "text_main": "#E6EDF3",
    "text_sub": "#8B949E",
    "gold": "#D29922",       # 更加沉稳的金色
    "gold_dim": "rgba(210, 153, 34, 0.15)",
    "red": "#F85149",
    "green": "#3FB950",
    "blue": "#58A6FF",
    "grid": "#21262D"
}

# Session State 初始化
if "bootstrap_returns" not in st.session_state:
    st.session_state["bootstrap_returns"] = None
if "fitted_normal_params" not in st.session_state:
    st.session_state["fitted_normal_params"] = None
if "input_model_choice" not in st.session_state:
    st.session_state["input_model_choice"] = "Normal"

# ==========================================
# OptionDataStore 缓存函数
# ==========================================
@st.cache_resource
def get_store():
    """
    获取 OptionDataStore 实例（使用 Streamlit 缓存）
    使用全量 50ETF 数据
    """
    store = OptionDataStore(
        instruments_path="data/50ETF/Filtered_OptionInstruments_510050.pkl",
        prices_path="data/50ETF/Filtered_OptionPrice_2020_2022.feather",
    )
    store.load()  # load once and cache object
    return store

# ==========================================
# Cached helpers for expiries / strikes
# ==========================================
# Removed get_expiries() and get_strikes() - now using inst_replayable directly
# ==========================================
# Cached helpers for HV / IV
# ==========================================
@st.cache_data
def cached_hv(symbol):
    # Access global cached store internally to avoid hashing store object
    store = get_store()
    return store.estimate_hv_iv(symbol)


@st.cache_data
def cached_iv(symbol):
    # Access global cached store internally to avoid hashing store object
    store = get_store()
    return store.estimate_implied_vol(symbol)

# 注入极简轻奢 CSS (Bloomberg Terminal Style)
st.markdown(f"""
    <style>
        /* 引入字体 */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap');

        /* 全局重置 */
        .stApp {{
            background-color: {COLORS['bg']};
            font-family: 'Inter', sans-serif;
            color: {COLORS['text_main']};
        }}
        
        /* 紧凑布局 */
        .block-container {{
            padding-top: 2rem !important;
            padding-bottom: 3rem !important;
            padding-left: 1.5rem !important;
            padding-right: 1.5rem !important;
        }}
        
        /* 侧边栏 */
        [data-testid="stSidebar"] {{
            background-color: #010409;
            border-right: 1px solid {COLORS['border']};
        }}
        
        /* 标题排版 */
        h1, h2, h3 {{
            font-family: 'Inter', sans-serif;
            font-weight: 400 !important;
            letter-spacing: 1px !important;
            text-transform: uppercase;
            color: {COLORS['text_main']};
        }}
        h4, h5, h6 {{
            color: {COLORS['text_sub']};
            font-weight: 500;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-top: 1rem;
        }}
        
        /* 输入框美化 */
        .stTextInput > div > div, .stNumberInput > div > div, .stSelectbox > div > div {{
            background-color: #0D1117;
            border: 1px solid {COLORS['border']};
            border-radius: 4px;
            color: {COLORS['text_main']};
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.9rem;
        }}
        .stTextInput > div > div:focus-within, .stNumberInput > div > div:focus-within {{
            border-color: {COLORS['gold']};
            box-shadow: none;
        }}
        
        /* 按钮美化 */
        .stButton button {{
            background: transparent;
            border: 1px solid {COLORS['border']};
            color: {COLORS['gold']};
            font-family: 'JetBrains Mono', monospace;
            font-weight: 500;
            text-transform: uppercase;
            border-radius: 4px;
            transition: all 0.2s;
        }}
        .stButton button:hover {{
            border-color: {COLORS['gold']};
            background: {COLORS['gold_dim']};
            color: {COLORS['gold']};
        }}
        .stButton button:active {{
            background: {COLORS['gold']};
            color: #000;
        }}

        /* Metric 卡片 */
        div[data-testid="metric-container"] {{
            background-color: {COLORS['card_bg']};
            border: 1px solid {COLORS['border']};
            padding: 10px 15px;
            border-radius: 6px;
        }}
        div[data-testid="metric-container"] label {{
            font-size: 0.7rem;
            letter-spacing: 1px;
            color: {COLORS['text_sub']};
        }}
        div[data-testid="metric-container"] div[data-testid="stMetricValue"] {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.5rem;
            color: {COLORS['text_main']};
        }}
        
        /* Tabs 样式 */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 24px;
            border-bottom: 1px solid {COLORS['border']};
        }}
        .stTabs [data-baseweb="tab"] {{
            height: 40px;
            white-space: pre-wrap;
            border-radius: 4px 4px 0 0;
            color: {COLORS['text_sub']};
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
        }}
        .stTabs [aria-selected="true"] {{
            color: {COLORS['gold']} !important;
            border-bottom-color: {COLORS['gold']} !important;
            background-color: transparent;
        }}
        
        /* Expander */
        .streamlit-expanderHeader {{
            background-color: {COLORS['card_bg']};
            color: {COLORS['text_main']};
            border: 1px solid {COLORS['border']};
            border-radius: 4px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
        }}
        
        /* 去除页脚 */
        footer {{visibility: hidden;}}
        #MainMenu {{visibility: hidden;}}
        
        /* 自定义分割线 */
        hr {{
            border: 0;
            border-top: 1px solid {COLORS['border']};
            margin: 1.5rem 0;
        }}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 高级绘图函数 (Plotly Refined)
# ==========================================

def get_chart_layout(height=400):
    return dict(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=height,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(
            showgrid=True, 
            gridcolor=COLORS['grid'], 
            gridwidth=1,
            linecolor=COLORS['border'], 
            tickfont=dict(family='JetBrains Mono', color=COLORS['text_sub'], size=10)
        ),
        yaxis=dict(
            showgrid=True, 
            gridcolor=COLORS['grid'], 
            gridwidth=1,
            zerolinecolor=COLORS['border'],
            tickfont=dict(family='JetBrains Mono', color=COLORS['text_sub'], size=10)
        ),
        legend=dict(
            orientation="h", 
            y=1.02, x=1, 
            xanchor="right", 
            font=dict(family="Inter", size=10, color=COLORS['text_sub']),
            bgcolor='rgba(0,0,0,0)'
        ),
        hovermode="x unified"
    )

def plot_monte_carlo_fan(dates, paths, median_path):
    dates_arr = np.asarray(dates)
    p95 = np.percentile(paths, 95, axis=1)
    p05 = np.percentile(paths, 5, axis=1)
    p75 = np.percentile(paths, 75, axis=1)
    p25 = np.percentile(paths, 25, axis=1)

    fig = go.Figure()
    
    # 90% Confidence Interval
    fig.add_trace(go.Scatter(
        x=np.concatenate([dates_arr, dates_arr[::-1]]),
        y=np.concatenate([p95, p05[::-1]]),
        fill='toself', fillcolor='rgba(210, 153, 34, 0.05)',
        line=dict(width=0), showlegend=False, hoverinfo='skip'
    ))

    # 50% Confidence Interval
    fig.add_trace(go.Scatter(
        x=np.concatenate([dates_arr, dates_arr[::-1]]),
        y=np.concatenate([p75, p25[::-1]]),
        fill='toself', fillcolor='rgba(210, 153, 34, 0.15)',
        line=dict(width=0), name='50% Conf. Interval'
    ))

    # Median
    fig.add_trace(go.Scatter(
        x=dates_arr, y=median_path, mode='lines',
        line=dict(color=COLORS['gold'], width=2),
        name='Median'
    ))

    fig.update_layout(**get_chart_layout(450))
    fig.update_layout(title="Projected Wealth Cone")
    return fig

def plot_nav_curve(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Portfolio'],
        mode='lines', name='Strategy',
        line=dict(color=COLORS['gold'], width=2),
        fill='tozeroy', fillcolor='rgba(210, 153, 34, 0.05)'
    ))
    fig.update_layout(**get_chart_layout(400))
    fig.update_layout(title="Net Asset Value")
    return fig

def render_hud_card(label, value, sub_value=None, sub_color=COLORS['text_sub']):
    """渲染 HTML 风格的 HUD 卡片 (Deprecated in favor of st.metric for this version but kept for compatibility)"""
    st.metric(label, value, sub_value)

def describe_input_model(model: dict | None) -> str:
    if not model: return "Default: Normal Distribution"
    params = model.get("params", {})
    params_text = ", ".join(f"{k}={v}" for k, v in params.items()) or "N/A"
    return f"Model: {model.get('dist_name', 'normal')} ({params_text})"

# ==========================================
# 3. Market View (50ETF Options)
# ==========================================

def render_market_view() -> None:
    """
    Market View UI for browsing 50ETF option contracts and inspecting price data.
    Bloomberg-style terminal layout with contract selector and visualization tabs.
    """
    # --- Top: Title & Caption ---
    st.markdown("### ❖ MARKET VIEW <span style='font-size:12px; color:#8B949E; border:1px solid #30363D; padding:2px 6px; border-radius:4px;'>50ETF OPTIONS</span>", unsafe_allow_html=True)
    st.caption("Browse real 50ETF option contracts and inspect their daily time series.")
    
    st.markdown("---")

    # --- Main Split Layout ---
    left_col, right_col = st.columns([1, 2.2], gap="large")

    # Initialize store
    store = get_store()
    # 构建可回放集合与 DataFrame（与 Derivatives Lab 保持一致）
    try:
        replayable_set = set(store.get_replayable_ids())
    except Exception:
        replayable_set = set()
    inst_df = store.instruments if hasattr(store, "instruments") else None
    if inst_df is None or len(replayable_set) == 0:
        st.error("No replayable contracts found in current dataset. Please check data source.")
        return
    inst_replayable = inst_df[inst_df["order_book_id"].astype(str).isin(replayable_set)]
    if inst_replayable.empty:
        st.error("No replayable contracts found in current dataset. Please check data source.")
        return

    # =========================================================
    # LEFT COLUMN: Contract Selector & Info Card
    # =========================================================
    with left_col:
        # --- Card 1: Contract Selector ---
        with st.container():
            st.markdown("#### Contract Selector")
            
            # 强制使用 inst_replayable 生成 expiry 列表
            expiries = sorted(inst_replayable["maturity_date"].unique())
            selected_expiry = st.selectbox("Expiry", expiries, key="mv_expiry")
            
            cp_label = st.selectbox("Call / Put", ["Call", "Put"], key="mv_cp")
            option_type = "call" if cp_label == "Call" else "put"
            
            opt_code = "C" if option_type == "call" else "P"
            # 强制使用 inst_replayable 生成 strike 列表
            # Use raw numpy values — do NOT convert to float
            strikes = list(
                inst_replayable[
                    (inst_replayable["maturity_date"] == selected_expiry)
                    & (inst_replayable["option_type"] == opt_code)
                ]["strike_price"].unique()
            )
            # Sort by float value but keep original types
            strikes = sorted(strikes, key=float)
            selected_strike = st.selectbox("Strike", strikes, key="mv_strike")
        
        st.markdown("---")
        
        # --- Card 2: Contract Information ---
        with st.container():
            st.markdown("#### Contract Information")
            
            # Resolve symbol (replayable only) without using find_symbol - 使用 inst_replayable
            # Use exact match (no float conversion) to preserve type consistency
            subset_mv = inst_replayable[
                (inst_replayable["maturity_date"] == selected_expiry)
                & (inst_replayable["option_type"].str.lower() == option_type.lower()[0])
                & (inst_replayable["strike_price"] == selected_strike)  # exact match
            ]
            if subset_mv.empty:
                st.error(
                    f"No replayable contract for expiry={selected_expiry}, "
                    f"type={option_type}, strike={selected_strike}"
                )
                return
            row_mv = subset_mv.sort_values("order_book_id").iloc[0]
            symbol = row_mv["symbol"]
            
            # Display contract info
            st.markdown(f"**Symbol:** `{symbol}`")
            
            contract_row = store.get_contract_row(symbol)
            if "contract_multiplier" in contract_row.index:
                contract_mult = contract_row["contract_multiplier"]
            else:
                contract_mult = 100
            # Replayable status
            replayable_label = "YES" if store.is_replayable(contract_row.get("order_book_id")) else "NO"
            
            # Contract details
            info_col1, info_col2 = st.columns(2)
            with info_col1:
                st.metric("Expiry", str(selected_expiry))
                st.metric("Strike", f"{selected_strike:.2f}")
            with info_col2:
                st.metric("Type", cp_label)
                st.metric("Multiplier", f"{contract_mult:.0f}")
            st.metric("Replayable", replayable_label)
            
            # HV estimate
            st.markdown("##### Volatility")
            hv_iv = store.estimate_hv_iv(symbol)
            if hv_iv is not None:
                st.metric("Historical Vol (HV)", f"{hv_iv:.2%}")
            else:
                st.info("Not enough data to compute HV.")

    # =========================================================
    # RIGHT COLUMN: Visualization Tabs
    # =========================================================
    with right_col:
        # Get price data
        df_price = store.get_price_history(symbol)
        
        if df_price is None or df_price.empty:
            st.info("No price history available for this symbol.")
            return
        
        # Create tabs
        tab_chart, tab_volume, tab_orderbook, tab_analysis = st.tabs([
            "Chart",
            "Volume",
            "Orderbook",
            "Analysis"
        ])
        
        # --- TAB 1: Chart (Candlestick) ---
        with tab_chart:
            st.caption("Daily OHLC candlestick chart for selected contract.")
            
            fig_price = go.Figure()
            fig_price.add_trace(
                go.Candlestick(
                    x=df_price["date"],
                    open=df_price["open"],
                    high=df_price["high"],
                    low=df_price["low"],
                    close=df_price["close"],
                    name="OHLC",
                )
            )
            fig_price.update_layout(
                **get_chart_layout(400),
                title=f"Daily Candlestick – {symbol}",
                xaxis_title="Date",
                yaxis_title="Price",
                xaxis_rangeslider_visible=False,
            )
            st.plotly_chart(fig_price, use_container_width=True)
            
            # Raw data preview (optional, can be collapsed)
            with st.expander("Raw Price Data (First 100 Rows)", expanded=False):
                st.dataframe(df_price.head(100), use_container_width=True)
        
        # --- TAB 2: Volume ---
        with tab_volume:
            st.caption("Daily trading volume for selected contract.")
            
            fig_vol = go.Figure()
            fig_vol.add_trace(
                go.Bar(
                    x=df_price["date"],
                    y=df_price["volume"],
                    name="Volume",
                    marker_color=COLORS["gold"],
                )
            )
            fig_vol.update_layout(
                **get_chart_layout(400),
                title="Daily Volume",
                xaxis_title="Date",
                yaxis_title="Volume",
            )
            st.plotly_chart(fig_vol, use_container_width=True)
        
        # --- TAB 3: Orderbook ---
        with tab_orderbook:
            st.caption("Top 5 levels of orderbook snapshot (latest available data).")
            
            latest = df_price.iloc[-1]
            
            # Detect bid1, bid2, ... ask1, ask2, ...
            bid_cols = [
                c for c in df_price.columns
                if c.lower().startswith("bid") and any(ch.isdigit() for ch in c)
            ]
            ask_cols = [
                c for c in df_price.columns
                if c.lower().startswith("ask") and any(ch.isdigit() for ch in c)
            ]
            
            if not bid_cols or not ask_cols:
                st.info("Orderbook fields not found in this dataset.")
            else:
                # Sort by level index (1~5)
                def level_num(c):
                    num = "".join(filter(str.isdigit, c))
                    return int(num) if num else 0
                
                bid_cols = sorted(bid_cols, key=level_num)
                ask_cols = sorted(ask_cols, key=level_num)
                
                levels = min(5, len(bid_cols), len(ask_cols))
                
                rows = []
                for i in range(levels):
                    rows.append({
                        "Level": i + 1,
                        "Bid Price": latest[bid_cols[i]] if pd.notna(latest[bid_cols[i]]) else "N/A",
                        "Ask Price": latest[ask_cols[i]] if pd.notna(latest[ask_cols[i]]) else "N/A",
                    })
                
                ob_df = pd.DataFrame(rows)
                st.dataframe(ob_df, use_container_width=True, hide_index=True)
        
        # --- TAB 4: Analysis ---
        with tab_analysis:
            st.markdown("#### Contract Analysis")
            
            # HV estimate (already shown in left panel, but also here for completeness)
            if hv_iv is not None:
                st.markdown(f"**Estimated Annualized Volatility (HV):** {hv_iv:.2%}")
            else:
                st.info("Not enough data to compute historical volatility.")
            
            # IV estimate
            st.markdown("---")
            iv_est = store.estimate_implied_vol(symbol)
            if iv_est is not None:
                st.markdown(f"**Estimated Implied Volatility (IV, market):** {iv_est:.2%}")
                if st.button("Use this IV in Derivatives Lab", key="use_iv_in_lab"):
                    st.session_state["real_contract_iv"] = float(iv_est)
                    st.success("Implied vol stored. Derivatives Lab will use it as default σ.")
            else:
                st.info("Cannot estimate implied volatility from current market snapshot.")
            
            # OptionLeg preview
            st.markdown("---")
            st.markdown("#### OptionLeg Object")
            preview_leg = OptionLeg(
                option_type=option_type,
                side="long",   # default; Derivatives Lab may allow override later
                strike=selected_strike,
                contract_size=contract_mult,
                symbol=symbol,
            )
            
            st.caption("This OptionLeg object can be passed into Derivatives Lab or Strategy Builder later.")
            
            st.caption("This OptionLeg object can be passed into Derivatives Lab or Strategy Builder later.")


# ==========================================
# 4. Derivatives Lab (UI 重构版)
# ==========================================

def render_derivatives_lab(mode: str = None) -> None:
    """
    Modernized Derivatives Lab UI
    Layout: Split View (Control Deck | Analysis Dashboard)
    
    Parameters
    ----------
    mode : str, optional
        Current system mode, used to determine if real contracts should be used
    """
    
    # --- TOP: Title & Caption ---
    st.markdown("### ❖ DERIVATIVES LAB <span style='font-size:12px; color:#8B949E; border:1px solid #30363D; padding:2px 6px; border-radius:4px;'>PRO</span>", unsafe_allow_html=True)
    if mode == "MARKET VIEW (50ETF Options)":
        st.caption("Using real 50ETF contracts for strategy and margin simulations.")
    else:
        st.caption("Build synthetic option structures and stress-test margin dynamics.")
    
    st.markdown("---")

    # --- MAIN SPLIT LAYOUT ---
    left_col, right_col = st.columns([1, 2.2], gap="large")

    # =========================================================
    # LEFT COLUMN: CONTROL DECK
    # =========================================================
    with left_col:
        # --- Card 1: Option & Strategy ---
        with st.container():
            st.markdown("#### Option & Strategy")
            
            # Underlying & Basic Params
            spot_price = st.number_input("Spot Price", value=100.0, step=0.5, format="%.2f")
            days_to_maturity = st.number_input("Days to Maturity", value=30, step=1)
            
            # Strategy Configuration
            strategy_name = st.selectbox(
                "Strategy Template",
                [
                    "Single Leg", "Vertical Spread (Bull Call)", "Vertical Spread (Bear Put)",
                    "Straddle", "Strangle", "Butterfly (Call)", "Iron Condor", "Custom (Manual Legs)"
                ]
            )
            
            # Dynamic Params
            spread_width = strangle_distance = wing_width = ic_width = ic_width2 = None
            
            # Base Params
            c_leg1, c_leg2 = st.columns(2)
            with c_leg1: 
                strike_price = st.number_input("Anchor Strike", value=100.0, step=1.0)
            with c_leg2: 
                contract_size = st.number_input("Contract Size", value=100, step=1)

            # Strategy Specific Inputs
            if strategy_name in ["Vertical Spread (Bull Call)", "Vertical Spread (Bear Put)"]:
                spread_width = st.number_input("Spread Width", value=5.0)
            elif strategy_name == "Strangle":
                strangle_distance = st.number_input("Strangle Dist", value=5.0)
            elif strategy_name == "Butterfly (Call)":
                wing_width = st.number_input("Wing Width", value=5.0)
            elif strategy_name == "Iron Condor":
                ic_c1, ic_c2 = st.columns(2)
                with ic_c1: ic_width = st.number_input("Short Width", value=5.0)
                with ic_c2: ic_width2 = st.number_input("Long Width", value=10.0)
            elif strategy_name == "Single Leg":
                c_opt1, c_opt2 = st.columns(2)
                with c_opt1: option_type = st.selectbox("Type", ["Call", "Put"])
                with c_opt2: position_side = st.selectbox("Side", ["Long", "Short"])
            else:
                # Custom defaults
                option_type = "Call"
                position_side = "Long"

            # --- Real Contract Selection (user toggle) ---
            use_real_contracts = st.checkbox(
                "Use real 50ETF contracts",
                value=(mode == "MARKET VIEW (50ETF Options)")
            )
            # 清除旧的 leg_* state，避免回显失效选项
            if use_real_contracts:
                for key in list(st.session_state.keys()):
                    if key.startswith("leg_"):
                        del st.session_state[key]
            real_leg_symbols = []
            real_leg_hvs = []
            real_leg_price_ids = []
            real_leg_ivs = []
            store = None
            has_unreplayable = False
            
            if use_real_contracts:
                st.markdown("### Assign Real Contracts to Strategy Legs")
                store = get_store()
                # 构建可回放集合与 DataFrame（直接使用 instruments ∩ prices）
                try:
                    replayable_set = set(store.get_replayable_ids())
                except Exception:
                    replayable_set = set()
                inst_df = store.instruments if hasattr(store, "instruments") else None
                if inst_df is None or len(replayable_set) == 0:
                    st.error("No replayable contracts found in current dataset. Please check data source.")
                    return
                inst_replayable = inst_df[inst_df["order_book_id"].astype(str).isin(replayable_set)]
                if inst_replayable.empty:
                    st.error("No replayable contracts found in current dataset. Please check data source.")
                    return

                # 强制使用 inst_replayable 生成 expiry 列表
                expiries = sorted(inst_replayable["maturity_date"].unique())
                
                # Determine number of legs needed
                num_legs = 1
                if strategy_name == "Vertical Spread (Bull Call)" or strategy_name == "Vertical Spread (Bear Put)":
                    num_legs = 2
                elif strategy_name == "Straddle" or strategy_name == "Strangle":
                    num_legs = 2
                elif strategy_name == "Butterfly (Call)":
                    num_legs = 3
                elif strategy_name == "Iron Condor":
                    num_legs = 4
                
                # 清理旧的 session_state key，避免使用过期的值
                # 清理旧格式的 key (leg_{idx}_expiry, leg_{idx}_cp, leg_{idx}_strike)
                for leg_idx in range(4):  # 最多清理 4 个 leg 的旧 key
                    old_keys = [
                        f"leg_{leg_idx}_expiry",
                        f"leg_{leg_idx}_cp",
                        f"leg_{leg_idx}_strike"
                    ]
                    for old_key in old_keys:
                        if old_key in st.session_state:
                            del st.session_state[old_key]
                
                # Collect real contract info for each leg
                for leg_idx in range(num_legs):
                    with st.expander(f"Leg {leg_idx + 1}", expanded=(leg_idx == 0)):
                        # 使用唯一 key 避免 session_state 冲突
                        leg_expiry = st.selectbox(
                            "Expiry",
                            expiries,
                            key=f"expiry_leg_{leg_idx}"
                        )
                        leg_cp = st.selectbox(
                            "Call/Put",
                            ["Call", "Put"],
                            key=f"cp_leg_{leg_idx}"
                        )
                        leg_option_type = "call" if leg_cp == "Call" else "put"
                        # Correct strike list: filter by expiry + option_type
                        # Use raw numpy values — do NOT convert to float
                        opt_code = "C" if leg_option_type == "call" else "P"
                        leg_strikes = list(
                            inst_replayable[
                                (inst_replayable["maturity_date"] == leg_expiry)
                                & (inst_replayable["option_type"] == opt_code)  # <-- Critical
                            ]["strike_price"].unique()
                        )
                        # Sort by float value but keep original types
                        leg_strikes = sorted(leg_strikes, key=float)
                        if len(leg_strikes) == 0:
                            st.error(f"No strikes available for Expiry={leg_expiry}, CP={opt_code}.")
                            has_unreplayable = True
                            continue
                        leg_strike = st.selectbox(
                            "Strike",
                            leg_strikes,
                            key=f"strike_leg_{leg_idx}"
                        )

                        # Guard against stale selections when dataset/caches change
                        if leg_expiry not in expiries:
                            st.error("Selected expiry is no longer available in the current dataset. Please re-select.")
                            has_unreplayable = True
                            continue
                        if leg_strike not in leg_strikes:
                            st.error("Selected strike is no longer available in the current dataset. Please re-select.")
                            has_unreplayable = True
                            continue
                        
                        # === 精确解析合约：确保找到正确的单条记录 ===
                        # Match contract by expiry + option_type + strike
                        # Use exact match (no float conversion) to preserve type consistency
                        leg_expiry_ts = pd.Timestamp(leg_expiry)
                        df_match = inst_replayable[
                            (inst_replayable["maturity_date"] == leg_expiry_ts)
                            & (inst_replayable["option_type"] == opt_code)
                            & (inst_replayable["strike_price"] == leg_strike)  # exact match
                        ]
                        
                        if df_match.empty:
                            st.error(
                                f"No contracts found for expiry={leg_expiry}, CP={opt_code}, strike={leg_strike}. "
                                f"This combination does not exist in the current dataset."
                            )
                            has_unreplayable = True
                            continue

                        # Always pick the lowest order_book_id for deterministic behavior
                        row = df_match.sort_values("order_book_id").iloc[0]
                        leg_symbol_human = row["symbol"]
                        leg_price_id = str(row["order_book_id"])

                        # First: verify actual price availability for replay
                        df_leg_price = store.get_price_history(leg_price_id)
                        if df_leg_price is None or df_leg_price.empty or "close" not in df_leg_price.columns:
                            st.warning(
                                f"Contract {leg_symbol_human} has no historical price data "
                                "and cannot be used for Historical Replay."
                            )
                            has_unreplayable = True
                            real_leg_symbols.append(None)
                            real_leg_price_ids.append(None)
                            real_leg_hvs.append(None)
                            real_leg_ivs.append(None)
                            continue

                        # If price exists → proceed normally
                        leg_hv = cached_hv(leg_symbol_human)
                        leg_iv = cached_iv(leg_symbol_human)

                        real_leg_symbols.append(leg_symbol_human)    # 人类可读符号
                        real_leg_price_ids.append(leg_price_id)      # 价格匹配用 ID
                        real_leg_hvs.append(leg_hv)
                        real_leg_ivs.append(leg_iv)

                        hv_text = f" | HV: {leg_hv:.2%}" if leg_hv else " | HV: N/A"
                        iv_text = f" | IV: {leg_iv:.2%}" if leg_iv else " | IV: N/A"
                        pid_text = f" | price_id: {leg_price_id}"
                        st.success(f"Symbol: {leg_symbol_human}{pid_text}{hv_text}{iv_text}")

                if has_unreplayable:
                    st.error("Some selected legs have no historical price data. Please re-select valid contracts.")
                    return

            # --- Logic: Build Strategy Legs ---
            def build_strategy_legs():
                legs = []
                if strategy_name == "Single Leg":
                    symbol_readable = real_leg_symbols[0] if use_real_contracts and len(real_leg_symbols) > 0 else None
                    if symbol_readable and store:
                        contract_row = store.get_contract_row(symbol_readable)
                        contract_mult = contract_row["contract_multiplier"] if "contract_multiplier" in contract_row.index else 100
                    else:
                        contract_mult = contract_size
                    legs = [OptionLeg(option_type, position_side, strike_price, contract_mult, symbol=symbol_readable)]
                elif strategy_name == "Vertical Spread (Bull Call)" and spread_width:
                    symbols_readable = real_leg_symbols[:2] if use_real_contracts and len(real_leg_symbols) >= 2 else [None, None]
                    legs = [
                        OptionLeg("call", "long", strike_price, contract_size, symbol=symbols_readable[0]),
                        OptionLeg("call", "short", strike_price + spread_width, contract_size, symbol=symbols_readable[1]),
                    ]
                elif strategy_name == "Vertical Spread (Bear Put)" and spread_width:
                    symbols_readable = real_leg_symbols[:2] if use_real_contracts and len(real_leg_symbols) >= 2 else [None, None]
                    legs = [
                        OptionLeg("put", "long", strike_price, contract_size, symbol=symbols_readable[0]),
                        OptionLeg("put", "short", strike_price - spread_width, contract_size, symbol=symbols_readable[1]),
                    ]
                elif strategy_name == "Straddle":
                    symbols_readable = real_leg_symbols[:2] if use_real_contracts and len(real_leg_symbols) >= 2 else [None, None]
                    legs = [
                        OptionLeg("call", "long", strike_price, contract_size, symbol=symbols_readable[0]),
                        OptionLeg("put", "long", strike_price, contract_size, symbol=symbols_readable[1]),
                    ]
                elif strategy_name == "Strangle" and strangle_distance:
                    symbols_readable = real_leg_symbols[:2] if use_real_contracts and len(real_leg_symbols) >= 2 else [None, None]
                    legs = [
                        OptionLeg("call", "long", strike_price + strangle_distance, contract_size, symbol=symbols_readable[0]),
                        OptionLeg("put", "long", strike_price - strangle_distance, contract_size, symbol=symbols_readable[1]),
                    ]
                elif strategy_name == "Butterfly (Call)" and wing_width:
                    symbols_readable = real_leg_symbols[:3] if use_real_contracts and len(real_leg_symbols) >= 3 else [None, None, None]
                    legs = [
                        OptionLeg("call", "long", strike_price - wing_width, contract_size, symbol=symbols_readable[0]),
                        OptionLeg("call", "short", strike_price, 2 * contract_size, symbol=symbols_readable[1]),
                        OptionLeg("call", "long", strike_price + wing_width, contract_size, symbol=symbols_readable[2]),
                    ]
                elif strategy_name == "Iron Condor" and ic_width and ic_width2:
                    symbols_readable = real_leg_symbols[:4] if use_real_contracts and len(real_leg_symbols) >= 4 else [None, None, None, None]
                    legs = [
                        OptionLeg("call", "short", strike_price + ic_width, contract_size, symbol=symbols_readable[0]),
                        OptionLeg("call", "long", strike_price + ic_width2, contract_size, symbol=symbols_readable[1]),
                        OptionLeg("put", "short", strike_price - ic_width, contract_size, symbol=symbols_readable[2]),
                        OptionLeg("put", "long", strike_price - ic_width2, contract_size, symbol=symbols_readable[3]),
                    ]
                else:
                    # Fallback / Custom
                    symbol_readable = real_leg_symbols[0] if use_real_contracts and real_leg_symbols else None
                    legs = [OptionLeg("call", "long", strike_price, contract_size, symbol=symbol_readable)]
                return legs
            
            strategy_legs = build_strategy_legs()
            
            # Update implied_vol from real contract IV (preferred) or HV if available
            if use_real_contracts and real_leg_ivs:
                valid_ivs = [iv for iv in real_leg_ivs if iv is not None]
                if valid_ivs:
                    avg_iv = float(sum(valid_ivs) / len(valid_ivs))
                    st.session_state["real_contract_iv"] = avg_iv
            
            if use_real_contracts and real_leg_hvs:
                valid_hvs = [hv for hv in real_leg_hvs if hv is not None]
                if valid_hvs:
                    avg_hv = sum(valid_hvs) / len(valid_hvs)
                    # Store in session state to update the input field
                    if 'real_contract_hv' not in st.session_state or st.session_state.get('update_hv', False):
                        st.session_state['real_contract_hv'] = avg_hv
                        st.session_state['update_hv'] = False
            
            # For pricing compatibility if Single Leg
            if strategy_name != "Single Leg":
                # Dummy values for single-leg functions to avoid errors, 
                # though multi-leg usually aggregates.
                option_type_calc = "Call" 
                position_side_calc = "Long"
            else:
                option_type_calc = option_type
                position_side_calc = position_side

        st.markdown("---")
        
        # --- Card 2: Pricing & Volatility ---
        with st.container():
            st.markdown("#### Pricing & Volatility")
            
            # Use IV (preferred) or HV from real contracts if available
            if mode == "MARKET VIEW (50ETF Options)":
                default_implied_vol = (
                    st.session_state.get("real_contract_iv")
                    or st.session_state.get("real_contract_hv")
                    or 0.20
                )
            else:
                default_implied_vol = 0.20
            implied_vol = st.number_input("Implied Vol (σ)", value=default_implied_vol, step=0.01, format="%.2f")
            risk_free_rate = st.number_input("Risk Free Rate (r)", value=0.02, step=0.005, format="%.3f")
            
            st.markdown("##### Volatility Model")
            dynamic_vol = st.checkbox("Dynamic Vol (Crash)", value=False)
            vol_sens = st.number_input("Vol Sensitivity (k)", 0.0, value=5.0) if dynamic_vol else 0.0
            
            st.markdown("##### Delta Hedging")
            enable_hedge = st.checkbox("Active Hedging", value=False)
            if enable_hedge:
                h1, h2 = st.columns(2)
                with h1: 
                    hedge_freq = st.number_input("Hedge Freq (Days)", 1, value=1)
                with h2: 
                    hedge_thr = st.number_input("Delta Threshold", 0.0, value=0.0)
            else:
                hedge_freq, hedge_thr = 1, 0.0

        st.markdown("---")
        
        # --- Card 3: Margin & Simulation ---
        with st.container():
            st.markdown("#### Margin & Simulation")
            
            st.markdown("##### Margin Rules")
            m1, m2 = st.columns(2)
            with m1: 
                initial_margin = st.number_input("Init Margin Rate", value=0.2, step=0.01)
            with m2: 
                maint_margin = st.number_input("Maint Margin Rate", value=0.1, step=0.01)
            
            scan_risk = st.number_input("Scan Risk Factor", value=0.20, step=0.01)
            min_margin = st.number_input("Min Margin Factor", value=0.10, step=0.01)
            ref_equity = st.number_input("Reference Equity", value=100000.0, step=10000.0)
            
            st.markdown("##### Simulation Parameters")
            sim_source = st.radio(
                "Simulation Source",
                ["Monte Carlo", "Historical Replay (50ETF)"],
                horizontal=True,
            )
            sim_mu = st.number_input("Drift (Daily)", value=0.0005, format="%.6f")
            sim_sigma = st.number_input("Vol (Daily)", value=0.02, format="%.4f")
            
            # Simulation controls
            sim_days = st.number_input("Path Duration (Days)", 10, 365, 60, key="path_days")
            mc_paths = st.number_input("MC Paths", 100, 5000, 500, key="mc_paths")

    # =========================================================
    # RIGHT COLUMN: ANALYSIS DASHBOARD
    # =========================================================
    with right_col:
        # Calculate Greeks for display
        T_years = days_to_maturity / 365.0
        calc_type = option_type_calc if strategy_name == "Single Leg" else "call"
        
        bs_p = bs_d = bs_g = bs_v = 0.0
        try:
            bs_p = float(np.squeeze(bs_price(spot_price, strike_price, T_years, risk_free_rate, implied_vol, calc_type)))
            bs_d = float(np.squeeze(bs_delta(spot_price, strike_price, T_years, risk_free_rate, implied_vol, calc_type)))
            bs_g = float(np.squeeze(bs_gamma(spot_price, strike_price, T_years, risk_free_rate, implied_vol)))
            bs_v = float(np.squeeze(bs_vega(spot_price, strike_price, T_years, risk_free_rate, implied_vol)))
        except:
            pass
        
        # Create tabs for different analysis views
        tab_pricing, tab_single, tab_mc, tab_worst = st.tabs([
            "Pricing & Greeks",
            "Single-Path Simulation",
            "Monte Carlo Risk",
            "Worst Paths"
        ])
        
        # --- TAB 1: Pricing & Greeks ---
        with tab_pricing:
            # Display Greeks Metrics
            st.markdown("#### Live Greeks")
            g1, g2, g3, g4 = st.columns(4)
            with g1: 
                st.metric("BS Price", f"${bs_p:.2f}")
            with g2: 
                st.metric("Delta", f"{bs_d:.3f}", delta_color="off")
            with g3: 
                st.metric("Gamma", f"{bs_g:.4f}", delta_color="off")
            with g4: 
                st.metric("Vega", f"{bs_v:.2f}", delta_color="off")
            
            # Payoff Chart
            st.markdown("#### Strategy Payoff")
            s_grid = np.linspace(0.5 * spot_price, 1.5 * spot_price, 200)
            payoff = np.zeros_like(s_grid)
            for leg in strategy_legs:
                intrinsic = np.maximum(s_grid - leg.strike, 0) if leg.option_type == "call" else np.maximum(leg.strike - s_grid, 0)
                payoff += leg.multiplier * intrinsic * leg.contract_size
            
            fig_payoff = go.Figure()
            fig_payoff.add_trace(go.Scatter(
                x=s_grid, y=payoff, mode="lines", 
                line=dict(color=COLORS['gold'], width=2), 
                fill='tozeroy', fillcolor='rgba(210, 153, 34, 0.1)',
                name="Payoff"
            ))
            fig_payoff.add_vline(x=spot_price, line=dict(color=COLORS['text_sub'], dash="dash"), annotation_text="Spot")
            fig_payoff.add_hline(y=0, line=dict(color=COLORS['border']))
            fig_payoff.update_layout(
                title="Strategy Payoff at Maturity",
                **get_chart_layout(400)
            )
            st.plotly_chart(fig_payoff, use_container_width=True)
            
            # Margin Analysis
            st.markdown("#### Margin Analysis")
            if st.button("Compute Margin Curve", key="btn_static", use_container_width=True):
                # Logic copied from original
                if position_side_calc != "Short" and strategy_name == "Single Leg":
                    st.warning("Switch side to 'Short' to see relevant margin data.")
                else:
                    s_grid_m = np.linspace(0.5 * strike_price, 1.5 * strike_price, 100)
                    # Simplified margin scan logic for the Anchor Leg (Short)
                    # For complex strategies, this needs full portfolio margin logic (backend dependent)
                    # Here we approximate using the single leg logic for demonstration or the first leg
                    
                    # Compute Price Curve
                    price_curve = bs_price(s_grid_m, strike_price, T_years, risk_free_rate, implied_vol, "call" if "Call" in strategy_name else "put")
                    otm = np.maximum(strike_price - s_grid_m, 0) if "Call" in strategy_name else np.maximum(s_grid_m - strike_price, 0)
                    
                    scan_part = price_curve + scan_risk * s_grid_m - otm
                    min_part = price_curve + min_margin * s_grid_m
                    margin_per_unit = np.maximum(np.maximum(scan_part, min_part), 0.0)
                    margin_per_contract = margin_per_unit * contract_size
                    
                    fig_margin = go.Figure()
                    fig_margin.add_trace(go.Scatter(x=s_grid_m, y=margin_per_contract, mode="lines", name="Margin Req", line=dict(color=COLORS['red'])))
                    fig_margin.add_hline(y=ref_equity, line=dict(color=COLORS['text_sub'], dash="dash"), annotation_text="Equity")
                    fig_margin.update_layout(title="Margin Req vs Spot", **get_chart_layout(300))
                    st.plotly_chart(fig_margin, use_container_width=True)

        # --- TAB 2: Single-Path Simulation ---
        with tab_single:
            run_path = st.button("▶ Run Single-Path Simulation", key="btn_path", type="primary", use_container_width=True)
            
            if run_path:
                # --- Build spot path from real underlying if available ---
                store = get_store()
                spot_series_real = None
                mu_real = sim_mu
                sigma_real = sim_sigma
                spot0_real = spot_price

                try:
                    spot_cache = getattr(store, "_price_cache", {})
                    spot_df = None
                    if isinstance(spot_cache, dict) and "510050.XSHG" in spot_cache:
                        spot_df = spot_cache["510050.XSHG"]
                    if spot_df is None:
                        spot_df = store.get_price_history("510050.XSHG")
                    if spot_df is not None and not spot_df.empty and "close" in spot_df.columns:
                        spot_series_real = spot_df["close"].dropna().values
                        if len(spot_series_real) >= 2:
                            ret_series = pd.Series(spot_series_real).pct_change().dropna()
                            if not ret_series.empty:
                                mu_real = float(ret_series.mean())
                                sigma_real = float(ret_series.std())
                            spot0_real = float(spot_series_real[0])
                except Exception:
                    spot_series_real = None

                simulator = OptionMarginSimulator(
                    option_type_calc, position_side_calc, strike_price, contract_size, spot0_real,
                    implied_vol, risk_free_rate, days_to_maturity, scan_risk, min_margin, maint_margin, 
                    mu_real, sigma_real, ref_equity,
                    enable_hedge=enable_hedge, hedge_frequency=hedge_freq, hedge_threshold=hedge_thr,
                    dynamic_vol=dynamic_vol, vol_sensitivity=vol_sens, legs=strategy_legs
                )
                
                if sim_source == "Historical Replay (50ETF)" and spot_series_real is not None and len(spot_series_real) >= 2:
                    res = simulator.run_single_path(min(sim_days, len(spot_series_real) - 1), spot_series=spot_series_real)
                else:
                    if sim_source == "Historical Replay (50ETF)" and (spot_series_real is None or len(spot_series_real) < 2):
                        st.warning("No real underlying data found; falling back to Monte Carlo spot simulation.")
                    res = simulator.run_single_path(sim_days)
                
                # Spot Price Path
                st.markdown("#### Spot Price Path")
                fig_spot = go.Figure()
                fig_spot.add_trace(go.Scatter(y=res['spot_path'], name='Spot', line=dict(color=COLORS['gold'], width=2)))
                fig_spot.update_layout(title="Spot Price Path", **get_chart_layout(300))
                st.plotly_chart(fig_spot, use_container_width=True)
                
                # Equity vs Margin
                st.markdown("#### Equity vs Margin")
                fig_eq = go.Figure()
                fig_eq.add_trace(go.Scatter(y=res['equity_path'], name='Equity', line=dict(color=COLORS['green'], width=2)))
                fig_eq.add_trace(go.Scatter(y=res['margin_path'], name='Margin', line=dict(color=COLORS['red'], width=2)))
                if res['liquidation_day']:
                    fig_eq.add_vline(x=res['liquidation_day'], line=dict(color='white', dash='dot'), annotation_text="Liquidation")
                fig_eq.update_layout(title="Equity vs Margin", **get_chart_layout(300))
                st.plotly_chart(fig_eq, use_container_width=True)
                
                # Margin Ratio
                if 'margin_ratio_path' in res:
                    st.markdown("#### Margin Ratio")
                    fig_ratio = go.Figure()
                    fig_ratio.add_trace(go.Scatter(y=res['margin_ratio_path'], name='Margin Ratio', line=dict(color=COLORS['blue'], width=2)))
                    fig_ratio.add_hline(y=maint_margin, line=dict(color=COLORS['red'], dash="dash"), annotation_text="Maint Threshold")
                    fig_ratio.update_layout(title="Margin Ratio", **get_chart_layout(250))
                    st.plotly_chart(fig_ratio, use_container_width=True)
                
                # Hedge P&L (if enabled)
                if enable_hedge and 'hedge_pnl_path' in res:
                    st.markdown("#### Hedge P&L")
                    fig_hedge = go.Figure()
                    fig_hedge.add_trace(go.Scatter(y=res['hedge_pnl_path'], name='Hedge P&L', line=dict(color=COLORS['gold'], width=2)))
                    fig_hedge.update_layout(title="Hedge P&L", **get_chart_layout(250))
                    st.plotly_chart(fig_hedge, use_container_width=True)

        # --- TAB 3: Monte Carlo Risk ---
        with tab_mc:
            run_mc = st.button("▶ Run Monte Carlo Simulation", key="btn_mc", type="primary", use_container_width=True)
            
            if run_mc:
                if sim_source == "Historical Replay (50ETF)":
                    st.warning("Historical Replay currently supports only single-path mode. Please use Single-Path Simulation tab.")
                else:
                    with st.spinner("Simulating Scenarios..."):
                        simulator = OptionMarginSimulator(
                            option_type_calc, position_side_calc, strike_price, contract_size, spot_price,
                            implied_vol, risk_free_rate, days_to_maturity, scan_risk, min_margin, maint_margin,
                            sim_mu, sim_sigma, ref_equity,
                            enable_hedge=enable_hedge, hedge_frequency=hedge_freq, hedge_threshold=hedge_thr,
                            dynamic_vol=dynamic_vol, vol_sensitivity=vol_sens, legs=strategy_legs
                        )
                        mc_res = simulator.run_monte_carlo(mc_paths, sim_days)
                        st.session_state['mc_result'] = mc_res
                    
                    # Metrics
                    breaches = (mc_res['liquidation_days'] < sim_days).mean()
                    final_eq = mc_res['equity_paths'][:, -1]
                    
                    st.markdown("#### Risk Metrics")
                    m1, m2, m3, m4 = st.columns(4)
                    with m1: 
                        st.metric("Margin Call Prob", f"{breaches:.1%}")
                    with m2: 
                        st.metric("Median Equity", f"${np.median(final_eq):,.0f}")
                    with m3: 
                        st.metric("Mean Equity", f"${np.mean(final_eq):,.0f}")
                    with m4: 
                        st.metric("CVaR (5%)", f"${np.percentile(final_eq, 5):,.0f}", delta_color="inverse")
                    
                    # Fan Chart
                    st.markdown("#### Equity Distribution Over Time")
                    st.plotly_chart(
                        plot_monte_carlo_fan(
                            np.arange(sim_days+1), 
                            mc_res['equity_paths'], 
                            np.median(mc_res['equity_paths'], axis=0)
                        ), 
                        use_container_width=True
                    )
                    
                    # Final Equity Distribution
                    st.markdown("#### Final Equity Distribution")
                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Histogram(x=final_eq, nbinsx=50, name="Final Equity", marker_color=COLORS['gold']))
                    fig_hist.add_vline(x=np.median(final_eq), line=dict(color=COLORS['green'], dash="dash"), annotation_text="Median")
                    fig_hist.add_vline(x=np.percentile(final_eq, 5), line=dict(color=COLORS['red'], dash="dash"), annotation_text="5th %ile")
                    fig_hist.update_layout(title="Final Equity Distribution", **get_chart_layout(300))
                    st.plotly_chart(fig_hist, use_container_width=True)
        
        # --- TAB 4: Worst Paths ---
        with tab_worst:
            if st.session_state.get('mc_result', None) is not None:
                mc_res = st.session_state['mc_result']
                final_eq = mc_res['equity_paths'][:, -1]
                worst_indices = np.argsort(final_eq)[:5]
                
                st.markdown("#### Worst Case Scenarios")
                st.caption(f"Showing {len(worst_indices)} worst paths out of {mc_paths} simulations")
                
                # Worst Equity Paths
                fig_worst = go.Figure()
                for idx in worst_indices:
                    fig_worst.add_trace(go.Scatter(
                        y=mc_res['equity_paths'][idx], 
                        mode='lines', 
                        line=dict(width=1, color=COLORS['red']), 
                        name=f"Worst Path {idx}",
                        opacity=0.7
                    ))
                fig_worst.add_trace(go.Scatter(
                    y=mc_res['equity_paths'].mean(axis=0), 
                    mode='lines', 
                    line=dict(color=COLORS['gold'], width=2), 
                    name="Average Path"
                ))
                fig_worst.update_layout(title="Worst Equity Paths vs Average", **get_chart_layout(400))
                st.plotly_chart(fig_worst, use_container_width=True)
                
                # Worst Spot Paths
                st.markdown("#### Worst Case Spot Price Paths")
                fig_worst_spot = go.Figure()
                for idx in worst_indices:
                    fig_worst_spot.add_trace(go.Scatter(
                        y=mc_res['spot_paths'][idx], 
                        mode='lines', 
                        line=dict(width=1, color=COLORS['red']), 
                        name=f"Path {idx}",
                        opacity=0.7
                    ))
                fig_worst_spot.add_trace(go.Scatter(
                    y=mc_res['spot_paths'].mean(axis=0), 
                    mode='lines', 
                    line=dict(color=COLORS['gold'], width=2), 
                    name="Average"
                ))
                fig_worst_spot.update_layout(title="Worst Spot Price Paths", **get_chart_layout(300))
                st.plotly_chart(fig_worst_spot, use_container_width=True)
            else:
                st.info("Run Monte Carlo simulation first to see worst paths.")

# ==========================================
# 5. 侧边栏控制台 (Control Panel)
# ==========================================
st.sidebar.markdown("## INVEST SIM <span style='font-size:10px; opacity:0.5'>PRO</span>", unsafe_allow_html=True)
st.sidebar.markdown("---")

# 模式选择
mode = st.sidebar.radio(
    "SYSTEM MODE",
    [
        "BACKTEST (Historical)",
        "PROJECTION (Monte Carlo)",
        "DERIVATIVES LAB (Options / Margin)",
        "MARKET VIEW (50ETF Options)",
    ],
    label_visibility="collapsed",
)

if mode not in ["DERIVATIVES LAB (Options / Margin)", "MARKET VIEW (50ETF Options)"]:
    st.sidebar.markdown("### STRATEGY CONFIG")
    strategy_name_global = st.sidebar.selectbox("Algorithm", InvestSimBridge.get_available_strategies())

    # 动态参数
    strategy_params = {}
    if strategy_name_global == "Target Risk":
        strategy_params["target_vol"] = st.sidebar.slider("Target Volatility", 0.05, 0.4, 0.15, 0.01)
    elif strategy_name_global == "Adaptive Rebalance":
        strategy_params["threshold"] = st.sidebar.slider("Rebalance Threshold", 0.01, 0.1, 0.05)
else:
    # 在 Derivatives Lab 或 MARKET VIEW 模式下，不需要全局策略参数
    if mode == "DERIVATIVES LAB (Options / Margin)":
        strategy_name_global = "DerivativesLab"
    else:
        strategy_name_global = "MarketView"
    strategy_params = {}

st.sidebar.markdown("### PORTFOLIO SETTINGS")
initial_capital = st.sidebar.number_input("Initial Capital", value=100000, step=10000)
leverage = st.sidebar.slider("Leverage Ratio", 0.5, 3.0, 1.0, 0.1)
risk_free = st.sidebar.number_input("Risk Free Rate", 0.0, 0.1, 0.02, 0.005)

st.sidebar.markdown("---")
st.sidebar.caption(f"System Status: ONLINE\nBackend: v2.4.0 (Bridge)")

# ==========================================
# 6. 主界面逻辑 (Main View)
# ==========================================

# 页面标题
if mode not in ["DERIVATIVES LAB (Options / Margin)", "MARKET VIEW (50ETF Options)"]:
    st.title(mode.split(" ")[0])
    st.markdown(
        f"Strategy: <span style='color:{COLORS['gold']}'>{strategy_name_global}</span> "
        f"&nbsp;|&nbsp; Leverage: <span style='color:{COLORS['text_main']}'>{leverage}x</span>",
        unsafe_allow_html=True,
    )
st.markdown("###")  # Spacer

# ------------------------------------------
# SCENARIO A: 历史回测 (Backtest)
# ------------------------------------------
if mode == "BACKTEST (Historical)":
    
    # 文件上传区域
    with st.expander("DATA SOURCE SETTINGS", expanded=True):
        col_file, col_reb = st.columns([2, 1])
        with col_file:
            uploaded_file = st.file_uploader("Upload Market Data (CSV)", type=['csv'], label_visibility="collapsed")
            if not uploaded_file:
                st.caption("Using synthetic demonstration data stream.")
        with col_reb:
            reb_freq = st.number_input("Rebalance Days", 1, 252, 21)
            
        run_bt = st.button("EXECUTE BACKTEST", use_container_width=True)

    if run_bt:
        with st.spinner("PROCESSING HISTORICAL DATA..."):
            market_data = InvestSimBridge.load_market_data(uploaded_file)
            params = {
                "strategy": strategy_name_global,
                "leverage": leverage,
                "risk_free": risk_free,
                "capital": initial_capital,
                "rebalance_frequency": reb_freq,
                **strategy_params
            }
            bt_res = InvestSimBridge.run_backtest(params, market_data)
            st.session_state['bt_result'] = bt_res
            
            if 'Returns' in bt_res.df.columns:
                st.session_state['bootstrap_returns'] = bt_res.df['Returns'].dropna().to_numpy()
            elif 'Portfolio' in bt_res.df.columns:
                portfolio_returns = bt_res.df['Portfolio'].pct_change().dropna().to_numpy()
                st.session_state['bootstrap_returns'] = portfolio_returns

    if 'bt_result' in st.session_state:
        res = st.session_state['bt_result']
        metrics = res.metrics
        
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Total Return", f"{metrics['total_return']:.2%}", f"CAGR: {metrics.get('annualized_return', 0):.2%}")
        with c2: st.metric("Sharpe Ratio", f"{metrics['sharpe']:.2f}")
        with c3: st.metric("Max Drawdown", f"{metrics['max_dd']:.2%}", delta_color="inverse")
        with c4: st.metric("Volatility", f"{metrics['volatility']:.2%}")

        col_main, col_side = st.columns([3, 1])
        with col_main:
            st.plotly_chart(plot_nav_curve(res.df), use_container_width=True)
        with col_side:
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=res.df.index, y=res.df['Drawdown'],
                fill='tozeroy', line=dict(color=COLORS['red'], width=1),
                fillcolor='rgba(248, 81, 73, 0.1)'
            ))
            fig_dd.update_layout(**get_chart_layout(200))
            fig_dd.update_layout(title="Drawdown", yaxis=dict(showgrid=False, tickformat=".0%"))
            st.plotly_chart(fig_dd, use_container_width=True)

# ------------------------------------------
# SCENARIO B: 蒙特卡洛模拟 (Projection)
# ------------------------------------------
elif mode == "PROJECTION (Monte Carlo)":
    with st.expander("SIMULATION PARAMETERS", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1: sim_years = st.number_input("Horizon (Years)", 1, 50, 10)
        with c2: num_trials = st.number_input("Monte Carlo Trials", 100, 5000, 1000)
        with c3: annual_cont = st.number_input("Annual Contribution", 0, 1000000, 0)
        input_choices = ["Normal", "Student-t", "Bootstrap"]
        default_choice = st.session_state.get("input_model_choice", "Normal")
        if default_choice not in input_choices: default_choice = "Normal"
        with c4:
            input_model_type = st.selectbox("Return Dist", input_choices, index=input_choices.index(default_choice))
        
        run_mc = st.button("RUN SIMULATION", use_container_width=True)

    if run_mc:
        with st.spinner("CALCULATING PROBABILITY PATHS..."):
            dist_name_map = {"Normal": "normal", "Student-t": "student_t", "Bootstrap": "empirical_bootstrap"}
            dist_name = dist_name_map.get(input_model_type, "normal")
            
            dist_params = {}
            if dist_name == "normal":
                fitted_params = st.session_state.get("fitted_normal_params")
                dist_params = fitted_params or {"mean": 0.0005, "vol": 0.02}
            elif dist_name == "student_t":
                dist_params = {"mean": 0.0, "df": 5.0, "scale": 0.02}
            elif dist_name == "empirical_bootstrap":
                bootstrap_returns = st.session_state.get("bootstrap_returns")
                if bootstrap_returns is None or len(bootstrap_returns) == 0:
                    st.warning("Bootstrap requires historical returns from Backtest.")
                    dist_name = "normal"
                    dist_params = {"mean": 0.0005, "vol": 0.02}
                else:
                    dist_params = {"historical_returns": bootstrap_returns.tolist()}
            
            input_model_config = {"dist_name": dist_name, "params": dist_params}
            params = {
                "strategy": strategy_name_global,
                "leverage": leverage,
                "capital": initial_capital,
                "duration": sim_years,
                "num_trials": num_trials,
                "annual_contribution": annual_cont,
                "input_model": input_model_config,
                **strategy_params
            }
            mc_res = InvestSimBridge.run_forward_simulation(params)
            st.session_state['mc_result'] = mc_res

    if 'mc_result' in st.session_state:
        res = st.session_state['mc_result']
        final_values = res['paths'][-1]
        median_val = np.median(final_values)
        p05_val = np.percentile(final_values, 5)
        breakeven_balance = initial_capital + annual_cont * sim_years
        gain = (median_val / breakeven_balance) - 1
        
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Expected Outcome", f"${median_val:,.0f}", f"{gain:+.1%} vs Invested")
        with c2: st.metric("Worst Case (95% VaR)", f"${p05_val:,.0f}", delta_color="inverse")
        with c3: st.metric("Success Prob", f"{np.mean(final_values > breakeven_balance):.1%}")

        st.plotly_chart(plot_monte_carlo_fan(res['dates'], res['paths'], res['median']), use_container_width=True)
        st.caption(describe_input_model(res.get("input_model")))

# ------------------------------------------
# SCENARIO C: Derivatives Lab (Refactored)
# ------------------------------------------
elif mode == "DERIVATIVES LAB (Options / Margin)":
    render_derivatives_lab(mode=mode)

# ------------------------------------------
# SCENARIO D: Market View (50ETF Options)
# ------------------------------------------
elif mode == "MARKET VIEW (50ETF Options)":
    render_market_view()