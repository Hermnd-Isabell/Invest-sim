import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from statistics import NormalDist

# 引入后端桥接 (保持原有引用)
from bridge import InvestSimBridge
from invest_sim.backend.input_modeling.fitting import fit_normal
from invest_sim.option_simulator import (
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
# 3. Derivatives Lab (UI 重构版)
# ==========================================

def render_derivatives_lab() -> None:
    """
    Modernized Derivatives Lab UI
    Layout: Split View (Control Deck | Analysis Dashboard)
    """
    
    # --- HEADER: Market Ticker ---
    # 使用 Container 模拟顶部状态栏
    with st.container():
        h1, h2, h3, h4 = st.columns([1.5, 1, 1, 1])
        with h1:
            st.markdown("### ❖ DERIVATIVES LAB <span style='font-size:12px; color:#8B949E; border:1px solid #30363D; padding:2px 6px; border-radius:4px;'>PRO</span>", unsafe_allow_html=True)
        with h2:
            spot_price = st.number_input("SPOT PRICE", value=100.0, step=0.5, format="%.2f")
        with h3:
            implied_vol = st.number_input("IMPLIED VOL (σ)", value=0.20, step=0.01, format="%.2f")
        with h4:
            days_to_maturity = st.number_input("DAYS TO EXP", value=30, step=1)
    
    st.markdown("---")

    # --- MAIN SPLIT LAYOUT ---
    col_controls, col_dashboard = st.columns([1, 2.2], gap="large")

    # =========================================================
    # LEFT PANEL: CONTROL DECK
    # =========================================================
    with col_controls:
        # 1. Strategy Configuration
        st.markdown("##### 🛠 STRATEGY CONFIG")
        with st.container():
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
            with c_leg1: strike_price = st.number_input("Anchor Strike", value=100.0, step=1.0)
            with c_leg2: contract_size = st.number_input("Size", value=100, step=1)

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

            # --- Logic: Build Strategy Legs ---
            def build_strategy_legs():
                legs = []
                if strategy_name == "Single Leg":
                    legs = [OptionLeg(option_type, position_side, strike_price, contract_size)]
                elif strategy_name == "Vertical Spread (Bull Call)" and spread_width:
                    legs = [
                        OptionLeg("call", "long", strike_price, contract_size),
                        OptionLeg("call", "short", strike_price + spread_width, contract_size),
                    ]
                elif strategy_name == "Vertical Spread (Bear Put)" and spread_width:
                    legs = [
                        OptionLeg("put", "long", strike_price, contract_size),
                        OptionLeg("put", "short", strike_price - spread_width, contract_size),
                    ]
                elif strategy_name == "Straddle":
                    legs = [
                        OptionLeg("call", "long", strike_price, contract_size),
                        OptionLeg("put", "long", strike_price, contract_size),
                    ]
                elif strategy_name == "Strangle" and strangle_distance:
                    legs = [
                        OptionLeg("call", "long", strike_price + strangle_distance, contract_size),
                        OptionLeg("put", "long", strike_price - strangle_distance, contract_size),
                    ]
                elif strategy_name == "Butterfly (Call)" and wing_width:
                    legs = [
                        OptionLeg("call", "long", strike_price - wing_width, contract_size),
                        OptionLeg("call", "short", strike_price, 2 * contract_size),
                        OptionLeg("call", "long", strike_price + wing_width, contract_size),
                    ]
                elif strategy_name == "Iron Condor" and ic_width and ic_width2:
                    legs = [
                        OptionLeg("call", "short", strike_price + ic_width, contract_size),
                        OptionLeg("call", "long", strike_price + ic_width2, contract_size),
                        OptionLeg("put", "short", strike_price - ic_width, contract_size),
                        OptionLeg("put", "long", strike_price - ic_width2, contract_size),
                    ]
                else:
                    # Fallback / Custom
                    legs = [OptionLeg("call", "long", strike_price, contract_size)]
                return legs
            
            strategy_legs = build_strategy_legs()
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

        # 2. Advanced Environment Configs
        with st.expander("⚙️ RISK & MARGIN PARAMETERS", expanded=False):
            st.caption("Environment")
            risk_free_rate = st.number_input("Risk Free Rate (r)", value=0.02, step=0.005, format="%.3f")
            
            st.caption("Margin Rules")
            m1, m2 = st.columns(2)
            with m1: initial_margin = st.number_input("Init Margin", value=0.2)
            with m2: maint_margin = st.number_input("Maint Margin", value=0.1)
            
            scan_risk = st.number_input("Scan Risk Factor", value=0.20)
            min_margin = st.number_input("Min Margin Factor", value=0.10)
            
            st.caption("Delta Hedging")
            enable_hedge = st.checkbox("Active Hedging", value=False)
            if enable_hedge:
                h1, h2 = st.columns(2)
                with h1: hedge_freq = st.number_input("Freq (Days)", 1, value=1)
                with h2: hedge_thr = st.number_input("Delta Thr", 0.0, value=0.0)
            else:
                hedge_freq, hedge_thr = 1, 0.0

            st.caption("Volatility Model")
            dynamic_vol = st.checkbox("Dynamic Vol (Crash)", value=False)
            vol_sens = st.number_input("Vol Sensitivity (k)", 0.0, value=5.0) if dynamic_vol else 0.0
        
        with st.expander("🎲 SIMULATION ASSUMPTIONS", expanded=False):
            st.caption("Random Walk Parameters")
            sim_mu = st.number_input("Drift (Daily)", value=0.0005, format="%.6f")
            sim_sigma = st.number_input("Vol (Daily)", value=0.02, format="%.4f")
            ref_equity = st.number_input("Ref. Equity", value=100000.0, step=10000.0)

    # =========================================================
    # RIGHT PANEL: ANALYSIS DASHBOARD
    # =========================================================
    with col_dashboard:
        
        # --- SECTION 1: LIVE GREEKS & PAYOFF (Interactive) ---
        T_years = days_to_maturity / 365.0
        
        # Calculate Greeks "On the Fly" for Anchor Leg (for display purposes)
        # Note: True multi-leg Greeks are complex sums, here we show Anchor or indicative
        bs_p = bs_d = bs_g = bs_v = 0.0
        
        # Try to calculate BS for the "Anchor" strike/type
        # If multi-leg, we use the first leg or the anchor input
        calc_type = option_type_calc if strategy_name == "Single Leg" else "call" # Default to Call for generic view
        
        try:
            bs_p = float(np.squeeze(bs_price(spot_price, strike_price, T_years, risk_free_rate, implied_vol, calc_type)))
            bs_d = float(np.squeeze(bs_delta(spot_price, strike_price, T_years, risk_free_rate, implied_vol, calc_type)))
            bs_g = float(np.squeeze(bs_gamma(spot_price, strike_price, T_years, risk_free_rate, implied_vol)))
            bs_v = float(np.squeeze(bs_vega(spot_price, strike_price, T_years, risk_free_rate, implied_vol)))
        except:
            pass
        
        # Display Greeks
        st.markdown(f"##### ⚡ LIVE METRICS (Anchor: {calc_type.title()} @ {strike_price})")
        g1, g2, g3, g4 = st.columns(4)
        with g1: st.metric("BS Price", f"${bs_p:.2f}")
        with g2: st.metric("Delta", f"{bs_d:.3f}", delta_color="off")
        with g3: st.metric("Gamma", f"{bs_g:.4f}", delta_color="off")
        with g4: st.metric("Vega", f"{bs_v:.2f}", delta_color="off")

        # Payoff Chart (Always visible)
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
            **get_chart_layout(300)
        )
        st.plotly_chart(fig_payoff, use_container_width=True)

        # --- SECTION 2: SIMULATION ENGINE (Tabs) ---
        st.markdown("##### 🔬 SIMULATION LAB")
        
        tab_static, tab_path, tab_mc = st.tabs(["📊 MARGIN ANALYSIS", "📈 PATH SIMULATOR", "🎲 MONTE CARLO"])
        
        # --- TAB 1: STATIC MARGIN ---
        with tab_static:
            st.caption("Analyze Short Option Margin Requirements vs Underlying Price.")
            
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

        # --- TAB 2: SINGLE PATH ---
        with tab_path:
            p_col1, p_col2 = st.columns(2)
            with p_col1: sim_days = st.number_input("Duration (Days)", 10, 365, 60, key="path_days")
            with p_col2: 
                st.markdown(f"<div style='padding-top:28px'></div>", unsafe_allow_html=True)
                run_path = st.button("▶ Run Path Simulation", key="btn_path", use_container_width=True)
            
            if run_path:
                simulator = OptionMarginSimulator(
                    option_type_calc, position_side_calc, strike_price, contract_size, spot_price,
                    implied_vol, risk_free_rate, days_to_maturity, scan_risk, min_margin, maint_margin, 
                    sim_mu, sim_sigma, ref_equity,
                    enable_hedge=enable_hedge, hedge_frequency=hedge_freq, hedge_threshold=hedge_thr,
                    dynamic_vol=dynamic_vol, vol_sensitivity=vol_sens, legs=strategy_legs
                )
                res = simulator.run_single_path(sim_days)
                
                # Plotting
                c1, c2 = st.columns(2)
                with c1:
                    fig_spot = go.Figure()
                    fig_spot.add_trace(go.Scatter(y=res['spot_path'], name='Spot', line=dict(color=COLORS['gold'])))
                    fig_spot.update_layout(title="Spot Price Path", **get_chart_layout(250))
                    st.plotly_chart(fig_spot, use_container_width=True)
                with c2:
                    fig_eq = go.Figure()
                    fig_eq.add_trace(go.Scatter(y=res['equity_path'], name='Equity', line=dict(color=COLORS['green'])))
                    fig_eq.add_trace(go.Scatter(y=res['margin_path'], name='Margin', line=dict(color=COLORS['red'])))
                    if res['liquidation_day']:
                        fig_eq.add_vline(x=res['liquidation_day'], line=dict(color='white', dash='dot'))
                    fig_eq.update_layout(title="Equity vs Margin", **get_chart_layout(250))
                    st.plotly_chart(fig_eq, use_container_width=True)

        # --- TAB 3: MONTE CARLO ---
        with tab_mc:
            mc_c1, mc_c2 = st.columns(2)
            with mc_c1: mc_paths = st.number_input("Paths", 100, 5000, 500)
            with mc_c2: 
                st.markdown(f"<div style='padding-top:28px'></div>", unsafe_allow_html=True)
                run_mc = st.button("▶ Run Monte Carlo", key="btn_mc", type="primary", use_container_width=True)
            
            if run_mc:
                with st.spinner("Simulating Scenarios..."):
                    simulator = OptionMarginSimulator(
                        option_type_calc, position_side_calc, strike_price, contract_size, spot_price,
                        implied_vol, risk_free_rate, days_to_maturity, scan_risk, min_margin, maint_margin,
                        sim_mu, sim_sigma, ref_equity,
                        enable_hedge=enable_hedge, hedge_frequency=hedge_freq, hedge_threshold=hedge_thr,
                        dynamic_vol=dynamic_vol, vol_sensitivity=vol_sens, legs=strategy_legs
                    )
                    mc_days_input = sim_days # Reuse from prev tab or add new input
                    mc_res = simulator.run_monte_carlo(mc_paths, mc_days_input)
                    
                    # Metrics
                    breaches = (mc_res['liquidation_days'] < mc_days_input).mean()
                    final_eq = mc_res['equity_paths'][:, -1]
                    
                    m1, m2, m3 = st.columns(3)
                    with m1: st.metric("Margin Call Prob", f"{breaches:.1%}")
                    with m2: st.metric("Median Equity", f"${np.median(final_eq):,.0f}")
                    with m3: st.metric("CVaR (5%)", f"${np.percentile(final_eq, 5):,.0f}", delta_color="inverse")
                    
                    # Fan Chart
                    st.plotly_chart(
                        plot_monte_carlo_fan(
                            np.arange(mc_days_input+1), 
                            mc_res['equity_paths'], 
                            np.median(mc_res['equity_paths'], axis=0)
                        ), 
                        use_container_width=True
                    )
                    
                    # Worst Paths
                    st.markdown("###### Worst Case Scenarios")
                    worst_indices = np.argsort(final_eq)[:3]
                    fig_worst = go.Figure()
                    for idx in worst_indices:
                        fig_worst.add_trace(go.Scatter(y=mc_res['equity_paths'][idx], mode='lines', line=dict(width=1), name=f"Path {idx}"))
                    fig_worst.add_trace(go.Scatter(y=mc_res['equity_paths'].mean(axis=0), mode='lines', line=dict(color=COLORS['gold'], width=2), name="Avg"))
                    fig_worst.update_layout(title="Worst Equity Paths", **get_chart_layout(250))
                    st.plotly_chart(fig_worst, use_container_width=True)

# ==========================================
# 4. 侧边栏控制台 (Control Panel)
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
    ],
    label_visibility="collapsed",
)

if mode != "DERIVATIVES LAB (Options / Margin)":
    st.sidebar.markdown("### STRATEGY CONFIG")
    strategy_name_global = st.sidebar.selectbox("Algorithm", InvestSimBridge.get_available_strategies())

    # 动态参数
    strategy_params = {}
    if strategy_name_global == "Target Risk":
        strategy_params["target_vol"] = st.sidebar.slider("Target Volatility", 0.05, 0.4, 0.15, 0.01)
    elif strategy_name_global == "Adaptive Rebalance":
        strategy_params["threshold"] = st.sidebar.slider("Rebalance Threshold", 0.01, 0.1, 0.05)

    st.sidebar.markdown("### PORTFOLIO SETTINGS")
    initial_capital = st.sidebar.number_input("Initial Capital", value=100000, step=10000)
    leverage = st.sidebar.slider("Leverage Ratio", 0.5, 3.0, 1.0, 0.1)
    risk_free = st.sidebar.number_input("Risk Free Rate", 0.0, 0.1, 0.02, 0.005)

st.sidebar.markdown("---")
st.sidebar.caption(f"System Status: ONLINE\nBackend: v2.4.0 (Bridge)")

# ==========================================
# 5. 主界面逻辑 (Main View)
# ==========================================

# 页面标题
if mode != "DERIVATIVES LAB (Options / Margin)":
    st.title(mode.split(" ")[0])
    st.markdown(f"Strategy: <span style='color:{COLORS['gold']}'>{strategy_name_global}</span> &nbsp;|&nbsp; Leverage: <span style='color:{COLORS['text_main']}'>{leverage}x</span>", unsafe_allow_html=True)
    st.markdown("###") # Spacer

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
else:
    render_derivatives_lab()