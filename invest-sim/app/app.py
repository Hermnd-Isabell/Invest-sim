import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from statistics import NormalDist

# 引入后端桥接
from bridge import InvestSimBridge
from invest_sim.backend.input_modeling.fitting import fit_normal
from invest_sim.option_simulator import (
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

# 调色板：黑金流光 (Black Gold & Cyber Grey)
COLORS = {
    "bg_start": "#050505",
    "bg_end": "#141414",
    "card_bg": "rgba(30, 35, 45, 0.6)", # 磨砂玻璃底色
    "card_border": "rgba(212, 175, 55, 0.2)",
    "text_main": "#E0E0E0",
    "text_sub": "#8B929E",
    "gold": "#D4AF37",  # 香槟金
    "gold_glow": "rgba(212, 175, 55, 0.4)",
    "red": "#FF4B4B",
    "green": "#00CC96",
    "grid": "#1F2229"
}

if "bootstrap_returns" not in st.session_state:
    st.session_state["bootstrap_returns"] = None
if "fitted_normal_params" not in st.session_state:
    st.session_state["fitted_normal_params"] = None
if "input_model_choice" not in st.session_state:
    st.session_state["input_model_choice"] = "Normal"
# 注入极简轻奢 CSS
st.markdown(f"""
    <style>
        /* 引入字体 (Inter & JetBrains Mono) */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=JetBrains+Mono:wght@400;700&display=swap');

        /* 全局重置 */
        .stApp {{
            background: linear-gradient(135deg, {COLORS['bg_start']} 0%, {COLORS['bg_end']} 100%);
            font-family: 'Inter', sans-serif;
        }}
        
        /* 侧边栏 */
        [data-testid="stSidebar"] {{
            background-color: #0A0C10;
            border-right: 1px solid #1F2229;
        }}
        
        /* 标题排版 */
        h1, h2, h3 {{
            font-weight: 300 !important;
            letter-spacing: 2px !important;
            color: {COLORS['text_main']};
            text-transform: uppercase;
        }}
        
        /* Glassmorphism Card (磨砂卡片) */
        .glass-card {{
            background: {COLORS['card_bg']};
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid {COLORS['card_border']};
            border-radius: 4px;
            padding: 20px;
            margin-bottom: 20px;
            transition: all 0.3s ease;
        }}
        .glass-card:hover {{
            border-color: {COLORS['gold']};
            box-shadow: 0 0 15px {COLORS['gold_glow']};
        }}

        /* 指标数字 */
        .metric-value {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 28px;
            color: #F8F9FA;
            font-weight: 600;
            text-shadow: 0 0 10px rgba(255,255,255,0.1);
        }}
        .metric-label {{
            font-size: 10px;
            color: {COLORS['text_sub']};
            letter-spacing: 1.5px;
            text-transform: uppercase;
            margin-bottom: 6px;
        }}
        .metric-sub {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            margin-top: 4px;
        }}
        
        /* Streamlit 组件美化 */
        .stButton button {{
            background: transparent;
            border: 1px solid {COLORS['gold']};
            color: {COLORS['gold']};
            font-family: 'JetBrains Mono', monospace;
            text-transform: uppercase;
            letter-spacing: 1px;
            transition: 0.3s;
        }}
        .stButton button:hover {{
            background: {COLORS['gold']};
            color: #000;
            box-shadow: 0 0 15px {COLORS['gold']};
        }}
        div[data-baseweb="select"] > div {{
            background-color: #121418;
            border-color: #2D3038;
            color: white;
        }}
        
        /* 去除页脚 */
        footer {{visibility: hidden;}}
        #MainMenu {{visibility: hidden;}}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 高级绘图函数 (Plotly Refined)
# ==========================================

def get_chart_layout(height=450):
    return dict(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=height,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(showgrid=False, linecolor=COLORS['grid'], tickfont=dict(family='JetBrains Mono', color=COLORS['text_sub'])),
        yaxis=dict(showgrid=True, gridcolor=COLORS['grid'], tickfont=dict(family='JetBrains Mono', color=COLORS['text_sub'])),
        legend=dict(orientation="h", y=1.05, x=1, xanchor="right", font=dict(family="Inter", size=10)),
        hovermode="x unified"
    )

def plot_monte_carlo_fan(dates, paths, median_path):
    dates_arr = np.asarray(dates)
    # 颜色更细腻的扇形
    p95 = np.percentile(paths, 95, axis=1)
    p05 = np.percentile(paths, 5, axis=1)
    p75 = np.percentile(paths, 75, axis=1)
    p25 = np.percentile(paths, 25, axis=1)

    fig = go.Figure()
    
    # 外层区间 (95% Confidence) - 极淡
    fig.add_trace(go.Scatter(
        x=np.concatenate([dates_arr, dates_arr[::-1]]),
        y=np.concatenate([p95, p05[::-1]]),
        fill='toself', fillcolor='rgba(212, 175, 55, 0.05)',
        line=dict(width=0), showlegend=False, hoverinfo='skip'
    ))

    # 内层区间 (50% Confidence) - 稍亮
    fig.add_trace(go.Scatter(
        x=np.concatenate([dates_arr, dates_arr[::-1]]),
        y=np.concatenate([p75, p25[::-1]]),
        fill='toself', fillcolor='rgba(212, 175, 55, 0.15)',
        line=dict(width=0), name='50% Confidence Interval'
    ))

    # 中位数 (发光效果)
    fig.add_trace(go.Scatter(
        x=dates_arr, y=median_path, mode='lines',
        line=dict(color=COLORS['gold'], width=2),
        name='Median Projection'
    ))
    # 光晕层
    fig.add_trace(go.Scatter(
        x=dates_arr, y=median_path, mode='lines',
        line=dict(color=COLORS['gold'], width=6, dash='solid'),
        opacity=0.2, hoverinfo='skip', showlegend=False
    ))

    fig.update_layout(**get_chart_layout(500))
    fig.update_yaxes(title="Projected Wealth")
    return fig

def plot_nav_curve(df):
    fig = go.Figure()
    
    # 策略曲线 (渐变填充)
    # 注意: Plotly 在 Python 中对渐变支持有限，这里用透明度模拟
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Portfolio'],
        mode='lines', name='Strategy',
        line=dict(color=COLORS['gold'], width=2),
        fill='tozeroy', fillcolor='rgba(212, 175, 55, 0.08)'
    ))
    
    fig.update_layout(**get_chart_layout(400))
    fig.update_yaxes(title="Net Asset Value")
    return fig

def render_hud_card(label, value, sub_value=None, sub_color=COLORS['text_sub']):
    """渲染 HUD 风格的指标卡片"""
    sub_html = f'<div class="metric-sub" style="color:{sub_color}">{sub_value}</div>' if sub_value else ""
    st.markdown(f"""
        <div class="glass-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {sub_html}
        </div>
    """, unsafe_allow_html=True)


def describe_input_model(model: dict | None) -> str:
    if not model:
        return "输入模型：默认 normal 分布。"
    params = model.get("params", {})
    params_text = ", ".join(f"{key}={value}" for key, value in params.items()) or "无参数"
    return f"输入模型：{model.get('dist_name', 'normal')}（参数：{params_text}）"


def render_derivatives_lab() -> None:
    """UI scaffold for the Derivatives Lab page."""
    st.markdown("### DERIVATIVES LAB (Options / Margin)")
    st.caption("Build option structures, configure margin assumptions, and stage simulations.")

    st.markdown("#### (A) OPTION BUILDER")
    with st.container():
        opt_cols = st.columns(3)
        with opt_cols[0]:
            underlying_asset = st.selectbox("Select Underlying Asset", ["SPX", "NDX", "Gold", "Custom"])
        with opt_cols[1]:
            option_type = st.selectbox("Option Type", ["Call", "Put"])
        with opt_cols[2]:
            position_side = st.selectbox("Position Side", ["Long", "Short"])

        opt_cols_secondary = st.columns(4)
        with opt_cols_secondary[0]:
            spot_price = st.number_input("Underlying Spot Price", value=100.0, step=1.0, min_value=0.0)
        with opt_cols_secondary[1]:
            strike_price = st.number_input("Strike Price", value=100.0, step=1.0, min_value=0.0)
        with opt_cols_secondary[2]:
            maturity_date = st.date_input("Maturity Date", value=datetime.today())
        with opt_cols_secondary[3]:
            contract_size = st.number_input("Contract Size", value=100, step=1, min_value=1)

    st.markdown("#### OPTION PRICING INPUTS")
    pricing_cols = st.columns(3)
    with pricing_cols[0]:
        implied_vol = st.number_input("Implied Volatility (σ)", min_value=0.0001, value=0.2, step=0.01, format="%.4f")
    with pricing_cols[1]:
        risk_free_rate = st.number_input("Risk-Free Rate (r)", value=0.02, step=0.005, format="%.3f")
    with pricing_cols[2]:
        days_to_maturity = st.number_input("Days to Maturity", min_value=1, value=30, step=1)
    time_to_maturity_years = days_to_maturity / 365.0

    st.markdown("#### (B) MARGIN & FEES CONFIG")
    margin_cols = st.columns(4)
    with margin_cols[0]:
        initial_margin_rate = st.number_input("Initial Margin Rate", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
    with margin_cols[1]:
        maintenance_margin_rate = st.number_input("Maintenance Margin Rate", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
    with margin_cols[2]:
        commission_per_contract = st.number_input("Commission / Contract", value=1.0, step=0.1)
    with margin_cols[3]:
        slippage_bps = st.number_input("Slippage (bps)", value=5, step=1)
    margin_factor_cols = st.columns(2)
    with margin_factor_cols[0]:
        scan_risk_factor = st.number_input(
            "Scan Risk Factor",
            value=0.20,
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            format="%.3f",
        )
    with margin_factor_cols[1]:
        min_margin_factor = st.number_input(
            "Min Margin Factor",
            value=0.10,
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            format="%.3f",
        )
    st.radio("Liquidation Rule", ["Force Liquidation", "Margin Call"], horizontal=True)

    st.markdown("#### (C) INPUT MODELING SECTION")
    modeling_cols = st.columns(3)
    with modeling_cols[0]:
        st.selectbox("Distribution Type", ["Normal", "Bootstrap"])
    with modeling_cols[1]:
        st.number_input("Mean (μ)", value=0.0, step=0.0001, format="%.4f")
    with modeling_cols[2]:
        st.number_input("Volatility (σ)", value=0.2, step=0.0001, format="%.4f")

    st.markdown("#### (D) SIMULATION SETTINGS")
    sim_cols = st.columns(4)
    with sim_cols[0]:
        st.number_input("Monte Carlo Paths", min_value=1, value=1000, step=100)
    with sim_cols[1]:
        st.number_input("Years", min_value=1, value=5, step=1)
    with sim_cols[2]:
        st.number_input("Time Step (Days)", min_value=1, value=21, step=1)
    with sim_cols[3]:
        st.number_input("Initial Capital", min_value=0, value=100000, step=1000)
    reference_equity = st.number_input(
        "Reference Account Equity (for margin ratio view)",
        value=100_000.0,
        min_value=0.0,
        step=10_000.0,
    )

    st.markdown("#### (E) VISUAL PLACEHOLDERS")
    vis_cols_top = st.columns(2)
    with vis_cols_top[0]:
        st.markdown("##### Payoff Diagram Area")
        plot_request = st.button("Plot Payoff", use_container_width=True)
        if plot_request:
            if strike_price <= 0 or contract_size <= 0:
                st.warning("Strike price and contract size must both be positive.")
            else:
                s_grid = np.linspace(0.3 * strike_price, 2.0 * strike_price, 200)
                if option_type == "Call":
                    intrinsic = np.maximum(s_grid - strike_price, 0)
                else:
                    intrinsic = np.maximum(strike_price - s_grid, 0)
                multiplier = 1 if position_side == "Long" else -1
                payoff = intrinsic * contract_size * multiplier

                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(x=s_grid, y=payoff, mode="lines", line=dict(color=COLORS["gold"]))
                )
                fig.add_hline(y=0, line=dict(color="gray", width=1))
                fig.update_layout(**get_chart_layout(320))
                fig.update_layout(title="Option Payoff at Maturity", xaxis_title="Underlying Price", yaxis_title="Payoff")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Configure the option above and click Plot Payoff.")
    with vis_cols_top[1]:
        st.markdown("##### Black-Scholes Pricing & Greeks")
        compute_request = st.button("Compute BS Price & Greeks", use_container_width=True)
        if compute_request:
            if strike_price <= 0 or spot_price <= 0:
                st.warning("Spot price and strike price must both be positive.")
            else:
                T = time_to_maturity_years
                sigma = implied_vol
                r = risk_free_rate
                price_val = float(np.squeeze(bs_price(spot_price, strike_price, T, r, sigma, option_type)))
                delta_val = float(np.squeeze(bs_delta(spot_price, strike_price, T, r, sigma, option_type)))
                gamma_val = float(np.squeeze(bs_gamma(spot_price, strike_price, T, r, sigma)))
                vega_val = float(np.squeeze(bs_vega(spot_price, strike_price, T, r, sigma)))

                metric_cols = st.columns(4)
                metric_cols[0].metric("BS Price", f"${price_val:,.2f}")
                metric_cols[1].metric("Delta", f"{delta_val:.3f}")
                metric_cols[2].metric("Gamma", f"{gamma_val:.6f}")
                metric_cols[3].metric("Vega", f"{vega_val:.2f}")

                s_grid = np.linspace(0.3 * strike_price, 2.0 * strike_price, 200)
                price_curve = bs_price(s_grid, strike_price, T, r, sigma, option_type)
                delta_curve = bs_delta(s_grid, strike_price, T, r, sigma, option_type)
                gamma_curve = bs_gamma(s_grid, strike_price, T, r, sigma)
                vega_curve = bs_vega(s_grid, strike_price, T, r, sigma)

                chart_cols_top = st.columns(2)
                with chart_cols_top[0]:
                    fig_price = go.Figure()
                    fig_price.add_trace(go.Scatter(x=s_grid, y=price_curve, mode="lines", line=dict(color=COLORS["gold"])))
                    fig_price.update_layout(
                        **get_chart_layout(300),
                        title="Black-Scholes Price vs Spot",
                        xaxis_title="Underlying Price",
                        yaxis_title="Option Price",
                    )
                    st.plotly_chart(fig_price, use_container_width=True)
                with chart_cols_top[1]:
                    fig_delta = go.Figure()
                    fig_delta.add_trace(go.Scatter(x=s_grid, y=delta_curve, mode="lines", line=dict(color=COLORS["green"])))
                    fig_delta.update_layout(
                        **get_chart_layout(300),
                        title="Delta vs Spot",
                        xaxis_title="Underlying Price",
                        yaxis_title="Delta",
                    )
                    st.plotly_chart(fig_delta, use_container_width=True)

                chart_cols_bottom = st.columns(2)
                with chart_cols_bottom[0]:
                    fig_gamma = go.Figure()
                    fig_gamma.add_trace(go.Scatter(x=s_grid, y=gamma_curve, mode="lines", line=dict(color=COLORS["text_main"])))
                    fig_gamma.update_layout(
                        **get_chart_layout(300),
                        title="Gamma vs Spot",
                        xaxis_title="Underlying Price",
                        yaxis_title="Gamma",
                    )
                    st.plotly_chart(fig_gamma, use_container_width=True)
                with chart_cols_bottom[1]:
                    fig_vega = go.Figure()
                    fig_vega.add_trace(go.Scatter(x=s_grid, y=vega_curve, mode="lines", line=dict(color=COLORS["red"])))
                    fig_vega.update_layout(
                        **get_chart_layout(300),
                        title="Vega vs Spot",
                        xaxis_title="Underlying Price",
                        yaxis_title="Vega",
                    )
                    st.plotly_chart(fig_vega, use_container_width=True)
        else:
            st.info("Configure pricing inputs above and click Compute BS Price & Greeks.")

    vis_cols_bottom = st.columns(2)
    with vis_cols_bottom[0]:
        st.markdown("##### Short Option Margin Requirement")
        margin_button = st.button("Compute Margin Curve", use_container_width=True)
        if margin_button:
            if strike_price <= 0 or contract_size <= 0:
                st.warning("Strike and contract size must be positive to compute a margin curve.")
            elif position_side != "Short":
                st.info("Margin curve is most relevant for SHORT positions. Switch Position Side to Short to see margin requirements.")
            else:
                T = max(time_to_maturity_years, 1e-9)
                sigma = max(implied_vol, 1e-6)
                r = risk_free_rate
                s_grid = np.linspace(0.3 * strike_price, 2.0 * strike_price, 200)
                option_type_lower = option_type.lower()
                price_curve = bs_price(s_grid, strike_price, T, r, sigma, option_type_lower)

                if option_type_lower == "call":
                    otm = np.maximum(strike_price - s_grid, 0)
                else:
                    otm = np.maximum(s_grid - strike_price, 0)

                scan_part = price_curve + scan_risk_factor * s_grid - otm
                min_part = price_curve + min_margin_factor * s_grid
                margin_per_unit = np.maximum(np.maximum(scan_part, min_part), 0.0)
                margin_per_contract = margin_per_unit * contract_size

                fig_margin = go.Figure()
                fig_margin.add_trace(
                    go.Scatter(
                        x=s_grid,
                        y=margin_per_contract,
                        mode="lines",
                        name="Margin per Contract",
                        line=dict(color=COLORS["red"]),
                    )
                )
                if reference_equity > 0:
                    margin_ratio = reference_equity / np.maximum(margin_per_contract, 1e-8)
                    fig_margin.add_hline(
                        y=reference_equity,
                        line=dict(color="gray", width=1, dash="dash"),
                        annotation_text="Reference Equity",
                        annotation_position="top left",
                    )
                    st.caption(
                        f"Reference equity covers between {margin_ratio.min():.2f}x and {margin_ratio.max():.2f}x of required margin across the grid."
                    )

                fig_margin.update_layout(
                    **get_chart_layout(320),
                    title="Short Option Margin vs Underlying Price",
                    xaxis_title="Underlying Price",
                    yaxis_title="Margin Required per Contract",
                )
                st.plotly_chart(fig_margin, use_container_width=True)
        else:
            st.info("Configure short option inputs above and click Compute Margin Curve.")
    with vis_cols_bottom[1]:
        st.markdown("##### P&L Distribution Area")
        st.info("Placeholder for P&L distribution.")

    # ------------------------------------------
    # Single-Path Daily Simulation
    # ------------------------------------------
    st.markdown("### Single-Path Daily Simulation")
    sim_param_cols = st.columns(3)
    with sim_param_cols[0]:
        simulate_days = st.number_input("Simulation Days", min_value=1, value=60, step=1)
    with sim_param_cols[1]:
        daily_return_mean = st.number_input("Daily Return Mean (μ)", value=0.0005, format="%.6f")
    with sim_param_cols[2]:
        daily_return_vol = st.number_input("Daily Return Vol (σ)", min_value=0.0, value=0.02, format="%.4f")
    run_single_sim = st.button("Run Single-Path Simulation", use_container_width=True)

    if run_single_sim:
        if strike_price <= 0 or contract_size <= 0:
            st.warning("Strike and contract size must be positive to run the simulation.")
        else:
            simulator = OptionMarginSimulator(
                option_type,
                position_side,
                strike_price,
                contract_size,
                spot_price,
                implied_vol,
                risk_free_rate,
                days_to_maturity,
                scan_risk_factor,
                min_margin_factor,
                maintenance_margin_rate,
                daily_return_mean,
                daily_return_vol,
                reference_equity,
            )
            result_single = simulator.run_single_path(simulate_days)
            spot_path = result_single["spot_path"]
            option_price_path = result_single["option_price_path"]
            equity_path = result_single["equity_path"]
            margin_path = result_single["margin_path"]
            margin_ratio_path = result_single["margin_ratio_path"]
            liquidation_day = result_single["liquidation_day"]
            days_axis = np.arange(len(spot_path))
            vis_cols_sim = st.columns(2)
            with vis_cols_sim[0]:
                fig_sim1 = go.Figure()
                fig_sim1.add_trace(
                    go.Scatter(
                        x=days_axis,
                        y=spot_path,
                        mode="lines",
                        name="Spot Price",
                        line=dict(color=COLORS["gold"]),
                    )
                )
                fig_sim1.add_trace(
                    go.Scatter(
                        x=days_axis,
                        y=option_price_path,
                        mode="lines",
                        name="Option Price",
                        line=dict(color=COLORS["text_sub"]),
                    )
                )
                fig_sim1.update_layout(
                    **get_chart_layout(320),
                    title="Spot & Option Price Path",
                    xaxis_title="Day",
                    yaxis_title="Value",
                )
                st.plotly_chart(fig_sim1, use_container_width=True)

            with vis_cols_sim[1]:
                fig_sim2 = go.Figure()
                fig_sim2.add_trace(
                    go.Scatter(
                        x=days_axis,
                        y=equity_path,
                        mode="lines",
                        name="Equity",
                        line=dict(color=COLORS["green"]),
                    )
                )
                fig_sim2.add_trace(
                    go.Scatter(
                        x=days_axis,
                        y=margin_path,
                        mode="lines",
                        name="Margin Requirement",
                        line=dict(color=COLORS["red"]),
                    )
                )
                if liquidation_day is not None:
                    fig_sim2.add_vline(
                        x=liquidation_day,
                        line=dict(color=COLORS["red"], dash="dash"),
                        annotation_text="Liquidation",
                        annotation_position="top right",
                    )
                fig_sim2.update_layout(
                    **get_chart_layout(320),
                    title="Equity vs Margin Requirement",
                    xaxis_title="Day",
                    yaxis_title="Value",
                )
                st.plotly_chart(fig_sim2, use_container_width=True)

            fig_ratio = go.Figure()
            fig_ratio.add_trace(
                go.Scatter(
                    x=days_axis,
                    y=margin_ratio_path,
                    mode="lines",
                    name="Margin Ratio",
                    line=dict(color=COLORS["text_main"]),
                )
            )
            fig_ratio.add_hline(
                y=maintenance_margin_rate,
                line=dict(color="gray", dash="dash"),
                annotation_text="Maintenance Margin",
                annotation_position="top left",
            )
            if liquidation_day is not None:
                fig_ratio.add_vline(
                    x=liquidation_day,
                    line=dict(color=COLORS["red"], dash="dot"),
                    annotation_text="Liquidation",
                    annotation_position="top right",
                )
            fig_ratio.update_layout(
                **get_chart_layout(280),
                title="Margin Ratio Path",
                xaxis_title="Day",
                yaxis_title="Equity / Margin",
            )
            st.plotly_chart(fig_ratio, use_container_width=True)

    # ------------------------------------------
    # Monte Carlo Margin Simulation
    # ------------------------------------------
    st.markdown("### Monte Carlo Margin Simulation")
    mc_cols = st.columns(2)
    with mc_cols[0]:
        num_paths = st.number_input(
            "Number of Simulation Paths",
            min_value=10,
            max_value=5000,
            value=500,
            step=50,
        )
    with mc_cols[1]:
        mc_days = st.number_input(
            "Simulation Days (Monte Carlo)",
            min_value=1,
            value=60,
            step=1,
        )
    run_mc_sim = st.button("Run Monte Carlo Simulation", use_container_width=True)

    if run_mc_sim:
        if strike_price <= 0 or contract_size <= 0:
            st.warning("Strike and contract size must be positive to run Monte Carlo.")
        elif implied_vol <= 0:
            st.warning("Implied volatility must be positive for Monte Carlo pricing.")
        else:
            simulator = OptionMarginSimulator(
                option_type,
                position_side,
                strike_price,
                contract_size,
                spot_price,
                implied_vol,
                risk_free_rate,
                days_to_maturity,
                scan_risk_factor,
                min_margin_factor,
                maintenance_margin_rate,
                daily_return_mean,
                daily_return_vol,
                reference_equity,
            )
            result_mc = simulator.run_monte_carlo(num_paths, mc_days)
            spot_paths = result_mc["spot_paths"]
            option_price_paths = result_mc["option_price_paths"]
            equity_paths = result_mc["equity_paths"]
            margin_paths = result_mc["margin_paths"]
            margin_ratio_paths = result_mc["margin_ratio_paths"]
            liquidation_days = result_mc["liquidation_days"]
            n = spot_paths.shape[0]
            T = spot_paths.shape[1] - 1

            margin_breached = liquidation_days < T
            margin_call_prob = margin_breached.mean()
            avg_liquidation_day = (
                float(liquidation_days[margin_breached].mean())
                if margin_breached.any()
                else None
            )
            final_equity = equity_paths[:, -1]
            q05 = np.quantile(equity_paths, 0.05, axis=0)
            q50 = np.quantile(equity_paths, 0.50, axis=0)
            q95 = np.quantile(equity_paths, 0.95, axis=0)
            days_axis = np.arange(T + 1)

            if position_side != "Short":
                st.info("For long positions, margin is not binding; Monte Carlo reflects P&L only.")

            mc_cols_top = st.columns(2)
            with mc_cols_top[0]:
                fig_fan = go.Figure()
                fig_fan.add_trace(
                    go.Scatter(
                        x=days_axis,
                        y=q95,
                        mode="lines",
                        line=dict(width=0),
                        showlegend=False,
                    )
                )
                fig_fan.add_trace(
                    go.Scatter(
                        x=days_axis,
                        y=q05,
                        mode="lines",
                        fill="tonexty",
                        name="90% Interval",
                        line=dict(width=0),
                    )
                )
                fig_fan.add_trace(
                    go.Scatter(
                        x=days_axis,
                        y=q50,
                        mode="lines",
                        name="Median Equity",
                        line=dict(color=COLORS["gold"]),
                    )
                )
                fig_fan.update_layout(
                    **get_chart_layout(320),
                    title="Equity Fan Chart (Monte Carlo)",
                    xaxis_title="Day",
                    yaxis_title="Equity",
                )
                st.plotly_chart(fig_fan, use_container_width=True)

            with mc_cols_top[1]:
                fig_hist = go.Figure()
                fig_hist.add_trace(
                    go.Histogram(
                        x=final_equity,
                        nbinsx=40,
                        marker=dict(color=COLORS["green"]),
                    )
                )
                fig_hist.update_layout(
                    **get_chart_layout(320),
                    title="Final Equity Distribution",
                    xaxis_title="Final Equity",
                    yaxis_title="Count",
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                st.metric("Margin Call Probability", f"{margin_call_prob:.1%}")
            with c2:
                if avg_liquidation_day is not None:
                    st.metric("Avg Liquidation Day (conditional)", f"{avg_liquidation_day:.1f}")
                else:
                    st.metric("Avg Liquidation Day (conditional)", "No Margin Calls")

            initial_equity = reference_equity
            mean_final_equity = float(np.mean(final_equity))
            median_final_equity = float(np.median(final_equity))
            std_final_equity = float(np.std(final_equity))
            p5_final = float(np.quantile(final_equity, 0.05))
            p1_final = float(np.quantile(final_equity, 0.01))
            p95_final = float(np.quantile(final_equity, 0.95))
            loss_p5 = initial_equity - p5_final
            loss_p1 = initial_equity - p1_final

            max_drawdowns = np.zeros(n)
            for j in range(n):
                eq = equity_paths[j]
                running_max = np.maximum.accumulate(eq)
                drawdowns = (running_max - eq) / np.maximum(running_max, 1e-8)
                max_drawdowns[j] = np.max(drawdowns)
            mean_max_drawdown = float(np.mean(max_drawdowns))
            p95_max_drawdown = float(np.quantile(max_drawdowns, 0.95))

            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Mean Final Equity", f"{mean_final_equity:,.0f}", help="Average ending equity across all paths.")
            with m2:
                st.metric("Median Final Equity", f"{median_final_equity:,.0f}", help="Median ending equity across all paths.")
            with m3:
                st.metric("Std of Final Equity", f"{std_final_equity:,.0f}", help="Standard deviation of ending equity.")

            t1, t2, t3 = st.columns(3)
            with t1:
                st.metric("5% Worst-case Equity (P5)", f"{p5_final:,.0f}", help="5th percentile of final equity.")
            with t2:
                st.metric("1% Worst-case Equity (P1)", f"{p1_final:,.0f}", help="1st percentile of final equity.")
            with t3:
                st.metric("5% Worst-case Loss", f"{loss_p5:,.0f}", help="Initial equity minus 5th percentile final equity.")

            d1, d2 = st.columns(2)
            with d1:
                st.metric("Mean Max Drawdown", f"{mean_max_drawdown:.1%}", help="Average maximum drawdown across paths.")
            with d2:
                st.metric("95% Max Drawdown (P95)", f"{p95_max_drawdown:.1%}", help="95th percentile of maximum drawdown.")

            stats_df = pd.DataFrame(
                {
                    "Metric": [
                        "Mean Final Equity",
                        "Median Final Equity",
                        "Std Final Equity",
                        "P5 Final Equity",
                        "P1 Final Equity",
                        "Margin Call Probability",
                    ],
                    "Value": [
                        mean_final_equity,
                        median_final_equity,
                        std_final_equity,
                        p5_final,
                        p1_final,
                        margin_call_prob,
                    ],
                }
            )
            st.dataframe(stats_df, use_container_width=True, hide_index=True)

# ==========================================
# 3. 侧边栏控制台 (Control Panel)
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

st.sidebar.markdown("### STRATEGY CONFIG")
strategy_name = st.sidebar.selectbox("Algorithm", InvestSimBridge.get_available_strategies())

# 动态参数
strategy_params = {}
if strategy_name == "Target Risk":
    strategy_params["target_vol"] = st.sidebar.slider("Target Volatility", 0.05, 0.4, 0.15, 0.01)
elif strategy_name == "Adaptive Rebalance":
    strategy_params["threshold"] = st.sidebar.slider("Rebalance Threshold", 0.01, 0.1, 0.05)

st.sidebar.markdown("### PORTFOLIO SETTINGS")
initial_capital = st.sidebar.number_input("Initial Capital", value=100000, step=10000)
leverage = st.sidebar.slider("Leverage Ratio", 0.5, 3.0, 1.0, 0.1)
risk_free = st.sidebar.number_input("Risk Free Rate", 0.0, 0.1, 0.02, 0.005)

st.sidebar.markdown("---")
st.sidebar.caption(f"System Status: ONLINE\nBackend: v2.4.0 (Bridge)")

# ==========================================
# 4. 主界面逻辑 (Main View)
# ==========================================

# 页面标题
st.title(mode.split(" ")[0])
st.markdown(f"Strategy: <span style='color:{COLORS['gold']}'>{strategy_name}</span> &nbsp;|&nbsp; Leverage: <span style='color:{COLORS['text_main']}'>{leverage}x</span>", unsafe_allow_html=True)
st.markdown("###") # Spacer

# ------------------------------------------
# INPUT MODELING LAB
# ------------------------------------------
st.markdown("### INPUT MODELING LAB")
with st.expander("上传历史数据并拟合 Normal 参数", expanded=False):
    st.caption("上传价格或收益 CSV，自动拟合正态分布参数，并使用直方图 + QQ Plot 进行检验。")
    uploaded_returns = st.file_uploader(
        "上传 CSV（第一列可为日期/索引，需至少包含一列价格/收益数据）",
        type=["csv"],
        key="fit_file_uploader",
    )
    if uploaded_returns:
        try:
            fit_df = pd.read_csv(uploaded_returns)
        except Exception as exc:
            st.error(f"无法读取文件：{exc}")
            fit_df = None

        if fit_df is not None:
            numeric_cols = fit_df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                st.warning("未检测到任何数值列，请上传包含价格或收益的 CSV。")
            else:
                selected_series = st.selectbox(
                    "选择用于拟合的列",
                    numeric_cols,
                    key="fit_col_select",
                )
                data_mode = st.radio(
                    "该列为",
                    ["价格序列", "收益序列"],
                    horizontal=True,
                    key="fit_data_mode",
                )
                series = pd.Series(fit_df[selected_series]).dropna()
                if data_mode == "价格序列":
                    returns_series = series.pct_change().dropna()
                else:
                    returns_series = series

                if returns_series.empty:
                    st.warning("无法计算有效的收益序列，请检查数据。")
                else:
                    fitted = fit_normal(returns_series.to_numpy())
                    st.success(
                        f"Normal 拟合结果：mean={fitted['mean']:.6f}，vol={fitted['vol']:.6f}"
                    )

                    if st.button("应用拟合参数 (Normal)", key="apply_fit_normal"):
                        st.session_state["fitted_normal_params"] = fitted
                        st.session_state["input_model_choice"] = "Normal"
                        st.success("已保存拟合参数，并设置 Normal Input Model。")

                    sigma = max(fitted["vol"], 1e-8)
                    dist = NormalDist(mu=fitted["mean"], sigma=sigma)

                    hist_fig = go.Figure()
                    hist_fig.add_trace(
                        go.Histogram(
                            x=returns_series,
                            histnorm="probability density",
                            nbinsx=60,
                            opacity=0.75,
                            marker_color=COLORS["gold"],
                            name="Empirical",
                        )
                    )
                    x_range = np.linspace(
                        returns_series.min(), returns_series.max(), 200
                    )
                    pdf_values = np.array([dist.pdf(x) for x in x_range])
                    hist_fig.add_trace(
                        go.Scatter(
                            x=x_range,
                            y=pdf_values,
                            mode="lines",
                            line=dict(color="#FFFFFF", width=2),
                            name="Fitted Normal",
                        )
                    )
                    hist_fig.update_layout(
                        title="收益直方图 + Normal 拟合曲线",
                        bargap=0.05,
                        **get_chart_layout(320),
                    )
                    st.plotly_chart(hist_fig, width="stretch")

                    sorted_returns = np.sort(returns_series)
                    probs = (np.arange(1, len(sorted_returns) + 1) - 0.5) / len(
                        sorted_returns
                    )
                    theoretical = np.array([dist.inv_cdf(p) for p in probs])
                    qq_fig = go.Figure()
                    qq_fig.add_trace(
                        go.Scatter(
                            x=theoretical,
                            y=sorted_returns,
                            mode="markers",
                            marker=dict(color=COLORS["gold"]),
                            name="Empirical Quantiles",
                        )
                    )
                    min_val = min(theoretical.min(), sorted_returns.min())
                    max_val = max(theoretical.max(), sorted_returns.max())
                    qq_fig.add_trace(
                        go.Scatter(
                            x=[min_val, max_val],
                            y=[min_val, max_val],
                            mode="lines",
                            line=dict(color=COLORS["text_sub"], dash="dash"),
                            name="y = x",
                        )
                    )
                    qq_fig.update_layout(
                        title="Normal QQ Plot",
                        **get_chart_layout(320),
                    )
                    st.plotly_chart(qq_fig, width="stretch")

# ------------------------------------------
# ------------------------------------------
# SCENARIO A: 历史回测 (Backtest)
# ------------------------------------------
if mode == "BACKTEST (Historical)":
    
    # 文件上传区域 (隐藏式设计)
    with st.expander("DATA SOURCE SETTINGS", expanded=True):
        col_file, col_reb = st.columns([2, 1])
        with col_file:
            uploaded_file = st.file_uploader("Upload Market Data (CSV)", type=['csv'], label_visibility="collapsed")
            if not uploaded_file:
                st.caption("Using synthetic demonstration data stream.")
        with col_reb:
            reb_freq = st.number_input("Rebalance Days", 1, 252, 21)
            
        run_bt = st.button("EXECUTE BACKTEST", width="stretch")

    if run_bt:
        with st.spinner("PROCESSING HISTORICAL DATA..."):
            # 调用 Bridge
            market_data = InvestSimBridge.load_market_data(uploaded_file)
            params = {
                "strategy": strategy_name,
                "leverage": leverage,
                "risk_free": risk_free,
                "capital": initial_capital,
                "rebalance_frequency": reb_freq,
                **strategy_params
            }
            bt_res = InvestSimBridge.run_backtest(params, market_data)
            
            # 保存到 Session State 防止刷新丢失
            st.session_state['bt_result'] = bt_res
            
            # 保存历史收益用于 Bootstrap 分布
            if 'Returns' in bt_res.df.columns:
                st.session_state['bootstrap_returns'] = bt_res.df['Returns'].dropna().to_numpy()
            elif 'Portfolio' in bt_res.df.columns:
                # 如果没有 Returns 列，从 Portfolio 计算收益率
                portfolio_returns = bt_res.df['Portfolio'].pct_change().dropna().to_numpy()
                st.session_state['bootstrap_returns'] = portfolio_returns

    # 结果展示
    if 'bt_result' in st.session_state:
        res = st.session_state['bt_result']
        metrics = res.metrics
        
        # HUD 核心指标
        c1, c2, c3, c4 = st.columns(4)
        with c1: render_hud_card("TOTAL RETURN", f"{metrics['total_return']:.2%}", f"CAGR: {metrics.get('annualized_return', 0):.2%}", COLORS['gold'])
        with c2: render_hud_card("SHARPE RATIO", f"{metrics['sharpe']:.2f}")
        with c3: render_hud_card("MAX DRAWDOWN", f"{metrics['max_dd']:.2%}", "Peak-to-Valley", COLORS['red'])
        with c4: render_hud_card("VOLATILITY", f"{metrics['volatility']:.2%}")

        # 主图表
        col_main, col_side = st.columns([3, 1])
        
        with col_main:
            st.markdown("#### NET ASSET VALUE")
            st.plotly_chart(plot_nav_curve(res.df), width="stretch")
        
        with col_side:
            st.markdown("#### DRAWDOWN")
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
                x=res.df.index, y=res.df['Drawdown'],
                fill='tozeroy', line=dict(color=COLORS['red'], width=1),
                fillcolor='rgba(255, 75, 75, 0.1)'
            ))
        fig_dd.update_layout(**get_chart_layout(200))
        fig_dd.update_layout(yaxis=dict(showgrid=False, tickformat=".0%"))
        st.plotly_chart(fig_dd, width="stretch")
            
        st.info("Performance calculated based on daily close prices adjusted for dividends.")

# ------------------------------------------
# SCENARIO B: 蒙特卡洛模拟 (Projection)
# ------------------------------------------
elif mode == "PROJECTION (Monte Carlo)":
    # 模拟设置
    with st.expander("SIMULATION PARAMETERS", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1: sim_years = st.number_input("Horizon (Years)", 1, 50, 10)
        with c2: num_trials = st.number_input("Monte Carlo Trials", 100, 5000, 1000)
        with c3: annual_cont = st.number_input("Annual Contribution", 0, 1000000, 0)
        input_choices = ["Normal", "Student-t", "Bootstrap"]
        default_choice = st.session_state.get("input_model_choice", "Normal")
        if default_choice not in input_choices:
            default_choice = "Normal"
        with c4:
            input_model_type = st.selectbox(
                "Return Dist",
                input_choices,
                index=input_choices.index(default_choice),
            )
        st.session_state["input_model_choice"] = input_model_type
        
        run_mc = st.button("RUN SIMULATION", width="stretch")

    if run_mc:
        with st.spinner("CALCULATING PROBABILITY PATHS..."):
            # 构造 Bridge 参数
            dist_name_map = {
                "Normal": "normal",
                "Student-t": "student_t",
                "Bootstrap": "empirical_bootstrap"
            }
            dist_name = dist_name_map.get(input_model_type, "normal")
            
            # 构建分布参数
            dist_params = {}
            if dist_name == "normal":
                fitted_params = st.session_state.get("fitted_normal_params")
                dist_params = fitted_params or {"mean": 0.0005, "vol": 0.02}
            elif dist_name == "student_t":
                dist_params = {"mean": 0.0, "df": 5.0, "scale": 0.02}
            elif dist_name == "empirical_bootstrap":
                # 尝试从 session_state 获取历史收益
                bootstrap_returns = st.session_state.get("bootstrap_returns")
                if bootstrap_returns is None or len(bootstrap_returns) == 0:
                    st.warning("Bootstrap 模式需要历史收益数据，将使用默认 normal 分布。")
                    dist_name = "normal"
                    dist_params = {"mean": 0.0005, "vol": 0.02}
                else:
                    dist_params = {"historical_returns": bootstrap_returns.tolist()}
            
            input_model_config = {"dist_name": dist_name, "params": dist_params}
            
            params = {
                "strategy": strategy_name,
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
        
        # 分析结果
        final_values = res['paths'][-1]
        median_val = np.median(final_values)
        p05_val = np.percentile(final_values, 5)
        breakeven_balance = initial_capital + annual_cont * sim_years
        gain = (median_val / breakeven_balance) - 1
        
        # HUD 指标
        c1, c2, c3 = st.columns(3)
        with c1: render_hud_card("EXPECTED OUTCOME (MEDIAN)", f"${median_val:,.0f}", f"+{gain:.1%} vs Contributions", COLORS['gold'])
        with c2: render_hud_card("WORST CASE (95% VaR)", f"${p05_val:,.0f}", "Bottom 5% Scenario", COLORS['red'])
        with c3: render_hud_card("SUCCESS PROBABILITY", f"{np.mean(final_values > breakeven_balance):.1%}", "> Initial + Contributions", COLORS['green'])

        # 扇形图
        st.markdown("#### WEALTH PROJECTION CONE")
        st.plotly_chart(plot_monte_carlo_fan(res['dates'], res['paths'], res['median']), width="stretch")
    
        st.warning("DISCLAIMER: Projections are hypothetical and do not guarantee future results.")
        st.caption(describe_input_model(res.get("input_model")))

# ------------------------------------------
# SCENARIO C: Derivatives Lab UI scaffold
# ------------------------------------------
else:
    render_derivatives_lab()