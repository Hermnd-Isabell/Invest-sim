import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta
from bridge import InvestSimBridge  # Import our bridge

# ==========================================
# 1. Config & Style (Quiet Luxury)
# ==========================================
st.set_page_config(page_title="INVEST SIM | PRO", layout="wide", initial_sidebar_state="expanded")

COLORS = {
    "bg": "#0E1117",
    "card": "#1E232D",
    "gold": "#D4AF37",
    "grey": "#8B929E",
    "red": "#FF6B6B",
    "grid": "#262730"
}

st.markdown(f"""
    <style>
        .stApp {{ background-color: {COLORS['bg']}; font-family: 'Helvetica Neue', sans-serif; }}
        [data-testid="stSidebar"] {{ background-color: #161B22; border-right: 1px solid {COLORS['grid']}; }}
        h1, h2, h3, h4 {{ color: #E0E0E0; font-weight: 300 !important; letter-spacing: 1px; }}
        
        /* Metric Card Style */
        div.metric-card {{
            background-color: {COLORS['card']};
            border: 1px solid #303642;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
            transition: all 0.3s ease;
        }}
        div.metric-card:hover {{ border-color: {COLORS['gold']}; transform: translateY(-2px); }}
        .metric-label {{ color: {COLORS['grey']}; font-size: 11px; text-transform: uppercase; letter-spacing: 1.5px; }}
        .metric-value {{ color: #F0F2F6; font-size: 24px; font-weight: 400; margin-top: 5px; }}
        
        /* Button Style */
        .stButton button {{
            background-color: transparent;
            border: 1px solid {COLORS['gold']};
            color: {COLORS['gold']};
            border-radius: 4px;
        }}
        .stButton button:hover {{ background-color: {COLORS['gold']}; color: {COLORS['bg']}; }}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Sidebar Controls
# ==========================================
st.sidebar.title("INVEST SIM")
st.sidebar.caption("Professional Quantitative Suite")

st.sidebar.markdown("### Strategy Configuration")
strat_name = st.sidebar.selectbox("Strategy Type", InvestSimBridge.get_available_strategies())

# Dynamic Inputs based on Strategy
strat_params = {}
if strat_name == "Target Risk":
    strat_params['target_vol'] = st.sidebar.slider("Target Volatility", 0.05, 0.4, 0.15)
elif strat_name == "Adaptive Rebalance":
    strat_params['threshold'] = st.sidebar.slider("Rebalance Threshold", 0.01, 0.10, 0.05)
else:
    st.sidebar.info("Fixed Weights: Defined in Config")

st.sidebar.markdown("---")
st.sidebar.markdown("### Global Settings")
leverage = st.sidebar.slider("Leverage Ratio", 0.5, 3.0, 1.0, 0.1)
risk_free = st.sidebar.number_input("Risk Free Rate", 0.0, 0.1, 0.02)

# ==========================================
# 3. Helper Functions
# ==========================================
def render_metric(label, value, fmt="{:.2%}"):
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{fmt.format(value)}</div>
        </div>
    """, unsafe_allow_html=True)

def plot_fan_chart(dates, paths, median_path):
    # Calculate quantiles
    p95 = np.percentile(paths, 95, axis=1)
    p05 = np.percentile(paths, 5, axis=1)
    
    fig = go.Figure()
    # Fan Area
    fig.add_trace(go.Scatter(
        x=np.concatenate([dates, dates[::-1]]),
        y=np.concatenate([p95, p05[::-1]]),
        fill='toself', fillcolor='rgba(212, 175, 55, 0.15)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% Confidence'
    ))
    # Median
    fig.add_trace(go.Scatter(
        x=dates, y=median_path, mode='lines',
        line=dict(color=COLORS['gold'], width=2),
        name='Median Projection'
    ))
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, linecolor=COLORS['grid']),
        yaxis=dict(showgrid=True, gridcolor=COLORS['grid']),
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right")
    )
    return fig

# ==========================================
# 4. Main Layout
# ==========================================
tab_mc, tab_bt = st.tabs(["🔮 FORWARD SIMULATION", "📜 HISTORICAL BACKTEST"])

# --- Tab 1: Monte Carlo ---
with tab_mc:
    col_in1, col_in2 = st.columns(2)
    with col_in1:
        sim_years = st.slider("Simulation Horizon (Years)", 1, 30, 10)
    with col_in2:
        init_capital = st.number_input("Initial Capital", value=100000)
    
    if st.button("RUN SIMULATION", key="btn_sim"):
        with st.spinner("Running Monte Carlo Engine..."):
            # Call Bridge
            params = {
                "duration": sim_years, 
                "capital": init_capital, 
                "strategy": strat_name,
                "leverage": leverage,
                **strat_params
            }
            res = InvestSimBridge.run_forward_simulation(params)
            
            # Display
            st.markdown("#### Projected Wealth Distribution")
            st.plotly_chart(plot_fan_chart(res['dates'], res['paths'], res['median']), use_container_width=True)
            
            # Metrics
            c1, c2, c3 = st.columns(3)
            final_median = res['median'][-1]
            with c1: render_metric("Expected Final Value", final_median, "${:,.0f}")
            with c2: render_metric("CAGR (Median)", (final_median/init_capital)**(1/sim_years)-1)
            with c3: render_metric("VaR (95%)", init_capital - np.percentile(res['paths'][-1], 5), "${:,.0f}")

# --- Tab 2: Backtest ---
with tab_bt:
    st.markdown("#### Market Data Input")
    uploaded_file = st.file_uploader("Upload Price CSV (Date index, Asset columns)", type=['csv'])
    
    if st.button("RUN BACKTEST", key="btn_bt"):
        with st.spinner("Backtesting Historical Data..."):
            # 1. Load Data
            market_data = InvestSimBridge.load_market_data(uploaded_file)
            
            # 2. Run Backtest via Bridge
            bt_params = {"strategy": strat_name, "leverage": leverage, **strat_params}
            result = InvestSimBridge.run_backtest(bt_params, market_data)
            
            # 3. Metrics Row
            m1, m2, m3, m4 = st.columns(4)
            with m1: render_metric("Total Return", result.metrics['total_return'])
            with m2: render_metric("Sharpe Ratio", result.metrics['sharpe'], "{:.2f}")
            with m3: render_metric("Max Drawdown", result.metrics['max_dd'])
            with m4: render_metric("Volatility", result.metrics['volatility'])
            
            # 4. Charts
            st.markdown("### Net Asset Value")
            fig_nav = go.Figure()
            fig_nav.add_trace(go.Scatter(
                x=result.df.index, y=result.df['Portfolio'],
                line=dict(color=COLORS['gold'], width=2),
                fill='tozeroy', fillcolor='rgba(212, 175, 55, 0.05)',
                name='Portfolio'
            ))
            fig_nav.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False, linecolor=COLORS['grid']),
                yaxis=dict(showgrid=True, gridcolor=COLORS['grid']),
                margin=dict(l=0, r=0, t=10, b=0)
            )
            st.plotly_chart(fig_nav, use_container_width=True)
            
            with st.expander("Drawdown Analysis", expanded=True):
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(
                    x=result.df.index, y=result.df['Drawdown'],
                    line=dict(color=COLORS['red'], width=1),
                    fill='tozeroy', fillcolor='rgba(255, 107, 107, 0.1)',
                    name='Drawdown'
                ))
                fig_dd.update_layout(
                    template="plotly_dark", height=200,
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    yaxis=dict(tickformat=".1%", gridcolor=COLORS['grid']),
                    xaxis=dict(showgrid=False), margin=dict(t=0, b=0)
                )
                st.plotly_chart(fig_dd, use_container_width=True)