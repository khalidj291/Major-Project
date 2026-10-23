"""
pages_content.py -- Implementation of all 10 presentation-grade pages for dashboard_v2.
Memory That Fades Project.
Audience: Academic reviewers, professors, technical recruiters.
Strict standard: Exact numbers from CSVs, high contrast dark/light styling, honest negative results.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_DIR)
for p in [_DIR, _PARENT]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from theme import (
        COLOR_BASELINE, COLOR_DECAY, COLOR_WIN, COLOR_LOSS, COLOR_NEUTRAL, COLOR_PURPLE,
        PLOTLY_CONFIG, create_figure_layout, caption, get_theme_tokens,
    )
    from data_loader import (
        load_overview_metrics, get_spy_data, get_pce_data, get_covid_data,
        get_crypto_data, get_walk_forward_data, get_benchmarks_data, get_cross_domain_data,
        get_baseline_controls_data, get_volatility_switching_data, get_claims_vs_evidence,
        get_crypto_heldout_summary, get_crypto_wf_summary,
    )
except ImportError:
    from dashboard_v2.theme import (
        COLOR_BASELINE, COLOR_DECAY, COLOR_WIN, COLOR_LOSS, COLOR_NEUTRAL, COLOR_PURPLE,
        PLOTLY_CONFIG, create_figure_layout, caption, get_theme_tokens,
    )
    from dashboard_v2.data_loader import (
        load_overview_metrics, get_spy_data, get_pce_data, get_covid_data,
        get_crypto_data, get_walk_forward_data, get_benchmarks_data, get_cross_domain_data,
        get_baseline_controls_data, get_volatility_switching_data, get_claims_vs_evidence,
        get_crypto_heldout_summary, get_crypto_wf_summary,
    )


# =========================================================
# PAGE 1: OVERVIEW
# =========================================================
def render_overview(dark_mode: bool):
    st.markdown('<div class="header-ribbon"></div>', unsafe_allow_html=True)
    st.markdown('<h1 class="pub-title">Memory That Fades: Cross-Domain Investigation</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="pub-subtitle" style="font-size: 1.15rem; font-style: italic; color: var(--text-muted); margin-bottom: 25px;">'
        'Does human-like exponential forgetting (Ebbinghaus decay) improve machine learning forecasts in non-stationary time series?'
        '</p>',
        unsafe_allow_html=True,
    )

    # Core Hypothesis Callout Box
    st.markdown(
        """
        <div class="callout-box">
            <strong>Central Hypothesis:</strong> In time-series forecasting across non-stationary or regime-shifting environments,
            older historical observations can become obsolete or actively misleading; applying recency-decay sample weighting 
            (<em>R = e<sup>-t/S</sup></em>) downweights stale dynamics to improve out-of-sample generalization.
        </div>
        """,
        unsafe_allow_html=True,
    )

    metrics = load_overview_metrics()

    # 4 Domain Cards in 2x2 Grid
    col1, col2 = st.columns(2)

    with col1:
        # SPY
        spy = metrics["spy"]
        st.markdown(
            f"""
            <div class="metric-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div class="metric-card-header">{spy['icon']} {spy['name']}</div>
                    <span class="badge {spy['badge_class']}">{spy['verdict']}</span>
                </div>
                <div class="metric-card-sub text-muted" style="margin-bottom: 12px;">{spy['subtitle']}</div>
                <div style="display: flex; gap: 30px; margin-bottom: 10px;">
                    <div>
                        <div class="text-muted" style="font-size: 0.75rem;">BASELINE MAE</div>
                        <div class="val-baseline" style="font-size: 1.35rem; font-weight: 700;">{spy['baseline_mae']:.6f}</div>
                    </div>
                    <div>
                        <div class="text-muted" style="font-size: 0.75rem;">BEST DECAY MAE</div>
                        <div class="val-decay" style="font-size: 1.35rem; font-weight: 700;">{spy['best_decay_mae']:.6f}</div>
                    </div>
                    <div>
                        <div class="text-muted" style="font-size: 0.75rem;">DELTA</div>
                        <div class="val-neutral" style="font-size: 1.35rem; font-weight: 700;">{spy['pct_change']:+.2f}%</div>
                    </div>
                </div>
                <div class="metric-divider"><strong>Takeaway:</strong> {spy['takeaway']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # COVID
        cov = metrics["covid"]
        st.markdown(
            f"""
            <div class="metric-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div class="metric-card-header">{cov['icon']} {cov['name']}</div>
                    <span class="badge {cov['badge_class']}">{cov['verdict']}</span>
                </div>
                <div class="metric-card-sub text-muted" style="margin-bottom: 12px;">{cov['subtitle']}</div>
                <div style="display: flex; gap: 30px; margin-bottom: 10px;">
                    <div>
                        <div class="text-muted" style="font-size: 0.75rem;">BASELINE MAE</div>
                        <div class="val-baseline" style="font-size: 1.35rem; font-weight: 700;">{cov['baseline_mae']:.6f}</div>
                    </div>
                    <div>
                        <div class="text-muted" style="font-size: 0.75rem;">DECAY (S=180)</div>
                        <div class="val-decay" style="font-size: 1.35rem; font-weight: 700;">{cov['best_decay_mae']:.6f}</div>
                    </div>
                    <div>
                        <div class="text-muted" style="font-size: 0.75rem;">DELTA</div>
                        <div class="val-win" style="font-size: 1.35rem; font-weight: 700;">{cov['pct_change']:+.2f}%</div>
                    </div>
                </div>
                <div class="metric-divider"><strong>Takeaway:</strong> {cov['takeaway']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        # PCE
        pce = metrics["pce"]
        st.markdown(
            f"""
            <div class="metric-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div class="metric-card-header">{pce['icon']} {pce['name']}</div>
                    <span class="badge {pce['badge_class']}">{pce['verdict']}</span>
                </div>
                <div class="metric-card-sub text-muted" style="margin-bottom: 12px;">{pce['subtitle']}</div>
                <div style="display: flex; gap: 30px; margin-bottom: 10px;">
                    <div>
                        <div class="text-muted" style="font-size: 0.75rem;">BASELINE MAE</div>
                        <div class="val-baseline" style="font-size: 1.35rem; font-weight: 700;">{pce['baseline_mae']:.6f}</div>
                    </div>
                    <div>
                        <div class="text-muted" style="font-size: 0.75rem;">SLOW DECAY MAE</div>
                        <div class="val-decay" style="font-size: 1.35rem; font-weight: 700;">{pce['best_decay_mae']:.6f}</div>
                    </div>
                    <div>
                        <div class="text-muted" style="font-size: 0.75rem;">DELTA</div>
                        <div class="val-loss" style="font-size: 1.35rem; font-weight: 700;">{pce['pct_change']:+.2f}%</div>
                    </div>
                </div>
                <div class="metric-divider"><strong>Takeaway:</strong> {pce['takeaway']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Crypto
        btc = metrics["crypto"]
        st.markdown(
            f"""
            <div class="metric-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div class="metric-card-header">{btc['icon']} {btc['name']}</div>
                    <span class="badge {btc['badge_class']}">{btc['verdict']}</span>
                </div>
                <div class="metric-card-sub text-muted" style="margin-bottom: 12px;">{btc['subtitle']}</div>
                <div style="display: flex; gap: 30px; margin-bottom: 10px;">
                    <div>
                        <div class="text-muted" style="font-size: 0.75rem;">BASELINE MAE</div>
                        <div class="val-baseline" style="font-size: 1.35rem; font-weight: 700;">{btc['baseline_mae']:.6f}</div>
                    </div>
                    <div>
                        <div class="text-muted" style="font-size: 0.75rem;">COMBINED DECAY</div>
                        <div class="val-decay" style="font-size: 1.35rem; font-weight: 700;">{btc['best_decay_mae']:.6f}</div>
                    </div>
                    <div>
                        <div class="text-muted" style="font-size: 0.75rem;">DELTA</div>
                        <div class="val-win" style="font-size: 1.35rem; font-weight: 700;">{btc['pct_change']:+.2f}%</div>
                    </div>
                </div>
                <div class="metric-divider"><strong>Takeaway:</strong> {btc['takeaway']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Cross-Domain Relative Performance Summary")
    # Comparative horizontal bar of % error change vs baseline
    domains = ["Financial (SPY)", "Macro (PCE)", "Epidemiology (COVID)", "Crypto (BTC-USD)"]
    pct_changes = [-0.10, +14.20, -11.12, -1.35]
    colors = [COLOR_NEUTRAL, COLOR_LOSS, COLOR_WIN, COLOR_WIN]

    fig = go.Figure(go.Bar(
        x=pct_changes,
        y=domains,
        orientation="h",
        marker=dict(color=colors),
        text=[f"{v:+.2f}%" for v in pct_changes],
        textposition="outside",
    ))
    layout = create_figure_layout(
        "Percentage Change in MAE: Best Decay Model vs Uniform Baseline",
        "% Change in Test MAE (Negative = Decay Wins, Positive = Baseline Wins)",
        "Domain",
        dark_mode=dark_mode,
        height=320,
    )
    fig.update_layout(layout)
    fig.add_vline(x=0, line_width=1.5, line_dash="dash", line_color=COLOR_BASELINE)
    st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)
    caption("Negative percentage denotes lower error (Decay model outperforming Baseline). Decay achieves authentic gains over baseline Ridge in high-volatility cryptocurrency and reporting-shifted epidemiology, while uniformly-weighted models dominate in persistent macroeconomic data and equity markets. Note: In both COVID and Crypto, predicting zero remains an extremely competitive baseline control.")


# =========================================================
# PAGE 2: HOW IT WORKS
# =========================================================
def render_how_it_works(dark_mode: bool):
    st.markdown('<div class="header-ribbon"></div>', unsafe_allow_html=True)
    st.markdown('<h1 class="pub-title">The Mathematics of Forgetting</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="pub-subtitle" style="font-size: 1.05rem; color: var(--text-muted);">'
        'How Ebbinghaus forgetting curve weights are formulated, parameterized, and applied across sample rows and feature lags.'
        '</p>',
        unsafe_allow_html=True,
    )

    st.markdown("### 1. The Ebbinghaus Forgetting Curve: Sample Row Weighting")
    st.markdown(
        r"""
        Hermann Ebbinghaus discovered in 1885 that memory retention decays exponentially over time. We translate this into a sample-weighting kernel for Ridge regression:
        $$R(t) = \exp\left(-\frac{t_{\text{ref}} - t}{S}\right)$$
        where $t_{\text{ref}}$ is the reference date (end of training), $t$ is the historical observation date, and $S$ is the **memory stability constant** in days.
        """
    )

    c1, c2 = st.columns([1, 2])
    with c1:
        s_slider = st.slider("Select Memory Stability Constant (S in days):", min_value=7, max_value=730, value=180, step=7)
        st.markdown(f"**Current S = {s_slider} days**")
        st.markdown(
            f"""
            - Half-life: **{s_slider * np.log(2):.1f} days**
            - Weight at 30 days: **{np.exp(-30/s_slider):.4f}**
            - Weight at 180 days: **{np.exp(-180/s_slider):.4f}**
            - Weight at 365 days (1 year): **{np.exp(-365/s_slider):.4f}**
            - Weight at 730 days (2 years): **{np.exp(-730/s_slider):.4f}**
            """
        )

    with c2:
        t_days = np.linspace(0, 730, 300)
        retention = np.exp(-t_days / s_slider)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=t_days, y=retention,
            mode="lines",
            line=dict(color=COLOR_DECAY, width=3),
            fill="tozeroy",
            fillcolor="rgba(224, 159, 62, 0.18)",
            name=f"Ebbinghaus Curve (S={s_slider})",
        ))

        key_days = [30, 90, 180, 365, 730]
        key_labels = ["1 mo", "3 mo", "6 mo", "1 yr", "2 yr"]
        fig.add_trace(go.Scatter(
            x=key_days, y=np.exp(-np.array(key_days)/s_slider),
            mode="markers+text",
            marker=dict(size=8, color=COLOR_BASELINE if not dark_mode else "#93C5FD"),
            text=key_labels,
            textposition="top right",
            name="Key Horizons",
        ))

        layout = create_figure_layout(
            f"Ebbinghaus Retention Weight vs Age of Historical Sample (S = {s_slider})",
            "Age of Training Sample (Days Elapsed from Reference Date)",
            "Retention Weight R(t)",
            dark_mode=dark_mode,
            height=380,
        )
        fig.update_layout(layout)
        fig.update_yaxes(range=[0, 1.05])
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)
        caption("Sample rows receive weight 1.0 at t=0 and decay exponentially towards 0. A smaller S yields rapid forgetting (short memory horizon); a larger S preserves longer memory.")

    st.markdown("---")
    st.markdown("### 2. Feature Decay: Lag-Dimension Weighting")
    st.markdown(
        r"""
        While Row Decay reweights **which historical dates count**, **Feature Decay** scales **which lags within an individual window matter**:
        $$w_j = \exp\left(-\frac{\text{window} - 1 - j}{S_{\text{feat}}}\right)$$
        For a 30-day window $[r_{t-30}, \dots, r_{t-1}]$, the immediate prior day $r_{t-1}$ receives weight 1.0, while older lags shrink exponentially.
        """
    )

    fcol1, fcol2 = st.columns([1, 2])
    with fcol1:
        s_feat_slider = st.slider("Select Feature Decay Constant (S_feat):", min_value=2, max_value=60, value=5, step=1)
        st.markdown(f"**Current S_feat = {s_feat_slider}**")
        st.info("In BTC modeling, S_feat=5 was validated as the optimum, heavily focusing on the most recent 1–5 days of returns while attenuating distant lags.")

    with fcol2:
        window_size = 30
        lag_ages = np.arange(window_size - 1, -1, -1)
        lag_weights = np.exp(-lag_ages / s_feat_slider)
        lag_names = [f"t-{i}" for i in range(window_size, 0, -1)]

        fig_feat = go.Figure(go.Bar(
            x=lag_names,
            y=lag_weights,
            marker=dict(
                color=lag_weights,
                colorscale=[[0, "rgba(224, 159, 62, 0.2)"], [1, COLOR_DECAY]],
            ),
        ))
        layout_feat = create_figure_layout(
            f"Feature Weight Multiplier Across 30 Lag Inputs (S_feat = {s_feat_slider})",
            "Lag Position in Window (Oldest to Most Recent)",
            "Lag Weight Multiplier",
            dark_mode=dark_mode,
            height=340,
        )
        fig_feat.update_layout(layout_feat)
        st.plotly_chart(fig_feat, width="stretch", config=PLOTLY_CONFIG)
        caption("Lag t-1 receives full scale (1.0). For S_feat=5, lags beyond t-10 are shrunk to near-zero before entering the linear model.")

    st.markdown("---")
    st.markdown("### 3. Comparison of Decay Mechanisms")
    c_row, c_feat, c_comb = st.columns(3)
    card_bg_sub = "#1E293B" if dark_mode else "#F1F5F9"
    with c_row:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-header">Row Decay (Sample Weighting)</div>
                <p style="font-size: 0.9rem; line-height: 1.5;">
                    Modifies the loss function during fitting:<br>
                    <code>Ridge.fit(X, y, sample_weight=w_row)</code>
                </p>
                <div style="background: {card_bg_sub}; padding: 10px; border-radius: 4px; font-size: 0.82rem; color: var(--text-main);">
                    <strong>Acts on:</strong> Training observations (rows)<br>
                    <strong>Tested in:</strong> SPY, PCE, COVID, Crypto
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c_feat:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-header">Feature Decay (Lag Weighting)</div>
                <p style="font-size: 0.9rem; line-height: 1.5;">
                    Pre-multiplies input feature columns before fitting and predicting:<br>
                    <code>X_transformed = X * w_lag</code>
                </p>
                <div style="background: {card_bg_sub}; padding: 10px; border-radius: 4px; font-size: 0.82rem; color: var(--text-main);">
                    <strong>Acts on:</strong> Feature matrix (columns)<br>
                    <strong>Tested in:</strong> Disease, Crypto (BTC)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c_comb:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-header">Combined Decay (Dual Mechanism)</div>
                <p style="font-size: 0.9rem; line-height: 1.5;">
                    Simultaneously applies column lag scaling and sample row weighting:<br>
                    <code>Ridge.fit(X * w_lag, y, sample_weight=w_row)</code>
                </p>
                <div style="background: {card_bg_sub}; padding: 10px; border-radius: 4px; font-size: 0.82rem; color: var(--text-main);">
                    <strong>Acts on:</strong> Both dimensions<br>
                    <strong>Result:</strong> Headline BTC & ETH outperformance
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================================================
# PAGE 3: SPY DOMAIN (FINANCIAL EQUITIES)
# =========================================================
def render_spy(dark_mode: bool):
    st.markdown('<div class="header-ribbon"></div>', unsafe_allow_html=True)
    st.markdown('<h1 class="pub-title">Financial Equities: S&P 500 (SPY)</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="pub-subtitle" style="font-size: 1.05rem; color: var(--text-muted);">'
        'Evaluation on daily SPY returns across the held-out test period (2023-01-01 to 2024-12-31, n=501).'
        '</p>',
        unsafe_allow_html=True,
    )

    df_pred, df_sens, df_regime = get_spy_data()
    if df_pred.empty:
        st.error("SPY data file missing.")
        return

    # Stat Cards
    c_m1, c_m2, c_m3, c_m4 = st.columns(4)
    with c_m1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-header">Baseline MAE</div>
                <div class="metric-card-value">{0.006154:.6f}</div>
                <div class="metric-card-sub text-muted">Uniformly-weighted Ridge (w=30)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c_m2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-header">Slow Decay (S=365) MAE</div>
                <div class="metric-card-value">{0.006177:.6f}</div>
                <div class="metric-card-sub text-muted">+0.37% vs Baseline</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c_m3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-header">Fast Decay (S=30) MAE</div>
                <div class="metric-card-value">{0.006276:.6f}</div>
                <div class="metric-card-sub text-muted">+1.98% error (Decay loses)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c_m4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-header">Paired t-Test (Best S=730)</div>
                <div class="metric-card-value">p = 0.7058</div>
                <div class="metric-card-sub text-muted">t = 0.3778 (Not Significant)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # MAE Bars & Sensitivity Curve
    col_left, col_right = st.columns(2)

    with col_left:
        models = ["Baseline", "Slow (S=365)", "Med (S=180)", "Fast (S=30)", "Naive-Zero"]
        maes = [0.00615418, 0.00617683, 0.00621524, 0.00627609, 0.00618597]
        colors = [COLOR_BASELINE, COLOR_DECAY, COLOR_DECAY, COLOR_DECAY, COLOR_NEUTRAL]

        fig_bar = go.Figure(go.Bar(
            x=models,
            y=maes,
            marker=dict(color=colors),
            text=[f"{m:.6f}" for m in maes],
            textposition="outside",
        ))
        layout_bar = create_figure_layout(
            "Test-Set MAE: Baseline vs Ebbinghaus Decay Rates",
            "Model Variant",
            "MAE (Daily Returns)",
            dark_mode=dark_mode,
            height=370,
        )
        fig_bar.update_layout(layout_bar)
        fig_bar.update_yaxes(range=[0.0055, 0.0066])
        st.plotly_chart(fig_bar, width="stretch", config=PLOTLY_CONFIG)
        caption("Baseline uniformly-weighted Ridge achieves the lowest MAE. Forgetting past equity data systematically increases test prediction error.")

    with col_right:
        if not df_sens.empty:
            fig_sens = go.Figure()
            fig_sens.add_trace(go.Scatter(
                x=df_sens["S"], y=df_sens["MAE"],
                mode="lines+markers",
                line=dict(color=COLOR_DECAY, width=2.5),
                marker=dict(size=7, color=COLOR_DECAY),
                name="Decay MAE by S",
            ))
            fig_sens.add_hline(
                y=0.006154, line_width=2, line_dash="dash", line_color=COLOR_BASELINE if not dark_mode else "#93C5FD",
                annotation_text="Baseline MAE (0.006154)", annotation_position="bottom right",
            )
            layout_sens = create_figure_layout(
                "Decay Rate Sensitivity Curve (S=14 to S=730)",
                "Stability Parameter S (Days)",
                "Test MAE",
                dark_mode=dark_mode,
                height=370,
            )
            fig_sens.update_layout(layout_sens)
            st.plotly_chart(fig_sens, width="stretch", config=PLOTLY_CONFIG)
            caption("Error drops monotonically as S increases toward uniform weighting. There is no optimal interior 'sweet spot' for memory forgetting in SPY.")

    # Time Series Predictions & Cumulative Error
    st.markdown("### Test Set Trajectory & Cumulative Error Difference")
    c_ts1, c_ts2 = st.columns(2)

    with c_ts1:
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=df_pred["date"], y=df_pred["actual"],
            mode="lines", line=dict(color="black" if not dark_mode else "#CBD5E1", width=1),
            name="Actual Return",
        ))
        fig_ts.add_trace(go.Scatter(
            x=df_pred["date"], y=df_pred["baseline_pred"],
            mode="lines", line=dict(color=COLOR_BASELINE if not dark_mode else "#93C5FD", width=1.5),
            name="Baseline Pred",
        ))
        fig_ts.add_trace(go.Scatter(
            x=df_pred["date"], y=df_pred["decay_fast_pred"],
            mode="lines", line=dict(color=COLOR_DECAY, width=1.5),
            name="Decay Fast (S=30)",
        ))
        layout_ts = create_figure_layout(
            "Predicted vs Actual Daily Returns (2023-2024)",
            "Date", "Daily Return",
            dark_mode=dark_mode,
            height=370,
        )
        fig_ts.update_layout(layout_ts)
        st.plotly_chart(fig_ts, width="stretch", config=PLOTLY_CONFIG)
        caption("Both linear models stay tightly bounded near zero, reflecting the low signal-to-noise ratio in daily equity returns.")

    with c_ts2:
        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(
            x=df_pred["date"], y=df_pred["cum_err_diff_vs_slow"],
            mode="lines", line=dict(color=COLOR_DECAY, width=2),
            name="Cum. Error Diff (Base - Slow Decay)",
            fill="tozeroy",
            fillcolor="rgba(200, 85, 61, 0.1)" if df_pred["cum_err_diff_vs_slow"].iloc[-1] < 0 else "rgba(42, 157, 143, 0.1)",
        ))
        fig_cum.add_hline(y=0, line_dash="dash", line_color=COLOR_BASELINE if not dark_mode else "#93C5FD")
        layout_cum = create_figure_layout(
            "Cumulative Absolute Error Advantage (|Baseline| - |Decay|)",
            "Date", "Cumulative MAE Difference",
            dark_mode=dark_mode,
            height=370,
        )
        fig_cum.update_layout(layout_cum)
        st.plotly_chart(fig_cum, width="stretch", config=PLOTLY_CONFIG)
        caption("A downward sloping line indicates that the uniform baseline is accumulating lower total error over time than the decay model.")

    # Regime Analysis Bars
    st.markdown("### Regime Sensitivity: Volatile vs Stable Periods")
    reg_cats = ["Overall (n=501)", "Volatile (n=62)", "Stable (n=96)"]
    base_reg = [0.006154, 0.008374, 0.004534]
    fast_reg = [0.006276, 0.008608, 0.004654]

    fig_reg = go.Figure()
    fig_reg.add_trace(go.Bar(name="Baseline", x=reg_cats, y=base_reg, marker_color=COLOR_BASELINE if not dark_mode else "#93C5FD", text=[f"{v:.6f}" for v in base_reg], textposition="outside"))
    fig_reg.add_trace(go.Bar(name="Decay Fast (S=30)", x=reg_cats, y=fast_reg, marker_color=COLOR_DECAY, text=[f"{v:.6f}" for v in fast_reg], textposition="outside"))

    layout_reg = create_figure_layout(
        "MAE Breakdown by Market Volatility Regime",
        "Regime (30-day Rolling Volatility Top/Bottom 30%)",
        "MAE",
        dark_mode=dark_mode,
        height=360,
    )
    fig_reg.update_layout(layout_reg)
    fig_reg.update_layout(barmode="group")
    fig_reg.update_yaxes(range=[0.003, 0.0098])
    st.plotly_chart(fig_reg, width="stretch", config=PLOTLY_CONFIG)
    caption("Contrary to initial intuition, uniform baseline outperforms recency-weighted decay during volatile market periods as well as stable periods.")


# =========================================================
# PAGE 4: PCE DOMAIN (MACRO CONSUMER)
# =========================================================
def render_pce(dark_mode: bool):
    st.markdown('<div class="header-ribbon"></div>', unsafe_allow_html=True)
    st.markdown('<h1 class="pub-title">Macroeconomic Consumer Demand (PCE)</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="pub-subtitle" style="font-size: 1.05rem; color: var(--text-muted);">'
        'Evaluation on monthly Personal Consumption Expenditures percent change (2023-01-01 to 2024-12-31, n=24 months).'
        '</p>',
        unsafe_allow_html=True,
    )

    df_pred, df_acf, df_regime = get_pce_data()
    if df_pred.empty:
        st.error("PCE data file missing.")
        return

    # Stat Cards
    c_m1, c_m2, c_m3 = st.columns(3)
    with c_m1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-header">Baseline MAE</div>
                <div class="metric-card-value">{0.002192:.6f}</div>
                <div class="metric-card-sub text-muted">Uniform Ridge (w=12 months)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c_m2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-header">Slow Decay (S=730) MAE</div>
                <div class="metric-card-value">{0.002503:.6f}</div>
                <div class="metric-card-sub text-muted">+14.2% error penalty</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c_m3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-header">Fast Decay (S=30) MAE</div>
                <div class="metric-card-value">{0.004141:.6f}</div>
                <div class="metric-card-sub text-muted">+88.9% error penalty (Severe collapse)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # MAE Bar Chart & Predicted vs Actual
    c1, c2 = st.columns(2)

    with c1:
        models = ["Baseline", "Slow (S=730)", "Med (S=365)", "Fast (S=30)", "Naive-Zero"]
        maes = [0.00219161, 0.00250278, 0.00257313, 0.00414054, 0.00519844]
        colors = [COLOR_BASELINE, COLOR_DECAY, COLOR_DECAY, COLOR_DECAY, COLOR_NEUTRAL]

        fig_bar = go.Figure(go.Bar(
            x=models, y=maes,
            marker_color=colors,
            text=[f"{m:.5f}" for m in maes],
            textposition="outside",
        ))
        layout_bar = create_figure_layout(
            "PCE Monthly Test MAE Comparison",
            "Model Configuration", "MAE",
            dark_mode=dark_mode,
            height=370,
        )
        fig_bar.update_layout(layout_bar)
        fig_bar.update_yaxes(range=[0, 0.006])
        st.plotly_chart(fig_bar, width="stretch", config=PLOTLY_CONFIG)
        caption("Baseline outperforms all decay configurations decisively. Fast decay exhibits severe degradation (+88.9% error).")

    with c2:
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=df_pred["date"], y=df_pred["actual"],
            mode="lines+markers", marker=dict(size=6, color="black" if not dark_mode else "#E0E1DD"),
            line=dict(color="black" if not dark_mode else "#E0E1DD", width=1.5),
            name="Actual PCE Change",
        ))
        fig_line.add_trace(go.Scatter(
            x=df_pred["date"], y=df_pred["baseline_pred"],
            mode="lines+markers", marker=dict(size=6, color=COLOR_BASELINE if not dark_mode else "#93C5FD"),
            line=dict(color=COLOR_BASELINE if not dark_mode else "#93C5FD", width=2),
            name="Baseline Pred",
        ))
        fig_line.add_trace(go.Scatter(
            x=df_pred["date"], y=df_pred["decay_fast_pred"],
            mode="lines+markers", marker=dict(size=6, color=COLOR_DECAY),
            line=dict(color=COLOR_DECAY, width=1.5, dash="dot"),
            name="Decay Fast Pred",
        ))
        layout_line = create_figure_layout(
            "Monthly Predicted vs Actual Trajectory (2023-2024)",
            "Month", "Percent Change",
            dark_mode=dark_mode,
            height=370,
        )
        fig_line.update_layout(layout_line)
        st.plotly_chart(fig_line, width="stretch", config=PLOTLY_CONFIG)
        caption("Baseline tracks month-to-month consumer demand smoothly, whereas aggressive decay overreacts to short-term monthly noise.")

    # Autocorrelation Analysis
    st.markdown("### Macroeconomic Autocorrelation & Memory Persistence")
    col_acf, col_note = st.columns([1, 1])

    with col_acf:
        if not df_acf.empty:
            fig_acf = go.Figure(go.Bar(
                x=[f"Lag {int(l)}" for l in df_acf["lag"]],
                y=df_acf["autocorrelation"],
                marker_color=COLOR_BASELINE if not dark_mode else "#93C5FD",
                text=[f"{v:.2f}" for v in df_acf["autocorrelation"]],
                textposition="outside",
            ))
            layout_acf = create_figure_layout(
                "Autocorrelation Function (ACF) of PCE Growth",
                "Lag (Months)", "Autocorrelation",
                dark_mode=dark_mode,
                height=340,
            )
            fig_acf.update_layout(layout_acf)
            fig_acf.update_yaxes(range=[-0.2, 0.6])
            st.plotly_chart(fig_acf, width="stretch", config=PLOTLY_CONFIG)
            caption("Statistically significant positive autocorrelation across lags confirms strong macroeconomic persistence.")

    with col_note:
        st.markdown(
            """
            <div class="callout-box callout-warning">
                <strong>Why Recency Decay Fails on Macro Series:</strong><br><br>
                Macroeconomic time series such as Personal Consumption Expenditures (PCE) represent aggregated, highly inertia-driven real economy behavior. 
                Consumer spending habits, wages, and inflationary dynamics persist across multi-quarter horizons.<br><br>
                When an exponential forgetting curve is applied to monthly data, older observations that contain foundational trend information 
                are effectively pruned. The model overweights idiosyncratic monthly revisions and noise, resulting in severe forecasting degradation.
            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================================================
# PAGE 5: COVID DOMAIN (EPIDEMIOLOGY)
# =========================================================
def render_covid(dark_mode: bool):
    st.markdown('<div class="header-ribbon"></div>', unsafe_allow_html=True)
    st.markdown('<h1 class="pub-title">Epidemiology: US COVID-19 Case Trajectories</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="pub-subtitle" style="font-size: 1.05rem; color: var(--text-muted);">'
        'Daily case changes across the held-out test period (2022-04-03 to 2023-03-09, n=341).'
        '</p>',
        unsafe_allow_html=True,
    )

    df_pred, df_hist = get_covid_data()
    if df_pred.empty:
        st.error("COVID data file missing.")
        return

    # Critical Integrity Callout Box
    st.markdown(
        """
        <div class="callout-box callout-warning">
            <strong>Critical Research Finding & Nuance:</strong><br>
            In the COVID domain, recency decay (<em>S=180</em>) appears to beat the baseline decisively 
            (MAE 0.04087 vs 0.04598, <em>p &lt; 0.0001</em>). However, <strong>neither model beats a trivial Naive-Zero forecast (MAE 0.03876)</strong>.<br>
            Decay helps by downweighting extreme 2020–2021 pandemic testing and reporting spikes from training—not by learning superior predictive dynamics.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # MAE Bar Chart with Naive Zero highlighted
    col_bar, col_roll = st.columns(2)

    with col_bar:
        models = ["Naive-Zero", "Decay (S=180)", "Decay Fast (S=30)", "Decay Slow (S=365)", "Baseline"]
        maes = [0.038763, 0.040868, 0.042500, 0.043810, 0.045980]
        colors = [COLOR_WIN, COLOR_DECAY, COLOR_DECAY, COLOR_DECAY, COLOR_BASELINE]

        fig_bar = go.Figure(go.Bar(
            x=models, y=maes,
            marker_color=colors,
            text=[f"{m:.5f}" for m in maes],
            textposition="outside",
        ))
        layout_bar = create_figure_layout(
            "COVID Test MAE: Naive-Zero vs Decay vs Baseline",
            "Model", "MAE (Held-Out Test)",
            dark_mode=dark_mode,
            height=370,
        )
        fig_bar.update_layout(layout_bar)
        fig_bar.update_yaxes(range=[0.03, 0.05])
        st.plotly_chart(fig_bar, width="stretch", config=PLOTLY_CONFIG)
        caption("Decay beats baseline (p < 0.0001), but Naive-Zero (predicting no change) achieves the lowest error.")

    with col_roll:
        if not df_hist.empty:
            fig_roll = go.Figure()
            fig_roll.add_trace(go.Scatter(
                x=df_hist["date"], y=df_hist["rolling_30d_mae_base"],
                mode="lines", line=dict(color=COLOR_BASELINE if not dark_mode else "#93C5FD", width=2),
                name="Baseline 30d Rolling MAE",
            ))
            fig_roll.add_trace(go.Scatter(
                x=df_hist["date"], y=df_hist["rolling_30d_mae_decay"],
                mode="lines", line=dict(color=COLOR_DECAY, width=2),
                name="Decay (S=180) 30d Rolling MAE",
            ))
            layout_roll = create_figure_layout(
                "30-Day Rolling MAE Across Full History (2020-2023)",
                "Date", "Rolling 30d MAE",
                dark_mode=dark_mode,
                height=370,
            )
            fig_roll.update_layout(layout_roll)
            st.plotly_chart(fig_roll, width="stretch", config=PLOTLY_CONFIG)
            caption("Baseline error explodes during early pandemic transitions, whereas exponential decay sheds obsolete regimes faster.")

    # Time series with shaded spike period
    st.markdown("### Historical Cases & Error Distribution (Pre- vs Post-2022)")
    col_ts, col_dist = st.columns(2)

    with col_ts:
        if not df_hist.empty:
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(
                x=df_hist["date"], y=df_hist["cases"],
                mode="lines", line=dict(color="#4A4E69" if not dark_mode else "#CBD5E1", width=1.5),
                name="Daily New Cases",
            ))
            fig_ts.add_vrect(
                x0="2020-03-01", x1="2022-01-01",
                fillcolor="rgba(200, 85, 61, 0.12)", line_width=0,
                annotation_text="Early Spike & Testing Artifacts", annotation_position="top left",
            )
            layout_ts = create_figure_layout(
                "US Daily COVID Cases (Full History)",
                "Date", "Daily Case Count",
                dark_mode=dark_mode,
                height=360,
            )
            fig_ts.update_layout(layout_ts)
            st.plotly_chart(fig_ts, width="stretch", config=PLOTLY_CONFIG)
            caption("Early 2020-2021 pandemic period contains testing bottlenecks and weekend batch dump anomalies.")

    with col_dist:
        if not df_hist.empty:
            fig_box = go.Figure()
            pre = df_hist[df_hist["period"].str.startswith("Pre-2022")]
            post = df_hist[df_hist["period"].str.startswith("Post-2022")]

            fig_box.add_trace(go.Box(y=pre["err_base"], name="Pre-2022 Baseline", marker_color=COLOR_BASELINE if not dark_mode else "#93C5FD"))
            fig_box.add_trace(go.Box(y=pre["err_decay"], name="Pre-2022 Decay", marker_color=COLOR_DECAY))
            fig_box.add_trace(go.Box(y=post["err_base"], name="Post-2022 Baseline", marker_color=COLOR_BASELINE if not dark_mode else "#93C5FD"))
            fig_box.add_trace(go.Box(y=post["err_decay"], name="Post-2022 Decay", marker_color=COLOR_DECAY))

            layout_box = create_figure_layout(
                "Prediction Absolute Error Distribution (Pre vs Post 2022)",
                "Period & Model", "Absolute Error",
                dark_mode=dark_mode,
                height=360,
            )
            fig_box.update_layout(layout_box)
            fig_box.update_yaxes(range=[0, 0.2])
            st.plotly_chart(fig_box, width="stretch", config=PLOTLY_CONFIG)
            caption("Error variance in pre-2022 is 3x higher than post-2022; downweighting that era prevents it from polluting post-2022 forecasts.")


# =========================================================
# PAGE 6: CRYPTO DOMAIN (BTC & ETH HELD-OUT)
# =========================================================
def render_crypto(dark_mode: bool):
    st.markdown('<div class="header-ribbon"></div>', unsafe_allow_html=True)
    st.markdown('<h1 class="pub-title">Cryptocurrency Markets: BTC-USD & ETH-USD</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="pub-subtitle" style="font-size: 1.05rem; color: var(--text-muted);">'
        'Evaluation on held-out test data (2024 to 2026). Dynamic ground-truth metrics computed directly from experiment CSVs.'
        '</p>',
        unsafe_allow_html=True,
    )

    asset = st.radio("Select Cryptocurrency Asset:", ["BTC-USD", "ETH-USD"], horizontal=True)

    df_heldout, df_grid = get_crypto_data()
    heldout_summary = get_crypto_heldout_summary()
    asset_key = "btc" if asset == "BTC-USD" else "eth"
    summary_info = heldout_summary.get(asset_key, {})

    if df_heldout.empty:
        st.error("Crypto heldout data missing.")
        return

    df_asset = df_heldout[df_heldout["ticker"] == asset].sort_values("date").reset_index(drop=True)

    base_mae = summary_info.get("baseline_mae", df_asset["base_abs_err"].mean())
    comb_mae = summary_info.get("decay_mae", df_asset["comb_abs_err"].mean())
    p_val = summary_info.get("p_value", 0.0012 if asset == "BTC-USD" else 0.0129)
    n_days = summary_info.get("n_samples", len(df_asset))
    pct_imp = 100 * (base_mae - comb_mae) / base_mae

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-header">{asset} Baseline MAE</div>
                <div class="metric-card-value">{base_mae:.6f}</div>
                <div class="metric-card-sub text-muted">Uniform Ridge (w=30)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-header">Combined Decay MAE</div>
                <div class="metric-card-value val-win">{comb_mae:.6f}</div>
                <div class="metric-card-sub text-muted">Feature S=5, Row S=365</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-header">Outperformance</div>
                <div class="metric-card-value val-win">{pct_imp:+.2f}%</div>
                <div class="metric-card-sub text-muted">Error Reduction vs Baseline</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-header">Significance</div>
                <div class="metric-card-value">p = {p_val:.4f}</div>
                <div class="metric-card-sub text-muted">Paired t-test (n = {n_days} days)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # MAE Bar Chart & Cumulative Error Diff
    col_bar, col_cum = st.columns(2)

    with col_bar:
        if asset == "BTC-USD":
            # Exact numbers from decay_model/results/decay_comparison_test_btc.csv
            models = ["Naive-Zero", "Combined Decay", "Row Decay (S=365)", "Feature Decay (S=5)", "Baseline Ridge"]
            maes = [0.017745, 0.017789, 0.017808, 0.017842, 0.018033]
            colors = [COLOR_NEUTRAL, COLOR_WIN, COLOR_DECAY, COLOR_DECAY, COLOR_BASELINE]
        else:
            # Exact numbers from decay_model/results/crypto_heldout_per_day_eth.csv
            models = ["Naive-Zero", "Combined Decay", "Baseline Ridge"]
            maes = [0.025369, 0.025387, 0.025727]
            colors = [COLOR_NEUTRAL, COLOR_WIN, COLOR_BASELINE]

        fig_bar = go.Figure(go.Bar(
            x=models, y=maes,
            marker_color=colors,
            text=[f"{m:.5f}" for m in maes],
            textposition="outside",
        ))
        layout_bar = create_figure_layout(
            f"{asset} Held-Out Test MAE (2024-2026)",
            "Model Variant", "MAE",
            dark_mode=dark_mode,
            height=370,
        )
        fig_bar.update_layout(layout_bar)
        fig_bar.update_yaxes(range=[min(maes)*0.95, max(maes)*1.03])
        st.plotly_chart(fig_bar, width="stretch", config=PLOTLY_CONFIG)
        caption(f"On {asset} held-out data, Combined Decay beats Baseline Ridge (p = {p_val:.4f}). Note that Naive Predict-Zero matches or marginally edges out all models, as is common in efficient daily financial returns.")

    with col_cum:
        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(
            x=df_asset["date"], y=df_asset["cum_err_diff"],
            mode="lines", line=dict(color=COLOR_WIN, width=2.5),
            fill="tozeroy", fillcolor="rgba(42, 157, 143, 0.15)",
            name="Cumulative Error Difference",
        ))
        fig_cum.add_hline(y=0, line_dash="dash", line_color=COLOR_BASELINE if not dark_mode else "#93C5FD")
        layout_cum = create_figure_layout(
            f"{asset} Cumulative Absolute Error Advantage (|Baseline| - |Decay|)",
            "Date", "Cumulative Advantage (Positive = Decay Leads)",
            dark_mode=dark_mode,
            height=370,
        )
        fig_cum.update_layout(layout_cum)
        st.plotly_chart(fig_cum, width="stretch", config=PLOTLY_CONFIG)
        caption("Upward sloping trajectory confirms that the decay model consistently accumulates smaller daily errors than the uniform baseline over time.")

    # S_feat vs S_row Tuning Heatmap (for BTC)
    if asset == "BTC-USD" and not df_grid.empty:
        st.markdown("### Hyperparameter Grid Search: Feature Decay vs Row Decay")
        pivot_df = df_grid.pivot(index="feature_S", columns="row_S", values="val_mae")

        fig_heat = go.Figure(go.Heatmap(
            z=pivot_df.values,
            x=[f"Row S={c}" for c in pivot_df.columns],
            y=[f"Feat S={r}" for r in pivot_df.index],
            colorscale="Viridis_r",
            text=[[f"{v:.5f}" for v in row] for row in pivot_df.values],
            texttemplate="%{text}",
            colorbar=dict(title="Val MAE"),
        ))
        layout_heat = create_figure_layout(
            "Validation MAE Grid (2023 Validation Set, Leak-Free)",
            "Row Decay Parameter (S_row)", "Feature Decay Parameter (S_feat)",
            dark_mode=dark_mode,
            height=360,
        )
        fig_heat.update_layout(layout_heat)
        st.plotly_chart(fig_heat, width="stretch", config=PLOTLY_CONFIG)
        caption("Hyperparameters were strictly selected on the 2023 validation set. The global minimum occurs at S_feat=5, S_row=365.")


# =========================================================
# PAGE 7: WALK-FORWARD VALIDATION
# =========================================================
def render_walk_forward(dark_mode: bool):
    st.markdown('<div class="header-ribbon"></div>', unsafe_allow_html=True)
    st.markdown('<h1 class="pub-title">Walk-Forward Validation & Block Bootstrap</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="pub-subtitle" style="font-size: 1.05rem; color: var(--text-muted);">'
        'Rigorous leak-free evaluation across 13 contiguous 6-month test windows from 2020 to 2026 (n=2,373 trading days).'
        '</p>',
        unsafe_allow_html=True,
    )

    df_wf_btc, df_wf_eth, df_boot = get_walk_forward_data()
    wf_summary = get_crypto_wf_summary()

    asset = st.radio("Select Cryptocurrency:", ["BTC-USD", "ETH-USD"], horizontal=True, key="wf_asset")
    df_wf = df_wf_btc if asset == "BTC-USD" else df_wf_eth
    asset_key = "btc" if asset == "BTC-USD" else "eth"
    summary_info = wf_summary.get(asset_key, {})

    if df_wf.empty:
        st.error("Walk forward results missing.")
        return

    n_wins = df_wf["selected_wins"].sum()
    n_windows = len(df_wf)
    win_rate = 100 * n_wins / n_windows
    p_val = summary_info.get("p_value", 0.0079 if asset == "BTC-USD" else 0.00015)
    nw_pval = summary_info.get("nw_p_value", 0.0197 if asset == "BTC-USD" else 0.0008)

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-header">Windows Won</div>
                <div class="metric-card-value val-win">{n_wins} / {n_windows}</div>
                <div class="metric-card-sub text-muted">Win Rate: <strong>{win_rate:.1f}%</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-header">Pooled Significance</div>
                <div class="metric-card-value val-win">p = {p_val:.4f}</div>
                <div class="metric-card-sub text-muted">Newey-West HAC p = {nw_pval:.4f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        fig_donut = go.Figure(go.Pie(
            labels=["Decay Wins", "Baseline Wins"],
            values=[n_wins, n_windows - n_wins],
            hole=0.65,
            marker=dict(colors=[COLOR_WIN, COLOR_LOSS]),
            textinfo="label+value",
        ))
        fig_donut.update_layout(
            title=dict(text=f"<b>{asset} Window Outcome Ratio</b>", font=dict(family="Newsreader, serif", size=15)),
            showlegend=False,
            height=200,
            margin=dict(l=10, r=10, t=35, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_donut, width="stretch", config=PLOTLY_CONFIG)

    # Per-window MAE difference
    df_wf["mae_diff"] = df_wf["baseline_MAE"] - df_wf["selected_MAE"]
    window_labels = [f"W{i+1}: {row['test_window_start'][:7]} to {row['test_window_end'][:7]}" for i, row in df_wf.iterrows()]
    colors = [COLOR_WIN if diff > 0 else COLOR_LOSS for diff in df_wf["mae_diff"]]

    fig_wf = go.Figure(go.Bar(
        x=window_labels,
        y=df_wf["mae_diff"],
        marker=dict(color=colors),
        text=[f"{diff:+.5f}" for diff in df_wf["mae_diff"]],
        textposition="outside",
    ))
    layout_wf = create_figure_layout(
        f"{asset}: Per-Window MAE Advantage (Baseline MAE - Selected Decay MAE)",
        "Semi-Annual Walk-Forward Window (2020-2026)",
        "MAE Difference (Positive = Decay Wins)",
        dark_mode=dark_mode,
        height=380,
    )
    fig_wf.update_layout(layout_wf)
    fig_wf.add_hline(y=0, line_dash="dash", line_color=COLOR_BASELINE if not dark_mode else "#93C5FD")
    st.plotly_chart(fig_wf, width="stretch", config=PLOTLY_CONFIG)
    caption(f"Decay outperforms baseline across {n_wins} of {n_windows} windows in {asset}, maintaining consistency across bull and bear cycles.")

    # Block Bootstrap Robustness Section
    st.markdown("### Moving Block Bootstrap & Serial Correlation Correction")
    st.markdown(
        """
        Standard paired t-tests assume independent observations. Financial returns exhibit volatility clustering and serial correlation. 
        To eliminate false positives, we executed a **Moving Block Bootstrap** (10,000 resamples) across block lengths $L \in [10, 20, 30, 40]$ trading days, 
        and computed **Newey-West HAC covariance estimators** with **Holm-Bonferroni multi-asset family adjustment**.
        """
    )

    if not df_boot.empty:
        df_boot_asset = df_boot[df_boot["asset"] == asset]
        col_ci, col_tbl = st.columns([3, 2])

        with col_ci:
            fig_ci = go.Figure()
            fig_ci.add_trace(go.Scatter(
                x=[f"L={l}d" for l in df_boot_asset["block_length"]],
                y=df_boot_asset["observed_d_bar"],
                mode="markers",
                marker=dict(size=10, color=COLOR_WIN),
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=df_boot_asset["ci_upper_95"] - df_boot_asset["observed_d_bar"],
                    arrayminus=df_boot_asset["observed_d_bar"] - df_boot_asset["ci_lower_95"],
                    color=COLOR_BASELINE if not dark_mode else "#93C5FD",
                    thickness=2,
                    width=8,
                ),
                name="Observed Mean Diff & 95% CI",
            ))
            fig_ci.add_hline(y=0, line_dash="dash", line_color=COLOR_LOSS, annotation_text="Null H0 (No Edge)", annotation_position="bottom right")
            layout_ci = create_figure_layout(
                f"{asset} Block Bootstrap 95% Confidence Intervals",
                "Block Length (Preserved Autocorrelation Horizon)",
                "Mean Error Reduction (Baseline - Decay)",
                dark_mode=dark_mode,
                height=340,
            )
            fig_ci.update_layout(layout_ci)
            st.plotly_chart(fig_ci, width="stretch", config=PLOTLY_CONFIG)
            caption("The 95% confidence interval stays entirely above zero across all tested block lengths, confirming that decay outperformance is robust to return autocorrelation.")

        with col_tbl:
            st.markdown(f"**Statistical Robustness Summary ({asset})**")
            table_display = df_boot_asset[["block_length", "bootstrap_p_value", "ci_lower_95", "ci_upper_95"]].copy()
            table_display.columns = ["Block Len (L)", "Bootstrap p-val", "95% CI Lower", "95% CI Upper"]
            st.dataframe(table_display.style.format({
                "Bootstrap p-val": "{:.4f}",
                "95% CI Lower": "{:.6f}",
                "95% CI Upper": "{:.6f}",
            }), width="stretch")
            st.markdown(
                f"""
                <div class="callout-box" style="margin-top: 10px; padding: 10px 14px; font-size: 0.83rem;">
                    <strong>HAC Standard Errors:</strong><br>
                    • Paired t-test p: <strong>{p_val:.4f}</strong><br>
                    • Newey-West HAC p: <strong>{nw_pval:.4f}</strong><br>
                    • Holm-Bonferroni Adjusted p: <strong>{'0.0395' if asset == 'BTC-USD' else '0.0016'}</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # 13 Windows Detailed Table
    with st.expander("View Full 13-Window Detailed Table"):
        display_wf = df_wf[["test_window_start", "test_window_end", "selected_method", "baseline_MAE", "selected_MAE", "selected_wins"]].copy()
        display_wf["test_window"] = display_wf["test_window_start"] + " to " + display_wf["test_window_end"]
        display_wf["pct_diff"] = 100 * (display_wf["baseline_MAE"] - display_wf["selected_MAE"]) / display_wf["baseline_MAE"]
        display_wf = display_wf[["test_window", "selected_method", "baseline_MAE", "selected_MAE", "pct_diff", "selected_wins"]]
        display_wf.columns = ["Test Window", "Selected Method", "Baseline MAE", "Decay MAE", "% Improvement", "Decay Won"]
        st.dataframe(display_wf.style.format({
            "Baseline MAE": "{:.6f}",
            "Decay MAE": "{:.6f}",
            "% Improvement": "{:+.2f}%",
        }), width="stretch")


# =========================================================
# PAGE 8: BENCHMARKS & TRADING SIMULATION
# =========================================================
def render_benchmarks(dark_mode: bool):
    st.markdown('<div class="header-ribbon"></div>', unsafe_allow_html=True)
    st.markdown('<h1 class="pub-title">Established Benchmarks & Trading Simulation</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="pub-subtitle" style="font-size: 1.05rem; color: var(--text-muted);">'
        'Comprehensive benchmarking: GARCH, Moving Average, RSI, Baseline Controls (Predict-Zero, Tuned Ridge, Rolling Windows), Volatility Switching, and Net-of-Fees Trading.'
        '</p>',
        unsafe_allow_html=True,
    )

    (
        df_garch, df_rsi, df_ma_btc, df_ma_eth,
        df_backtest_btc, df_backtest_eth, df_equity, df_turnover,
    ) = get_benchmarks_data()
    controls_data = get_baseline_controls_data()
    vol_summary, vol_windows, vol_per_day = get_volatility_switching_data()

    # Model MAE Benchmarking Cards
    st.markdown("### 1. Comparative MAE vs Established Quantitative Benchmarks")
    asset_bm = st.radio("Select Benchmark Asset:", ["BTC-USD", "ETH-USD"], horizontal=True, key="bm_asset_select")
    df_backtest = df_backtest_btc if asset_bm == "BTC-USD" else df_backtest_eth

    c1, c2, c3 = st.columns(3)
    if asset_bm == "BTC-USD":
        with c1:
            st.markdown(
                """
                <div class="metric-card">
                    <div class="metric-card-header">vs GARCH(1,1) + AR(1)</div>
                    <div class="metric-card-value">Parity (p = 0.8597)</div>
                    <div class="metric-card-sub text-muted">Decay MAE: 0.021144 | GARCH: 0.021139<br>Both beat uniform baseline (p = 0.0031)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                """
                <div class="metric-card">
                    <div class="metric-card-header">vs 20/50 MA Crossover</div>
                    <div class="metric-card-value">Parity (p = 0.7413)</div>
                    <div class="metric-card-sub text-muted">Decay MAE: 0.021144 | MA Lin: 0.021157<br>Directional hit rate tied at 50.6%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                """
                <div class="metric-card">
                    <div class="metric-card-header">vs RSI(14) Momentum</div>
                    <div class="metric-card-value">Parity (p = 0.1210)</div>
                    <div class="metric-card-sub text-muted">Decay MAE: 0.021144 | RSI: 0.021227<br>Directional accuracy: 50.6% vs 48.7%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        with c1:
            st.markdown(
                """
                <div class="metric-card">
                    <div class="metric-card-header">vs GARCH(1,1) + AR(1)</div>
                    <div class="metric-card-value">Parity (p = 0.9657)</div>
                    <div class="metric-card-sub text-muted">Decay MAE: 0.028786 | GARCH: 0.028781<br>Both beat uniform baseline (p = 0.0002)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                """
                <div class="metric-card">
                    <div class="metric-card-header">vs 20/50 MA Crossover</div>
                    <div class="metric-card-value">Parity (p = 0.8931)</div>
                    <div class="metric-card-sub text-muted">Decay MAE: 0.028786 | MA Lin: 0.028795<br>Both linear models closely track</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                """
                <div class="metric-card">
                    <div class="metric-card-header">vs RSI(14) Momentum</div>
                    <div class="metric-card-value">Parity (p = 0.3540)</div>
                    <div class="metric-card-sub text-muted">Decay MAE: 0.028786 | RSI: 0.028905<br>Directional accuracy: 50.4% vs 48.5%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Baseline Controls Section
    st.markdown("### 2. Baseline Controls: Predict-Zero, Tuned Ridge, and Rolling Windows")
    st.markdown(
        """
        To rigorously isolate whether decay weighting provides unique statistical value beyond simple hyperparameter tuning or window truncation, 
        we compared it against multiple essential baseline controls across all 2,373 trading days (2020–2026):
        """
    )

    ctrl_key = "btc" if asset_bm == "BTC-USD" else "eth"
    df_ctrl = controls_data.get(ctrl_key, pd.DataFrame())

    if not df_ctrl.empty:
        col_c_chart, col_c_tbl = st.columns([3, 2])
        with col_c_chart:
            # Sort models by MAE
            df_ctrl_sorted = df_ctrl.sort_values("mae").reset_index(drop=True)
            colors_ctrl = [
                COLOR_WIN if m == "Selected Decay" else COLOR_NEUTRAL if "Zero" in m else COLOR_PURPLE if "Tuned" in m else COLOR_BASELINE
                for m in df_ctrl_sorted["model"]
            ]
            fig_ctrl = go.Figure(go.Bar(
                x=df_ctrl_sorted["model"],
                y=df_ctrl_sorted["mae"],
                marker_color=colors_ctrl,
                text=[f"{v:.5f}" for v in df_ctrl_sorted["mae"]],
                textposition="outside",
            ))
            layout_ctrl = create_figure_layout(
                f"{asset_bm} Baseline Controls Comparison (2020-2026, n=2,373)",
                "Control Model", "MAE",
                dark_mode=dark_mode,
                height=350,
            )
            fig_ctrl.update_layout(layout_ctrl)
            fig_ctrl.update_yaxes(range=[df_ctrl_sorted["mae"].min() * 0.98, df_ctrl_sorted["mae"].max() * 1.02])
            st.plotly_chart(fig_ctrl, width="stretch", config=PLOTLY_CONFIG)
            caption("Decay beats the fixed Baseline, Rolling Window, and Tuned Ridge. Predict-Zero achieves the lowest absolute error, confirming near-martingale return efficiency.")

        with col_c_tbl:
            st.markdown(f"**Baseline Controls Table ({asset_bm})**")
            tbl_disp = df_ctrl_sorted[["model", "mae", "pct_vs_baseline"]].copy()
            tbl_disp.columns = ["Model Variant", "MAE", "Delta vs Baseline"]
            st.dataframe(tbl_disp.style.format({
                "MAE": "{:.6f}",
                "Delta vs Baseline": "{:+.2f}%",
            }), width="stretch")

    # Volatility Switching Negative Result Panel
    st.markdown("### 3. Volatility-Switching Regime Model (Negative Result)")
    st.markdown(
        """
        <div class="callout-box callout-warning">
            <strong>Empirical Finding (Negative Result):</strong> An intuitive adaptive extension is to dynamically switch between 
            rapid decay (<em>S=30</em>) in high-volatility regimes and slow decay (<em>S=365</em>) in calm regimes based on trailing 30-day realized volatility. 
            However, our empirical backtest demonstrates that <strong>regime-switching fails to beat static combined decay</strong> (MAE 0.021154 vs 0.021132). 
            Lagged volatility estimates switch regimes too late, lagging turning points.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not vol_summary.empty:
        vol_disp = vol_summary.copy()
        st.dataframe(vol_disp, width="stretch")

    # Net-of-Fees Backtesting Section
    st.markdown("### 4. Net-of-Fees Trading Simulation (2020-2026)")
    st.markdown(
        """
        <div class="callout-box">
            <strong>Important Reality Check on Cumulative Returns:</strong><br>
            In unconstrained financial returns, passive <strong>Buy & Hold (7.1x return, Sharpe 0.84)</strong> and simple trend-following 
            <strong>20/50 MA Crossover (5.9x return, Sharpe 0.92)</strong> outperform the Decay ML strategy (2.2x return, Sharpe 0.60).<br>
            The genuine victory of the Decay Model is <strong>overcoming the catastrophic turnover churn of Baseline Ridge</strong>: 
            Baseline Ridge churned 733 times and collapsed to -95% equity after transaction fees, whereas Decay executed only 23 trades, 
            preserving capital and maintaining positive Sharpe across all fee tiers.
        </div>
        """,
        unsafe_allow_html=True,
    )

    fee_choice = st.selectbox(
        "Select Transaction Fee Level:",
        options=[0.0005, 0.001, 0.0025, 0.005],
        index=1,
        format_func=lambda x: f"{x*100:.2f}% per trade ({'0.10% Binance Spot Taker Fee' if x == 0.001 else 'High Friction' if x > 0.001 else 'Low Friction'})",
    )

    if not df_equity.empty:
        df_eq_sub = df_equity[df_equity["fee_rate"] == fee_choice]

        col_eq, col_dd = st.columns(2)

        with col_eq:
            fig_eq = go.Figure()
            strats = ["buy_and_hold", "decay_selected", "ma_signal", "baseline"]
            labels = {"buy_and_hold": "Buy & Hold", "decay_selected": "Decay Strategy", "ma_signal": "20/50 MA Crossover", "baseline": "Baseline Ridge"}
            colors = {"buy_and_hold": "#4A4E69" if not dark_mode else "#94A3B8", "decay_selected": COLOR_DECAY, "ma_signal": COLOR_WIN, "baseline": COLOR_BASELINE if not dark_mode else "#93C5FD"}

            for s in strats:
                sub = df_eq_sub[df_eq_sub["strategy"] == s]
                fig_eq.add_trace(go.Scatter(
                    x=sub["date"], y=sub["equity"],
                    mode="lines",
                    line=dict(color=colors[s], width=2 if s != "baseline" else 1.5),
                    name=labels[s],
                ))
            layout_eq = create_figure_layout(
                f"Cumulative Strategy Equity (Net of {fee_choice*100:.2f}% Fee)",
                "Date", "Portfolio Equity (Start = $1.00)",
                dark_mode=dark_mode,
                height=380,
            )
            fig_eq.update_layout(layout_eq)
            fig_eq.update_yaxes(type="log")
            st.plotly_chart(fig_eq, width="stretch", config=PLOTLY_CONFIG)
            caption("Decay strategy preserves capital and generates steady positive equity, while Baseline Ridge gets wiped out by 733 fee-paying trades.")

        with col_dd:
            fig_dd = go.Figure()
            for s in ["decay_selected", "baseline", "ma_signal"]:
                sub = df_eq_sub[df_eq_sub["strategy"] == s]
                fig_dd.add_trace(go.Scatter(
                    x=sub["date"], y=sub["drawdown"] * 100,
                    mode="lines",
                    line=dict(color=colors[s], width=1.5),
                    name=labels[s],
                ))
            layout_dd = create_figure_layout(
                "Historical Drawdown Trajectory (%)",
                "Date", "Drawdown (%)",
                dark_mode=dark_mode,
                height=380,
            )
            fig_dd.update_layout(layout_dd)
            st.plotly_chart(fig_dd, width="stretch", config=PLOTLY_CONFIG)
            caption("Drawdowns highlight that Baseline Ridge experiences extreme -84% to -96% drawdowns under real transaction costs.")

    # Sharpe vs Transaction Fee & Turnover Breakdown
    col_shp, col_trn = st.columns(2)

    with col_shp:
        if not df_backtest_btc.empty:
            fig_shp = go.Figure()
            for s in ["buy_and_hold", "decay_selected", "ma_signal", "baseline"]:
                sub = df_backtest_btc[df_backtest_btc["strategy"] == s]
                fig_shp.add_trace(go.Scatter(
                    x=sub["fee_rate"] * 100, y=sub["annualized_sharpe"],
                    mode="lines+markers",
                    line=dict(color=colors[s], width=2),
                    name=labels[s],
                ))
            layout_shp = create_figure_layout(
                "Annualized Sharpe Ratio vs Transaction Fee Drag",
                "Fee Rate (% per trade)", "Annualized Sharpe Ratio",
                dark_mode=dark_mode,
                height=350,
            )
            fig_shp.update_layout(layout_shp)
            fig_shp.add_hline(y=0, line_dash="dash", line_color=COLOR_LOSS)
            st.plotly_chart(fig_shp, width="stretch", config=PLOTLY_CONFIG)
            caption("Baseline Sharpe collapses to negative (-0.41) at 50 bps fees; Decay strategy Sharpe stays robust (~0.58 to 0.61).")

    with col_trn:
        if not df_backtest_btc.empty:
            trades_df = df_backtest_btc[df_backtest_btc["fee_rate"] == 0.001]
            fig_trn = go.Figure(go.Bar(
                x=[labels[s] for s in trades_df["strategy"]],
                y=trades_df["n_trades"],
                marker_color=[colors[s] for s in trades_df["strategy"]],
                text=trades_df["n_trades"],
                textposition="outside",
            ))
            layout_trn = create_figure_layout(
                "Total Trade Churn Over 2,373 Trading Days",
                "Strategy", "Number of Trades Executed",
                dark_mode=dark_mode,
                height=350,
            )
            fig_trn.update_layout(layout_trn)
            st.plotly_chart(fig_trn, width="stretch", config=PLOTLY_CONFIG)
            caption("Baseline executes 733 trades (swapping positions almost every 3 days), bleeding capital in friction. Decay strategy trades only 23 times.")


# =========================================================
# PAGE 9: METHODOLOGY & INTEGRITY
# =========================================================
def render_methodology(dark_mode: bool):
    st.markdown('<div class="header-ribbon"></div>', unsafe_allow_html=True)
    st.markdown('<h1 class="pub-title">Methodology & Research Integrity</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="pub-subtitle" style="font-size: 1.05rem; color: var(--text-muted);">'
        'Full transparency: post-mortem documentation of two major bugs resolved during integration, and the Claims vs Evidence matrix.'
        '</p>',
        unsafe_allow_html=True,
    )

    # Bug Post-Mortem Cards
    st.markdown("### Critical Bugs Caught, Documented, and Re-verified")
    col_b1, col_b2 = st.columns(2)
    card_bg_sub = "#1E293B" if dark_mode else "#F1F5F9"

    with col_b1:
        st.markdown(
            f"""
            <div class="metric-card" style="border-top: 4px solid {COLOR_LOSS};">
                <div class="metric-card-header" style="color: {COLOR_LOSS};">1. Target-Offset Alignment Bug</div>
                <p style="font-size: 0.92rem; line-height: 1.55;">
                    <strong>The Bug:</strong> An early implementation of the decay windowing function set the prediction target 
                    to <code>returns[i+1]</code>, while baseline targeted <code>returns[i]</code> for the identical feature window 
                    <code>returns[i-w:i]</code>. This inadvertent 1-day gap meant the two models were answering slightly different questions.
                </p>
                <p style="font-size: 0.92rem; line-height: 1.55;">
                    <strong>The Consequence:</strong> Under the buggy code, decay appeared to outperform baseline in volatile equity regimes. 
                    Once synchronized (target = <code>returns[i]</code>), the result reversed: baseline won both volatile and stable periods.
                </p>
                <div style="background: {card_bg_sub}; padding: 10px; border-radius: 4px; font-size: 0.83rem; color: var(--text-main);">
                    <strong>Resolution:</strong> Consolidated all windowing logic across all script copies into a single tested module 
                    (<code>windowing.py</code>). Retrained and re-verified every model.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_b2:
        st.markdown(
            f"""
            <div class="metric-card" style="border-top: 4px solid {COLOR_LOSS};">
                <div class="metric-card-header" style="color: {COLOR_LOSS};">2. Multi-Ticker Data Concatenation Leak</div>
                <p style="font-size: 0.92rem; line-height: 1.55;">
                    <strong>The Bug:</strong> <code>data_processed.csv</code> contained stacked rows from AAPL, BTC-USD, and SPY. 
                    Several sensitivity scripts initially read the file without an explicit <code>ticker == 'SPY'</code> filter, 
                    silently interleaving BTC's 3x volatility into SPY's rolling calculations.
                </p>
                <p style="font-size: 0.92rem; line-height: 1.55;">
                    <strong>The Consequence:</strong> SPY baseline MAE was inflated to ~0.0123 (double its true value of ~0.00615).
                </p>
                <div style="background: {card_bg_sub}; padding: 10px; border-radius: 4px; font-size: 0.83rem; color: var(--text-main);">
                    <strong>Resolution:</strong> Enforced strict mandatory ticker assertions in <code>add_regime_labels.py</code> and 
                    <code>windowing.py</code>; regenerated and re-verified all downstream CSVs.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Claims vs Evidence Matrix Table
    st.markdown("### Claims vs. Evidence Matrix (Auditing All 10 Hypotheses)")
    claims_df = get_claims_vs_evidence()
    if not claims_df.empty:
        st.dataframe(claims_df, width="stretch")

    # Safeguards List
    st.markdown("### Core Research Safeguards Implemented")
    st.markdown(
        """
        - **Leak-Free Rolling Hyperparameter Selection**: In walk-forward testing, all decay parameters ($S, \alpha$) are tuned exclusively on preceding validation folds—never on test data.
        - **Contiguous Walk-Forward Windows**: Validated zero date gaps or overlap across all 13 contiguous six-month evaluation windows.
        - **Moving Block Bootstrap**: Implemented 10,000 resamples across multi-day blocks to ensure statistical significance isn't an artifact of autocorrelation or volatility clustering.
        - **Newey-West HAC & Holm Corrections**: Standard errors robust to heteroskedasticity and autocorrelation, with family-wise error rate control.
        - **Real Market Friction**: Backtests account for spot taker fees (5 to 50 bps) and position churn rather than reporting hypothetical gross returns.
        """
    )


# =========================================================
# PAGE 10: CONCLUSIONS & FUTURE WORK
# =========================================================
def render_conclusions(dark_mode: bool):
    st.markdown('<div class="header-ribbon"></div>', unsafe_allow_html=True)
    st.markdown('<h1 class="pub-title">Synthesis, Conclusions & Future Work</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="pub-subtitle" style="font-size: 1.05rem; color: var(--text-muted);">'
        'Cross-domain synthesis and grounded roadmap for temporal memory modeling.'
        '</p>',
        unsafe_allow_html=True,
    )

    # Master Table
    st.markdown("### Master Cross-Domain Results Matrix")
    master_data = [
        {"Domain": "Financial (SPY)", "Data Frequency": "Daily", "Test Window": "2023-2024 (n=501)", "Baseline MAE": 0.006154, "Best Decay MAE": 0.006148, "Delta (%)": "-0.10%", "p-value": "0.7058", "Verdict": "No Benefit / Parity"},
        {"Domain": "Consumer (PCE)", "Data Frequency": "Monthly", "Test Window": "2023-2024 (n=24)", "Baseline MAE": 0.002192, "Best Decay MAE": 0.002503, "Delta (%)": "+14.20%", "p-value": "< 0.05", "Verdict": "Decay Hurts (+14.2%)"},
        {"Domain": "Disease (COVID)", "Data Frequency": "Daily", "Test Window": "2022-2023 (n=341)", "Baseline MAE": 0.045980, "Best Decay MAE": 0.040868, "Delta (%)": "-11.12%", "p-value": "< 0.0001", "Verdict": "Artifact Rejection (Beaten by Naive-Zero 0.03876)"},
        {"Domain": "Crypto (BTC Held-Out)", "Data Frequency": "Daily", "Test Window": "2024-2026 (n=949)", "Baseline MAE": 0.018033, "Best Decay MAE": 0.017789, "Delta (%)": "-1.35%", "p-value": "0.0012", "Verdict": "Decay Wins vs Baseline (Matches Naive-Zero 0.017745)"},
        {"Domain": "Crypto (ETH Held-Out)", "Data Frequency": "Daily", "Test Window": "2024-2026 (n=912)", "Baseline MAE": 0.025727, "Best Decay MAE": 0.025387, "Delta (%)": "-1.32%", "p-value": "0.0129", "Verdict": "Decay Wins vs Baseline (Matches Naive-Zero 0.025369)"},
        {"Domain": "Crypto (BTC Walk-Forward)", "Data Frequency": "Daily", "Test Window": "2020-2026 (n=2,373)", "Baseline MAE": 0.021334, "Best Decay MAE": 0.021144, "Delta (%)": "-0.89%", "p-value": "0.0079", "Verdict": "Decay Wins 10/13 Windows (HAC p = 0.0197)"},
        {"Domain": "Crypto (ETH Walk-Forward)", "Data Frequency": "Daily", "Test Window": "2020-2026 (n=2,373)", "Baseline MAE": 0.029211, "Best Decay MAE": 0.028786, "Delta (%)": "-1.46%", "p-value": "0.00015", "Verdict": "Decay Wins 11/13 Windows (HAC p = 0.0008)"},
    ]
    st.dataframe(pd.DataFrame(master_data), width="stretch")

    # 4 Plain Language Conclusions
    st.markdown("### Core Empirical Takeaways")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-card-header">1. Memory Value is Strictly Domain-Specific</div>
                <p style="font-size: 0.92rem; line-height: 1.55;">
                    There is no universal benefit to exponential forgetting in time-series machine learning. 
                    In series characterized by strong macroeconomic persistence (PCE), forgetting is strictly destructive. 
                    In near-martingale asset prices (equities), forgetting prunes sample size without improving signal.
                </p>
            </div>
            <div class="metric-card">
                <div class="metric-card-header">2. The Dual-Decay Mechanism in Crypto</div>
                <p style="font-size: 0.92rem; line-height: 1.55;">
                    In crypto markets, true outperformance over baseline Ridge requires combining <em>feature-dimension lag decay</em> (S_feat=5) 
                    with <em>sample-dimension row decay</em> (S_row=365). Lag decay filters noise from distant days within the window, 
                    while row decay enables adaptation across multi-year cycles.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-card-header">3. Always Audit Naive & Control Baselines</div>
                <p style="font-size: 0.92rem; line-height: 1.55;">
                    In epidemiology (COVID), decay beat baseline by 11.1% simply because it discarded obsolete early-pandemic reporting noise. 
                    In financial returns, predicting zero is an exceptionally difficult benchmark to beat. Evaluating against naive controls 
                    prevents celebrating false predictive prowess.
                </p>
            </div>
            <div class="metric-card">
                <div class="metric-card-header">4. Transaction Churn Decides Practical Utility</div>
                <p style="font-size: 0.92rem; line-height: 1.55;">
                    Evaluating models on MAE alone is dangerous for financial applications. Uniform baseline generated 733 trades and suffered -95% drawdown after fees; 
                    the smoother decay model generated only 23 trades, avoiding fee destruction and preserving positive Sharpe ratios.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Future Work & Engineering Roadmap
    st.markdown("### Engineering Roadmap & Next Steps")
    st.markdown(
        """
        1. **Non-Linear Tree Ensembles (LightGBM / XGBoost with Sample Weights)**:
           Extend the Ebbinghaus row-weighting kernel to gradient boosted decision trees to test if non-linear interactions benefit from temporal decay.
        2. **Deep Temporal Sequence Models (LSTM & Temporal Fusion Transformers)**:
           Incorporate continuous-time exponential decay directly into the multi-head attention kernels ($Attention(Q, K, V) \odot \exp(-\Delta t / S)$).
        3. **Learned Dynamic Stability Constant ($S_t$)**:
           Replace discrete grid search with an end-to-end meta-learner that predicts time-varying memory horizon $S_t$ conditioned on realized volatility and liquidity regimes.
        4. **Direct Utility / Sharpe Loss Optimization**:
           Transition from symmetrical $L_1$/$L_2$ regression objectives to asymmetric directional and transaction-cost-aware Sharpe ratio loss functions.
        """
    )
