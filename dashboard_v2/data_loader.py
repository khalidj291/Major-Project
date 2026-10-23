"""
data_loader.py -- Cached, robust data loader for dashboard_v2.
Provides graceful fallbacks if any optional file is missing.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
LOCAL_DATA_DIR = os.path.join(SCRIPT_DIR, "data")
DECAY_RESULTS_DIR = os.path.join(PROJECT_ROOT, "decay_model", "results")


@st.cache_data
def load_csv_safe(path: str, parse_dates=None) -> pd.DataFrame:
    """Safely load CSV file, returns empty DataFrame with warning if missing."""
    if os.path.exists(path):
        try:
            return pd.read_csv(path, parse_dates=parse_dates)
        except Exception as e:
            st.warning(f"Error loading {os.path.basename(path)}: {e}")
            return pd.DataFrame()
    return pd.DataFrame()


@st.cache_data
def load_overview_metrics():
    """Load core summary metrics across all 4 domains."""
    return {
        "spy": {
            "name": "Financial Equities (SPY)",
            "subtitle": "Daily returns, 501 test days (2023-2024)",
            "baseline_mae": 0.006154,
            "best_decay_mae": 0.006148, # S=730 (p=0.7058)
            "pct_change": -0.10,
            "verdict": "No Benefit / Parity",
            "badge_class": "badge-parity",
            "takeaway": "Baseline uniformly-weighted Ridge wins on all standard S (30, 180, 365). S=730 is statistically indistinguishable (p=0.71).",
            "icon": "📈",
        },
        "pce": {
            "name": "Macro Consumer Demand (PCE)",
            "subtitle": "Monthly percent changes, 24 test months (2023-2024)",
            "baseline_mae": 0.002192,
            "best_decay_mae": 0.002503, # Slow S=730
            "pct_change": +14.20,
            "verdict": "Decay Hurts",
            "badge_class": "badge-loss",
            "takeaway": "Strong macroeconomic persistence; aggressive forgetting severely impairs forecasting accuracy (+88.9% error for fast decay).",
            "icon": "🛒",
        },
        "covid": {
            "name": "Epidemiology (COVID-19 Cases)",
            "subtitle": "Daily new cases, 341 held-out test days (2022-2023)",
            "baseline_mae": 0.045980,
            "best_decay_mae": 0.040868, # S=180
            "pct_change": -11.12,
            "verdict": "Mixed / Artifact Rejection",
            "badge_class": "badge-mixed",
            "takeaway": "Decay beats baseline (p < 0.0001) by pruning 2020 reporting artifacts, but neither beats trivial naive-zero (0.038763).",
            "icon": "🦠",
        },
        "crypto": {
            "name": "Cryptocurrency (BTC / ETH)",
            "subtitle": "Daily returns, 13 walk-forward windows & held-out test",
            "baseline_mae": 0.018033,
            "best_decay_mae": 0.017789, # Combined Decay
            "pct_change": -1.35,
            "verdict": "Decay Wins",
            "badge_class": "badge-win",
            "takeaway": "Statistically significant improvement on held-out test (p=0.0012) and 10 of 13 walk-forward windows won (p=0.0079).",
            "icon": "₿",
        },
    }


@st.cache_data
def get_spy_data():
    df_pred = load_csv_safe(os.path.join(LOCAL_DATA_DIR, "spy_predictions.csv"), parse_dates=["date"])
    df_sens = load_csv_safe(os.path.join(DECAY_RESULTS_DIR, "decay_rate_sensitivity.csv"))
    df_regime = load_csv_safe(os.path.join(DECAY_RESULTS_DIR, "regime_analysis_financial.csv"))
    return df_pred, df_sens, df_regime


@st.cache_data
def get_pce_data():
    df_pred = load_csv_safe(os.path.join(LOCAL_DATA_DIR, "pce_predictions.csv"), parse_dates=["date"])
    df_acf = load_csv_safe(os.path.join(LOCAL_DATA_DIR, "pce_acf.csv"))
    df_regime = load_csv_safe(os.path.join(DECAY_RESULTS_DIR, "regime_analysis_consumer.csv"))
    return df_pred, df_acf, df_regime


@st.cache_data
def get_covid_data():
    df_pred = load_csv_safe(os.path.join(LOCAL_DATA_DIR, "covid_predictions.csv"), parse_dates=["date"])
    df_hist = load_csv_safe(os.path.join(LOCAL_DATA_DIR, "covid_history.csv"), parse_dates=["date"])
    return df_pred, df_hist


@st.cache_data
def get_crypto_data():
    df_heldout = load_csv_safe(os.path.join(LOCAL_DATA_DIR, "crypto_heldout_predictions.csv"), parse_dates=["date"])
    df_grid = load_csv_safe(os.path.join(LOCAL_DATA_DIR, "crypto_tuning_grid.csv"))
    return df_heldout, df_grid


@st.cache_data
def get_walk_forward_data():
    df_wf_btc = load_csv_safe(os.path.join(DECAY_RESULTS_DIR, "walk_forward_results_btc.csv"))
    df_wf_eth = load_csv_safe(os.path.join(DECAY_RESULTS_DIR, "walk_forward_results_eth.csv"))
    df_boot = load_csv_safe(os.path.join(LOCAL_DATA_DIR, "block_bootstrap_summary.csv"))
    return df_wf_btc, df_wf_eth, df_boot


@st.cache_data
def get_benchmarks_data():
    df_garch = load_csv_safe(os.path.join(DECAY_RESULTS_DIR, "garch_comparison_summary_btc.csv"))
    df_rsi = load_csv_safe(os.path.join(DECAY_RESULTS_DIR, "rsi_comparison_btc.csv"))
    df_backtest_btc = load_csv_safe(os.path.join(DECAY_RESULTS_DIR, "trading_cost_backtest_btc.csv"))
    df_backtest_eth = load_csv_safe(os.path.join(DECAY_RESULTS_DIR, "trading_cost_backtest_eth.csv"))
    df_equity = load_csv_safe(os.path.join(LOCAL_DATA_DIR, "trading_equity_curves.csv"), parse_dates=["date"])
    df_turnover = load_csv_safe(os.path.join(DECAY_RESULTS_DIR, "turnover_matched_backtest_btc.csv"))
    return df_garch, df_rsi, df_backtest_btc, df_backtest_eth, df_equity, df_turnover


@st.cache_data
def get_cross_domain_data():
    return load_csv_safe(os.path.join(DECAY_RESULTS_DIR, "cross_domain_comparison.csv"))
