"""
app.py -- Main application entrypoint for Memory That Fades presentation dashboard (v2).
Audience: Professors, Recruiters, and Reviewers.
Run via: streamlit run dashboard_v2/app.py
"""

import streamlit as st
import os
import sys

# Ensure local imports work reliably
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from theme import inject_custom_css
from pages_content import (
    render_overview,
    render_how_it_works,
    render_spy,
    render_pce,
    render_covid,
    render_crypto,
    render_walk_forward,
    render_benchmarks,
    render_methodology,
    render_conclusions,
)

# Page Configuration
st.set_page_config(
    page_title="Memory That Fades — Research Dashboard",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    # Initialize and toggle dark mode first so CSS is applied early
    dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=False)
    inject_custom_css(dark_mode=dark_mode)

    # Sidebar Header
    st.sidebar.markdown(
        """
        <div style="padding: 10px 0 16px 0; border-bottom: 1px solid var(--border-color, #E0E0E0); margin-bottom: 15px;">
            <div class="sidebar-header-title">
                Memory That Fades
            </div>
            <div class="sidebar-header-sub">
                Results & Integrity Dashboard
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar Navigation
    pages = [
        "1. Overview & Summary",
        "2. How It Works (Decay Math)",
        "3. Financial Equities (SPY)",
        "4. Macro Consumer (PCE)",
        "5. Epidemiology (COVID-19)",
        "6. Cryptocurrency (BTC / ETH)",
        "7. Walk-Forward Validation",
        "8. Benchmarks & Trading",
        "9. Methodology & Integrity",
        "10. Conclusions & Roadmap",
    ]

    selected_page = st.sidebar.radio("Navigate Research Pages:", pages, index=0)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div class="text-muted" style="font-size: 0.8rem; line-height: 1.5;">
            <strong>Research Standard:</strong><br>
            • Real experiment data only<br>
            • Negative results highlighted<br>
            • Leak-free walk-forward validation<br>
            • High-DPI PNG export on every chart
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Page Routing
    if selected_page == "1. Overview & Summary":
        render_overview(dark_mode=dark_mode)
    elif selected_page == "2. How It Works (Decay Math)":
        render_how_it_works(dark_mode=dark_mode)
    elif selected_page == "3. Financial Equities (SPY)":
        render_spy(dark_mode=dark_mode)
    elif selected_page == "4. Macro Consumer (PCE)":
        render_pce(dark_mode=dark_mode)
    elif selected_page == "5. Epidemiology (COVID-19)":
        render_covid(dark_mode=dark_mode)
    elif selected_page == "6. Cryptocurrency (BTC / ETH)":
        render_crypto(dark_mode=dark_mode)
    elif selected_page == "7. Walk-Forward Validation":
        render_walk_forward(dark_mode=dark_mode)
    elif selected_page == "8. Benchmarks & Trading":
        render_benchmarks(dark_mode=dark_mode)
    elif selected_page == "9. Methodology & Integrity":
        render_methodology(dark_mode=dark_mode)
    elif selected_page == "10. Conclusions & Roadmap":
        render_conclusions(dark_mode=dark_mode)


if __name__ == "__main__":
    main()
