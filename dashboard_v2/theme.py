"""
theme.py -- Presentation-grade styling, color tokens, and Plotly helpers for dashboard_v2.
Audience: External reviewers, professors, and technical recruiters.
Clean academic aesthetic: Serif headers, clean sans UI, high contrast, strict color conventions.
"""

import plotly.graph_objects as go
import streamlit as st

# Strict Palette Definition for Charts (Standardized publication tokens)
COLOR_BASELINE = "#14213D"   # Ink Navy -- Always Baseline on every chart
COLOR_DECAY = "#E09F3E"      # Warm Amber -- Always Decay Model on every chart
COLOR_WIN = "#2A9D8F"        # Teal -- Outperformance / Gains / Wins
COLOR_LOSS = "#C8553D"       # Muted Red -- Underperformance / Volatile / Losses
COLOR_NEUTRAL = "#64748B"    # Slate Grey -- Neutral references / Naive Zero
COLOR_PURPLE = "#8B5CF6"     # Purple for Tuned Ridge / Rolling Controls

# Plotly High-DPI and PNG Export Configuration
PLOTLY_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "toImageButtonOptions": {
        "format": "png",
        "filename": "memory_that_fades_chart",
        "height": 550,
        "width": 950,
        "scale": 2.5,  # High-DPI publication grade export
    },
}


def get_theme_tokens(dark_mode: bool = False):
    """Returns dynamic CSS tokens ensuring 100% readability in both light & dark mode."""
    if dark_mode:
        return {
            "bg": "#0B132B",
            "card_bg": "#1E293B",
            "text_main": "#F8FAFC",
            "text_muted": "#94A3B8",
            "border": "#334155",
            "baseline_text": "#93C5FD",  # Bright soft blue for dark mode
            "decay_text": "#FBBF24",     # Bright amber for dark mode
            "win_text": "#2DD4BF",       # Crisp teal
            "loss_text": "#FB7185",      # Crisp rose red
            "neutral_text": "#CBD5E1",
            "callout_bg": "#1E293B",
            "callout_border": "#3B82F6",
        }
    else:
        return {
            "bg": "#FAF9F6",
            "card_bg": "#FFFFFF",
            "text_main": "#0F172A",
            "text_muted": "#475569",
            "border": "#E2E8F0",
            "baseline_text": "#14213D",  # Deep navy for light mode
            "decay_text": "#D97706",     # Warm dark amber
            "win_text": "#0D9488",       # Deep teal
            "loss_text": "#BE123C",      # Deep red
            "neutral_text": "#64748B",
            "callout_bg": "#F8FAFC",
            "callout_border": "#14213D",
        }


def inject_custom_css(dark_mode: bool = False):
    """Inject polished typography and CSS styling with high-contrast text guarantee."""
    t = get_theme_tokens(dark_mode)

    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@500;700&family=Inter:wght@300;400;500;600;700&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,600;1,6..72,400&display=swap');

    :root {{
        --bg-main: {t['bg']};
        --card-bg: {t['card_bg']};
        --text-main: {t['text_main']};
        --text-muted: {t['text_muted']};
        --border-color: {t['border']};
        --baseline-text: {t['baseline_text']};
        --decay-text: {t['decay_text']};
        --win-text: {t['win_text']};
        --loss-text: {t['loss_text']};
        --neutral-text: {t['neutral_text']};
    }}

    html, body, [class*="css"], .stMarkdown, p, div, span {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }}

    .stApp {{
        background-color: {t['bg']};
        color: {t['text_main']};
    }}

    /* Text & Metric Utility Classes */
    .text-main {{ color: {t['text_main']} !important; }}
    .text-muted {{ color: {t['text_muted']} !important; }}
    .val-baseline {{ color: {t['baseline_text']} !important; }}
    .val-decay {{ color: {t['decay_text']} !important; }}
    .val-win {{ color: {t['win_text']} !important; }}
    .val-loss {{ color: {t['loss_text']} !important; }}
    .val-neutral {{ color: {t['neutral_text']} !important; }}
    .metric-divider {{ border-top: 1px solid {t['border']}; padding-top: 8px; font-size: 0.85rem; color: {t['text_main']}; }}

    .sidebar-header-title {{
        font-family: 'Newsreader', Georgia, serif;
        font-size: 1.25rem;
        font-weight: 700;
        color: {t['text_main']} !important;
    }}
    .sidebar-header-sub {{
        font-size: 0.8rem;
        color: {t['text_muted']} !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 3px;
    }}

    /* Headers */
    h1, h2, h3, h4, .pub-title, .pub-subtitle {{
        font-family: 'Newsreader', 'Georgia', serif !important;
        color: {t['text_main']} !important;
        letter-spacing: -0.015em;
    }}

    /* Card Containers */
    .metric-card {{
        background: {t['card_bg']};
        border: 1px solid {t['border']};
        border-radius: 8px;
        padding: 20px 22px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, {'0.25' if dark_mode else '0.04'});
        margin-bottom: 18px;
        color: {t['text_main']};
    }}

    .metric-card-header {{
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: {t['text_muted']};
        margin-bottom: 8px;
    }}

    .metric-card-value {{
        font-size: 2.0rem;
        font-weight: 700;
        font-family: 'Inter', sans-serif;
        color: {t['text_main']};
        line-height: 1.15;
        margin-bottom: 6px;
    }}

    .metric-card-sub {{
        font-size: 0.85rem;
        color: {t['text_muted']};
    }}

    /* Sidebar text fix */
    [data-testid="stSidebar"] {{
        background-color: {'#080E21' if dark_mode else '#F1F5F9'};
        border-right: 1px solid {t['border']};
    }}
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] div, 
    [data-testid="stSidebar"] span, 
    [data-testid="stSidebar"] label {{
        color: {t['text_main']} !important;
    }}

    /* Verdict Badges */
    .badge {{
        display: inline-block;
        padding: 4px 10px;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 4px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    .badge-win {{
        background-color: {'rgba(45, 212, 191, 0.2)' if dark_mode else 'rgba(13, 148, 136, 0.12)'};
        color: {t['win_text']};
        border: 1px solid {t['win_text']};
    }}
    .badge-loss {{
        background-color: {'rgba(251, 113, 133, 0.2)' if dark_mode else 'rgba(190, 18, 60, 0.12)'};
        color: {t['loss_text']};
        border: 1px solid {t['loss_text']};
    }}
    .badge-parity {{
        background-color: {'rgba(148, 163, 184, 0.2)' if dark_mode else 'rgba(100, 116, 139, 0.12)'};
        color: {t['text_muted']};
        border: 1px solid {t['text_muted']};
    }}
    .badge-mixed {{
        background-color: {'rgba(251, 191, 36, 0.2)' if dark_mode else 'rgba(217, 119, 6, 0.12)'};
        color: {t['decay_text']};
        border: 1px solid {t['decay_text']};
    }}

    /* Top Ribbon Header Motif */
    .header-ribbon {{
        background: linear-gradient(90deg, #14213D 0%, #2A9D8F 40%, #E09F3E 75%, rgba(224, 159, 62, 0.15) 100%);
        height: 5px;
        border-radius: 2px;
        margin-bottom: 24px;
    }}

    /* Scientific Callout Box */
    .callout-box {{
        background-color: {t['callout_bg']};
        border-left: 4px solid {t['callout_border']};
        padding: 16px 20px;
        border-radius: 0 6px 6px 0;
        margin: 16px 0 20px 0;
        font-size: 0.93rem;
        line-height: 1.55;
        color: {t['text_main']};
        border-top: 1px solid {t['border']};
        border-right: 1px solid {t['border']};
        border-bottom: 1px solid {t['border']};
    }}

    .callout-warning {{
        border-left-color: {t['loss_text']};
        background-color: {'rgba(251, 113, 133, 0.08)' if dark_mode else '#FFF1F2'};
    }}

    .callout-success {{
        border-left-color: {t['win_text']};
        background-color: {'rgba(45, 212, 191, 0.08)' if dark_mode else '#F0FDFA'};
    }}

    /* Chart Caption */
    .chart-caption {{
        font-size: 0.84rem;
        color: {t['text_muted']};
        font-style: italic;
        margin-top: -6px;
        margin-bottom: 22px;
        line-height: 1.45;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def create_figure_layout(title: str, x_title: str, y_title: str, dark_mode: bool = False, height: int = 450) -> go.Layout:
    """Standardized publication-grade Plotly layout."""
    t = get_theme_tokens(dark_mode)
    plot_bg = "#111827" if dark_mode else "#FBFBFC"
    grid_color = "#334155" if dark_mode else "#E2E8F0"

    return go.Layout(
        title={
            "text": f"<b>{title}</b>",
            "x": 0.02,
            "xanchor": "left",
            "font": {"family": "Newsreader, Georgia, serif", "size": 17, "color": t["text_main"]},
        },
        xaxis={
            "title": {"text": x_title, "font": {"family": "Inter, sans-serif", "size": 12, "color": t["text_main"]}},
            "gridcolor": grid_color,
            "zerolinecolor": grid_color,
            "showgrid": True,
            "tickfont": {"family": "Inter, sans-serif", "size": 11, "color": t["text_muted"]},
        },
        yaxis={
            "title": {"text": y_title, "font": {"family": "Inter, sans-serif", "size": 12, "color": t["text_main"]}},
            "gridcolor": grid_color,
            "zerolinecolor": grid_color,
            "showgrid": True,
            "tickfont": {"family": "Inter, sans-serif", "size": 11, "color": t["text_muted"]},
        },
        paper_bgcolor=t["card_bg"],
        plot_bgcolor=plot_bg,
        font={"family": "Inter, sans-serif", "color": t["text_main"]},
        margin={"l": 55, "r": 25, "t": 55, "b": 50},
        height=height,
        legend={
            "bgcolor": "rgba(255, 255, 255, 0.7)" if not dark_mode else "rgba(30, 41, 59, 0.8)",
            "bordercolor": grid_color,
            "borderwidth": 1,
            "font": {"size": 11, "color": t["text_main"]},
        },
        hoverlabel={"font": {"family": "Inter, sans-serif", "size": 11}},
    )


def caption(text: str):
    """Render a standardized explanatory caption underneath every chart."""
    st.markdown(f'<div class="chart-caption"><strong>Takeaway:</strong> {text}</div>', unsafe_allow_html=True)
