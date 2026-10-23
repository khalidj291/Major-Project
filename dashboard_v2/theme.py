"""
theme.py -- Presentation-grade styling, color tokens, and Plotly helpers for dashboard_v2.
Audience: External reviewers, professors, and technical recruiters.
Clean academic aesthetic: Serif headers, clean sans UI, high contrast, strict color conventions.
"""

import plotly.graph_objects as go
import streamlit as st

# Strict Palette Definition
COLOR_BASELINE = "#14213D"   # Ink Navy -- Always Baseline on every chart
COLOR_DECAY = "#E09F3E"      # Warm Amber -- Always Decay Model on every chart
COLOR_WIN = "#2A9D8F"        # Teal -- Outperformance / Gains / Wins
COLOR_LOSS = "#C8553D"       # Muted Red -- Underperformance / Volatile / Losses
COLOR_NEUTRAL = "#8D99AE"    # Slate Grey -- Neutral references / Naive Zero
COLOR_BG_LIGHT = "#FFFFFF"   # Clean White
COLOR_BG_CARD = "#F8F9FA"    # Soft Off-white / Card Background
COLOR_TEXT_MAIN = "#14213D"  # Dark Charcoal Navy
COLOR_TEXT_MUTED = "#5C677D" # Secondary Muted Text
COLOR_BORDER = "#E5E5E5"     # Subtle Grey Border

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


def inject_custom_css(dark_mode: bool = False):
    """Inject polished typography and CSS styling."""
    bg_color = "#0B132B" if dark_mode else "#FAF9F6"
    card_bg = "#1C2541" if dark_mode else "#FFFFFF"
    text_color = "#E0E1DD" if dark_mode else "#14213D"
    muted_text = "#9A8C98" if dark_mode else "#5C677D"
    border_color = "#3A506B" if dark_mode else "#E9ECEF"

    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@500;700&family=Inter:wght@300;400;500;600;700&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,600;1,6..72,400&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: {text_color};
    }}

    .stApp {{
        background-color: {bg_color};
    }}

    /* Publication Page Titles */
    h1, .pub-title {{
        font-family: 'Newsreader', 'Georgia', serif !important;
        font-weight: 600 !important;
        color: {text_color} !important;
        letter-spacing: -0.02em;
    }}

    h2, h3, .pub-subtitle {{
        font-family: 'Newsreader', 'Georgia', serif !important;
        font-weight: 500 !important;
        color: {text_color} !important;
    }}

    /* Card Containers */
    .metric-card {{
        background: {card_bg};
        border: 1px solid {border_color};
        border-radius: 8px;
        padding: 20px 22px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        margin-bottom: 18px;
    }}

    .metric-card-header {{
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: {muted_text};
        margin-bottom: 8px;
    }}

    .metric-card-value {{
        font-size: 2.1rem;
        font-weight: 700;
        font-family: 'Inter', sans-serif;
        color: {text_color};
        line-height: 1.1;
        margin-bottom: 6px;
    }}

    .metric-card-sub {{
        font-size: 0.85rem;
        color: {muted_text};
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
        background-color: rgba(42, 157, 143, 0.15);
        color: #2A9D8F;
        border: 1px solid #2A9D8F;
    }}
    .badge-loss {{
        background-color: rgba(200, 85, 61, 0.15);
        color: #C8553D;
        border: 1px solid #C8553D;
    }}
    .badge-parity {{
        background-color: rgba(141, 153, 174, 0.15);
        color: #5C677D;
        border: 1px solid #8D99AE;
    }}
    .badge-mixed {{
        background-color: rgba(224, 159, 62, 0.15);
        color: #D48B28;
        border: 1px solid #E09F3E;
    }}

    /* Top Ribbon Header Motif */
    .header-ribbon {{
        background: linear-gradient(90deg, #14213D 0%, #2A9D8F 40%, #E09F3E 75%, rgba(224, 159, 62, 0.1) 100%);
        height: 5px;
        border-radius: 2px;
        margin-bottom: 24px;
    }}

    /* Scientific Callout Box */
    .callout-box {{
        background-color: {'rgba(28, 37, 65, 0.5)' if dark_mode else '#F4F6F9'};
        border-left: 4px solid #14213D;
        padding: 16px 20px;
        border-radius: 0 6px 6px 0;
        margin: 16px 0 20px 0;
        font-size: 0.93rem;
        line-height: 1.55;
    }}

    .callout-warning {{
        border-left-color: #C8553D;
        background-color: {'rgba(200, 85, 61, 0.1)' if dark_mode else '#FDF6F5'};
    }}

    .callout-success {{
        border-left-color: #2A9D8F;
        background-color: {'rgba(42, 157, 143, 0.1)' if dark_mode else '#F3FAF8'};
    }}

    /* Chart Caption */
    .chart-caption {{
        font-size: 0.84rem;
        color: {muted_text};
        font-style: italic;
        margin-top: -6px;
        margin-bottom: 22px;
        line-height: 1.4;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def create_figure_layout(title: str, x_title: str, y_title: str, dark_mode: bool = False, height: int = 450) -> go.Layout:
    """Standardized publication-grade Plotly layout."""
    bg_color = "#1C2541" if dark_mode else "#FFFFFF"
    plot_bg = "#151E34" if dark_mode else "#FBFBFC"
    font_color = "#E0E1DD" if dark_mode else "#14213D"
    grid_color = "#2E3A59" if dark_mode else "#E9ECEF"

    return go.Layout(
        title={
            "text": f"<b>{title}</b>",
            "x": 0.02,
            "xanchor": "left",
            "font": {"family": "Newsreader, Georgia, serif", "size": 17, "color": font_color},
        },
        xaxis={
            "title": {"text": x_title, "font": {"family": "Inter, sans-serif", "size": 12, "color": font_color}},
            "gridcolor": grid_color,
            "zerolinecolor": grid_color,
            "showgrid": True,
            "tickfont": {"family": "Inter, sans-serif", "size": 11, "color": font_color},
        },
        yaxis={
            "title": {"text": y_title, "font": {"family": "Inter, sans-serif", "size": 12, "color": font_color}},
            "gridcolor": grid_color,
            "zerolinecolor": grid_color,
            "showgrid": True,
            "tickfont": {"family": "Inter, sans-serif", "size": 11, "color": font_color},
        },
        paper_bgcolor=bg_color,
        plot_bgcolor=plot_bg,
        font={"family": "Inter, sans-serif", "color": font_color},
        margin={"l": 55, "r": 25, "t": 55, "b": 50},
        height=height,
        legend={
            "bgcolor": "rgba(255, 255, 255, 0.7)" if not dark_mode else "rgba(28, 37, 65, 0.8)",
            "bordercolor": grid_color,
            "borderwidth": 1,
            "font": {"size": 11},
        },
        hoverlabel={"font": {"family": "Inter, sans-serif", "size": 11}},
    )


def caption(text: str):
    """Render a standardized explanatory caption underneath every chart."""
    st.markdown(f'<div class="chart-caption"><strong>Takeaway:</strong> {text}</div>', unsafe_allow_html=True)
