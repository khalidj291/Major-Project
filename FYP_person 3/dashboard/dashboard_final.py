"""
Memory That Fades — Live Comparison Dashboard (FINAL, financial domain)
Person 3

Real baseline + fast + medium + slow predictions, real regime timeline,
real regime-split comparison bar chart, Ebbinghaus curves panel.

Consumer domain toggle is wired but disabled until data_consumer.csv and
model_baseline_consumer.pkl are delivered by Person 1 -- see the
CONSUMER_DATA_AVAILABLE flag below.

Run: python dashboard_final.py
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.widgets import Slider, Button
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
# ensure local models/ is on path before importing model helper modules
sys.path.append(os.path.join(PROJECT_ROOT, "models"))
from ebbinghaus import ebbinghaus_weight

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "data_processed.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

TICKER = "SPY"
WINDOW = 30
TEST_START = pd.Timestamp("2023-01-01")

# Flip this on once Person 1 delivers data_consumer.csv + model_baseline_consumer.pkl
CONSUMER_DATA_AVAILABLE = os.path.exists(os.path.join(PROJECT_ROOT, "data", "data_consumer.csv"))

MODEL_FILES = {
    "Baseline": "model_baseline.pkl",
    "Decay Fast": "model_decay_fast.pkl",
    "Decay Medium": "model_decay_medium.pkl",
    "Decay Slow": "model_decay_slow.pkl",
}
COLORS = {
    "Baseline": "grey",
    "Decay Fast": "#d62728",
    "Decay Medium": "#ff7f0e",
    "Decay Slow": "#1f77b4",
}
REGIME_COLORS = {"volatile": "red", "stable": "green", "neutral": "grey"}


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

def load_ticker_df(ticker=TICKER):
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df[df["ticker"] == ticker].sort_values("date").reset_index(drop=True)
    return df


def build_all_samples(df, window=WINDOW):
    """All samples (not just test) so the top-left panel can show the full series
    if wanted later; here we still restrict display to the test window."""
    returns = df["returns"].values
    dates = df["date"].values
    close = df["close"].values
    regimes = df["regime"].values
    X, y, sample_dates, prices, sample_regimes = [], [], [], [], []
    for i in range(window, len(returns) - 1):
        X.append(returns[i - window:i])
        y.append(returns[i + 1])
        sample_dates.append(dates[i + 1])
        prices.append(close[i + 1])
        sample_regimes.append(regimes[i + 1])
    X = np.array(X)
    y = np.array(y)
    sample_dates = pd.to_datetime(sample_dates)
    prices = np.array(prices)
    sample_regimes = np.array(sample_regimes)
    mask = sample_dates >= TEST_START
    return X[mask], y[mask], sample_dates[mask], prices[mask], sample_regimes[mask]


def load_all_predictions(X):
    preds = {}
    for name, fname in MODEL_FILES.items():
        with open(os.path.join(MODELS_DIR, fname), "rb") as f:
            model = pickle.load(f)
        preds[name] = model.predict(X).flatten()
    return preds


def mae_rmse(actual, pred):
    err = actual - pred
    return np.abs(err).mean(), np.sqrt((err ** 2).mean())


# ---------------------------------------------------------------------------
# DASHBOARD
# ---------------------------------------------------------------------------

def build_dashboard():
    df = load_ticker_df()
    X, y, dates, prices, regimes = build_all_samples(df)
    preds = load_all_predictions(X)

    volatile_mask = regimes == "volatile"
    stable_mask = regimes == "stable"

    plt.rcParams["font.size"] = 10
    fig = plt.figure(figsize=(15, 12), facecolor="white")
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.7], hspace=0.55, wspace=0.28)
    ax_pred = fig.add_subplot(gs[0, 0])
    ax_table = fig.add_subplot(gs[0, 1])
    ax_regime = fig.add_subplot(gs[1, 0])
    ax_bar = fig.add_subplot(gs[1, 1])
    ax_curves = fig.add_subplot(gs[2, :])

    fig.suptitle("Memory That Fades — Live Comparison Dashboard  (SPY, financial domain)",
                  fontsize=16, fontweight="bold", y=0.98)

    # --- Top-left: real predictions vs actual, all 4 models ---
    ax_pred.plot(dates, y, label="Actual", color="black", linewidth=1.3)
    for name in MODEL_FILES:
        ax_pred.plot(dates, preds[name], label=name, color=COLORS[name], alpha=0.75, linewidth=1)
    ax_pred.set_title("Model Predictions vs Actual — SPY returns (2023-2024)")
    ax_pred.legend(fontsize=7, loc="upper left", ncol=2)
    ax_pred.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax_pred.tick_params(axis="x", rotation=30)

    # --- Top-right: real metrics table ---
    ax_table.axis("off")
    ax_table.set_title("Metrics Table — MAE / RMSE (real)", loc="left")
    table_rows = []
    for name in MODEL_FILES:
        mae, rmse = mae_rmse(y, preds[name])
        table_rows.append([name, f"{mae:.5f}", f"{rmse:.5f}"])
    table = ax_table.table(cellText=table_rows, colLabels=["Model", "MAE", "RMSE"],
                            cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.9)

    # --- Bottom-left: real regime timeline ---
    ax_regime.plot(dates, prices, color="black", linewidth=1)
    ax_regime.set_title("Market Regime Timeline — SPY (real)")
    for i in range(1, len(dates)):
        c = REGIME_COLORS.get(regimes[i], "grey")
        alpha = 0.15 if regimes[i] != "neutral" else 0.05
        ax_regime.axvspan(dates[i - 1], dates[i], color=c, alpha=alpha)
    ax_regime.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax_regime.tick_params(axis="x", rotation=30)

    # --- Bottom-right: real regime comparison bar chart (the headline panel) ---
    models_order = list(MODEL_FILES.keys())
    volatile_maes = [mae_rmse(y[volatile_mask], preds[m][volatile_mask])[0] for m in models_order]
    stable_maes = [mae_rmse(y[stable_mask], preds[m][stable_mask])[0] for m in models_order]
    x = np.arange(len(models_order))
    width = 0.35
    bars1 = ax_bar.bar(x - width / 2, volatile_maes, width, label=f"Volatile (n={volatile_mask.sum()})", color="#d62728")
    bars2 = ax_bar.bar(x + width / 2, stable_maes, width, label=f"Stable (n={stable_mask.sum()})", color="#2ca02c", alpha=0.75)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(models_order, fontsize=8)
    ax_bar.set_ylabel("MAE")
    ax_bar.set_title("Model Comparison by Regime (real)")
    ax_bar.legend(fontsize=7)
    for bars in (bars1, bars2):
        for b in bars:
            ax_bar.annotate(f"{b.get_height():.4f}",
                             xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                             xytext=(0, 2), textcoords="offset points",
                             ha="center", fontsize=6.5)

    # --- Full-width: Ebbinghaus decay curves ---
    reference_date = pd.Timestamp("2022-12-31")
    days_ago = np.arange(0, 1500)
    curve_dates = [reference_date - pd.Timedelta(days=int(d)) for d in days_ago]
    curve_specs = [("Fast decay (S=30)", 30, "#d62728"),
                    ("Medium decay (S=180)", 180, "#ff7f0e"),
                    ("Slow decay (S=365)", 365, "#1f77b4")]
    for label, S, color in curve_specs:
        weights = [ebbinghaus_weight(d, reference_date, S) for d in curve_dates]
        ax_curves.plot(days_ago, weights, label=label, color=color, linewidth=2)
    ax_curves.set_xlabel("Days ago")
    ax_curves.set_ylabel("Weight (retention)")
    ax_curves.set_title("Ebbinghaus Forgetting Curves — Sample Weight vs Data Age")
    ax_curves.legend(fontsize=9)
    ax_curves.grid(alpha=0.3)

    # --- Footer ---
    fig.text(0.5, 0.005, "Memory That Fades  |  Team of 3  |  All results on held-out 2023-2024 SPY test set",
              ha="center", fontsize=8, color="grey")

    # --- Domain toggle button (stubbed until consumer data arrives) ---
    domain_ax = fig.add_axes([0.85, 0.955, 0.12, 0.03])
    domain_label = "Switch to Consumer" if CONSUMER_DATA_AVAILABLE else "Consumer (pending data)"
    domain_button = Button(domain_ax, domain_label, color="#eeeeee" if not CONSUMER_DATA_AVAILABLE else "#cde7d8")
    if not CONSUMER_DATA_AVAILABLE:
        domain_button.label.set_color("grey")

    def on_domain_click(event):
        if not CONSUMER_DATA_AVAILABLE:
            print("Consumer domain data not yet available -- waiting on data_consumer.csv "
                  "and model_baseline_consumer.pkl from Person 1.")
        else:
            print("Consumer domain switching not yet wired -- placeholder for post-delivery update.")

    domain_button.on_clicked(on_domain_click)

    # --- Date range slider ---
    slider_ax = fig.add_axes([0.15, 0.045, 0.55, 0.015])
    n = len(dates)
    date_slider = Slider(slider_ax, "Start idx", 0, n - 20, valinit=0, valstep=1)

    def update(val):
        start = int(date_slider.val)
        for name in MODEL_FILES:
            mae, rmse = mae_rmse(y[start:], preds[name][start:])
            row_idx = models_order.index(name) + 1
            table.get_celld()[(row_idx, 1)].get_text().set_text(f"{mae:.5f}")
            table.get_celld()[(row_idx, 2)].get_text().set_text(f"{rmse:.5f}")
        ax_pred.set_xlim(dates[start], dates[-1])
        fig.canvas.draw_idle()

    date_slider.on_changed(update)

    return fig, date_slider, domain_button  # keep refs alive


if __name__ == "__main__":
    fig, _slider, _button = build_dashboard()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(RESULTS_DIR, "dashboard_final_preview.png"), dpi=150, bbox_inches="tight")
    print("Final dashboard rendered. Preview saved to results/dashboard_final_preview.png")
    plt.show()  
