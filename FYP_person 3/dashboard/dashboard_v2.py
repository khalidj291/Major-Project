"""
Memory That Fades — Live Comparison Dashboard
Person 3 — Day 1, Step 4: real baseline model + real data connected.
Decay models (fast/medium/slow) are still placeholders until Person 2 delivers them.
Run: python dashboard_v2.py
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.widgets import Slider

DATA_PATH = r"C:\Users\koshe\OneDrive\Desktop\FY PROJECT\data\data_processed.csv"
BASELINE_MODEL_PATH = r"C:\Users\koshe\OneDrive\Desktop\FY PROJECT\models\model_baseline.pkl"
TICKER = "SPY"
WINDOW = 30
TEST_START = "2023-01-01"


# ---------------------------------------------------------------------------
# REAL DATA LOADING
# ---------------------------------------------------------------------------

def load_ticker_data(path=DATA_PATH, ticker=TICKER):
    df = pd.read_csv(path, parse_dates=["date"])
    df = df[df["ticker"] == ticker].sort_values("date").reset_index(drop=True)
    return df


def build_features(df, window=WINDOW):
    returns = df["returns"].values
    dates = df["date"].values
    X, y, sample_dates, prices = [], [], [], []
    close = df["close"].values
    regime = df["regime"].values if "regime" in df.columns else None
    regimes_out = []
    for i in range(window, len(returns) - 1):
        X.append(returns[i - window:i])
        y.append(returns[i + 1])
        sample_dates.append(dates[i + 1])
        prices.append(close[i + 1])
        if regime is not None:
            regimes_out.append(regime[i + 1])
    X = np.array(X)
    y = np.array(y)
    sample_dates = pd.to_datetime(sample_dates)
    prices = np.array(prices)
    regimes_out = np.array(regimes_out) if regime is not None else None
    return X, y, sample_dates, prices, regimes_out


def get_real_baseline_predictions():
    df = load_ticker_data()
    X, y, sample_dates, prices, regimes = build_features(df)

    with open(BASELINE_MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    preds = model.predict(X).flatten()

    # restrict everything to the test window for display
    mask = sample_dates >= pd.Timestamp(TEST_START)
    return (sample_dates[mask], y[mask], preds[mask], prices[mask],
            regimes[mask] if regimes is not None else None)


def compute_mae_rmse(actual, pred):
    err = np.abs(actual - pred)
    mae = err.mean()
    rmse = np.sqrt(((actual - pred) ** 2).mean())
    return mae, rmse


# ---------------------------------------------------------------------------
# PLACEHOLDER DATA for the 3 decay models — replace once Person 2 delivers
# ---------------------------------------------------------------------------

def make_placeholder_decay_preds(actual, seed, noise_scale):
    rng = np.random.default_rng(seed)
    return actual + rng.normal(0, noise_scale, len(actual))


def make_placeholder_bar_data():
    models = ["Baseline", "Fast", "Medium", "Slow"]
    volatile_mae = [0.028, 0.016, 0.020, 0.026]
    stable_mae = [0.015, 0.017, 0.014, 0.012]
    return models, volatile_mae, stable_mae


# ---------------------------------------------------------------------------
# DASHBOARD
# ---------------------------------------------------------------------------

def build_dashboard():
    dates, actual, baseline_pred, prices, regimes = get_real_baseline_predictions()

    # placeholder decay model predictions until Person 2 delivers real ones
    fast_pred = make_placeholder_decay_preds(actual, 1, 0.004)
    medium_pred = make_placeholder_decay_preds(actual, 2, 0.005)
    slow_pred = make_placeholder_decay_preds(actual, 3, 0.006)

    baseline_mae, baseline_rmse = compute_mae_rmse(actual, baseline_pred)
    fast_mae, fast_rmse = compute_mae_rmse(actual, fast_pred)
    medium_mae, medium_rmse = compute_mae_rmse(actual, medium_pred)
    slow_mae, slow_rmse = compute_mae_rmse(actual, slow_pred)

    metrics = {
        "Baseline": (baseline_mae, baseline_rmse),
        "Decay Fast (placeholder)": (fast_mae, fast_rmse),
        "Decay Medium (placeholder)": (medium_mae, medium_rmse),
        "Decay Slow (placeholder)": (slow_mae, slow_rmse),
    }

    models_bar, volatile_mae, stable_mae = make_placeholder_bar_data()

    plt.rcParams["font.size"] = 10
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5), facecolor="white")
    fig.suptitle("Memory That Fades — Live Comparison Dashboard  (SPY, real baseline)",
                  fontsize=15, fontweight="bold")

    # --- Top-left: predictions vs actual (REAL baseline, placeholder decay) ---
    ax_pred = axes[0, 0]
    line_actual, = ax_pred.plot(dates, actual, label="Actual", color="black", linewidth=1.3)
    line_baseline, = ax_pred.plot(dates, baseline_pred, label="Baseline (real)", color="grey", alpha=0.8)
    line_fast, = ax_pred.plot(dates, fast_pred, label="Decay Fast (placeholder)", color="#d62728", alpha=0.7)
    line_medium, = ax_pred.plot(dates, medium_pred, label="Decay Medium (placeholder)", color="#ff7f0e", alpha=0.7)
    line_slow, = ax_pred.plot(dates, slow_pred, label="Decay Slow (placeholder)", color="#1f77b4", alpha=0.7)
    ax_pred.set_title("Model Predictions vs Actual — SPY returns")
    ax_pred.legend(fontsize=7, loc="upper left")
    ax_pred.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax_pred.tick_params(axis="x", rotation=30)

    # --- Top-right: metrics table ---
    ax_table = axes[0, 1]
    ax_table.axis("off")
    ax_table.set_title("Metrics Table (MAE / RMSE)", loc="left")
    table_data = [[name, f"{mae:.4f}", f"{rmse:.4f}"] for name, (mae, rmse) in metrics.items()]
    table = ax_table.table(
        cellText=table_data,
        colLabels=["Model", "MAE", "RMSE"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)

    # --- Bottom-left: real regime timeline ---
    ax_regime = axes[1, 0]
    ax_regime.plot(dates, prices, color="black", linewidth=1)
    ax_regime.set_title("Market Regime Timeline — SPY (real)")
    regime_colors = {"volatile": "red", "stable": "green", "neutral": "grey"}
    for i in range(1, len(dates)):
        c = regime_colors.get(regimes[i], "grey")
        alpha = 0.12 if regimes[i] != "neutral" else 0.04
        ax_regime.axvspan(dates[i - 1], dates[i], color=c, alpha=alpha)
    ax_regime.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax_regime.tick_params(axis="x", rotation=30)

    # --- Bottom-right: regime comparison bar chart (placeholder until Day 3) ---
    ax_bar = axes[1, 1]
    x = np.arange(len(models_bar))
    width = 0.35
    bars1 = ax_bar.bar(x - width / 2, volatile_mae, width, label="Volatile", color="#d62728")
    bars2 = ax_bar.bar(x + width / 2, stable_mae, width, label="Stable", color="#2ca02c", alpha=0.7)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(models_bar)
    ax_bar.set_ylabel("MAE")
    ax_bar.set_title("Model Comparison by Regime (placeholder — Day 3)")
    ax_bar.legend(fontsize=8)
    for bars in (bars1, bars2):
        for b in bars:
            ax_bar.annotate(f"{b.get_height():.3f}",
                             xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                             xytext=(0, 2), textcoords="offset points",
                             ha="center", fontsize=7)

    fig.tight_layout(rect=[0, 0.06, 1, 0.96])

    # --- Interactive date range slider (controls top-left + metrics table) ---
    slider_ax = fig.add_axes([0.15, 0.015, 0.7, 0.02])
    n = len(dates)
    date_slider = Slider(slider_ax, "Start index", 0, n - 20, valinit=0, valstep=1)

    def update(val):
        start = int(date_slider.val)
        sub_actual = actual[start:]
        sub_baseline = baseline_pred[start:]
        mae, rmse = compute_mae_rmse(sub_actual, sub_baseline)
        table.get_celld()[(1, 1)].get_text().set_text(f"{mae:.4f}")
        table.get_celld()[(1, 2)].get_text().set_text(f"{rmse:.4f}")
        ax_pred.set_xlim(dates[start], dates[-1])
        fig.canvas.draw_idle()

    date_slider.on_changed(update)

    return fig, date_slider  # keep slider referenced so it isn't garbage collected


if __name__ == "__main__":
    fig, _slider = build_dashboard()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/dashboard_v2_preview.png", dpi=150)
    print("Dashboard rendered with REAL baseline model + REAL data.")
    print("Preview saved to results/dashboard_v2_preview.png")
    plt.show()  # uncomment when running with a display, to use the slider live
