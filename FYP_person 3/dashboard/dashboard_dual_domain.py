"""
Memory That Fades — Live Comparison Dashboard (FINAL, dual-domain)
Person 3 — Day 4 complete

Toggle button switches between financial (SPY, daily, regime-labeled) and
consumer (PCE, monthly, no regime labels yet) domains. All panels redraw.

Run: python dashboard_dual_domain.py
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.widgets import Slider, Button

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(os.path.join(PROJECT_ROOT, "models"))
from ebbinghaus import ebbinghaus_weight

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
TEST_START = pd.Timestamp("2023-01-01")

COLORS = {
    "Baseline": "grey",
    "Decay Fast": "#d62728",
    "Decay Medium": "#ff7f0e",
    "Decay Slow": "#1f77b4",
}
REGIME_COLORS = {"volatile": "red", "stable": "green", "neutral": "grey"}


# ---------------------------------------------------------------------------
# FINANCIAL DOMAIN (daily, SPY, has regime labels)
# ---------------------------------------------------------------------------

FIN_MODEL_FILES = {
    "Baseline": "model_baseline.pkl",
    "Decay Fast": "model_decay_fast.pkl",
    "Decay Medium": "model_decay_medium.pkl",
    "Decay Slow": "model_decay_slow.pkl",
}
FIN_WINDOW = 30


def load_financial_domain():
    df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "data_processed.csv"), parse_dates=["date"])
    df = df[df["ticker"] == "SPY"].sort_values("date").reset_index(drop=True)

    returns = df["returns"].values
    dates = df["date"].values
    close = df["close"].values
    regime = df["regime"].values

    X, y, sample_dates, prices, regimes = [], [], [], [], []
    for i in range(FIN_WINDOW, len(returns) - 1):
        X.append(returns[i - FIN_WINDOW:i])
        y.append(returns[i + 1])
        sample_dates.append(dates[i + 1])
        prices.append(close[i + 1])
        regimes.append(regime[i + 1])
    X, y = np.array(X), np.array(y)
    sample_dates = pd.to_datetime(sample_dates)
    prices, regimes = np.array(prices), np.array(regimes)
    mask = sample_dates >= TEST_START
    X, y, sample_dates, prices, regimes = X[mask], y[mask], sample_dates[mask], prices[mask], regimes[mask]

    preds = {}
    for name, fname in FIN_MODEL_FILES.items():
        with open(os.path.join(MODELS_DIR, fname), "rb") as f:
            model = pickle.load(f)
        preds[name] = model.predict(X).flatten()

    return {
        "domain_label": "Financial (SPY, daily)",
        "dates": sample_dates, "actual": y, "preds": preds, "prices": prices,
        "regimes": regimes, "has_regime": True,
    }


# ---------------------------------------------------------------------------
# CONSUMER DOMAIN (monthly, PCE, no regime labels)
# ---------------------------------------------------------------------------

CONS_MODEL_FILES = {
    "Baseline": "model_baseline_consumer.pkl",
    "Decay Fast": "model_decay_fast_consumer.pkl",
    "Decay Medium": "model_decay_medium_consumer.pkl",
    "Decay Slow": "model_decay_slow_consumer.pkl",
}
CONS_WINDOW = 12


def load_consumer_domain():
    consumer_regime_path = os.path.join(PROJECT_ROOT, "data", "data_consumer_with_regime.csv")
    consumer_plain_path = os.path.join(PROJECT_ROOT, "data", "data_consumer.csv")
    has_regime_file = os.path.exists(consumer_regime_path)
    path = consumer_regime_path if has_regime_file else consumer_plain_path

    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    returns = df["returns"].values
    dates = df["date"].values
    close = df["close"].values
    regime = df["regime"].values if has_regime_file else None

    X, y, sample_dates, prices, regimes = [], [], [], [], []
    for i in range(CONS_WINDOW, len(returns) - 1):
        X.append(returns[i - CONS_WINDOW:i])
        y.append(returns[i + 1])
        sample_dates.append(dates[i + 1])
        prices.append(close[i + 1])
        if has_regime_file:
            regimes.append(regime[i + 1])
    X, y = np.array(X), np.array(y)
    sample_dates = pd.to_datetime(sample_dates)
    prices = np.array(prices)
    regimes = np.array(regimes) if has_regime_file else None
    mask = sample_dates >= TEST_START
    X, y, sample_dates, prices = X[mask], y[mask], sample_dates[mask], prices[mask]
    if has_regime_file:
        regimes = regimes[mask]

    preds = {}
    for name, fname in CONS_MODEL_FILES.items():
        with open(os.path.join(MODELS_DIR, fname), "rb") as f:
            model = pickle.load(f)
        preds[name] = model.predict(X).flatten()

    return {
        "domain_label": "Consumer (PCE, monthly)",
        "dates": sample_dates, "actual": y, "preds": preds, "prices": prices,
        "regimes": regimes, "has_regime": has_regime_file,
    }


def mae_rmse(actual, pred):
    err = actual - pred
    return np.abs(err).mean(), np.sqrt((err ** 2).mean())


# ---------------------------------------------------------------------------
# DASHBOARD
# ---------------------------------------------------------------------------

def build_dashboard():
    domains = {"financial": load_financial_domain(), "consumer": load_consumer_domain()}
    state = {"domain": "financial"}

    plt.rcParams["font.size"] = 10
    fig = plt.figure(figsize=(15, 12), facecolor="white")
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.7], hspace=0.55, wspace=0.28)
    ax_pred = fig.add_subplot(gs[0, 0])
    ax_table = fig.add_subplot(gs[0, 1])
    ax_regime = fig.add_subplot(gs[1, 0])
    ax_bar = fig.add_subplot(gs[1, 1])
    ax_curves = fig.add_subplot(gs[2, :])

    title = fig.suptitle("Memory That Fades — Live Comparison Dashboard", fontsize=16, fontweight="bold", y=0.98)

    def draw(domain_key):
        d = domains[domain_key]
        dates, y, preds, prices = d["dates"], d["actual"], d["preds"], d["prices"]

        title.set_text(f"Memory That Fades — Live Comparison Dashboard  ({d['domain_label']})")

        # --- predictions panel ---
        ax_pred.clear()
        ax_pred.plot(dates, y, label="Actual", color="black", linewidth=1.3)
        for name in FIN_MODEL_FILES:
            ax_pred.plot(dates, preds[name], label=name, color=COLORS[name], alpha=0.75, linewidth=1)
        ax_pred.set_title(f"Model Predictions vs Actual — {d['domain_label']}")
        ax_pred.legend(fontsize=7, loc="upper left", ncol=2)
        ax_pred.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax_pred.tick_params(axis="x", rotation=30)

        # --- metrics table ---
        ax_table.clear()
        ax_table.axis("off")
        ax_table.set_title("Metrics Table — MAE / RMSE (real)", loc="left")
        rows = []
        for name in FIN_MODEL_FILES:
            mae, rmse = mae_rmse(y, preds[name])
            rows.append([name, f"{mae:.5f}", f"{rmse:.5f}"])
        table = ax_table.table(cellText=rows, colLabels=["Model", "MAE", "RMSE"], cellLoc="center", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.9)

        # --- regime timeline / price panel ---
        ax_regime.clear()
        ax_regime.plot(dates, prices, color="black", linewidth=1)
        if d["has_regime"]:
            regimes = d["regimes"]
            ax_regime.set_title(f"Market Regime Timeline — {d['domain_label']}")
            for i in range(1, len(dates)):
                c = REGIME_COLORS.get(regimes[i], "grey")
                alpha = 0.15 if regimes[i] != "neutral" else 0.05
                ax_regime.axvspan(dates[i - 1], dates[i], color=c, alpha=alpha)
        else:
            ax_regime.set_title(f"Price Level — {d['domain_label']}  (no regime labels available yet)")
        ax_regime.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax_regime.tick_params(axis="x", rotation=30)

        # --- comparison bar chart ---
        ax_bar.clear()
        models_order = list(FIN_MODEL_FILES.keys())
        if d["has_regime"]:
            regimes = d["regimes"]
            vol_mask, stab_mask = regimes == "volatile", regimes == "stable"
            vol_maes = [mae_rmse(y[vol_mask], preds[m][vol_mask])[0] for m in models_order]
            stab_maes = [mae_rmse(y[stab_mask], preds[m][stab_mask])[0] for m in models_order]
            x = np.arange(len(models_order))
            width = 0.35
            b1 = ax_bar.bar(x - width / 2, vol_maes, width, label=f"Volatile (n={vol_mask.sum()})", color="#d62728")
            b2 = ax_bar.bar(x + width / 2, stab_maes, width, label=f"Stable (n={stab_mask.sum()})", color="#2ca02c", alpha=0.75)
            for bars in (b1, b2):
                for b in bars:
                    ax_bar.annotate(f"{b.get_height():.4f}", xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                                     xytext=(0, 2), textcoords="offset points", ha="center", fontsize=6.5)
            ax_bar.set_title("Model Comparison by Regime (real)")
            ax_bar.legend(fontsize=7)
        else:
            overall_maes = [mae_rmse(y, preds[m])[0] for m in models_order]
            bars = ax_bar.bar(models_order, overall_maes, color=[COLORS[m] for m in models_order], alpha=0.85)
            for b in bars:
                ax_bar.annotate(f"{b.get_height():.4f}", xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                                 xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7)
            ax_bar.set_title("Overall Model Comparison (no regime labels yet — consumer domain)")
        ax_bar.set_xticks(range(len(models_order)))
        ax_bar.set_xticklabels(models_order, fontsize=8)
        ax_bar.set_ylabel("MAE")

        fig.canvas.draw_idle()

    # --- Ebbinghaus curves panel (domain-independent, drawn once) ---
    reference_date = pd.Timestamp("2022-12-31")
    days_ago = np.arange(0, 1500)
    curve_dates = [reference_date - pd.Timedelta(days=int(d)) for d in days_ago]
    for label, S, color in [("Fast decay (S=30)", 30, "#d62728"),
                             ("Medium decay (S=180)", 180, "#ff7f0e"),
                             ("Slow decay (S=365)", 365, "#1f77b4")]:
        weights = [ebbinghaus_weight(d, reference_date, S) for d in curve_dates]
        ax_curves.plot(days_ago, weights, label=label, color=color, linewidth=2)
    ax_curves.set_xlabel("Days ago")
    ax_curves.set_ylabel("Weight (retention)")
    ax_curves.set_title("Ebbinghaus Forgetting Curves — Sample Weight vs Data Age")
    ax_curves.legend(fontsize=9)
    ax_curves.grid(alpha=0.3)

    fig.text(0.5, 0.005, "Memory That Fades  |  Team of 3  |  All results on held-out 2023-2024 test set",
              ha="center", fontsize=8, color="grey")

    # --- domain toggle button, now fully functional ---
    domain_ax = fig.add_axes([0.83, 0.955, 0.14, 0.03])
    domain_button = Button(domain_ax, "Switch to Consumer", color="#cde7d8")

    def on_domain_click(event):
        state["domain"] = "consumer" if state["domain"] == "financial" else "financial"
        domain_button.label.set_text(
            "Switch to Financial" if state["domain"] == "consumer" else "Switch to Consumer"
        )
        draw(state["domain"])

    domain_button.on_clicked(on_domain_click)

    draw("financial")
    return fig, domain_button, draw, state


if __name__ == "__main__":
    fig, _button, draw, state = build_dashboard()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    draw("financial")
    plt.savefig(os.path.join(RESULTS_DIR, "dashboard_dual_financial_view.png"), dpi=150, bbox_inches="tight")
    print("Saved financial-view screenshot.")

    draw("consumer")
    plt.savefig(os.path.join(RESULTS_DIR, "dashboard_dual_consumer_view.png"), dpi=150, bbox_inches="tight")
    print("Saved consumer-view screenshot.")

    print("Run with plt.show() uncommented, then click the toggle button to switch live.")
    plt.show()  # uncomment when running with a display
