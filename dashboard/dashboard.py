"""
Memory That Fades — Live Comparison Dashboard
Person 3

Consolidated from dashboard_v2.py / dashboard_final.py / dashboard_dual_domain.py
(the most complete of the three -- dual_domain had a real financial+consumer
toggle already working; v2 and final were earlier, less complete drafts).

Fixes applied during integration into the main repo structure:
  - Paths now point at the real, single-source project structure
    (baseline_model/models, decay_model/models, data/) instead of a
    self-contained local copy that had drifted out of sync.
  - Windowing had the same one-day feature/target offset bug that was
    found and fixed everywhere else in the project (target was
    returns[i+1] instead of returns[i]) -- now imports the shared,
    tested windowing.py used by every other script instead of a local
    reimplementation.
  - Added the disease domain (didn't exist yet when this was first built).

Run: python dashboard/dashboard.py
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
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # dashboard/ sits directly under project root
sys.path.append(os.path.join(PROJECT_ROOT, "decay_model"))
from ebbinghaus import ebbinghaus_weight
from windowing import make_windows  # shared, tested -- see decay_model/windowing.py

BASELINE_MODELS_DIR = os.path.join(PROJECT_ROOT, "baseline_model", "models")
DECAY_MODELS_DIR = os.path.join(PROJECT_ROOT, "decay_model", "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

COLORS = {
    "Baseline": "grey",
    "Decay Fast": "#d62728",
    "Decay Medium": "#ff7f0e",
    "Decay Slow": "#1f77b4",
}
REGIME_COLORS = {"volatile": "red", "stable": "green", "neutral": "grey"}


def _load_models(model_suffix):
    """model_suffix: '' for financial, '_consumer', or '_disease'."""
    files = {
        "Baseline": os.path.join(BASELINE_MODELS_DIR, f"model_baseline{model_suffix}.pkl"),
        "Decay Fast": os.path.join(DECAY_MODELS_DIR, f"model_decay_fast{model_suffix}.pkl"),
        "Decay Medium": os.path.join(DECAY_MODELS_DIR, f"model_decay_medium{model_suffix}.pkl"),
        "Decay Slow": os.path.join(DECAY_MODELS_DIR, f"model_decay_slow{model_suffix}.pkl"),
    }
    models = {}
    for name, path in files.items():
        with open(path, "rb") as f:
            models[name] = pickle.load(f)
    return models


def _predict_all(models, X):
    return {name: m.predict(X).flatten() for name, m in models.items()}


def load_financial_domain():
    df = pd.read_csv(os.path.join(DATA_DIR, "data_processed.csv"), parse_dates=["date"])
    df = df[df["ticker"] == "SPY"].sort_values("date").reset_index(drop=True)

    X, y, sample_dates = make_windows(df, "2023-01-01", "2024-12-31", 30, ticker="SPY")
    # price + regime lookups aligned to sample_dates, for the timeline/bar panels
    lookup = df.set_index("date")
    prices = lookup.loc[pd.to_datetime(sample_dates), "close"].values
    regimes = lookup.loc[pd.to_datetime(sample_dates), "regime"].values

    models = _load_models("")
    preds = _predict_all(models, X)
    return {
        "domain_label": "Financial (SPY, daily)",
        "dates": pd.to_datetime(sample_dates), "actual": y.flatten(), "preds": preds,
        "prices": prices, "regimes": regimes, "has_regime": True,
    }


def load_consumer_domain():
    consumer_regime_path = os.path.join(DATA_DIR, "data_consumer_with_regime.csv")
    consumer_plain_path = os.path.join(DATA_DIR, "data_consumer.csv")
    has_regime_file = os.path.exists(consumer_regime_path)
    path = consumer_regime_path if has_regime_file else consumer_plain_path

    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    X, y, sample_dates = make_windows(df, "2023-01-01", "2024-12-31", 12)

    lookup = df.set_index("date")
    prices = lookup.loc[pd.to_datetime(sample_dates), "close"].values
    regimes = lookup.loc[pd.to_datetime(sample_dates), "regime"].values if has_regime_file else None

    models = _load_models("_consumer")
    preds = _predict_all(models, X)
    return {
        "domain_label": "Consumer (PCE, monthly)",
        "dates": pd.to_datetime(sample_dates), "actual": y.flatten(), "preds": preds,
        "prices": prices, "regimes": regimes, "has_regime": has_regime_file,
    }


def load_disease_domain():
    df = pd.read_csv(os.path.join(DATA_DIR, "data_disease.csv"), parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    # This domain's own test boundary -- see train_baseline_disease.py for why
    # it differs from the 2023-2024 boundary the other domains use (OWID's US
    # case data only runs through March 2023).
    X, y, sample_dates = make_windows(df, "2022-04-03", df["date"].max(), 30)

    lookup = df.set_index("date")
    prices = lookup.loc[pd.to_datetime(sample_dates), "close"].values

    models = _load_models("_disease")
    preds = _predict_all(models, X)
    return {
        "domain_label": "Disease (US COVID cases, daily)",
        "dates": pd.to_datetime(sample_dates), "actual": y.flatten(), "preds": preds,
        "prices": prices, "regimes": None, "has_regime": False,
    }


def mae_rmse(actual, pred):
    err = actual - pred
    return np.abs(err).mean(), np.sqrt((err ** 2).mean())


def build_dashboard():
    domains = {
        "financial": load_financial_domain(),
        "consumer": load_consumer_domain(),
        "disease": load_disease_domain(),
    }
    domain_order = ["financial", "consumer", "disease"]
    state = {"domain": "financial"}

    plt.rcParams["font.size"] = 10
    fig = plt.figure(figsize=(15, 12), facecolor="white")
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.7], hspace=0.55, wspace=0.28)
    ax_pred = fig.add_subplot(gs[0, 0])
    ax_table = fig.add_subplot(gs[0, 1])
    ax_regime = fig.add_subplot(gs[1, 0])
    ax_bar = fig.add_subplot(gs[1, 1])
    ax_curves = fig.add_subplot(gs[2, :])

    title = fig.suptitle("Memory That Fades — Live Comparison Dashboard", fontsize=16, fontweight="bold", y=0.995)

    def draw(domain_key):
        d = domains[domain_key]
        dates, y, preds, prices = d["dates"], d["actual"], d["preds"], d["prices"]
        model_names = list(preds.keys())

        title.set_text(f"Memory That Fades — Live Comparison Dashboard  ({d['domain_label']})")

        ax_pred.clear()
        ax_pred.plot(dates, y, label="Actual", color="black", linewidth=1.3)
        for name in model_names:
            ax_pred.plot(dates, preds[name], label=name, color=COLORS[name], alpha=0.75, linewidth=1)
        ax_pred.set_title(f"Predictions vs Actual — {d['domain_label']}", fontsize=10)
        ax_pred.legend(fontsize=7, loc="upper left", ncol=2)
        ax_pred.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax_pred.tick_params(axis="x", rotation=30)

        ax_table.clear()
        ax_table.axis("off")
        ax_table.set_title("Metrics Table — MAE / RMSE (real)", loc="left")
        rows = []
        for name in model_names:
            mae, rmse = mae_rmse(y, preds[name])
            rows.append([name, f"{mae:.5f}", f"{rmse:.5f}"])
        table = ax_table.table(cellText=rows, colLabels=["Model", "MAE", "RMSE"], cellLoc="center", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.9)

        ax_regime.clear()
        ax_regime.plot(dates, prices, color="black", linewidth=1)
        if d["has_regime"]:
            regimes = d["regimes"]
            ax_regime.set_title(f"Regime Timeline — {d['domain_label']}", fontsize=10)
            for i in range(1, len(dates)):
                c = REGIME_COLORS.get(regimes[i], "grey")
                alpha = 0.15 if regimes[i] != "neutral" else 0.05
                ax_regime.axvspan(dates[i - 1], dates[i], color=c, alpha=alpha)
        else:
            ax_regime.set_title(f"Price/Case Level — {d['domain_label']}", fontsize=10)
            ax_regime.text(0.02, 0.03, "(no regime labels for this domain)",
                            transform=ax_regime.transAxes, fontsize=7.5, color="grey", style="italic")
        ax_regime.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax_regime.tick_params(axis="x", rotation=30)

        ax_bar.clear()
        if d["has_regime"]:
            regimes = d["regimes"]
            vol_mask, stab_mask = regimes == "volatile", regimes == "stable"
            vol_maes = [mae_rmse(y[vol_mask], preds[m][vol_mask])[0] for m in model_names]
            stab_maes = [mae_rmse(y[stab_mask], preds[m][stab_mask])[0] for m in model_names]
            x = np.arange(len(model_names))
            width = 0.35
            b1 = ax_bar.bar(x - width / 2, vol_maes, width, label=f"Volatile (n={vol_mask.sum()})", color="#d62728")
            b2 = ax_bar.bar(x + width / 2, stab_maes, width, label=f"Stable (n={stab_mask.sum()})", color="#2ca02c", alpha=0.75)
            for bars in (b1, b2):
                for b in bars:
                    ax_bar.annotate(f"{b.get_height():.4f}", xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                                     xytext=(0, 2), textcoords="offset points", ha="center", fontsize=6.5)
            ax_bar.set_title("Comparison by Regime (real)", fontsize=10)
            ax_bar.legend(fontsize=7)
        else:
            overall_maes = [mae_rmse(y, preds[m])[0] for m in model_names]
            bars = ax_bar.bar(model_names, overall_maes, color=[COLORS[m] for m in model_names], alpha=0.85)
            for b in bars:
                ax_bar.annotate(f"{b.get_height():.4f}", xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                                 xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7)
            ax_bar.set_title(f"Overall Comparison — {d['domain_label']}", fontsize=10)
        ax_bar.set_xticks(range(len(model_names)))
        ax_bar.set_xticklabels(model_names, fontsize=8)
        ax_bar.set_ylabel("MAE")

        fig.canvas.draw_idle()

    reference_date = pd.Timestamp("2022-12-31")
    days_ago = np.arange(0, 1500)
    curve_dates = [reference_date - pd.Timedelta(days=int(dd)) for dd in days_ago]
    for label, S, color in [("Fast decay (S=30)", 30, "#d62728"),
                             ("Medium decay (S=180)", 180, "#ff7f0e"),
                             ("Slow decay (S=365)", 365, "#1f77b4")]:
        weights = [ebbinghaus_weight(dd, reference_date, S) for dd in curve_dates]
        ax_curves.plot(days_ago, weights, label=label, color=color, linewidth=2)
    ax_curves.set_xlabel("Days ago")
    ax_curves.set_ylabel("Weight (retention)")
    ax_curves.set_title("Ebbinghaus Forgetting Curves — Sample Weight vs Data Age")
    ax_curves.legend(fontsize=9)
    ax_curves.grid(alpha=0.3)

    fig.text(0.5, 0.005, "Memory That Fades  |  Team of 3  |  Financial/Consumer: 2023-2024 test set  |  Disease: 2022-04 to 2023-03 test set",
              ha="center", fontsize=8, color="grey")

    domain_ax = fig.add_axes([0.83, 0.925, 0.14, 0.035])
    domain_button = Button(domain_ax, "Switch domain \u2192 Consumer", color="#cde7d8", hovercolor="#a8d5b8")
    domain_button.label.set_fontsize(9)

    def on_domain_click(event):
        idx = domain_order.index(state["domain"])
        state["domain"] = domain_order[(idx + 1) % len(domain_order)]
        next_domain = domain_order[(domain_order.index(state["domain"]) + 1) % len(domain_order)]
        domain_button.label.set_text(f"Switch domain \u2192 {next_domain.capitalize()}")
        draw(state["domain"])

    domain_button.on_clicked(on_domain_click)

    draw("financial")
    return fig, domain_button, draw, state


if __name__ == "__main__":
    fig, _button, draw, state = build_dashboard()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for domain in ["financial", "consumer", "disease"]:
        draw(domain)
        plt.savefig(os.path.join(RESULTS_DIR, f"dashboard_{domain}_view.png"), dpi=150, bbox_inches="tight")
        print(f"Saved {domain}-view screenshot.")

    draw("financial")
    print("Run with plt.show() uncommented, then click the toggle button to cycle through domains.")
    plt.show()  # uncomment when running with a display