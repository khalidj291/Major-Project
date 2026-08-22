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
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # dashboard/ sits directly under project root
sys.path.append(os.path.join(PROJECT_ROOT, "decay_model"))
from ebbinghaus import ebbinghaus_weight
from windowing import make_windows  # shared, tested -- see decay_model/windowing.py
from feature_decay import apply_feature_decay  # new mechanism -- see decay_model/feature_decay.py

BASELINE_MODELS_DIR = os.path.join(PROJECT_ROOT, "baseline_model", "models")
DECAY_MODELS_DIR = os.path.join(PROJECT_ROOT, "decay_model", "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

COLORS = {
    "Baseline": "grey",
    "Decay Fast": "#d62728",
    "Decay Medium": "#ff7f0e",
    "Decay Slow": "#1f77b4",
    "Feature Decay": "#9467bd",
    "Row Decay": "#8c564b",
    "Combined Decay": "#2ca02c",
}
REGIME_COLORS = {"volatile": "red", "stable": "green", "neutral": "grey"}


def _load_models(model_suffix, include_feature_decay=False):
    """model_suffix: '' for financial, '_consumer', or '_disease'."""
    files = {
        "Baseline": os.path.join(BASELINE_MODELS_DIR, f"model_baseline{model_suffix}.pkl"),
        "Decay Fast": os.path.join(DECAY_MODELS_DIR, f"model_decay_fast{model_suffix}.pkl"),
        "Decay Medium": os.path.join(DECAY_MODELS_DIR, f"model_decay_medium{model_suffix}.pkl"),
        "Decay Slow": os.path.join(DECAY_MODELS_DIR, f"model_decay_slow{model_suffix}.pkl"),
    }
    if include_feature_decay:
        files["Feature Decay"] = os.path.join(DECAY_MODELS_DIR, f"model_feature_decay{model_suffix}.pkl")
    models = {}
    for name, path in files.items():
        if not os.path.exists(path):
            continue  # e.g. BTC has no Decay Fast/Medium/Slow -- only Baseline + Feature Decay exist for it
        with open(path, "rb") as f:
            models[name] = pickle.load(f)
    return models


def _predict_one(model_entry, X, window):
    """A plain model is a fitted Ridge -- predict directly.
    Dict-based models come in 3 shapes (see train_feature_decay_btc.py /
    train_row_decay_btc.py / train_combined_decay_btc.py):
      - {"model","S","window"}            -> feature-decay: X must be transformed
      - {"model","feature_S","row_S",...} -> combined: X transformed by feature_S only
                                              (row_S only affected training via sample_weight)
      - {"model","S"} only (no window)    -> row-decay: predict on raw X, no transform"""
    if isinstance(model_entry, dict) and "model" in model_entry:
        if "feature_S" in model_entry:
            X_transformed = apply_feature_decay(X, model_entry["window"], model_entry["feature_S"])
            return model_entry["model"].predict(X_transformed).flatten()
        if "window" in model_entry:
            X_transformed = apply_feature_decay(X, model_entry["window"], model_entry["S"])
            return model_entry["model"].predict(X_transformed).flatten()
        return model_entry["model"].predict(X).flatten()
    return model_entry.predict(X).flatten()


def _predict_all(models, X, window):
    return {name: _predict_one(m, X, window) for name, m in models.items()}


def load_financial_domain():
    df = pd.read_csv(os.path.join(DATA_DIR, "data_processed.csv"), parse_dates=["date"])
    df = df[df["ticker"] == "SPY"].sort_values("date").reset_index(drop=True)

    X, y, sample_dates = make_windows(df, "2023-01-01", "2024-12-31", 30, ticker="SPY")
    # price + regime lookups aligned to sample_dates, for the timeline/bar panels
    lookup = df.set_index("date")
    prices = lookup.loc[pd.to_datetime(sample_dates), "close"].values
    regimes = lookup.loc[pd.to_datetime(sample_dates), "regime"].values

    models = _load_models("")
    preds = _predict_all(models, X, 30)
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
    preds = _predict_all(models, X, 12)
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

    models = _load_models("_disease", include_feature_decay=True)
    preds = _predict_all(models, X, 30)
    return {
        "domain_label": "Disease (US COVID cases, daily)",
        "dates": pd.to_datetime(sample_dates), "actual": y.flatten(), "preds": preds,
        "prices": prices, "regimes": None, "has_regime": False,
    }


def _load_models_btc():
    """BTC now has 4 models: Baseline, Feature Decay, Row Decay, and the
    headline Combined Decay result."""
    files = {
        "Baseline": os.path.join(BASELINE_MODELS_DIR, "model_baseline_btc.pkl"),
        "Row Decay": os.path.join(DECAY_MODELS_DIR, "model_row_decay_btc.pkl"),
        "Feature Decay": os.path.join(DECAY_MODELS_DIR, "model_feature_decay_btc.pkl"),
        "Combined Decay": os.path.join(DECAY_MODELS_DIR, "model_combined_decay_btc.pkl"),
    }
    models = {}
    for name, path in files.items():
        if not os.path.exists(path):
            continue
        with open(path, "rb") as f:
            models[name] = pickle.load(f)
    return models


def load_btc_domain():
    df = pd.read_csv(os.path.join(DATA_DIR, "data_btc.csv"), parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    # UPDATED test period -- matches the retrained models (train through 2022, val=2023).
    # The old 2020 window is now INSIDE these models' training data -- using it here would
    # silently leak and show fake results, so this must stay in sync with the training scripts.
    X, y, sample_dates = make_windows(df, "2024-01-01", df["date"].max(), 30)

    lookup = df.set_index("date")
    prices = lookup.loc[pd.to_datetime(sample_dates), "close"].values

    models = _load_models_btc()
    preds = _predict_all(models, X, 30)
    return {
        "domain_label": "Crypto (BTC-USD, daily)",
        "dates": pd.to_datetime(sample_dates), "actual": y.flatten(), "preds": preds,
        "prices": prices, "regimes": None, "has_regime": False,
    }


def mae_rmse(actual, pred):
    err = actual - pred
    return np.abs(err).mean(), np.sqrt((err ** 2).mean())


MAX_MODEL_SLOTS = 5  # disease is the largest set (Baseline + Fast/Med/Slow + Feature Decay) --
                      # every domain's bar chart reserves this many slots so bar WIDTH and
                      # SPACING never change with domain. Fewer models just leave empty slots.


def build_dashboard():
    domains = {
        "financial": load_financial_domain(),
        "consumer": load_consumer_domain(),
        "disease": load_disease_domain(),
        "btc": load_btc_domain(),
    }
    domain_order = ["financial", "consumer", "disease", "btc"]
    state = {"domain": "financial"}

    plt.rcParams["font.size"] = 10
    fig = plt.figure(figsize=(15, 12), facecolor="white")
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.7], hspace=0.6, wspace=0.3)
    ax_pred = fig.add_subplot(gs[0, 0])
    ax_table = fig.add_subplot(gs[0, 1])
    ax_context = fig.add_subplot(gs[1, 0])
    ax_bar = fig.add_subplot(gs[1, 1])
    ax_curves = fig.add_subplot(gs[2, :])

    title = fig.suptitle("Memory That Fades — Live Comparison Dashboard", fontsize=16, fontweight="bold", y=0.995)

    def draw(domain_key):
        d = domains[domain_key]
        dates, y, preds, prices = d["dates"], d["actual"], d["preds"], d["prices"]
        model_names = list(preds.keys())
        n = len(model_names)

        title.set_text(f"Memory That Fades — Live Comparison Dashboard  ({d['domain_label']})")

        # --- predictions vs actual ---
        ax_pred.clear()
        ax_pred.plot(dates, y, label="Actual", color="black", linewidth=1.3)
        for name in model_names:
            ax_pred.plot(dates, preds[name], label=name, color=COLORS[name], alpha=0.75, linewidth=1)
        ax_pred.set_title(f"Predictions vs Actual — {d['domain_label']}", fontsize=10)
        # fixed legend shape every time: single column, small font, same corner -- height
        # grows/shrinks with model count but width and position never move
        ax_pred.legend(fontsize=6.5, loc="upper left", ncol=1, framealpha=0.9)
        ax_pred.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax_pred.tick_params(axis="x", rotation=30)

        # --- metrics table: fixed row height, top-anchored, so row COUNT changes total
        # height but never squishes/stretches individual rows differently per domain ---
        ax_table.clear()
        ax_table.axis("off")
        if domain_key == "btc" and "Baseline" in preds:
            # BTC-only: show % improvement vs baseline and p-value so the panel doesn't
            # have to eyeball small MAE differences -- other domains' tables untouched
            ax_table.set_title("Metrics Table — MAE / RMSE / % vs Baseline / p-value", loc="left", fontsize=10)
            base_err_row = np.abs(y - preds["Baseline"])
            base_mae = base_err_row.mean()
            rows = []
            for name in model_names:
                err = np.abs(y - preds[name])
                mae, rmse = mae_rmse(y, preds[name])
                pct = 100 * (base_mae - mae) / base_mae
                if name == "Baseline":
                    pct_str, p_str = "—", "—"
                else:
                    _, p_val = stats.ttest_rel(err, base_err_row)
                    pct_str = f"{pct:+.2f}%"
                    p_str = f"{p_val:.4f}" + (" *" if p_val < 0.05 else "")
                rows.append([name, f"{mae:.5f}", f"{rmse:.5f}", pct_str, p_str])
            col_labels = ["Model", "MAE", "RMSE", "% vs Baseline", "p-value"]
        else:
            ax_table.set_title("Metrics Table — MAE / RMSE (real)", loc="left", fontsize=10)
            rows = [[name, f"{mae_rmse(y, preds[name])[0]:.5f}", f"{mae_rmse(y, preds[name])[1]:.5f}"] for name in model_names]
            col_labels = ["Model", "MAE", "RMSE"]
        row_h = 0.14  # fixed height per row, identical across every domain
        table = ax_table.table(cellText=rows, colLabels=col_labels, cellLoc="center",
                                bbox=[0.0, 1.0 - row_h * (n + 1), 1.0, row_h * (n + 1)])
        table.auto_set_font_size(False)
        table.set_fontsize(9 if domain_key != "btc" else 8)

        # --- context panel: price/level line, with regime shading overlaid IF this
        # domain has it -- the plot itself is always the same shape either way ---
        ax_context.clear()
        ax_context.plot(dates, prices, color="black", linewidth=1)
        if d["has_regime"]:
            regimes = d["regimes"]
            ax_context.set_title(f"Price/Level (with regime shading) — {d['domain_label']}", fontsize=10)
            for i in range(1, len(dates)):
                c = REGIME_COLORS.get(regimes[i], "grey")
                alpha = 0.15 if regimes[i] != "neutral" else 0.05
                ax_context.axvspan(dates[i - 1], dates[i], color=c, alpha=alpha)
        else:
            ax_context.set_title(f"Price/Case Level — {d['domain_label']}", fontsize=10)
            ax_context.text(0.02, 0.03, "(no regime labels for this domain)",
                             transform=ax_context.transAxes, fontsize=7.5, color="grey", style="italic")
        ax_context.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax_context.tick_params(axis="x", rotation=30)

        # --- bar chart: ALWAYS the same chart type (overall MAE per model), ALWAYS
        # MAX_MODEL_SLOTS-wide x-axis and fixed bar width -- 2 models (BTC) and 5
        # models (disease) render at the identical bar width and spacing, just with
        # unused slots left empty rather than the bars stretching to fill the panel ---
        ax_bar.clear()
        overall_maes = [mae_rmse(y, preds[m])[0] for m in model_names]
        x = np.arange(n)
        bars = ax_bar.bar(x, overall_maes, width=0.55, color=[COLORS[m] for m in model_names], alpha=0.85)
        for b in bars:
            ax_bar.annotate(f"{b.get_height():.4f}", xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                             xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7.5)
        ax_bar.set_title(f"Overall Comparison — {d['domain_label']}", fontsize=10)
        ax_bar.set_xlim(-0.5, MAX_MODEL_SLOTS - 0.5)  # fixed regardless of n -- this is the key fix
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(model_names, fontsize=8, rotation=10 if n >= 4 else 0)
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

    fig.text(0.5, 0.005, "Memory That Fades  |  Team of 3  |  Financial/Consumer: 2023-2024 test set  |  Disease: 2022-04 to 2023-03 test set  |  BTC: 2024-2026 test set",
              ha="center", fontsize=8, color="grey")

    domain_ax = fig.add_axes((0.83, 0.925, 0.14, 0.035))
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

    for domain in ["financial", "consumer", "disease", "btc"]:
        draw(domain)
        plt.savefig(os.path.join(RESULTS_DIR, f"dashboard_{domain}_view.png"), dpi=150, bbox_inches="tight")
        print(f"Saved {domain}-view screenshot.")

    draw("financial")
    print("Run with plt.show() uncommented, then click the toggle button to cycle through domains.")
    # plt.show()
    plt.show()  # uncomment when running with a display