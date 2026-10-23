# Presentation-Grade Research Dashboard (v2)
## *Memory That Fades: Exponential Forgetting in Time Series ML*

Designed for external reviewers, professors, and technical recruiters. Built with **Streamlit** and **Plotly** to provide publication-grade interactivity, high-DPI chart exports, and uncompromising scientific honesty.

---

## 1. Quick Start

### Prerequisites
Ensure Python 3.11+ is available with required dependencies:
```bash
pip install streamlit plotly pandas scikit-learn matplotlib scipy
```

### Launch the Dashboard
Run the dashboard directly from the project root:
```bash
streamlit run dashboard_v2/app.py
```
Or with explicit Python launcher:
```bash
py -3.11 -m streamlit run dashboard_v2/app.py
```

### Optional Data Refresh
To re-verify and regenerate all derived dashboard CSVs from the original model `.pkl` files and raw datasets:
```bash
py -3.11 dashboard_v2/export_dashboard_data.py
```
*Note: All exported MAEs match the repository's reported metrics to within $10^{-9}$ to $10^{-6}$.*

---

## 2. Architecture & Data File Lineage

The dashboard is strictly decoupled from the core experiments: it does **not** modify any original models or training scripts. Every chart is backed by real repository data files:

| Page | Visualization / Chart | Metrics Displayed | Primary Source Data File |
|---|---|---|---|
| **1. Overview** | 4 Domain Metric Tiles | Baseline MAE, Decay MAE, % change, Verdict | `load_overview_metrics()` (`export_dashboard_data.py` verified) |
| **1. Overview** | Cross-Domain Relative Bar Chart | % Error Delta vs Baseline across 4 domains | `decay_model/results/cross_domain_comparison.csv` |
| **2. How It Works** | Interactive Ebbinghaus Curve | Retention weight $R(t) = e^{-t/S}$ over time | Mathematical formula with live $S$ slider |
| **2. How It Works** | Feature Decay Lag Bars | Lag weight multiplier across 30 inputs | `decay_model/feature_decay.py` formula with live $S_{\text{feat}}$ slider |
| **2. How It Works** | Mechanism Comparison Cards | Row vs Feature vs Combined decay architecture | Code structure in `ebbinghaus.py` & `feature_decay.py` |
| **3. Financial (SPY)** | MAE Bars (Baseline vs Decay) | Test MAE across $S \in [30, 180, 365]$, Naive-Zero | `dashboard_v2/data/spy_predictions.csv` |
| **3. Financial (SPY)** | Predicted vs Actual Series | 501 test days (2023-2024) daily returns | `dashboard_v2/data/spy_predictions.csv` |
| **3. Financial (SPY)** | Cumulative Absolute Error Diff | Cumulative $(\lvert\text{err}_{\text{base}}\rvert - \lvert\text{err}_{\text{decay}}\rvert)$ | `dashboard_v2/data/spy_predictions.csv` |
| **3. Financial (SPY)** | Regime Sensitivity Bars | Volatile ($n=62$) vs Stable ($n=96$) MAE | `decay_model/results/regime_analysis_financial.csv` |
| **3. Financial (SPY)** | Decay Rate Sensitivity Curve | MAE across $S=14$ to $S=730$ vs Baseline line | `decay_model/results/decay_rate_sensitivity.csv` |
| **3. Financial (SPY)** | Paired t-Test Stat Card | $t = 0.3778, p = 0.7058$ (Not Significant) | `decay_model/results/statistical_significance.txt` |
| **4. Consumer (PCE)** | MAE Bars (Baseline vs Decay) | Monthly test MAE across $S \in [30, 365, 730]$ | `dashboard_v2/data/pce_predictions.csv` |
| **4. Consumer (PCE)** | Predicted vs Actual Monthly Line| 24 test months (2023-2024) percent changes | `dashboard_v2/data/pce_predictions.csv` |
| **4. Consumer (PCE)** | Autocorrelation Function (ACF) | Lags 1 to 12 autocorrelation values | `dashboard_v2/data/pce_acf.csv` |
| **5. COVID-19** | MAE Bars with Naive-Zero | Naive-Zero (0.03876), Decay (0.04087), Baseline (0.04598) | `dashboard_v2/data/covid_predictions.csv` |
| **5. COVID-19** | Historical Cases & Shaded Spikes| Daily cases with 2020-2021 spike period shaded | `dashboard_v2/data/covid_history.csv` |
| **5. COVID-19** | 30-Day Rolling MAE Series | Trajectory of rolling error during regime shifts | `dashboard_v2/data/covid_history.csv` |
| **5. COVID-19** | Pre vs Post 2022 Error Boxplot | Error distribution before and after 2022 | `dashboard_v2/data/covid_history.csv` |
| **6. Crypto (BTC/ETH)** | Held-Out Test MAE Bars | Baseline, Row, Feature, and Combined MAE | `dashboard_v2/data/crypto_heldout_predictions.csv` |
| **6. Crypto (BTC/ETH)** | Cumulative Error Difference | Upward accumulation of decay advantage | `dashboard_v2/data/crypto_heldout_predictions.csv` |
| **6. Crypto (BTC/ETH)** | Hyperparameter Tuning Heatmap | $S_{\text{feat}} \in [5..60] \times S_{\text{row}} \in [90..730]$ Val MAE | `dashboard_v2/data/crypto_tuning_grid.csv` |
| **7. Walk-Forward** | Per-Window MAE Difference Bars | 13 contiguous 6-month windows (2020-2026) | `decay_model/results/walk_forward_results_btc.csv` / `_eth.csv` |
| **7. Walk-Forward** | Win / Loss Donut Chart | 10/13 wins (BTC), 11/13 wins (ETH) | `decay_model/results/walk_forward_summary_btc.csv` |
| **7. Walk-Forward** | Block Bootstrap 95% CIs | Confidence intervals across block lengths $L \in [10, 20, 30, 40]$ | `dashboard_v2/data/block_bootstrap_summary.csv` |
| **8. Benchmarks & Trading** | Technique Benchmark Cards | MAE vs GARCH(1,1), 20/50 MA, and RSI(14) | `decay_model/results/garch_comparison_summary_btc.csv`, `rsi_*.csv` |
| **8. Benchmarks & Trading** | Net-of-Fees Equity Curves | Growth of $1.00 net of fees (5 to 50 bps) | `dashboard_v2/data/trading_equity_curves.csv` |
| **8. Benchmarks & Trading** | Annualized Sharpe vs Fee Curve | Sharpe degradation across 5 to 50 bps | `decay_model/results/trading_cost_backtest_btc.csv` |
| **8. Benchmarks & Trading** | Total Trade Count Churn Bars | 733 trades (Baseline) vs 23 trades (Decay) | `decay_model/results/trading_cost_backtest_btc.csv` |
| **8. Benchmarks & Trading** | Historical Drawdown Trajectory | Maximum drawdowns over 2,373 days | `dashboard_v2/data/trading_equity_curves.csv` |
| **9. Methodology** | Bug Post-Mortem Cards | Target-offset and ticker-concatenation fixes | `logs/project_log.md` & `logs/data_documentation.md` |
| **10. Conclusions** | Master Cross-Domain Table | Complete performance & statistical verdict matrix | Consolidated across all domain test results |

---

## 3. Completeness Disclosure

**All requested charts and pages have been completely built.** No requested visualization was omitted or compromised.
- All 10 pages requested by the specification are active in the sidebar navigation.
- All charts feature high-DPI PNG export buttons (`scale: 2.5`).
- The strict color palette is preserved across every visualization: **Baseline is always Ink Navy (`#14213D`)**, **Decay is always Warm Amber (`#E09F3E`)**, **Wins are Teal (`#2A9D8F`)**, and **Losses are Muted Red (`#C8553D`)**.
