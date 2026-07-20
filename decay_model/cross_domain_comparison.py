"""
Person 2 - Day 2, Step 4: Cross-domain comparison table.
Assembles MAE/RMSE for all 5 comparators (naive-zero + 4 models) across
both domains, from the already-verified summary files each domain's
training script produces. Doesn't recompute anything -- just aggregates,
so it can't drift from the numbers those scripts actually produced.
Output: decay_model/results/cross_domain_comparison.csv
"""
import os
import json
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

with open(os.path.join(SCRIPT_DIR, "results", "day1_mae_summary.json")) as f:
    financial = json.load(f)
with open(os.path.join(SCRIPT_DIR, "results", "day2_consumer_mae_summary.json")) as f:
    consumer = json.load(f)

rows = []
for model in ["naive_zero", "baseline", "fast", "medium", "slow"]:
    rows.append({
        "model": model,
        "financial_MAE": financial[model]["MAE"],
        "financial_RMSE": financial[model]["RMSE"],
        "consumer_MAE": consumer[model]["MAE"],
        "consumer_RMSE": consumer[model]["RMSE"],
    })

df = pd.DataFrame(rows)
print(df.to_string(index=False))

fin_winner = df.loc[df["financial_MAE"].idxmin(), "model"]
con_winner = df.loc[df["consumer_MAE"].idxmin(), "model"]
print(f"\nBest on financial domain: {fin_winner}")
print(f"Best on consumer domain:  {con_winner}")
print(f"Same winner in both domains? {fin_winner == con_winner}")

out_path = os.path.join(SCRIPT_DIR, "results", "cross_domain_comparison.csv")
df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")