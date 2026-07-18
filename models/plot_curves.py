import os
import numpy as np
import matplotlib.pyplot as plt
from ebbinghaus import ebbinghaus_weight
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

reference_date = pd.Timestamp("2022-12-31")
days_ago = np.arange(0, 3001)
dates = [reference_date - pd.Timedelta(days=int(d)) for d in days_ago]

S_configs = [
    ("Fast decay (S=30)", 30, "#e63946"),
    ("Medium decay (S=180)", 180, "#f4a261"),
    ("Slow decay (S=365)", 365, "#2a9d8f"),
]

plt.figure(figsize=(10, 6))
for label, S, color in S_configs:
    weights = [ebbinghaus_weight(d, reference_date, S) for d in dates]
    plt.plot(days_ago, weights, label=label, color=color, linewidth=2.5)

plt.xlabel("Days ago", fontsize=12)
plt.ylabel("Weight (retention)", fontsize=12)
plt.title("Ebbinghaus Forgetting Curves — Sample Weight vs Data Age", fontsize=14, fontweight="bold")
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.ylim(-0.02, 1.02)
plt.tight_layout()
plt.savefig(os.path.join(PROJECT_ROOT, "results", "ebbinghaus_curves.png"), dpi=150)
print("Saved results/ebbinghaus_curves.png")