"""Holm step-down sensitivity analysis across the 25 interval estimates of Table 4 (5 contrasts x 5 columns).

Reads the percentile bootstrap intervals already written by the analysis scripts (no resampling is repeated):
  E1-E3 for R-A, H-A, B-A, C-B, C-A  -> outputs/relabel_eval, outputs/deconf, outputs/milk_eval
  untreated -> melanoma, nevus recall -> outputs/tradeoffs/contrasts.csv
Each two-sided p-value is approximated from its 95% interval assuming an approximately normal bootstrap
distribution: se = (ci_high - ci_low) / (2 * 1.96), p = 2 * (1 - Phi(|delta| / se)). Holm's step-down
procedure is then applied across the 25 tests. This is a sensitivity analysis, not a replacement for the
percentile intervals in Table 4.
Output: outputs/holm/table4_holm.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "holm"
OUT.mkdir(parents=True, exist_ok=True)
Z = norm.ppf(0.975)

ESTIMAND_COLS = {"E1_derm_recall_target": "E1 target recall", "E2_P1_target": "E2 target gap",
                 "E3_P1_control": "E3 untreated gap"}
TRADEOFF_COLS = {"untreated_derm_to_melanoma": "Untreated to melanoma", "NEV_derm_recall": "Nevus recall"}
ROWS = ["R-A", "H-A", "B-A", "C-B", "C-A"]

rows = []
rl = pd.read_csv(ROOT / "outputs/relabel_eval/derm7pt_results.csv").dropna(subset=["estimand"])
dc = pd.read_csv(ROOT / "outputs/deconf/derm7pt_primary.csv")
mk = pd.read_csv(ROOT / "outputs/milk_eval/derm7pt_results.csv").dropna(subset=["estimand"])
tr = pd.read_csv(ROOT / "outputs/tradeoffs/contrasts.csv")

for row in ROWS:
    for est, lab in ESTIMAND_COLS.items():
        if row == "R-A":
            r = rl[rl.estimand == est].iloc[0]
        elif row == "H-A":
            r = dc[dc.estimand == est].iloc[0]
        else:
            r = mk[(mk.row == row) & (mk.estimand == est)].iloc[0]
        rows.append({"contrast": row, "measure": lab, "delta": r.delta, "ci_low": r.ci_low, "ci_high": r.ci_high})
    for met, lab in TRADEOFF_COLS.items():
        r = tr[(tr.contrast == row) & (tr.metric == met)].iloc[0]
        rows.append({"contrast": row, "measure": lab, "delta": r.delta, "ci_low": r.ci_low, "ci_high": r.ci_high})

df = pd.DataFrame(rows)
df["se"] = (df.ci_high - df.ci_low) / (2 * Z)
df["p_approx"] = 2 * (1 - norm.cdf((df.delta / df.se).abs()))
m = len(df)
order = np.argsort(df.p_approx.to_numpy())
holm = np.empty(m)
running = 0.0
for rank, i in enumerate(order):
    running = max(running, min(1.0, (m - rank) * df.p_approx.iloc[i]))
    holm[i] = running
df["p_holm"] = holm
df["holm_significant"] = df.p_holm < 0.05
df.to_csv(OUT / "table4_holm.csv", index=False)

pd.set_option("display.width", 200)
print(df.round(4).to_string(index=False))
print(f"tests: {m}; Holm-significant (two-sided 0.05): {int(df.holm_significant.sum())}")
