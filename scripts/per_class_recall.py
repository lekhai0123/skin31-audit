"""EXPLORATORY (review, 04/10/2026): recall of the two target classes separately, by view, for every arm
(Derm7pt, B0 + training opt.). Contrasts with lesion-bootstrap CI (seed-averaged) and per-seed values."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "tradeoffs"
SEEDS, B, RNG = [42, 123, 2024], 2000, 20260930
names = np.array([p.name.lower() for p in sorted((ROOT / "data/skin31_clean/train").iterdir())])
cls = pd.read_csv(ROOT / "prereg/derm7pt_manifest.csv").cls.to_numpy()
F = {"A": "outputs/milk_eval/logits/derm7pt_A", "H": "outputs/deconf/logits/derm7pt_deconf",
     "B": "outputs/milk_eval/logits/derm7pt_B", "C": "outputs/milk_eval/logits/derm7pt_C",
     "R": "outputs/relabel_eval/logits/derm7pt_R"}
T = {"BCC": "basal cell carcinoma", "MEL": "melanoma"}
P = {}
for a, f in F.items():
    for s in SEEDS:
        z = np.load(ROOT / f"{f}_seed{s}.npz")
        P[(a, s)] = (names[z["clinic"].argmax(1)], names[z["derm"].argmax(1)])


def rec(a, s, idx):
    out = []
    for d, t in T.items():
        m = cls[idx] == d
        out += [(P[(a, s)][0][idx][m] == t).mean(), (P[(a, s)][1][idx][m] == t).mean()]
    return np.array(out)


LAB = ["BCC_clinical", "BCC_dermoscopic", "MEL_clinical", "MEL_dermoscopic"]
allx = np.arange(len(cls))
rows = [{"arm": a, **dict(zip(LAB, np.mean([rec(a, s, allx) for s in SEEDS], axis=0)))} for a in F]
rng = np.random.default_rng(RNG)
boots = []
while len(boots) < B:
    i = rng.integers(0, len(cls), len(cls))
    if (cls[i] == "BCC").any() and (cls[i] == "MEL").any():
        boots.append(i)
for x, y in [("H", "A"), ("C", "B"), ("R", "A")]:
    d = np.array([rec(x, s, allx) - rec(y, s, allx) for s in SEEDS])
    bs = np.array([np.mean([rec(x, s, i) - rec(y, s, i) for s in SEEDS], axis=0) for i in boots])
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    for k, lab in enumerate(LAB):
        rows.append({"arm": f"{x}-{y}", "metric": lab, "delta": d[:, k].mean(), "per_seed": d[:, k].round(3).tolist(),
                     "ci_low": lo[k], "ci_high": hi[k]})
r = pd.DataFrame(rows)
r.to_csv(OUT / "per_class_recall.csv", index=False)
pd.set_option("display.width", 200)
print(r.round(3).to_string(index=False))
