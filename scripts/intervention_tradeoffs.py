"""EXPLORATORY (04/10/2026, after review): trade-offs and seed variability for all training-data arms.
Arms (B0 + training opt., Derm7pt): orig (31-class), A corrected, H +HAM10000 derm, B +MILK10k clinical,
C +MILK10k derm, R relabel. Reads saved logits only.
Outputs (outputs/tradeoffs/):
  seed_table.csv     E1, E2, E3, P1 per arm and seed
  contrasts.csv      X - A (and C - B) with lesion-bootstrap CI (seed-averaged) and per-seed values for:
                     MEL dermoscopic recall (any melanoma output), MEL derm -> nevus, untreated derm -> melanoma,
                     untreated derm -> BCC, NEV derm recall, P1 all lesions
  test_paired.csv    strictly filtered test accuracy, X - A, image-bootstrap CI (seed-averaged)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from deconf_eval import estimands  # noqa: E402  (registered E1-E3 definitions)

OUT = ROOT / "outputs" / "tradeoffs"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS, B, RNG = [42, 123, 2024], 2000, 20260930
clean = np.array([p.name.lower() for p in sorted((ROOT / "data/skin31_clean/train").iterdir())])
orig = np.array([k.lower() for k, _ in sorted(json.loads((ROOT / "prereg/class_mapping.json").read_text())["class_to_idx"].items(),
                                              key=lambda kv: kv[1])])
ARMS = {"orig": ("outputs/derm7pt/logits/5_b0_trainopt", orig, {"melanoma", "seborrheic keratosis"}),
        "A": ("outputs/milk_eval/logits/derm7pt_A", clean, {"melanoma"}),
        "H": ("outputs/deconf/logits/derm7pt_deconf", clean, {"melanoma"}),
        "B": ("outputs/milk_eval/logits/derm7pt_B", clean, {"melanoma"}),
        "C": ("outputs/milk_eval/logits/derm7pt_C", clean, {"melanoma"}),
        "R": ("outputs/relabel_eval/logits/derm7pt_R", clean, {"melanoma"})}
ISIC = {"actinic keratosis", "dermatofibroma", "nevus", "pigmented benign keratosis", "squamous cell carcinoma",
        "vascular lesion", "seborrheic keratosis"}  # SK output counted as ISIC-derived for the 31-class model
cls = pd.read_csv(ROOT / "prereg/derm7pt_manifest.csv").cls.to_numpy()
UNTREATED = np.isin(cls, ["NEV", "SEK", "DF", "VASC"])

pred = {}
for a, (f, names, _) in ARMS.items():
    for s in SEEDS:
        z = np.load(ROOT / f"{f}_seed{s}.npz")
        pred[(a, s)] = (names[z["clinic"].argmax(1)], names[z["derm"].argmax(1)])


def metrics(a, s, idx):
    pc, pd_ = pred[(a, s)][0][idx], pred[(a, s)][1][idx]
    c = cls[idx]
    mel, unt, nev = c == "MEL", np.isin(c, ["NEV", "SEK", "DF", "VASC"]), c == "NEV"
    outs = ARMS[a][2]
    return np.array([np.isin(pd_[mel], list(outs)).mean(), (pd_[mel] == "nevus").mean(),
                     np.isin(pd_[unt], list(outs)).mean(), (pd_[unt] == "basal cell carcinoma").mean(),
                     (pd_[nev] == "nevus").mean(),
                     np.isin(pd_, list(ISIC)).mean() - np.isin(pc, list(ISIC)).mean()])


LAB = ["MEL_derm_recall", "MEL_derm_to_nevus", "untreated_derm_to_melanoma", "untreated_derm_to_BCC",
       "NEV_derm_recall", "P1_all"]

# seed table with registered estimands (clean arms only; the 31-class model has no comparable E-definitions)
rows = []
for a in ["A", "H", "B", "C", "R"]:
    for s in SEEDS:
        z = np.load(ROOT / f"{ARMS[a][0]}_seed{s}.npz")
        e = estimands(z["clinic"].argmax(1), z["derm"].argmax(1), cls, list(clean))
        rows.append({"arm": a, "seed": s, "E1": e[0], "E2": e[1], "E3": e[2], "P1_all": e[4]})
st = pd.DataFrame(rows)
st.to_csv(OUT / "seed_table.csv", index=False)
print(st.pivot(index="arm", columns="seed", values=["E1", "E2", "E3", "P1_all"]).round(3).to_string())

all_idx = np.arange(len(cls))
rng = np.random.default_rng(RNG)
boots = []
while len(boots) < B:
    i = rng.integers(0, len(cls), len(cls))
    if all((cls[i] == d).any() for d in ["MEL", "NEV", "SEK", "DF", "VASC", "BCC"]):
        boots.append(i)
crow = []
for x, y in [("R", "A"), ("R", "orig"), ("B", "A"), ("C", "A"), ("C", "B"), ("H", "A")]:
    per_seed = np.array([metrics(x, s, all_idx) - metrics(y, s, all_idx) for s in SEEDS])
    bs = np.array([np.mean([metrics(x, s, i) - metrics(y, s, i) for s in SEEDS], axis=0) for i in boots])
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    for k, lab in enumerate(LAB):
        crow.append({"contrast": f"{x}-{y}", "metric": lab, "x": np.mean([metrics(x, s, all_idx)[k] for s in SEEDS]),
                     "y": np.mean([metrics(y, s, all_idx)[k] for s in SEEDS]), "delta": per_seed[:, k].mean(),
                     "per_seed": per_seed[:, k].round(4).tolist(), "ci_low": lo[k], "ci_high": hi[k]})
ct = pd.DataFrame(crow)
ct.to_csv(OUT / "contrasts.csv", index=False)
pd.set_option("display.width", 220)
print(ct.round(4).to_string(index=False))

# strictly filtered test accuracy, paired by image
from torchvision import datasets  # noqa: E402

ty = np.array(datasets.ImageFolder(ROOT / "data/skin31_clean/test").targets)
tl = {"A": "outputs/milk_eval/logits/test_A", "H": "outputs/deconf/logits/test_deconf",
      "B": "outputs/milk_eval/logits/test_B", "C": "outputs/milk_eval/logits/test_C", "R": "outputs/relabel_eval/logits/test_R"}
corr = {a: np.array([np.load(ROOT / f"{f}_seed{s}.npy").argmax(1) == ty for s in SEEDS]) for a, f in tl.items()}
ib = [rng.integers(0, len(ty), len(ty)) for _ in range(B)]
trow = []
for x in ["H", "B", "C", "R"]:
    d = corr[x].mean(1) - corr["A"].mean(1)
    bs = [(corr[x][:, i].mean() - corr["A"][:, i].mean()) for i in ib]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    trow.append({"contrast": f"{x}-A", "acc_x": corr[x].mean(), "acc_A": corr["A"].mean(), "delta": d.mean(),
                 "per_seed": d.round(4).tolist(), "ci_low": lo, "ci_high": hi})
tt = pd.DataFrame(trow)
tt.to_csv(OUT / "test_paired.csv", index=False)
print(tt.round(4).to_string(index=False))
