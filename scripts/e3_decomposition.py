"""Post hoc, exploratory: decomposition of E3 (P1 of the untreated classes) into its two views.
E3 = r_derm - r_clin, where r_v is the fraction of untreated lesions (nevus, seborrheic keratosis, vascular lesion,
dermatofibroma; pooled over lesions) whose image in view v is predicted as an ISIC-derived class. A change in E3 can
come from the dermoscopic view, the clinical view or both. Contrasts are paired by lesion and seed, with lesion
bootstrap CIs (2,000 resamples, seed 20260930) for fixed checkpoints."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import pad_eval as P  # noqa: E402
from deconf_eval import ISIC6  # noqa: E402

O = ROOT / "outputs"
OUT = O / "e3_decomposition"
SEEDS, B, RNG_SEED = [42, 123, 2024], 2000, 20260930
UNT = ["NEV", "SEK", "VASC", "DF"]
D7 = {"A": O / "milk_eval/logits/derm7pt_A", "B": O / "milk_eval/logits/derm7pt_B", "C": O / "milk_eval/logits/derm7pt_C",
      "H": O / "deconf/logits/derm7pt_deconf", "R": O / "relabel_eval/logits/derm7pt_R",
      "vB": O / "vit_milk/logits/derm7pt_B", "vC": O / "vit_milk/logits/derm7pt_C"}
CONTRASTS = [("R-A", "R", "A"), ("H-A", "H", "A"), ("C-B (B0)", "C", "B"), ("C-B (ViT)", "vC", "vB")]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    names = np.array(P.class_names("clean"))
    isic = np.isin(names, list(ISIC6))
    cls = pd.read_csv(ROOT / "prereg" / "derm7pt_manifest.csv").cls.to_numpy()
    u = np.isin(cls, UNT)
    pr = {}
    for a, f in D7.items():
        for s in SEEDS:
            z = np.load(f"{f}_seed{s}.npz")
            pr[(a, s)] = (isic[z["clinic"].argmax(1)].astype(float), isic[z["derm"].argmax(1)].astype(float))

    def comps(x, z, idx):
        """per seed: (delta r_derm, delta r_clin, delta E3) on untreated lesions in idx"""
        out = []
        for s in SEEDS:
            m = u[idx]
            dc = pr[(x, s)][0][idx][m].mean() - pr[(z, s)][0][idx][m].mean()
            dd = pr[(x, s)][1][idx][m].mean() - pr[(z, s)][1][idx][m].mean()
            out.append((dd, dc, dd - dc))
        return np.array(out)

    rows = []
    allidx = np.arange(len(cls))
    for lab, x, z in CONTRASTS:
        c = comps(x, z, allidx)
        rng = np.random.default_rng(RNG_SEED)
        bs = []
        while len(bs) < B:
            i = rng.integers(0, len(cls), len(cls))
            if any((cls[i] == d).sum() == 0 for d in UNT):
                continue
            bs.append(comps(x, z, i).mean(0))
        lo, hi = np.percentile(np.array(bs), [2.5, 97.5], axis=0)
        m = c.mean(0)
        for k, nm in enumerate(["delta_r_derm", "delta_r_clin", "delta_E3"]):
            rows.append({"contrast": lab, "component": nm, "mean": m[k], "per_seed": [round(float(v), 4) for v in c[:, k]],
                         "ci_low": lo[k], "ci_high": hi[k]})
        rows.append({"contrast": lab, "component": "share_of_E3_change_from_clinical_view", "mean": -m[1] / m[2] if m[2] else np.nan,
                     "per_seed": None, "ci_low": np.nan, "ci_high": np.nan})
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "e3_decomposition.csv", index=False)
    lv = []
    for a in D7:
        lv.append({"arm": a, "r_derm_untreated": np.mean([pr[(a, s)][1][u].mean() for s in SEEDS]),
                   "r_clin_untreated": np.mean([pr[(a, s)][0][u].mean() for s in SEEDS])})
    pd.DataFrame(lv).to_csv(OUT / "untreated_isic_rate_by_arm.csv", index=False)
    pd.set_option("display.width", 220)
    print("untreated lesions:", int(u.sum()))
    print(pd.DataFrame(lv).round(3).to_string(index=False))
    print(r.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
