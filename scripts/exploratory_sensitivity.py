"""Exploratory sensitivity analyses (not pre-registered; added 03/10/2026 after review).
Reuses saved logits only; no new inference.

S-A  Derm7pt: softmax probability mass on ISIC-derived outputs, dermoscopic minus clinical image of the same case
     (threshold-free counterpart of P1). Case bootstrap as in derm7pt_eval.py.
S-B  PAD: metrics restricted to biopsy-confirmed images; ACK/NEV/SEK recall split by biopsy status.
     Patient bootstrap as in pad_eval.py.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from pad_eval import ISIC_NAMES, PAD6, class_names, metrics, patient_resamples  # noqa: E402

PRE, OUT = ROOT / "prereg", ROOT / "outputs" / "sensitivity"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [42, 123, 2024]
ORIG = ["1_b2_baseline", "2_b2_trainopt", "3_b2_fusion_teacher", "4_b0_baseline", "5_b0_trainopt",
        "6_b0_kd_nonfusion", "7_b0_kd_fusion"]
CLEAN = ["4_b0_baseline", "5_b0_trainopt"]
B, RNG_SEED = 2000, 20260930
on, cn = class_names("orig"), class_names("clean")


def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


# ---------- S-A: Derm7pt probability mass ----------
D7 = ROOT / "outputs" / "derm7pt" / "logits"
groups = {("orig", c): [D7 / f"{c}_seed{s}.npz" for s in SEEDS] for c in ORIG}
groups[("orig", "authors")] = [D7 / "authors_seed0.npz"]
groups.update({("clean", c): [D7 / f"clean_{c}_seed{s}.npz" for s in SEEDS] for c in CLEAN})
rows = []
for (kind, c), files in groups.items():
    names = on if kind == "orig" else cn
    isic = np.array([n in ISIC_NAMES for n in names])
    per = []
    for f in files:
        z = np.load(f)
        per.append((softmax(z["clinic"])[:, isic].sum(1), softmax(z["derm"])[:, isic].sum(1),
                    np.isin(z["clinic"].argmax(1), np.flatnonzero(isic)), np.isin(z["derm"].argmax(1), np.flatnonzero(isic))))
    n = len(per[0][0])
    vals = [d.mean() - cl.mean() for cl, d, _, _ in per]
    p1 = np.mean([dh.mean() - ch.mean() for _, _, ch, dh in per])
    rng = np.random.default_rng(RNG_SEED)
    bs = [np.mean([d[i].mean() - cl[i].mean() for cl, d, _, _ in per]) for i in (rng.integers(0, n, n) for _ in range(B))]
    rows.append({"model": f"{kind}:{c}", "n_cases": n, "mass_clinic": np.mean([p[0].mean() for p in per]),
                 "mass_derm": np.mean([p[1].mean() for p in per]), "delta_mass": np.mean(vals),
                 "ci_low": np.percentile(bs, 2.5), "ci_high": np.percentile(bs, 97.5),
                 "per_seed": " ".join(f"{v:.3f}" for v in vals), "P1_argmax": p1})
sa = pd.DataFrame(rows)
sa.to_csv(OUT / "derm7pt_probability_mass.csv", index=False)
print(sa.round(3).to_string(index=False))

# ---------- S-B: PAD biopsy-confirmed ----------
man = pd.read_csv(PRE / "pad_manifest.csv")
main = (man.exclude.fillna("") == "").to_numpy()
y_all = man.diagnostic.map({d: k for k, d in enumerate(PAD6)}).to_numpy()
bio = man.biopsed.astype(bool).to_numpy()
PL = ROOT / "outputs" / "pad" / "logits"
pgroups = {("orig", c): [PL / f"orig_{c}_seed{s}.npy" for s in SEEDS] for c in ["7_b0_kd_fusion", "5_b0_trainopt", "4_b0_baseline"]}
pgroups[("orig", "authors")] = [PL / "authors_authors_seed0.npy"]
pgroups.update({("clean", c): [PL / f"clean_{c}_seed{s}.npy" for s in SEEDS] for c in CLEAN})
mask = main & bio
y, pid = y_all[mask], man.patient_id.to_numpy()[mask]
res, rejected = patient_resamples(pid, y)
print(f"\nPAD biopsy-confirmed: {mask.sum()} images, {np.unique(pid).size} patients; "
      f"per class {dict(zip(PAD6, np.bincount(y, minlength=6)))}; rejected resamples {rejected}")
rows = []
for (kind, c), files in pgroups.items():
    names = on if kind == "orig" else cn
    zs = [np.load(f) for f in files]
    full = [metrics(z[main], names, y_all[main]) for z in zs]
    sub = [metrics(z[mask], names, y) for z in zs]
    r = {"model": f"{kind}:{c}"}
    for k in ["bal_acc_6way", "bal_acc_5way_noSEK", "macro_auroc_6way", "isic_rate"]:
        r[f"{k}_all"] = np.mean([m[k] for m in full])
        r[f"{k}_biopsy"] = np.mean([m[k] for m in sub])
    bs = [np.mean([metrics(z[mask][i], names, y[i])["bal_acc_6way"] for z in zs]) for i in res[:500]]
    r["bal_acc_6way_biopsy_ci"] = f"[{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}] (500 resamples)"
    for d in ["ACK", "NEV", "SEK"]:
        k = PAD6.index(d)
        for b, lab in [(True, "bx"), (False, "nobx")]:
            m2 = main & (bio == b) & (y_all == k)
            # recall in 6-way restricted space, as in metrics()
            rec = []
            for z in zs:
                cols = [(names.index(nm), j) for j, dx in enumerate(PAD6) for nm in
                        {"ACK": ["actinic keratosis"], "BCC": ["basal cell carcinoma"],
                         "MEL": ["melanoma", "seborrheic keratosis"], "NEV": ["nevus"],
                         "SEK": ["pigmented benign keratosis"], "SCC": ["squamous cell carcinoma"]}[dx] if nm in names]
                idx, owner = zip(*cols)
                pred = np.array(owner)[z[m2][:, list(idx)].argmax(1)]
                rec.append((pred == k).mean())
            r[f"recall_{d}_{lab}"] = np.mean(rec)
            r[f"n_{d}_{lab}"] = int(m2.sum())
    rows.append(r)
sb = pd.DataFrame(rows)
sb.to_csv(OUT / "pad_biopsy_confirmed.csv", index=False)
print(sb.round(3).T.to_string())
