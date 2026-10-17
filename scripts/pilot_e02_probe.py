"""Pilot for proposed experiment E-02 (balanced-view reference). Exploratory, NOT for the manuscript.

Question: how much of the paired-view routing (P1) disappears when the view of the training images is no
longer coupled to the diagnosis? Proxy: linear probe on frozen DINOv2 ViT-S/14 features of the 960 Derm7pt
lesions (outputs/dino_linear/derm7pt_{clinic,derm}.npz, rows in prereg/derm7pt_manifest.csv order).

Arms (same lesions; COUPLED and BALANCED have one image per lesion):
  COUPLED   BCC and MEL train lesions give their clinical image; NEV, SEK, VASC, DF give their dermoscopic image
            (mimics the benchmark: two clinical-only target classes, dermoscopic-only other classes).
  BALANCED  within every class exactly half of the train lesions give the clinical image, half the dermoscopic.
  BOTH      every train lesion gives both images (secondary).
Classifier: StandardScaler + multinomial logistic regression, C = 1, class_weight = "balanced", no tuning.
Evaluation: stratified 5-fold CV by lesion, repeated 10 times; both views of every test lesion are predicted.
G = {NEV, SEK, VASC, DF} (the dermoscopic-only classes of COUPLED).
  P1 = rate(dermoscopic image predicted in G) - rate(clinical image predicted in G), over all lesions
  E2 = P1 over BCC and MEL lesions; E3 = P1 over G lesions; E1 = dermoscopic recall of BCC and MEL (macro).
Decision rule fixed before running: residual share = P1(BALANCED) / P1(COUPLED);
  <= 0.25 low risk for E-02, 0.25-0.5 moderate, >= 0.5 high (skip E-02).
Usage: <datn python> scripts/pilot_e02_probe.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "pilot_e02"
CLASSES = ["BCC", "MEL", "NEV", "SEK", "VASC", "DF"]
TARGET, G = ["BCC", "MEL"], ["NEV", "SEK", "VASC", "DF"]
REPEATS, FOLDS, B, BOOT_SEED = 10, 5, 2000, 20260930


def fit_predict(Xtr, ytr, Xc_te, Xd_te):
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000))
    clf.fit(Xtr, ytr)
    return clf.predict(Xc_te), clf.predict(Xd_te)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(ROOT / "prereg" / "derm7pt_manifest.csv")
    Xc = np.load(ROOT / "outputs" / "dino_linear" / "derm7pt_clinic.npz")["X"]
    Xd = np.load(ROOT / "outputs" / "dino_linear" / "derm7pt_derm.npz")["X"]
    assert len(man) == len(Xc) == len(Xd) == 960
    y = man.cls.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    n = len(y)
    g_idx = [CLASSES.index(c) for c in G]
    arms = ["COUPLED", "BALANCED", "BOTH"]
    # pred[arm][view] : (REPEATS, n) predicted class index
    pred = {a: {v: np.full((REPEATS, n), -1) for v in ("clin", "derm")} for a in arms}
    for r in range(REPEATS):
        skf = StratifiedKFold(FOLDS, shuffle=True, random_state=1000 + r)
        for f, (tr, te) in enumerate(skf.split(np.zeros(n), y)):
            rng = np.random.default_rng(10_000 * r + f)
            # COUPLED
            use_clin = np.isin(y[tr], [CLASSES.index(c) for c in TARGET])
            Xtr = np.where(use_clin[:, None], Xc[tr], Xd[tr])
            pc, pd_ = fit_predict(Xtr, y[tr], Xc[te], Xd[te])
            pred["COUPLED"]["clin"][r, te], pred["COUPLED"]["derm"][r, te] = pc, pd_
            # BALANCED: exactly half of each class clinical (random which half)
            use_clin = np.zeros(len(tr), bool)
            for k in range(len(CLASSES)):
                pos = np.flatnonzero(y[tr] == k)
                pos = rng.permutation(pos)
                use_clin[pos[: len(pos) // 2]] = True
            Xtr = np.where(use_clin[:, None], Xc[tr], Xd[tr])
            pc, pd_ = fit_predict(Xtr, y[tr], Xc[te], Xd[te])
            pred["BALANCED"]["clin"][r, te], pred["BALANCED"]["derm"][r, te] = pc, pd_
            # BOTH views
            pc, pd_ = fit_predict(np.vstack([Xc[tr], Xd[tr]]), np.concatenate([y[tr], y[tr]]), Xc[te], Xd[te])
            pred["BOTH"]["clin"][r, te], pred["BOTH"]["derm"][r, te] = pc, pd_
    for a in arms:
        for v in ("clin", "derm"):
            assert (pred[a][v] >= 0).all()

    tmask = np.isin(y, [CLASSES.index(c) for c in TARGET])

    def endpoints(ing_c, ing_d, corr_d, corr_c, idx):
        """ing_*: per-lesion rate of prediction in G; corr_*: per-lesion rate of correct prediction."""
        diff = ing_d[idx] - ing_c[idx]
        yy = y[idx]
        p1 = diff.mean()
        e2 = diff[np.isin(yy, [0, 1])].mean()
        e3 = diff[~np.isin(yy, [0, 1])].mean()
        e1 = np.mean([corr_d[idx][yy == k].mean() for k in (0, 1)])
        c1 = np.mean([corr_c[idx][yy == k].mean() for k in (0, 1)])
        return np.array([p1, e2, e3, e1, c1])

    names = ["P1", "E2_target", "E3_untreated", "E1_derm_recall_target", "clin_recall_target"]
    per_lesion = {}
    for a in arms:
        ing_c = np.isin(pred[a]["clin"], g_idx).mean(0)
        ing_d = np.isin(pred[a]["derm"], g_idx).mean(0)
        corr_c = (pred[a]["clin"] == y).mean(0)
        corr_d = (pred[a]["derm"] == y).mean(0)
        per_lesion[a] = (ing_c, ing_d, corr_d, corr_c)
    allidx = np.arange(n)
    point = {a: endpoints(*per_lesion[a], allidx) for a in arms}
    # spread across repeats (each repeat is a full out-of-fold prediction of every lesion)
    rep = {a: np.array([endpoints(np.isin(pred[a]["clin"][r], g_idx).astype(float),
                                  np.isin(pred[a]["derm"][r], g_idx).astype(float),
                                  (pred[a]["derm"][r] == y).astype(float),
                                  (pred[a]["clin"][r] == y).astype(float), allidx) for r in range(REPEATS)])
           for a in arms}
    # lesion bootstrap (paired across arms)
    rng = np.random.default_rng(BOOT_SEED)
    boots = {a: [] for a in arms}
    for _ in range(B):
        i = rng.integers(0, n, n)
        if any((y[i] == k).sum() == 0 for k in range(len(CLASSES))):
            continue
        for a in arms:
            boots[a].append(endpoints(*per_lesion[a], i))
    boots = {a: np.array(v) for a, v in boots.items()}

    rows = []
    for a in arms:
        lo, hi = np.percentile(boots[a], [2.5, 97.5], axis=0)
        for j, nm in enumerate(names):
            rows.append(dict(arm=a, endpoint=nm, estimate=point[a][j], ci_low=lo[j], ci_high=hi[j],
                             repeat_min=rep[a][:, j].min(), repeat_max=rep[a][:, j].max()))
    for a in ["BALANCED", "BOTH"]:
        d = boots["COUPLED"] - boots[a]
        lo, hi = np.percentile(d, [2.5, 97.5], axis=0)
        for j, nm in enumerate(names):
            rows.append(dict(arm=f"COUPLED-{a}", endpoint=nm, estimate=point["COUPLED"][j] - point[a][j],
                             ci_low=lo[j], ci_high=hi[j], repeat_min=np.nan, repeat_max=np.nan))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "pilot_results.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))
    share = point["BALANCED"][0] / point["COUPLED"][0]
    share_e2 = point["BALANCED"][1] / point["COUPLED"][1]
    print(f"\nresidual share P1 = {share:.3f}; residual share E2 (target) = {share_e2:.3f}")
    print(f"bootstrap resamples used: {len(boots['COUPLED'])}")


if __name__ == "__main__":
    main()
