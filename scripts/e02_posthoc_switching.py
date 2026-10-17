"""E-02 post hoc descriptive reanalysis (not in the locked plan; the locked evaluation scripts/e02_eval.py is imported,
not changed). From the saved out-of-fold predictions of both arms:

a  = fraction of cases whose clinical image is predicted in T (BCC, MEL) and dermoscopic image in G (DF, NEV, SEK, VASC)
b  = fraction with the clinical image predicted in G and the dermoscopic image in T
P1 = a - b (the registered estimand; checked against e02_estimands.csv), Q = a + b (cross-group switching),
D  = fraction of cases whose two images receive different diagnoses (any change, also within a group).
Per arm and seed, per diagnosis, and COUPLED - BALANCED with the same paired case bootstrap as the registered analysis
(2,000 resamples, RNG seed 20260930, resamples lacking a diagnosis redrawn, seeds averaged within each resample).
Outputs: outputs/e02/posthoc/. Usage (datn environment): python scripts/e02_posthoc_switching.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e02_eval import ARMS, B, CLASSES, G, PRED, RES, RNG_SEED, ROOT, SEEDS, load_predictions  # noqa: E402

OUT = ROOT / "outputs" / "e02" / "posthoc"
NAMES = ["a", "b", "P1", "Q", "D"]


def parts(w):
    pc, pd_ = w["clinic"].to_numpy(), w["derm"].to_numpy()
    gc, gd = np.isin(pc, G), np.isin(pd_, G)
    return np.stack([(gd & ~gc), (gc & ~gd), gd.astype(float) - gc, (gd != gc), (pc != pd_)]).astype(float)


def main():
    cls, W = load_predictions(PRED)
    n = len(cls)
    X = {k: parts(w) for k, w in W.items()}  # 5 x n per arm and seed

    # registered P1 reproduced exactly
    reg = pd.read_csv(RES / "e02_estimands.csv").query("estimand == 'P1'").set_index(["arm", "seed"]).value
    for k, x in X.items():
        assert abs(x[2].mean() - reg[k]) < 1e-12, (k, x[2].mean(), reg[k])

    OUT.mkdir(parents=True, exist_ok=True)
    rows = [dict(arm=a, seed=s, **{nm: X[(a, s)][j].mean() for j, nm in enumerate(NAMES)}) for (a, s) in X]
    pd.DataFrame(rows).to_csv(OUT / "switching_per_seed.csv", index=False)

    per = []
    for a in ARMS:
        for k in CLASSES:
            m = cls == k
            per.append(dict(arm=a, diagnosis=k, n_cases=int(m.sum()),
                            **{nm: np.mean([X[(a, s)][j][m].mean() for s in SEEDS]) for j, nm in enumerate(NAMES)}))
    pd.DataFrame(per).to_csv(OUT / "switching_per_diagnosis.csv", index=False)

    lev = {a: np.mean([X[(a, s)].mean(axis=1) for s in SEEDS], axis=0) for a in ARMS}
    rng = np.random.default_rng(RNG_SEED)
    boots = []
    while len(boots) < B:
        i = rng.integers(0, n, n)
        if any((cls[i] == k).sum() == 0 for k in CLASSES):
            continue
        la = {a: np.mean([X[(a, s)][:, i].mean(axis=1) for s in SEEDS], axis=0) for a in ARMS}
        boots.append(np.concatenate([la["COUPLED"], la["BALANCED"], la["COUPLED"] - la["BALANCED"]]))
    lo, hi = np.percentile(np.array(boots), [2.5, 97.5], axis=0)
    m = len(NAMES)
    summ = []
    for j, nm in enumerate(NAMES):
        summ.append(dict(quantity=nm, coupled=lev["COUPLED"][j], coupled_lo=lo[j], coupled_hi=hi[j],
                         balanced=lev["BALANCED"][j], balanced_lo=lo[m + j], balanced_hi=hi[m + j],
                         difference=lev["COUPLED"][j] - lev["BALANCED"][j], diff_lo=lo[2 * m + j], diff_hi=hi[2 * m + j]))
    s = pd.DataFrame(summ)
    s.to_csv(OUT / "switching_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(s.round(4).to_string(index=False))
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(pd.DataFrame(per).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
