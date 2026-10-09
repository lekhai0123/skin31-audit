"""Exploratory: P1 of the original 31-output models recomputed with the six ISIC-derived outputs of the corrected
30-output models (the "seborrheic keratosis" output, which contains melanomas, left out of the ISIC-derived group),
so that corrected - original contrasts on Derm7pt use one class grouping. Seed-matched, lesion bootstrap as elsewhere."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import pad_eval as P  # noqa: E402
from deconf_eval import ISIC6  # noqa: E402

OUT = ROOT / "outputs" / "grouping_check"
SEEDS, B, RNG_SEED = [42, 123, 2024], 2000, 20260930
LOG = ROOT / "outputs" / "derm7pt" / "logits"
ISIC7 = ISIC6 | {"seborrheic keratosis"}


def preds(f, n):
    z = np.load(f)
    return z["clinic"][:, :n].argmax(1), z["derm"][:, :n].argmax(1)


def p1(pc, pd_, g):
    return g[pd_].mean() - g[pc].mean()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    on, cn = np.array(P.class_names("orig")), np.array(P.class_names("clean"))
    g7, g6o, g6c = np.isin(on, list(ISIC7)), np.isin(on, list(ISIC6)), np.isin(cn, list(ISIC6))
    n = len(pd.read_csv(ROOT / "prereg" / "derm7pt_manifest.csv"))
    rows = []
    for cfg in ("4_b0_baseline", "5_b0_trainopt"):
        o = [preds(LOG / f"{cfg}_seed{s}.npz", len(on)) for s in SEEDS]
        c = [preds(LOG / f"clean_{cfg}_seed{s}.npz", len(cn)) for s in SEEDS]
        for grp, go in (("seven-output (as reported)", g7), ("six-output (common)", g6o)):
            d = np.array([p1(*c[k], g6c) - p1(*o[k], go) for k in range(3)])
            rng = np.random.default_rng(RNG_SEED)
            bs = []
            for _ in range(B):
                i = rng.integers(0, n, n)
                bs.append(np.mean([p1(c[k][0][i], c[k][1][i], g6c) - p1(o[k][0][i], o[k][1][i], go) for k in range(3)]))
            lo, hi = np.percentile(bs, [2.5, 97.5])
            rows.append({"config": cfg, "original_grouping": grp,
                         "original_P1": np.mean([p1(*x, go) for x in o]), "corrected_P1": np.mean([p1(*x, g6c) for x in c]),
                         "delta": d.mean(), "per_seed": d.round(4).tolist(), "ci_low": lo, "ci_high": hi})
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "corrected_vs_original_P1.csv", index=False)
    print(r.round(3).to_string(index=False))

    # P1 of every original 31-output model with both groupings (Supplementary Table S4, columns P1 and P1^b)
    allrows = []
    cfgs = ["1_b2_baseline", "2_b2_trainopt", "3_b2_fusion_teacher", "4_b0_baseline", "5_b0_trainopt",
            "6_b0_kd_nonfusion", "7_b0_kd_fusion"]
    models = {c: [LOG / f"{c}_seed{s}.npz" for s in SEEDS] for c in cfgs}
    models["authors"] = [LOG / "authors_seed0.npz"]
    for name, files in models.items():
        o = [preds(f, len(on)) for f in files]
        allrows.append({"model": name, "P1_seven": np.mean([p1(*x, g7) for x in o]),
                        "P1_six": np.mean([p1(*x, g6o) for x in o]),
                        "P1_six_per_seed": [round(float(p1(*x, g6o)), 4) for x in o]})
    vit = []
    for s in SEEDS:
        z = np.load(ROOT / "outputs" / "vit" / f"seed{s}.npz")
        vit.append((z["d7c"][:, :len(on)].argmax(1), z["d7d"][:, :len(on)].argmax(1)))
    allrows.append({"model": "vit_s16", "P1_seven": np.mean([p1(*x, g7) for x in vit]),
                    "P1_six": np.mean([p1(*x, g6o) for x in vit]), "P1_six_per_seed": [round(float(p1(*x, g6o)), 4) for x in vit]})
    a = pd.DataFrame(allrows)
    a.to_csv(OUT / "original_models_P1_by_grouping.csv", index=False)
    print(a.round(3).to_string(index=False))

    # 95% lesion-bootstrap CIs of P1 for every model with both groupings (same resampling as derm7pt_eval.bootstrap:
    # estimand per seed within each resample, then mean over seeds; resamples without melanoma are skipped).
    # The seven-output CIs reproduce outputs/derm7pt/primary.csv; the six-output CIs are used in Figure 3c.
    mel = pd.read_csv(ROOT / "prereg" / "derm7pt_manifest.csv").cls.to_numpy() == "MEL"
    sets = {c: [preds(f, len(on)) for f in models[c]] for c in models}
    sets["vit_s16"] = vit
    sets["corrected_4_b0_baseline"] = [preds(LOG / f"clean_4_b0_baseline_seed{s}.npz", len(cn)) for s in SEEDS]
    sets["corrected_5_b0_trainopt"] = [preds(LOG / f"clean_5_b0_trainopt_seed{s}.npz", len(cn)) for s in SEEDS]
    rows = []
    for name, pr in sets.items():
        gs = {"seven": (g7 if not name.startswith("corrected") else g6c), "six": (g6o if not name.startswith("corrected") else g6c)}
        rng = np.random.default_rng(RNG_SEED)
        draws = []
        while len(draws) < B:
            i = rng.integers(0, n, n)
            if mel[i].sum() == 0:
                continue
            draws.append({k: np.mean([p1(x[0][i], x[1][i], g) for x in pr]) for k, g in gs.items()})
        for k, g in gs.items():
            lo, hi = np.percentile([d[k] for d in draws], [2.5, 97.5])
            rows.append({"model": name, "grouping": k, "P1": np.mean([p1(x[0], x[1], g) for x in pr]), "ci_low": lo, "ci_high": hi})
    c = pd.DataFrame(rows)
    c.to_csv(OUT / "P1_ci_by_grouping.csv", index=False)
    print(c.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
