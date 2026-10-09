"""EXPLORATORY (not pre-registered; added 04/10/2026 after review): relabel arm.
Arm R = corrected benchmark in which the 64 melanomas of the "Seborrheic keratosis" folder are moved to Melanoma
instead of being dropped (data/skin31_relabel, built by build_relabel_skin31.py). Compared with arm A (corrected,
checkpoints_clean) using the same estimands, bootstrap and seeds as osf.io/pe8q6 and osf.io/5ksja.
Reports: E1-E3 and P1 (all lesions) for R - A; per-diagnosis recall by view for BCC and melanoma; macro P1 over
diagnoses; PAD-UFES-20 six-diagnosis balanced accuracy; strictly filtered test accuracy.
Usage: python scripts/relabel_eval.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from torchvision import datasets

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import milk_eval as M  # noqa: E402  (load, contrast, estimands: same definitions as the registered analyses)
import pad_eval as P  # noqa: E402
from derm7pt_eval_clean import logits as d7_logits  # noqa: E402

M.ARMS["R"] = ROOT / "checkpoints_relabel"
OUT = ROOT / "outputs" / "relabel_eval"
SEEDS = M.SEEDS
DX = ["MEL", "BCC", "NEV", "SEK", "DF", "VASC"]
ISIC = {"actinic keratosis", "dermatofibroma", "nevus", "pigmented benign keratosis", "squamous cell carcinoma",
        "vascular lesion"}


def derm_logits(arm, man):
    src = M.OUT if arm == "A" else OUT  # arm A logits already exist from the MILK10k evaluation
    out = {}
    for s in SEEDS:
        f = src / "logits" / f"derm7pt_{arm}_seed{s}.npz"
        if not f.exists():
            f.parent.mkdir(parents=True, exist_ok=True)
            m = M.load(arm, s)
            np.savez(f, clinic=d7_logits(m, [M.D7 / p for p in man.clinic]), derm=d7_logits(m, [M.D7 / p for p in man.derm]))
        z = np.load(f)
        out[s] = (z["clinic"].argmax(1), z["derm"].argmax(1))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    names = np.array(P.class_names("clean"))
    assert sorted(p.name.lower() for p in (ROOT / "data/skin31_relabel/train").iterdir()) == sorted(names)
    man = pd.read_csv(M.PRE / "derm7pt_manifest.csv")
    cls = man.cls.to_numpy()
    preds = {}
    for arm in ("A", "R"):
        for s, v in derm_logits(arm, man).items():
            preds[(arm, s)] = v

    rows = []
    for arm in ("A", "R"):
        v = np.array([M.estimands(*preds[(arm, s)], cls, list(names)) for s in SEEDS])
        rows.append({"row": f"arm {arm}", **{l: v[:, k].mean() for k, l in enumerate(M.LABELS)}})
    d, (lo, hi) = M.contrast(preds, "R", "A", cls, list(names))
    for k, lab in enumerate(M.LABELS):
        rows.append({"row": "R-A", "estimand": lab, "delta": d[:, k].mean(), "per_seed": d[:, k].round(4).tolist(),
                     "ci_low": lo[k], "ci_high": hi[k]})
    # descriptive: original 31-class model of the same configuration contains the same 64 melanomas, but under
    # the "seborrheic keratosis" name; R vs original isolates the effect of filing them under the right class.
    orig_names = P.class_names("orig")
    orig = {}
    for s in SEEDS:
        zz = np.load(ROOT / "outputs" / "derm7pt" / "logits" / f"5_b0_trainopt_seed{s}.npz")
        orig[s] = (zz["clinic"].argmax(1), zz["derm"].argmax(1))
    # E-estimands do not count the "seborrheic keratosis" output as melanoma, so compare melanoma directly:
    # fraction of dermoscopic / clinical melanomas predicted as ANY melanoma output, and as nevus.
    mel = cls == "MEL"
    mel_rows = []
    for lab, nm, pr, outs in [("original (64 MEL filed as SK)", np.array(orig_names), orig, {"melanoma", "seborrheic keratosis"}),
                              ("A corrected (64 MEL dropped)", names, {s: preds[("A", s)] for s in SEEDS}, {"melanoma"}),
                              ("R relabel (64 MEL as melanoma)", names, {s: preds[("R", s)] for s in SEEDS}, {"melanoma"})]:
        v = [(np.isin(nm[pr[s][1][mel]], list(outs)).mean(), np.isin(nm[pr[s][0][mel]], list(outs)).mean(),
              (nm[pr[s][1][mel]] == "nevus").mean()) for s in SEEDS]
        mel_rows.append({"model": lab, "MEL_derm_as_melanoma": np.mean([x[0] for x in v]),
                         "MEL_clin_as_melanoma": np.mean([x[1] for x in v]), "MEL_derm_as_nevus": np.mean([x[2] for x in v])})
    pd.DataFrame(mel_rows).to_csv(OUT / "melanoma_routing.csv", index=False)
    print(pd.DataFrame(mel_rows).round(3).to_string(index=False))
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "derm7pt_results.csv", index=False)
    print(r.round(4).to_string(index=False))

    # per-diagnosis view: recall on each view and P1 by diagnosis (macro P1 = mean over diagnoses)
    target = {"MEL": {"melanoma"}, "BCC": {"basal cell carcinoma"}, "NEV": {"nevus"},
              "SEK": {"pigmented benign keratosis"}, "DF": {"dermatofibroma"}, "VASC": {"vascular lesion"}}
    per = []
    for arm in ("A", "R"):
        for s in SEEDS:
            pc, pd_ = names[preds[(arm, s)][0]], names[preds[(arm, s)][1]]
            for dx in DX:
                m = cls == dx
                per.append({"arm": arm, "seed": s, "dx": dx, "n": int(m.sum()),
                            "recall_clinic": np.isin(pc[m], list(target[dx])).mean(),
                            "recall_derm": np.isin(pd_[m], list(target[dx])).mean(),
                            "P1_dx": np.isin(pd_[m], list(ISIC)).mean() - np.isin(pc[m], list(ISIC)).mean()})
    per = pd.DataFrame(per)
    per.to_csv(OUT / "per_diagnosis_by_seed.csv", index=False)
    summ = per.groupby(["arm", "dx"])[["recall_clinic", "recall_derm", "P1_dx"]].mean().round(3)
    print(summ.to_string())
    print("macro P1:", per.groupby(["arm", "seed"]).P1_dx.mean().groupby("arm").mean().round(3).to_dict())

    # PAD-UFES-20 six-diagnosis balanced accuracy, patient bootstrap, R - A
    pm = pd.read_csv(M.PRE / "pad_manifest.csv")
    pm = pm[pm.exclude.fillna("") == ""].reset_index(drop=True)
    files = {p.name: p for p in P.PAD.rglob("*.png")}
    paths = [files[i] for i in pm.img_id]
    y = pm.diagnostic.map({d: k for k, d in enumerate(P.PAD6)}).to_numpy()
    z = {}
    for arm in ("A", "R"):
        for s in SEEDS:
            f = (M.OUT if arm == "A" else OUT) / "logits" / f"pad_{arm}_seed{s}.npy"
            if not f.exists():
                np.save(f, P.logits(M.load(arm, s), paths, 30))
            z[(arm, s)] = np.load(f)
    res, _ = P.patient_resamples(pm.patient_id.to_numpy(), y)
    prow = []
    for arm in ("A", "R"):
        ms = [P.metrics(z[(arm, s)], list(names), y) for s in SEEDS]
        prow.append({"row": f"arm {arm}", **{k: np.mean([x[k] for x in ms]) for k in ("bal_acc_6way", "macro_auroc_6way", "isic_rate")}})
    for m in ("bal_acc_6way", "macro_auroc_6way"):
        dd = [P.metrics(z[("R", s)], list(names), y)[m] - P.metrics(z[("A", s)], list(names), y)[m] for s in SEEDS]
        db = [np.mean([P.metrics(z[("R", s)][i], list(names), y[i])[m] - P.metrics(z[("A", s)][i], list(names), y[i])[m]
                       for s in SEEDS]) for i in res]
        lo, hi = np.percentile(db, [2.5, 97.5])
        prow.append({"row": f"R-A {m}", "delta": np.mean(dd), "per_seed": np.round(dd, 4).tolist(), "ci_low": lo, "ci_high": hi})
    pd.DataFrame(prow).to_csv(OUT / "pad_secondary.csv", index=False)
    print(pd.DataFrame(prow).round(4).to_string(index=False))

    # strictly filtered test set (same 863 images for A and R)
    test = datasets.ImageFolder(ROOT / "data" / "skin31_clean" / "test")
    tp, ty = [p for p, _ in test.samples], np.array(test.targets)
    acc = []
    for arm in ("A", "R"):
        for s in SEEDS:
            f = (M.OUT if arm == "A" else OUT) / "logits" / f"test_{arm}_seed{s}.npy"
            if not f.exists():
                np.save(f, P.logits(M.load(arm, s), tp, 30))
            acc.append({"arm": arm, "seed": s, "acc": (np.load(f).argmax(1) == ty).mean()})
    print(pd.DataFrame(acc).groupby("arm").acc.agg(["mean", "std"]).round(4).to_string())


if __name__ == "__main__":
    main()
