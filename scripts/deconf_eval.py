"""Evaluation for prereg/deconfound_analysis_plan.md. Deconfounded models (checkpoints_deconf/5_b0_trainopt/seed*)
vs corrected models (checkpoints_clean/5_b0_trainopt/seed*), seeds matched.
Usage: python scripts/deconf_eval.py --locked-commit <osf id>
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import pad_eval as P  # noqa: E402
from derm7pt_eval_clean import logits as d7_logits  # noqa: E402
from reproduce import Paths  # noqa: E402

PRE = ROOT / "prereg"
D7 = ROOT / "data" / "release_v0" / "images"
OUT = ROOT / "outputs" / "deconf"
SEEDS, B, RNG_SEED = [42, 123, 2024], 2000, 20260930
CFG = "5_b0_trainopt"
ISIC6 = {"actinic keratosis", "dermatofibroma", "nevus", "pigmented benign keratosis",
         "squamous cell carcinoma", "vascular lesion"}
TARGET, CONTROL = ["BCC", "MEL"], ["NEV", "SEK", "VASC", "DF"]
CORRECT = {"BCC": "basal cell carcinoma", "MEL": "melanoma"}


def load(kind, seed):
    folder = ROOT / ("checkpoints_deconf" if kind == "deconf" else "checkpoints_clean") / CFG / f"seed{seed}"
    m = timm.create_model("efficientnet_b0", pretrained=False, num_classes=30)
    missing, unexpected = m.load_state_dict(torch.load(next(folder.glob("*.pth")), map_location="cpu",
                                                       weights_only=True), strict=False)
    assert not missing and not unexpected
    return m.cuda().eval()


def estimands(pc, pd_, cls, names):
    names = np.array(names)
    isic = np.array([n in ISIC6 for n in names])
    t, n = np.isin(cls, TARGET), np.isin(cls, CONTROL)
    e1 = np.mean([(names[pd_[cls == d]] == CORRECT[d]).mean() for d in TARGET])
    e2 = isic[pd_[t]].mean() - isic[pc[t]].mean()
    e3 = isic[pd_[n]].mean() - isic[pc[n]].mean()
    clin = np.mean([(names[pc[cls == d]] == CORRECT[d]).mean() for d in TARGET])
    p1 = isic[pd_].mean() - isic[pc].mean()
    return np.array([e1, e2, e3, clin, p1])


def verdict(per_seed, lo, hi, expected):
    s = np.sign(expected)
    if all(np.sign(v) == s for v in per_seed) and (lo > 0 if s > 0 else hi < 0):
        return "supported (expected direction)"
    if all(np.sign(v) == -s for v in per_seed) and (hi < 0 if s > 0 else lo > 0):
        return "opposite direction"
    return "no consistent evidence of a change"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locked-commit", required=True)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "LOCKED_COMMIT.txt").write_text(args.locked_commit)
    names = P.class_names("clean")
    man = pd.read_csv(PRE / "derm7pt_manifest.csv")
    cls = man.cls.to_numpy()

    # ---- Derm7pt
    preds = {}
    for kind in ("deconf", "clean"):
        for s in SEEDS:
            f = OUT / "logits" / f"derm7pt_{kind}_seed{s}.npz"
            if not f.exists():
                f.parent.mkdir(parents=True, exist_ok=True)
                m = load(kind, s)
                np.savez(f, clinic=d7_logits(m, [D7 / p for p in man.clinic]), derm=d7_logits(m, [D7 / p for p in man.derm]))
            z = np.load(f)
            preds[(kind, s)] = (z["clinic"].argmax(1), z["derm"].argmax(1))
    point = {k: estimands(*v, cls, names) for k, v in preds.items()}
    diff_seed = np.array([point[("deconf", s)] - point[("clean", s)] for s in SEEDS])
    rng = np.random.default_rng(RNG_SEED)
    boots = []
    while len(boots) < B:
        i = rng.integers(0, len(cls), len(cls))
        if any((cls[i] == d).sum() == 0 for d in TARGET + CONTROL):
            continue
        boots.append(np.mean([estimands(preds[("deconf", s)][0][i], preds[("deconf", s)][1][i], cls[i], names)
                              - estimands(preds[("clean", s)][0][i], preds[("clean", s)][1][i], cls[i], names)
                              for s in SEEDS], axis=0))
    boots = np.array(boots)
    labels = ["E1_derm_recall_target", "E2_P1_target", "E3_P1_control", "S_clinical_recall_target", "S_P1_all"]
    expected = [1, -1, None, None, None]
    rows = []
    for k, lab in enumerate(labels):
        lo, hi = np.percentile(boots[:, k], [2.5, 97.5])
        rows.append({"estimand": lab,
                     "clean_mean": np.mean([point[("clean", s)][k] for s in SEEDS]),
                     "deconf_mean": np.mean([point[("deconf", s)][k] for s in SEEDS]),
                     "delta": diff_seed[:, k].mean(), "delta_per_seed": diff_seed[:, k].round(4).tolist(),
                     "ci_low": lo, "ci_high": hi,
                     "verdict": verdict(diff_seed[:, k], lo, hi, expected[k]) if expected[k] else
                     ("E3 rule: see plan §4" if lab.startswith("E3") else "descriptive")})
    r = pd.DataFrame(rows)
    e2, e3 = r.loc[1, "delta"], r.loc[2, "delta"]
    e3_supported_like_e2 = verdict(diff_seed[:, 2], r.loc[2, "ci_low"], r.loc[2, "ci_high"], -1).startswith("supported")
    r.loc[2, "verdict"] = ("specific" if (not e3_supported_like_e2 or abs(e3) < 0.5 * abs(e2)) else "not specific")
    r.to_csv(OUT / "derm7pt_primary.csv", index=False)
    print(r.round(4).to_string(index=False))

    # ---- PAD (secondary)
    pm = pd.read_csv(PRE / "pad_manifest.csv")
    pm = pm[pm.exclude.fillna("") == ""].reset_index(drop=True)
    files = {p.name: p for p in P.PAD.rglob("*.png")}
    paths = [files[i] for i in pm.img_id]
    y = pm.diagnostic.map({d: k for k, d in enumerate(P.PAD6)}).to_numpy()
    z = {}
    for kind in ("deconf", "clean"):
        for s in SEEDS:
            f = OUT / "logits" / f"pad_{kind}_seed{s}.npy"
            if not f.exists():
                np.save(f, P.logits(load(kind, s), paths, 30))
            z[(kind, s)] = np.load(f)
    res, _ = P.patient_resamples(pm.patient_id.to_numpy(), y)
    prow = []
    for m in ("bal_acc_6way", "macro_auroc_6way"):
        d_seed = [P.metrics(z[("deconf", s)], names, y)[m] - P.metrics(z[("clean", s)], names, y)[m] for s in SEEDS]
        d_boot = [np.mean([P.metrics(z[("deconf", s)][i], names, y[i])[m] - P.metrics(z[("clean", s)][i], names, y[i])[m]
                           for s in SEEDS]) for i in res]
        lo, hi = np.percentile(d_boot, [2.5, 97.5])
        prow.append({"metric": m, "delta": np.mean(d_seed), "per_seed": np.round(d_seed, 4).tolist(), "ci_low": lo, "ci_high": hi})
    for kind in ("clean", "deconf"):
        ms = [P.metrics(z[(kind, s)], names, y) for s in SEEDS]
        prow.append({"metric": f"{kind}_summary", **{k: np.mean([x[k] for x in ms]) for k in ms[0]}})
    pd.DataFrame(prow).to_csv(OUT / "pad_secondary.csv", index=False)
    print(pd.DataFrame(prow)[["metric", "delta", "ci_low", "ci_high", "bal_acc_6way", "macro_auroc_6way", "isic_rate"]].round(4).to_string(index=False))

    # ---- corrected test set (secondary)
    from torchvision import datasets
    test = datasets.ImageFolder(ROOT / "data" / "skin31_clean" / "test")
    tp, ty = [p for p, _ in test.samples], np.array(test.targets)
    acc = []
    for kind in ("clean", "deconf"):
        for s in SEEDS:
            f = OUT / "logits" / f"test_{kind}_seed{s}.npy"
            if not f.exists():
                np.save(f, P.logits(load(kind, s), tp, 30))
            pr = np.load(f).argmax(1)
            acc.append({"model": kind, "seed": s, "acc": (pr == ty).mean(),
                        "bal_acc": np.mean([(pr[ty == k] == k).mean() for k in np.unique(ty)])})
    a = pd.DataFrame(acc)
    a.to_csv(OUT / "test_secondary.csv", index=False)
    print(a.groupby("model")[["acc", "bal_acc"]].agg(["mean", "std"]).round(4).to_string())


if __name__ == "__main__":
    main()
