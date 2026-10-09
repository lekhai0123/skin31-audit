"""Evaluation for prereg/milk_analysis_plan.md (source- and lesion-matched control, MILK10k).
Arms (EfficientNet-B0 + training optimization, seeds 42/123/2024):
  A  corrected data                      checkpoints_clean/5_b0_trainopt
  B  + MILK10k clinical close-ups       checkpoints_milkB/5_b0_trainopt
  C  + MILK10k dermoscopic, same lesions checkpoints_milkC/5_b0_trainopt
Primary: C - B on E1 and E2 (Derm7pt). Secondary: E3 for C - B; all estimands for B - A and C - A; PAD and test.
Usage: python scripts/milk_eval.py --locked-commit <osf id>
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from torchvision import datasets

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import pad_eval as P  # noqa: E402
from deconf_eval import estimands, verdict  # noqa: E402  (same definitions as osf.io/pe8q6)
from derm7pt_eval_clean import logits as d7_logits  # noqa: E402

PRE = ROOT / "prereg"
D7 = ROOT / "data" / "release_v0" / "images"
OUT = ROOT / "outputs" / "milk_eval"
SEEDS, B, RNG_SEED = [42, 123, 2024], 2000, 20260930
ARMS = {"A": ROOT / "checkpoints_clean", "B": ROOT / "checkpoints_milkB", "C": ROOT / "checkpoints_milkC"}
LABELS = ["E1_derm_recall_target", "E2_P1_target", "E3_P1_control", "S_clinical_recall_target", "S_P1_all"]
EXPECT = {"C-B": [1, -1, None, None, None]}


def load(arm, seed):
    folder = ARMS[arm] / "5_b0_trainopt" / f"seed{seed}"
    m = timm.create_model("efficientnet_b0", pretrained=False, num_classes=30)
    missing, unexpected = m.load_state_dict(torch.load(next(folder.glob("*.pth")), map_location="cpu",
                                                       weights_only=True), strict=False)
    assert not missing and not unexpected
    return m.cuda().eval()


def contrast(preds, x, y, cls, names):
    """Paired over cases and seed-matched: estimands(x) - estimands(y)."""
    d_seed = np.array([estimands(*preds[(x, s)], cls, names) - estimands(*preds[(y, s)], cls, names) for s in SEEDS])
    rng = np.random.default_rng(RNG_SEED)
    boots = []
    while len(boots) < B:
        i = rng.integers(0, len(cls), len(cls))
        if any((cls[i] == d).sum() == 0 for d in ("BCC", "MEL", "NEV", "SEK", "VASC", "DF")):
            continue
        boots.append(np.mean([estimands(preds[(x, s)][0][i], preds[(x, s)][1][i], cls[i], names)
                              - estimands(preds[(y, s)][0][i], preds[(y, s)][1][i], cls[i], names) for s in SEEDS], axis=0))
    return d_seed, np.percentile(np.array(boots), [2.5, 97.5], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locked-commit", required=True)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "LOCKED_COMMIT.txt").write_text(args.locked_commit)
    names = P.class_names("clean")
    man = pd.read_csv(PRE / "derm7pt_manifest.csv")
    cls = man.cls.to_numpy()

    preds = {}
    for arm in ARMS:
        for s in SEEDS:
            f = OUT / "logits" / f"derm7pt_{arm}_seed{s}.npz"
            if not f.exists():
                f.parent.mkdir(parents=True, exist_ok=True)
                m = load(arm, s)
                np.savez(f, clinic=d7_logits(m, [D7 / p for p in man.clinic]), derm=d7_logits(m, [D7 / p for p in man.derm]))
            z = np.load(f)
            preds[(arm, s)] = (z["clinic"].argmax(1), z["derm"].argmax(1))

    rows = []
    for arm in ARMS:
        v = np.array([estimands(*preds[(arm, s)], cls, names) for s in SEEDS])
        rows.append({"row": f"arm {arm}", **{l: v[:, k].mean() for k, l in enumerate(LABELS)}})
    for x, y in (("C", "B"), ("B", "A"), ("C", "A")):
        d, (lo, hi) = contrast(preds, x, y, cls, names)
        for k, lab in enumerate(LABELS):
            exp = EXPECT.get(f"{x}-{y}", [None] * 5)[k]
            rows.append({"row": f"{x}-{y}", "estimand": lab, "delta": d[:, k].mean(), "per_seed": d[:, k].round(4).tolist(),
                         "ci_low": lo[k], "ci_high": hi[k],
                         "verdict": verdict(d[:, k], lo[k], hi[k], exp) if exp else "descriptive"})
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "derm7pt_results.csv", index=False)
    print(r.round(4).to_string(index=False))

    # PAD (secondary): C - B
    pm = pd.read_csv(PRE / "pad_manifest.csv")
    pm = pm[pm.exclude.fillna("") == ""].reset_index(drop=True)
    files = {p.name: p for p in P.PAD.rglob("*.png")}
    paths = [files[i] for i in pm.img_id]
    y = pm.diagnostic.map({d: k for k, d in enumerate(P.PAD6)}).to_numpy()
    z = {}
    for arm in ARMS:
        for s in SEEDS:
            f = OUT / "logits" / f"pad_{arm}_seed{s}.npy"
            if not f.exists():
                np.save(f, P.logits(load(arm, s), paths, 30))
            z[(arm, s)] = np.load(f)
    res, _ = P.patient_resamples(pm.patient_id.to_numpy(), y)
    prow = []
    for arm in ARMS:
        ms = [P.metrics(z[(arm, s)], names, y) for s in SEEDS]
        prow.append({"row": f"arm {arm}", **{k: np.mean([x[k] for x in ms]) for k in ("bal_acc_6way", "macro_auroc_6way", "isic_rate")}})
    for m in ("bal_acc_6way", "macro_auroc_6way"):
        d = [P.metrics(z[("C", s)], names, y)[m] - P.metrics(z[("B", s)], names, y)[m] for s in SEEDS]
        db = [np.mean([P.metrics(z[("C", s)][i], names, y[i])[m] - P.metrics(z[("B", s)][i], names, y[i])[m] for s in SEEDS]) for i in res]
        lo, hi = np.percentile(db, [2.5, 97.5])
        prow.append({"row": f"C-B {m}", "delta": np.mean(d), "ci_low": lo, "ci_high": hi})
    pd.DataFrame(prow).to_csv(OUT / "pad_secondary.csv", index=False)
    print(pd.DataFrame(prow).round(4).to_string(index=False))

    # corrected test set (secondary)
    test = datasets.ImageFolder(ROOT / "data" / "skin31_clean" / "test")
    tp, ty = [p for p, _ in test.samples], np.array(test.targets)
    acc = []
    for arm in ARMS:
        for s in SEEDS:
            f = OUT / "logits" / f"test_{arm}_seed{s}.npy"
            if not f.exists():
                np.save(f, P.logits(load(arm, s), tp, 30))
            pr = np.load(f).argmax(1)
            acc.append({"arm": arm, "seed": s, "acc": (pr == ty).mean()})
    print(pd.DataFrame(acc).groupby("arm").acc.agg(["mean", "std"]).round(4).to_string())


if __name__ == "__main__":
    main()
