"""Exploratory (not pre-registered): MILK10k control (osf.io/5ksja) repeated with ViT-S/16.
Arms B (+ MILK10k clinical) and C (+ MILK10k dermoscopic, same lesions), seeds 42/123/2024, same data as the
EfficientNet-B0 arms; ViT recipe as in the ViT-S/16 analysis (AdamW 1e-4, cosine, no label smoothing/CutMix/SAM).
Same estimands and bootstrap as milk_eval.py; also the trade-offs of intervention_tradeoffs.py."""
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
from deconf_eval import estimands  # noqa: E402
from derm7pt_eval_clean import logits as d7_logits  # noqa: E402

PRE = ROOT / "prereg"
D7 = ROOT / "data" / "release_v0" / "images"
OUT = ROOT / "outputs" / "vit_milk"
SEEDS, B, RNG_SEED = [42, 123, 2024], 2000, 20260930
ARMS = {"B": ROOT / "checkpoints_milkB" / "ViTmilkB", "C": ROOT / "checkpoints_milkC" / "ViTmilkC"}
LABELS = ["E1", "E2", "E3", "clin_recall_target", "P1_all", "untreated_derm_to_MEL", "NEV_derm_recall", "MEL_derm_recall"]
UNTREATED = ["NEV", "SEK", "VASC", "DF"]


def load(arm, seed):
    m = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=30)
    missing, unexpected = m.load_state_dict(torch.load(next((ARMS[arm] / f"seed{seed}").glob("*.pth")), map_location="cpu",
                                                       weights_only=True), strict=False)
    assert not missing and not unexpected
    return m.cuda().eval()


def stats(pc, pd_, cls, names):
    names = np.array(names)
    u = np.isin(cls, UNTREATED)
    extra = [(names[pd_[u]] == "melanoma").mean(), (names[pd_[cls == "NEV"]] == "nevus").mean(),
             (names[pd_[cls == "MEL"]] == "melanoma").mean()]
    return np.concatenate([estimands(pc, pd_, cls, names), extra])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
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
        v = np.array([stats(*preds[(arm, s)], cls, names) for s in SEEDS])
        rows.append({"row": f"arm {arm}", **{l: v[:, k].mean() for k, l in enumerate(LABELS)},
                     **{f"{l}_seeds": v[:, k].round(4).tolist() for k, l in enumerate(LABELS)}})
    d = np.array([stats(*preds[("C", s)], cls, names) - stats(*preds[("B", s)], cls, names) for s in SEEDS])
    rng = np.random.default_rng(RNG_SEED)
    boots = []
    while len(boots) < B:
        i = rng.integers(0, len(cls), len(cls))
        if any((cls[i] == c).sum() == 0 for c in ("BCC", "MEL", *UNTREATED)):
            continue
        boots.append(np.mean([stats(preds[("C", s)][0][i], preds[("C", s)][1][i], cls[i], names)
                              - stats(preds[("B", s)][0][i], preds[("B", s)][1][i], cls[i], names) for s in SEEDS], axis=0))
    lo, hi = np.percentile(np.array(boots), [2.5, 97.5], axis=0)
    for k, l in enumerate(LABELS):
        rows.append({"row": "C-B", "estimand": l, "delta": d[:, k].mean(), "per_seed": d[:, k].round(4).tolist(),
                     "ci_low": lo[k], "ci_high": hi[k]})
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "derm7pt_results.csv", index=False)
    print(r[r.row == "C-B"][["estimand", "delta", "per_seed", "ci_low", "ci_high"]].round(4).to_string(index=False))
    print(r[r.row != "C-B"][["row", *LABELS]].round(4).to_string(index=False))

    pm = pd.read_csv(PRE / "pad_manifest.csv")
    pm = pm[pm.exclude.fillna("") == ""].reset_index(drop=True)
    files = {p.name: p for p in P.PAD.rglob("*.png")}
    paths = [files[i] for i in pm.img_id]
    y = pm.diagnostic.map({c: k for k, c in enumerate(P.PAD6)}).to_numpy()
    z = {}
    for arm in ARMS:
        for s in SEEDS:
            f = OUT / "logits" / f"pad_{arm}_seed{s}.npy"
            if not f.exists():
                np.save(f, P.logits(load(arm, s), paths, 30))
            z[(arm, s)] = np.load(f)
    res, _ = P.patient_resamples(pm.patient_id.to_numpy(), y)
    m = "bal_acc_6way"
    prow = [{"row": f"arm {a}", m: np.mean([P.metrics(z[(a, s)], names, y)[m] for s in SEEDS])} for a in ARMS]
    dd = [P.metrics(z[("C", s)], names, y)[m] - P.metrics(z[("B", s)], names, y)[m] for s in SEEDS]
    db = [np.mean([P.metrics(z[("C", s)][i], names, y[i])[m] - P.metrics(z[("B", s)][i], names, y[i])[m] for s in SEEDS]) for i in res]
    prow.append({"row": "C-B", "delta": np.mean(dd), "per_seed": np.round(dd, 4).tolist(),
                 "ci_low": np.percentile(db, 2.5), "ci_high": np.percentile(db, 97.5)})
    pd.DataFrame(prow).to_csv(OUT / "pad.csv", index=False)
    print(pd.DataFrame(prow).round(4).to_string(index=False))

    test = datasets.ImageFolder(ROOT / "data" / "skin31_clean" / "test")
    tp, ty = [p for p, _ in test.samples], np.array(test.targets)
    acc = []
    for arm in ARMS:
        for s in SEEDS:
            f = OUT / "logits" / f"test_{arm}_seed{s}.npy"
            if not f.exists():
                np.save(f, P.logits(load(arm, s), tp, 30))
            acc.append({"arm": arm, "seed": s, "acc": (np.load(f).argmax(1) == ty).mean()})
    a = pd.DataFrame(acc)
    a.to_csv(OUT / "test.csv", index=False)
    print(a.groupby("arm").acc.agg(["mean", "std"]).round(4).to_string())


if __name__ == "__main__":
    main()
