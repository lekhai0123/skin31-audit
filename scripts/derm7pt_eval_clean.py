"""Derm7pt evaluation of the models retrained on the corrected dataset (30 outputs).

Applies the estimands of prereg/derm7pt_analysis_plan.md (osf.io/643ht) unchanged. Documented deviation: the
registered script indexes outputs by number for 31-output models; here outputs are resolved by class name.
The corrected models have no "Seborrheic keratosis" output, so in P2 the terms for that output are 0 and P2 reduces
to P(pred = Melanoma | clinic) - P(pred = Melanoma | derm).
Also reports, descriptively (not registered), the corrected-minus-original difference for the same configurations.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from pad_eval import class_names, load_clean  # noqa: E402
from skin_disease.transforms import build_eval_transform  # noqa: E402

PRE = ROOT / "prereg"
D7 = ROOT / "data" / "release_v0" / "images"
OUT = ROOT / "outputs" / "derm7pt"
ISIC_NAMES = {"actinic keratosis", "dermatofibroma", "nevus", "pigmented benign keratosis",
              "seborrheic keratosis", "squamous cell carcinoma", "vascular lesion"}
CORRECT = {"MEL": {"melanoma", "seborrheic keratosis"}, "BCC": {"basal cell carcinoma"}, "NEV": {"nevus"},
           "VASC": {"vascular lesion"}, "DF": {"dermatofibroma"}, "SEK": {"pigmented benign keratosis"}}
SEEDS, B, RNG_SEED = [42, 123, 2024], 2000, 20260930


def estimands(pc, pd_, cls, names):
    isic = np.array([n in ISIC_NAMES for n in names])
    mc, md = (np.array(names)[pc] == "melanoma"), (np.array(names)[pd_] == "melanoma")
    sc = (np.array(names)[pc] == "seborrheic keratosis")
    sd = (np.array(names)[pd_] == "seborrheic keratosis")
    mel = cls == "MEL"
    p1 = isic[pd_].mean() - isic[pc].mean()
    p2 = sd[mel].mean() - sc[mel].mean() + mc[mel].mean() - md[mel].mean()
    return p1, p2


def boot(preds, cls, names):
    rng = np.random.default_rng(RNG_SEED)
    out = []
    while len(out) < B:
        idx = rng.integers(0, len(cls), len(cls))
        if (cls[idx] == "MEL").sum() == 0:
            continue
        out.append(np.mean([estimands(pc[idx], pd_[idx], cls[idx], names) for pc, pd_ in preds], axis=0))
    return np.array(out)


@torch.inference_mode()
def logits(model, paths):
    tf = build_eval_transform(224)
    out = []
    for i in range(0, len(paths), 32):
        x = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i + 32]]).cuda()
        out.append(model(x).float().cpu())
    return torch.cat(out).numpy()


def main():
    man = pd.read_csv(PRE / "derm7pt_manifest.csv")
    cls = man.cls.to_numpy()
    cn, on = class_names("clean"), class_names("orig")
    rows, sec = [], []
    for c in ["4_b0_baseline", "5_b0_trainopt"]:
        preds_c, preds_o = [], []
        for s in SEEDS:
            f = OUT / "logits" / f"clean_{c}_seed{s}.npz"
            if not f.exists():
                m = load_clean(c, s)
                np.savez(f, clinic=logits(m, [D7 / p for p in man.clinic]), derm=logits(m, [D7 / p for p in man.derm]))
            z = np.load(f)
            preds_c.append((z["clinic"].argmax(1), z["derm"].argmax(1)))
            zo = np.load(OUT / "logits" / f"{c}_seed{s}.npz")
            preds_o.append((zo["clinic"].argmax(1), zo["derm"].argmax(1)))
        bc, bo = boot(preds_c, cls, cn), boot(preds_o, cls, on)
        for k, name in enumerate(["P1", "P2"]):
            vc = [estimands(pc, pd_, cls, cn)[k] for pc, pd_ in preds_c]
            vo = [estimands(pc, pd_, cls, on)[k] for pc, pd_ in preds_o]
            lo, hi = np.percentile(bc[:, k], [2.5, 97.5])
            verdict = ("supported" if all(v > 0 for v in vc) and lo > 0 else
                       "reversed" if all(v < 0 for v in vc) and hi < 0 else "no consistent evidence")
            rows.append({"config": c, "estimand": name, "corrected_mean": np.mean(vc), "corrected_sd": np.std(vc, ddof=1),
                         "ci_low": lo, "ci_high": hi, "verdict": verdict, "original_mean": np.mean(vo),
                         "corrected_minus_original": np.mean(vc) - np.mean(vo)})
        for s, (pc, pd_) in zip(SEEDS, preds_c):
            for dx, ok in CORRECT.items():
                m = cls == dx
                sec.append({"config": c, "seed": s, "dx": dx,
                            "recall_clinic": np.isin(np.array(cn)[pc[m]], list(ok)).mean(),
                            "recall_derm": np.isin(np.array(cn)[pd_[m]], list(ok)).mean(),
                            "isic_rate_clinic": np.mean([cn[p] in ISIC_NAMES for p in pc[m]]),
                            "isic_rate_derm": np.mean([cn[p] in ISIC_NAMES for p in pd_[m]])})
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "primary_clean.csv", index=False)
    s = pd.DataFrame(sec)
    s.to_csv(OUT / "secondary_clean.csv", index=False)
    print(r.round(3).to_string(index=False))
    print(s.groupby(["config", "dx"])[["recall_clinic", "recall_derm", "isic_rate_clinic", "isic_rate_derm"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
