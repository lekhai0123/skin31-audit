"""R4 (exploratory, not pre-registered): ViT-S/16 trained on the original 31-class dataset.
Applies the Derm7pt estimands P1/P2 of prereg/derm7pt_analysis_plan.md and the PAD metrics of
prereg/pad_analysis_plan.md, plus accuracy on the original and corrected test sets."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from torchvision import datasets

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import derm7pt_eval as D  # noqa: E402
import pad_eval as P  # noqa: E402
from derm7pt_eval_clean import logits as raw_logits  # noqa: E402

OUT = ROOT / "outputs" / "vit"
CK = ROOT / "checkpoints_vit" / "vit_s16"
SEEDS = [s for s in (42, 123, 2024) if list((CK / f"seed{s}").glob("*.pth"))]


def load(seed):
    m = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=31)
    missing, unexpected = m.load_state_dict(torch.load(next((CK / f"seed{seed}").glob("*.pth")), map_location="cpu",
                                                       weights_only=True), strict=False)
    assert not missing and not unexpected
    return m.cuda().eval()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    names = P.class_names("orig")
    man = pd.read_csv(ROOT / "prereg" / "derm7pt_manifest.csv")
    cls = man.cls.to_numpy()
    pm = pd.read_csv(ROOT / "prereg" / "pad_manifest.csv")
    pm = pm[pm.exclude.fillna("") == ""].reset_index(drop=True)
    files = {p.name: p for p in P.PAD.rglob("*.png")}
    pad_paths = [files[i] for i in pm.img_id]
    y_pad = pm.diagnostic.map({d: k for k, d in enumerate(P.PAD6)}).to_numpy()
    t_orig = datasets.ImageFolder(ROOT / "data" / "Atlas dan ISIC2019 (31 classes)" / "test")
    flags = pd.read_csv(ROOT / "outputs" / "repro" / "skin31_test_flags.csv")

    preds, rows = [], []
    for s in SEEDS:
        f = OUT / f"seed{s}.npz"
        if not f.exists():
            m = load(s)
            np.savez(f, d7c=raw_logits(m, [D.D7 / p for p in man.clinic]), d7d=raw_logits(m, [D.D7 / p for p in man.derm]),
                     pad=P.logits(m, pad_paths, 31), test=P.logits(m, [p for p, _ in t_orig.samples], 31))
        z = np.load(f)
        pc, pd_ = z["d7c"][:, :31].argmax(1), z["d7d"][:, :31].argmax(1)
        preds.append((pc, pd_))
        p1, p2 = D.estimands(pc, pd_, cls)
        pm_ = P.metrics(z["pad"], names, y_pad)
        yt = np.array(t_orig.targets)
        pt = z["test"].argmax(1)
        rows.append({"seed": s, "P1": p1, "P2": p2, "pad_bal_acc_6way": pm_["bal_acc_6way"],
                     "pad_macro_auroc": pm_["macro_auroc_6way"], "pad_isic_rate": pm_["isic_rate"],
                     "test_acc_original": (pt == yt).mean(), "test_acc_strict": (pt == yt)[flags.strict.to_numpy()].mean()})
    r = pd.DataFrame(rows)
    lo, hi = D.bootstrap(preds, cls)  # already the 2.5 / 97.5 percentiles, per estimand
    r.to_csv(OUT / "vit_results.csv", index=False)
    print(r.round(4).to_string(index=False))
    print("P1 CI", np.round([lo[0], hi[0]], 3), "| P2 CI", np.round([lo[1], hi[1]], 3))


if __name__ == "__main__":
    main()
