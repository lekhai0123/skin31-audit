"""Internal (benchmark) accuracy behind Supplementary Table S4 and Results 3.1, from saved outputs:
the 21 original checkpoints on the original test split (994 images; fp32 logits from reproduce.py), the ViT-S/16
(outputs/vit/seed*.npz) and the authors' released model (outputs/authors_model/predictions.csv, train and test)."""
from pathlib import Path

import numpy as np
import pandas as pd
from torchvision import datasets

ROOT = Path(__file__).resolve().parents[1]
O = ROOT / "outputs"
SEEDS = [42, 123, 2024]
CFGS = ["1_b2_baseline", "2_b2_trainopt", "3_b2_fusion_teacher", "4_b0_baseline", "5_b0_trainopt",
        "6_b0_kd_nonfusion", "7_b0_kd_fusion"]


def main():
    y = np.array(datasets.ImageFolder(ROOT / "data" / "Atlas dan ISIC2019 (31 classes)" / "test").targets)
    rows = []
    for c in CFGS:
        acc = [(np.load(O / "repro" / c / f"seed{s}" / "skin31_test_fp32.npy").argmax(1) == y).mean() for s in SEEDS]
        rows.append({"model": c, "split": "test (994)", "acc_pct_mean": 100 * np.mean(acc),
                     "acc_pct_per_seed": [round(100 * a, 2) for a in acc]})
    acc = [(np.load(O / "vit" / f"seed{s}.npz")["test"][:, :31].argmax(1) == y).mean() for s in SEEDS]
    rows.append({"model": "vit_s16", "split": "test (994)", "acc_pct_mean": 100 * np.mean(acc),
                 "acc_pct_per_seed": [round(100 * a, 2) for a in acc]})
    a = pd.read_csv(O / "authors_model" / "predictions.csv")
    for sp, g in a.groupby("split"):
        rows.append({"model": "authors", "split": f"{sp} ({len(g)})", "acc_pct_mean": 100 * (g.label == g.pred).mean(),
                     "acc_pct_per_seed": None})
    r = pd.DataFrame(rows)
    r.to_csv(O / "repro" / "internal_accuracy_summary.csv", index=False)
    print(r.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
