"""How much of the reported skin31 error is label error? Uses saved test logits from reproduce.py
and the ISIC Archive audit from isic_label_audit.py. Reports per config/seed:
  - test images whose folder label disagrees with the ISIC Archive diagnosis
  - for mislabeled Nevus-folder images (really AK): how many were predicted AK
  - accuracy on the original test set vs the test set with label-disputed and train-leaked images removed
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "outputs" / "repro"
AUDIT = ROOT / "outputs" / "dup_check"
MODE = "amp"

idx = pd.read_csv(REPRO / "skin31_test_index.csv")
idx["id"] = idx.path.map(lambda p: Path(p).stem)
idx["folder"] = idx.path.map(lambda p: Path(p).parent.name.lower())
audit = pd.read_csv(AUDIT / "skin31_isic_label_audit.csv")
audit = audit[audit.split == "test"][["id", "folder", "diagnosis_3", "agrees"]]
idx = idx.merge(audit, on=["id", "folder"], how="left")
idx["disputed"] = idx.agrees.eq(False)

hashes = pd.read_csv(AUDIT / "hashes_skin31.csv")
train_pix = set(hashes[hashes.split == "train"].md5_pixel)
test_h = hashes[hashes.split == "test"].set_index("path").md5_pixel
idx["leaked"] = idx.path.map(test_h).isin(train_pix)
idx["clean"] = ~idx.disputed & ~idx.leaked

AK, NEV = 0, 18
nev_is_ak = (idx.folder == "nevus") & (idx.diagnosis_3 == "Solar or actinic keratosis")
print(f"test images: {len(idx)} | label-disputed: {idx.disputed.sum()} | leaked from train: {idx.leaked.sum()} "
      f"| clean: {idx.clean.sum()} | Nevus-folder images that are AK: {nev_is_ak.sum()}")
print(idx[idx.disputed].groupby(["folder", "diagnosis_3"]).size().to_string())

rows = []
for d in sorted(REPRO.glob("*/seed*")):
    f = d / f"skin31_test_{MODE}.npy"
    if not f.exists():
        continue
    p = np.load(f).argmax(1)
    y = idx.label.to_numpy()
    rows.append({
        "config": d.parent.name, "seed": d.name,
        "acc_all": 100 * (p == y).mean(),
        "acc_clean": 100 * (p == y)[idx.clean].mean(),
        "nevAK_pred_AK": int((p[nev_is_ak] == AK).sum()),
        "nevAK_pred_NEV": int((p[nev_is_ak] == NEV).sum()),
        "nevus_to_AK_total": int(((idx.label == NEV) & (p == AK)).sum()),
    })
r = pd.DataFrame(rows)
r.to_csv(REPRO / "label_error_impact.csv", index=False)
print(r.round(2).to_string(index=False))
print(r.groupby("config")[["acc_all", "acc_clean"]].agg(["mean", "std"]).round(2).to_string())
