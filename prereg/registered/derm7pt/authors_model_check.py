"""Evaluate the EfficientSkinDis authors' released weights (github.com/abdulrafay97/EfficientSkinDis, Weights.h5)
on the public Kaggle copy of the 31-class dataset.

A model memorizes its own training set (the paper reports ~0.98 train accuracy vs ~0.87 test), so near-train-level
accuracy on the public train folder and ~0.87 on the public test folder indicates the public split is the authors' split.
Also reports what the authors' model predicts for the mislabeled images.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import datasets, models, transforms

ROOT = Path(__file__).resolve().parents[1]
SKIN31 = ROOT / "data" / "Atlas dan ISIC2019 (31 classes)"
OUT = ROOT / "outputs" / "authors_model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_authors_model():
    # identical to app.py: `_fc` is attached but never called by torchvision's forward,
    # so predictions come from the 1000-way `classifier`; indices 0..30 are the 31 classes.
    model = models.efficientnet_b2(weights=None)
    model._fc = nn.Sequential(nn.BatchNorm1d(1024), nn.Linear(1024, 512), nn.ReLU(), nn.BatchNorm1d(512),
                              nn.Linear(512, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.4),
                              nn.Linear(128, 31))
    state = torch.load(ROOT / "ref_efficientskindis" / "Weights.h5", map_location="cpu", weights_only=True)
    print(model.load_state_dict(state, strict=True))
    return model.to(DEVICE).eval()


TF = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


@torch.inference_mode()
def predict(model, paths):
    out = []
    for i in range(0, len(paths), 32):
        x = torch.stack([TF(Image.open(p).convert("RGB")) for p in paths[i:i + 32]]).to(DEVICE)
        out.append(model(x).float().cpu())
    return torch.cat(out).numpy()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = load_authors_model()
    audit = pd.read_csv(ROOT / "outputs" / "dup_check" / "skin31_isic_label_audit.csv")
    audit["key"] = audit.split + "/" + audit.id
    rows = []
    for split in ("train", "test"):
        ds = datasets.ImageFolder(SKIN31 / split)
        paths = [p for p, _ in ds.samples]
        z = predict(model, paths)
        np.save(OUT / f"{split}_logits1000.npy", z)
        y = np.array(ds.targets)
        p31, p1000 = z[:, :31].argmax(1), z.argmax(1)
        print(f"{split}: n={len(y)} acc(31 logits)={100 * (p31 == y).mean():.2f}  "
              f"acc(all 1000)={100 * (p1000 == y).mean():.2f}  argmax outside 0..30: {(p1000 > 30).sum()}")
        for pth, yi, pi in zip(paths, y, p31):
            rows.append({"split": split, "path": pth, "label": ds.classes[yi], "pred": ds.classes[pi],
                         "id": Path(pth).stem, "correct": yi == pi})
    df = pd.DataFrame(rows)
    df["key"] = df.split + "/" + df.id
    df = df.merge(audit[["key", "label", "diagnosis_3", "agrees"]].rename(columns={"label": "folder"}),
                  left_on=["key", "label"], right_on=["key", "folder"], how="left")
    df.to_csv(OUT / "predictions.csv", index=False)

    print("\nPer-class accuracy, train vs test (authors' model):")
    print(df.pivot_table(index="label", columns="split", values="correct", aggfunc="mean").mul(100).round(1).to_string())
    bad = df[df.agrees.eq(False)]
    print("\nMislabeled images: what the authors' model predicts")
    print(bad.groupby(["split", "label", "diagnosis_3", "pred"]).size().to_string())


if __name__ == "__main__":
    main()
