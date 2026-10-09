"""Task 0.16: verify the local environment reproduces the paper's checkpoints.

(a) SHA256 is in checkpoints/checkpoint_manifest.csv
(b) every checkpoint loads with missing=0, unexpected=0
(c) skin31 test accuracy of all 21 checkpoints vs Supplementary Table S5
(d) PAD-UFES-20, seed 42, fusion-distilled B0 only (the seed already inspected),
    on the original uncleaned PAD: 6-way balanced accuracy 0.3016, macro AUROC ~0.641
(e) class_mapping.json

The original `evaluate` ran under torch.amp.autocast on CUDA, so every evaluation is
done twice: "amp" (as in the paper) and "fp32" (plan section 3.6). Logits are saved
for both. Derm7pt is deliberately NOT evaluated here.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, precision_score, roc_auc_score
from torchvision import datasets

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from skin_disease.models import efficientnet_b2_dual_level, efficientnet_b2_original  # noqa: E402
from skin_disease.transforms import build_eval_transform  # noqa: E402

SKIN31 = ROOT / "data" / "Atlas dan ISIC2019 (31 classes)"
PAD = ROOT / "data" / "archive"
CKPT = ROOT / "checkpoints"
OUT = ROOT / "outputs" / "repro"
SEEDS = [42, 123, 2024]

CONFIGS = {  # folder -> (architecture, Table S5 accuracy per seed 42/123/2024)
    "1_b2_baseline": ("b2", [83.90, 83.60, 84.61]),
    "2_b2_trainopt": ("b2", [89.64, 89.13, 90.14]),
    "3_b2_fusion_teacher": ("b2_fusion", [90.04, 89.54, 89.94]),
    "4_b0_baseline": ("b0", [85.31, 86.22, 85.51]),
    "5_b0_trainopt": ("b0", [88.83, 88.13, 89.94]),
    "6_b0_kd_nonfusion": ("b0", [89.13, 89.64, 89.54]),
    "7_b0_kd_fusion": ("b0", [89.44, 90.54, 89.74]),
}
PAD_MAP = {"ACK": 0, "BCC": 1, "MEL": 14, "NEV": 18, "SEK": 25, "SCC": 26}
SHARED6 = ["ACK", "BCC", "MEL", "NEV", "SEK", "SCC"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build(arch):
    if arch == "b2":
        return efficientnet_b2_original(num_classes=31, pretrained=False)
    if arch == "b2_fusion":
        return efficientnet_b2_dual_level(num_classes=31, pretrained=False)
    return timm.create_model("efficientnet_b0", pretrained=False, num_classes=31)


def load(folder, seed):
    path = next((CKPT / folder / f"seed{seed}").glob("*.pth"))
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = build(CONFIGS[folder][0])
    missing, unexpected = model.load_state_dict(state, strict=False)
    return model.to(DEVICE).eval(), len(missing), len(unexpected)


class Paths(torch.utils.data.Dataset):
    def __init__(self, paths):
        self.paths, self.tf = list(paths), build_eval_transform(224)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB"))


@torch.inference_mode()
def logits(model, paths, amp):
    loader = torch.utils.data.DataLoader(Paths(paths), batch_size=32, shuffle=False, num_workers=0)
    out = []
    for x in loader:
        with torch.amp.autocast("cuda", enabled=amp and DEVICE.type == "cuda"):
            out.append(model(x.to(DEVICE)).float().cpu())
    return torch.cat(out).numpy()


def check_class_order():
    train_classes = datasets.ImageFolder(SKIN31 / "train").classes
    test_classes = datasets.ImageFolder(SKIN31 / "test").classes
    assert len(train_classes) == 31
    assert [c.lower() for c in train_classes] == [c.lower() for c in test_classes], "train/test class order differs"
    expected = {0: "actinic keratosis", 1: "basal cell carcinoma", 14: "melanoma", 18: "nevus",
                25: "seborrheic keratosis", 26: "squamous cell carcinoma"}
    for i, name in expected.items():
        assert train_classes[i].lower() == name, (i, train_classes[i])
    mapping = {"class_to_idx": {c: i for i, c in enumerate(train_classes)}, "pad_to_idx": PAD_MAP}
    text = json.dumps(mapping, indent=2, ensure_ascii=False)
    (OUT / "class_mapping.json").write_text(text, encoding="utf-8")
    print("class_mapping.json sha256:", hashlib.sha256(text.encode()).hexdigest())
    return train_classes


def skin31_metrics(y, z):
    p = z.argmax(1)
    return {"acc": 100 * (p == y).mean(), "bal_acc": 100 * balanced_accuracy_score(y, p),
            "macro_prec": 100 * precision_score(y, p, average="macro", zero_division=0),
            "macro_f1": 100 * f1_score(y, p, average="macro")}


def pad_metrics(y6, z):
    z6 = z[:, [PAD_MAP[c] for c in SHARED6]]
    p6 = z6.argmax(1)
    prob = np.exp(z6 - z6.max(1, keepdims=True))
    prob /= prob.sum(1, keepdims=True)
    return {"bal_acc_6way": balanced_accuracy_score(y6, p6),
            "macro_auroc_6way": roc_auc_score(y6, prob, multi_class="ovr", average="macro"),
            "cm": confusion_matrix(y6, p6, labels=range(6))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("torch", torch.__version__, "| timm", timm.__version__, "|", DEVICE,
          torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "")
    check_class_order()

    test = datasets.ImageFolder(SKIN31 / "test")
    paths = [p for p, _ in test.samples]
    y = np.array(test.targets)
    pd.DataFrame({"path": paths, "label": y}).to_csv(OUT / "skin31_test_index.csv", index=False)

    rows = []
    for folder in CONFIGS:
        for k, seed in enumerate(SEEDS):
            model, n_miss, n_unexp = load(folder, seed)
            d = OUT / folder / f"seed{seed}"
            d.mkdir(parents=True, exist_ok=True)
            row = {"config": folder, "seed": seed, "missing": n_miss, "unexpected": n_unexp,
                   "paper_acc": CONFIGS[folder][1][k]}
            for mode in ("amp", "fp32"):
                z = logits(model, paths, amp=(mode == "amp"))
                np.save(d / f"skin31_test_{mode}.npy", z)
                for m, v in skin31_metrics(y, z).items():
                    row[f"{m}_{mode}"] = round(v, 2)
            row["n_pred_diff_amp_fp32"] = int((np.load(d / "skin31_test_amp.npy").argmax(1) != z.argmax(1)).sum())
            rows.append(row)
            print(f"{folder:22s} seed{seed:<5d} miss={n_miss} unexp={n_unexp} "
                  f"acc amp={row['acc_amp']:.2f} fp32={row['acc_fp32']:.2f} paper={row['paper_acc']:.2f}")
    df = pd.DataFrame(rows)
    df["match_amp"] = (df.acc_amp - df.paper_acc).abs() < 0.005
    df.to_csv(OUT / "skin31_reproduction.csv", index=False)
    print("\nAll loads clean:", bool((df.missing == 0).all() and (df.unexpected == 0).all()))
    print("Table S5 reproduced exactly (amp):", int(df.match_amp.sum()), "/ 21")
    print(df.groupby("config")[["acc_amp", "bal_acc_amp", "macro_prec_amp", "macro_f1_amp"]]
          .agg(["mean", "std"]).round(2).to_string())

    # (d) PAD, seed 42, fusion-distilled B0 only
    meta = pd.read_csv(PAD / "metadata.csv")
    files = {p.name: p for p in PAD.rglob("*.png")}
    pad_paths = [files[i] for i in meta.img_id]
    y6 = meta.diagnostic.map({c: i for i, c in enumerate(SHARED6)}).to_numpy()
    model, _, _ = load("7_b0_kd_fusion", 42)
    d = OUT / "7_b0_kd_fusion" / "seed42"
    print("\nPAD-UFES-20 (original, uncleaned), fusion-distilled B0 seed 42 | target BA=0.3016, AUROC~0.641")
    for mode in ("amp", "fp32"):
        z = logits(model, pad_paths, amp=(mode == "amp"))
        np.save(d / f"pad_{mode}.npy", z)
        r = pad_metrics(y6, z)
        print(f"  {mode}: BA_6way={r['bal_acc_6way']:.4f}  macroAUROC_6way={r['macro_auroc_6way']:.4f}")
        print("  confusion (rows true, cols pred):", SHARED6)
        print("\n".join("   " + " ".join(f"{v:5d}" for v in row) for row in r["cm"]))
    meta[["img_id", "patient_id", "diagnostic"]].to_csv(OUT / "pad_index.csv", index=False)


if __name__ == "__main__":
    main()
