"""R6: near-duplicate / same-lesion search between the corrected test set (863 images) and the training images,
using DINOv2 embeddings (pretrained, independent of our classifiers). Outputs the top match per test image and
contact sheets of the most similar pairs for visual review."""
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "skin31_clean"
OUT = ROOT / "outputs" / "embed_leak"
OUT.mkdir(parents=True, exist_ok=True)

model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=224).cuda().eval()
cfg = timm.data.resolve_data_config({"input_size": (3, 224, 224), "crop_pct": 1.0}, model=model)
tf = timm.data.create_transform(**cfg)


@torch.inference_mode()
def embed(paths):
    out = []
    for i in range(0, len(paths), 64):
        x = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i + 64]]).cuda()
        out.append(torch.nn.functional.normalize(model(x).float(), dim=1).cpu())
    return torch.cat(out).numpy()


test = sorted(p for p in (CLEAN / "test").rglob("*") if p.is_file())
train = sorted(p for p in (CLEAN / "train").rglob("*") if p.is_file())
et, er = embed(test), embed(train)
sim = et @ er.T
j = sim.argmax(1)
df = pd.DataFrame({"test": [str(p) for p in test], "test_class": [p.parent.name for p in test],
                   "train": [str(train[k]) for k in j], "train_class": [train[k].parent.name for k in j],
                   "cosine": sim[np.arange(len(test)), j]})
df["source"] = np.where(df.test.str.contains("ISIC_"), "ISIC", "Atlas")
df = df.sort_values("cosine", ascending=False).reset_index(drop=True)
df.to_csv(OUT / "top_match_per_test_image.csv", index=False)
print(df.cosine.describe().round(3).to_string())
for t in (0.95, 0.9, 0.85, 0.8):
    print(f"cosine >= {t}:", (df.cosine >= t).sum(), df[df.cosine >= t].source.value_counts().to_dict())

top = df.head(64)
W, H = 220, 165
for s in range(0, len(top), 8):
    ch = top.iloc[s:s + 8]
    sh = Image.new("RGB", (2 * W + 10, len(ch) * (H + 20)), "white")
    d = ImageDraw.Draw(sh)
    for k, r in enumerate(ch.itertuples()):
        y = k * (H + 20)
        for c, p in enumerate([r.test, r.train]):
            im = Image.open(p).convert("RGB")
            im.thumbnail((W, H))
            sh.paste(im, (c * (W + 10), y + 18))
        d.text((2, y + 2), f"#{s + k} cos={r.cosine:.3f} test:{r.test_class[:18]} / train:{r.train_class[:18]}", fill="black")
    sh.save(OUT / f"sheet_{s // 8:02d}.jpg", quality=80)
