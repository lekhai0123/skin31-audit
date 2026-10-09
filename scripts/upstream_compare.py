"""Task 0.12: trace skin31's ISIC part back to the Kaggle 'Skin Cancer ISIC 9 classes' dataset
(nodoubttome/skin-cancer9-classesisic) and check whether the label errors were inherited."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
UP = next((ROOT / "data" / "kaggle_cache").rglob("Skin cancer ISIC The International Skin Imaging Collaboration"))
OUT = ROOT / "outputs" / "dup_check"

rows = []
for p in sorted(UP.rglob("*")):
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        arr = np.asarray(Image.open(p).convert("RGB"))
        rows.append({"up_split": p.parent.parent.name, "up_label": p.parent.name.lower(), "id": p.stem,
                     "md5_pixel": hashlib.md5(str(arr.shape).encode() + arr.tobytes()).hexdigest()})
up = pd.DataFrame(rows)
up.to_csv(OUT / "hashes_upstream9.csv", index=False)
print("upstream images:", len(up))
print(pd.crosstab(up.up_label, up.up_split).to_string())

# 1. is the AK/Nevus double-filing already in the upstream?
multi = up.groupby("id").up_label.nunique()
print("\nupstream IDs filed under >1 class:", (multi > 1).sum())
print(up[up.id.isin(multi[multi > 1].index)].groupby("up_label").size().to_string())

# 2. map every skin31 ISIC image to its upstream copy by pixel hash
s = pd.read_csv(OUT / "hashes_skin31.csv")
s["id"] = s.path.map(lambda p: Path(p).stem)
s = s[s.id.str.startswith("ISIC_")]
s["label"] = s.label.str.lower()
m = s.merge(up, on="md5_pixel", how="left")
print("\nskin31 ISIC images:", len(s), "| found in upstream by pixel hash:", s.md5_pixel.isin(up.md5_pixel).sum())
print("skin31 folder vs upstream folder (same pixels):")
print(pd.crosstab(m.label, m.up_label.fillna("NOT FOUND")).to_string())

# 3. upstream labels vs ISIC Archive
audit = OUT / "isic_archive_all.csv"
if audit.exists():
    a = pd.read_csv(audit)
    ua = up.merge(a, on="id", how="left")
    print("\nupstream folder vs ISIC Archive diagnosis_3:")
    print(pd.crosstab(ua.diagnosis_3.fillna("not looked up"), ua.up_label).to_string())
