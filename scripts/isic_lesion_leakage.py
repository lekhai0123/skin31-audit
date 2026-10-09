"""Lesion-level train/test leakage in skin31's ISIC part, using lesion_id from the ISIC Archive API."""
import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "dup_check"
cache_path = OUT / "isic_lesion_ids.csv"

d = pd.read_csv(OUT / "skin31_isic_label_audit.csv")
cache = pd.read_csv(cache_path).set_index("id").lesion_id.to_dict() if cache_path.exists() else {}
for n, i in enumerate(sorted(set(d.id) - set(cache))):
    for attempt in range(3):
        try:
            j = json.load(urllib.request.urlopen(f"https://api.isic-archive.com/api/v2/images/{i}/", timeout=30))
            cache[i] = j["metadata"].get("clinical", {}).get("lesion_id")
            break
        except Exception:
            cache[i] = None
            time.sleep(2)
    time.sleep(0.15)
    if n % 200 == 199:
        pd.Series(cache, name="lesion_id").rename_axis("id").to_csv(cache_path)
pd.Series(cache, name="lesion_id").rename_axis("id").to_csv(cache_path)

d["lesion_id"] = d.id.map(cache)
print("images with lesion_id:", d.lesion_id.notna().sum(), "of", len(d))
g = d.dropna(subset=["lesion_id"]).groupby("lesion_id")
multi = g.id.nunique()
print("lesions with >1 distinct image:", (multi > 1).sum())
spl = g.split.agg(lambda s: "+".join(sorted(set(s))))
cross = spl[spl == "test+train"].index
t = d[(d.split == "test") & d.lesion_id.isin(cross)]
# exclude the exact-copy case (same id in both splits), already counted as exact duplicates
same_id = set(d[d.split == "train"].id)
t_other = t[~t.id.isin(same_id)]
print("test images whose lesion also appears in train:", t.id.nunique(),
      "| of which via a different image of the same lesion:", t_other.id.nunique())
print(t_other.groupby("folder").size().to_string())
d.to_csv(OUT / "skin31_isic_lesions.csv", index=False)
