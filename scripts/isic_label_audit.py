"""Look up every ISIC image in skin31 on the ISIC Archive API and compare with its folder label."""
import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "dup_check"
EXPECTED = {  # skin31 folder -> ISIC diagnosis_3 values that agree with it
    "actinic keratosis": {"Solar or actinic keratosis"},
    "dermatofibroma": {"Dermatofibroma"},
    "nevus": {"Nevus"},
    "pigmented benign keratosis": {"Pigmented benign keratosis"},
    "seborrheic keratosis": {"Seborrheic keratosis"},
    "squamous cell carcinoma": {"Squamous cell carcinoma, NOS", "Squamous cell carcinoma in situ"},
}

d = pd.read_csv(OUT / "skin31_isic_vs_gt.csv")
cache_path = OUT / "isic_archive_all.csv"
cache = pd.read_csv(cache_path).set_index("id").to_dict("index") if cache_path.exists() else {}
for i in sorted(set(d.id) - set(cache)):
    for attempt in range(3):
        try:
            j = json.load(urllib.request.urlopen(f"https://api.isic-archive.com/api/v2/images/{i}/", timeout=30))
            c = j["metadata"].get("clinical", {})
            cache[i] = {k: c.get(k) for k in ("diagnosis_1", "diagnosis_2", "diagnosis_3", "diagnosis_confirm_type")}
            break
        except Exception as e:
            cache[i] = {"diagnosis_3": f"ERR {e}"[:60]}
            time.sleep(2)
    time.sleep(0.15)
    if len(cache) % 200 == 0:
        pd.DataFrame.from_dict(cache, orient="index").rename_axis("id").to_csv(cache_path)
arch = pd.DataFrame.from_dict(cache, orient="index").rename_axis("id").reset_index()
arch.to_csv(cache_path, index=False)

m = d.merge(arch, on="id", how="left")
m["folder"] = m.label.str.lower()
m["agrees"] = [str(dx) in EXPECTED.get(f, {str(dx)}) for f, dx in zip(m.folder, m.diagnosis_3)]
m.to_csv(OUT / "skin31_isic_label_audit.csv", index=False)
print(pd.crosstab(m.folder, m.diagnosis_3.fillna("none")).T.to_string())
print("\nimages whose folder disagrees with ISIC Archive diagnosis:")
print(m[~m.agrees].groupby(["split", "folder"]).size().to_string())
print("total disagreeing:", (~m.agrees).sum(), "of", len(m))
