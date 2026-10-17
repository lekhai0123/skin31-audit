"""Image type of every ISIC-derived benchmark image from the ISIC Archive API (metadata.acquisition.image_type),
cached in outputs/e01/isic_image_type.csv and resumable. Same function as the `isic` step of the authors' working
script e01_modality_census.py, which was run on 17 October 2026 (13:15-13:35, UTC+7).
Usage: python scripts/isic_image_type.py
"""
import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "e01"


def isic():
    OUT.mkdir(parents=True, exist_ok=True)
    ids = sorted(set(pd.read_csv(ROOT / "outputs" / "dup_check" / "skin31_isic_label_audit.csv").id))
    cache_path = OUT / "isic_image_type.csv"
    cache = pd.read_csv(cache_path).set_index("id").to_dict("index") if cache_path.exists() else {}
    todo = [i for i in ids if i not in cache or pd.isna(cache[i].get("image_type"))]
    for n, i in enumerate(todo):
        rec = {"image_type": None, "dermoscopic_type": None}
        for attempt in range(3):
            try:
                j = json.load(urllib.request.urlopen(f"https://api.isic-archive.com/api/v2/images/{i}/", timeout=30))
                acq = j.get("metadata", {}).get("acquisition", {})
                rec = {"image_type": acq.get("image_type"), "dermoscopic_type": acq.get("dermoscopic_type")}
                break
            except Exception:
                time.sleep(2)
        cache[i] = rec
        time.sleep(0.15)
        if n % 200 == 199:
            pd.DataFrame.from_dict(cache, orient="index").rename_axis("id").to_csv(cache_path)
            print(f"{n + 1}/{len(todo)}", flush=True)
    pd.DataFrame.from_dict(cache, orient="index").rename_axis("id").to_csv(cache_path)
    t = pd.read_csv(cache_path)
    print("unique ISIC ids:", len(ids), "| image_type counts:", t.image_type.value_counts(dropna=False).to_dict())


if __name__ == "__main__":
    isic()
