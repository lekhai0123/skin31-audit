"""Find openly licensed (CC-0 / CC-BY) ISIC Archive images for the graphical abstract.
A: clinical close-ups of different diagnoses (stand-ins for the clinical atlas classes).
B: one lesion with both a clinical close-up and a dermoscopic image (same lesion_id).
Writes docs/ga_images/candidates/*.jpg, candidates.csv and contact sheets."""
import json
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "ga_images" / "candidates"
API = "https://api.isic-archive.com/api/v2/images/search/"
LIC = "(copyright_license:CC-0 OR copyright_license:CC-BY)"


def search(q, limit=100):
    url = API + "?" + urllib.parse.urlencode({"query": q, "limit": limit})
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)["results"]


def row(r, block):
    a, c = r["metadata"]["acquisition"], r["metadata"]["clinical"]
    return {"block": block, "isic_id": r["isic_id"], "license": r["copyright_license"], "attribution": r["attribution"],
            "image_type": a.get("image_type"), "diagnosis": c.get("diagnosis_3") or c.get("diagnosis_2") or c.get("diagnosis_1"),
            "confirm": c.get("diagnosis_confirm_type"), "lesion_id": c.get("lesion_id"),
            "thumb": r["files"]["thumbnail_256"]["url"], "full": r["files"]["full"]["url"]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    # A: clinical close-ups, several diagnoses
    for dx in ["Basal cell carcinoma", "Squamous cell carcinoma, NOS", "Solar or actinic keratosis",
               "Seborrheic keratosis", "Dermatofibroma", "Melanoma, NOS", "Hemangioma"]:
        try:
            res = search(f'image_type:"clinical: close-up" AND {LIC} AND diagnosis_3:"{dx}"', 8)
        except Exception as e:  # noqa: BLE001
            print("A", dx, e); continue
        rows += [row(r, "A") for r in res[:6]]
        print("A", dx, len(res))
    # B: clinical close-ups whose lesion also has a licensed dermoscopic image
    pairs = 0
    for dx in ["Nevus", "Melanoma, NOS", "Basal cell carcinoma"]:
        res = search(f'image_type:"clinical: close-up" AND {LIC} AND diagnosis_3:"{dx}" AND diagnosis_confirm_type:histopathology', 100)
        for r in res:
            lid = r["metadata"]["clinical"].get("lesion_id")
            if not lid:
                continue
            d = search(f'lesion_id:{lid} AND image_type:dermoscopic AND {LIC}', 5)
            if d:
                rows += [row(r, f"B{pairs}"), row(d[0], f"B{pairs}")]
                pairs += 1
            if pairs >= 4 * (["Nevus", "Melanoma, NOS", "Basal cell carcinoma"].index(dx) + 1):
                break
        print("B", dx, pairs)
    df = pd.DataFrame(rows).drop_duplicates("isic_id")
    for t in df.itertuples():
        f = OUT / f"{t.block}_{t.isic_id}.jpg"
        if not f.exists():
            urllib.request.urlretrieve(t.thumb, f)
    df.to_csv(OUT / "candidates.csv", index=False)

    def sheet(sub, name, ncol):
        w = 256; n = len(sub); nr = (n + ncol - 1) // ncol
        im = Image.new("RGB", (ncol * w, nr * (w + 30)), "white"); dr = ImageDraw.Draw(im)
        for k, t in enumerate(sub.itertuples()):
            x, y = (k % ncol) * w, (k // ncol) * (w + 30)
            im.paste(Image.open(OUT / f"{t.block}_{t.isic_id}.jpg").convert("RGB").resize((w, w)), (x, y))
            dr.text((x + 4, y + w + 2), f"{k}:{t.isic_id} {t.license}", fill="black")
            dr.text((x + 4, y + w + 15), f"{str(t.diagnosis)[:30]}", fill="black")
        im.save(OUT.parent / name)
    sheet(df[df.block == "A"], "sheet_A.jpg", 7)
    sheet(df[df.block != "A"], "sheet_B.jpg", 6)
    print(df.groupby("block").size())


if __name__ == "__main__":
    main()
