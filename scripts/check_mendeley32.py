"""Does the Mendeley Data '32 Curated Categories of Skin Disease Images' (2026) carry the label errors of the
31-class dataset? Match every image to skin31 by decoded-pixel MD5, else by pHash (Hamming <= 2) with identical size."""
import hashlib
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "mendeley32" / "extracted"
OUT = ROOT / "outputs" / "mendeley32"
OUT.mkdir(parents=True, exist_ok=True)

rows = []
for p in sorted(SRC.rglob("*")):
    if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
        continue
    im = Image.open(p).convert("RGB")
    a = np.asarray(im)
    rows.append({"path": str(p), "folder": p.parent.name, "split": p.parent.parent.name,
                 "md5_pixel": hashlib.md5(str(a.shape).encode() + a.tobytes()).hexdigest(),
                 "phash": int(str(imagehash.phash(im)), 16), "w": a.shape[1], "h": a.shape[0]})
m = pd.DataFrame(rows)
m["phash"] = m.phash.astype("uint64")
print("images:", len(m), "| folders:", m.folder.nunique())
print(m.groupby("folder").size().to_string())

s = pd.read_csv(ROOT / "outputs" / "dup_check" / "hashes_skin31.csv", dtype={"phash": "uint64"})
s["id"] = s.path.map(lambda x: Path(x).stem)
s["s31_label"] = s.label
s["s31_split"] = s.split
exact = m.merge(s[["md5_pixel", "id", "s31_label", "s31_split"]], on="md5_pixel", how="left")
matched = exact.dropna(subset=["id"])
unmatched = m[~m.path.isin(matched.path)]
sp = s.phash.to_numpy(np.uint64)
near = []
for r in unmatched.itertuples():
    d = np.array([bin(int(v)).count("1") for v in (sp ^ np.uint64(r.phash))])
    j = np.flatnonzero(d <= 2)
    if len(j):
        near.append({"path": r.path, "folder": r.folder, "id": s.id.iat[j[0]], "s31_label": s.s31_label.iat[j[0]], "hamming": int(d[j[0]])})
near = pd.DataFrame(near)
print("matched to skin31: exact", matched.path.nunique(), "| near (pHash<=2)", len(near), "| total images", len(m))

audit = pd.read_csv(ROOT / "outputs" / "dup_check" / "skin31_isic_label_audit.csv").drop_duplicates("id")
bad = audit[~audit.agrees.astype(bool)][["id", "folder", "diagnosis_3"]].rename(columns={"folder": "s31_folder"})
allm = pd.concat([matched[["path", "folder", "id"]], near[["path", "folder", "id"]] if len(near) else None])
hit = allm.merge(bad, on="id")
print("\nimages in the 32-class dataset that are disputed in skin31:", hit.id.nunique())
print(hit.groupby(["folder", "diagnosis_3"]).size().to_string())
conf = allm[allm.id.str.startswith("ISIC_")].groupby("id").folder.nunique()
print("ISIC IDs present under >1 folder in the 32-class dataset:", int((conf > 1).sum()))
allm.to_csv(OUT / "matches.csv", index=False)
hit.to_csv(OUT / "disputed_present.csv", index=False)
