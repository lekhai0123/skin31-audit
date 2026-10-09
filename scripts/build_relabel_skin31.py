"""Build the RELABEL variant of the corrected skin31 dataset (exploratory, added 04/10/2026 after review).

Identical to build_clean_skin31.py except that the images in the "Seborrheic keratosis" folder, which the ISIC Archive
diagnoses as (histopathology-confirmed) melanoma, are moved to the Melanoma class instead of being dropped.
The test set is the same strict clean test set (863 images), so results are directly comparable with skin31_clean.

Original description of the corrected dataset:

train: original train with
  - Nevus-folder images that the ISIC Archive diagnoses as actinic keratosis moved to Actinic keratosis
    (dropped when an identical copy is already in Actinic keratosis);
  - other label-disputed ISIC images dropped (the whole "Seborrheic keratosis" class = dermoscopic melanoma,
    and 1 Nevus-folder image without an ISIC diagnosis);
  - exact duplicates (same decoded pixels, same label) kept once.
test: the strict clean test set (outputs/repro/skin31_test_flags.csv, 863 images): no label-disputed images,
  no exact or lesion-level overlap with train, no Atlas same-lesion pairs.
Class folder names are taken from the original train folders, so train and test share one class order (30 classes).
"""
import hashlib
import shutil
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "Atlas dan ISIC2019 (31 classes)"
DST = ROOT / "data" / "skin31_relabel"
AUDIT = ROOT / "outputs" / "dup_check"
DROP_CLASS = "Seborrheic keratosis"

train_names = sorted(p.name for p in (SRC / "train").iterdir())
canon = {n.lower(): n for n in train_names}  # test "Actinic Keratosis" -> train "Actinic keratosis"

h = pd.read_csv(AUDIT / "hashes_skin31.csv")
h["id"] = h.path.map(lambda p: Path(p).stem)
h["folder"] = h.label.str.lower()
audit = pd.read_csv(AUDIT / "skin31_isic_label_audit.csv")[["id", "split", "folder", "diagnosis_3", "agrees"]]
h = h.merge(audit, on=["id", "split", "folder"], how="left")

rows = []
tr = h[h.split == "train"].copy()
for r in tr.itertuples():
    new, reason = canon[r.folder], "kept"
    if r.agrees is False or r.agrees == False:  # noqa: E712 (NaN for Atlas images)
        if r.folder == "nevus" and r.diagnosis_3 == "Solar or actinic keratosis":
            new, reason = canon["actinic keratosis"], "relabel Nevus->AK (ISIC Archive)"
        elif r.folder == "seborrheic keratosis" and str(r.diagnosis_3).startswith("Melanoma"):
            new, reason = canon["melanoma"], "relabel SK->Melanoma (ISIC Archive)"
        else:
            new, reason = None, "drop: label disputed"
    if new == DROP_CLASS:
        new, reason = None, "drop: class removed"
    rows.append({"split": "train", "src": r.path, "md5_pixel": r.md5_pixel, "label": new, "reason": reason})
t = pd.DataFrame(rows)
dup = t.label.notna() & t.duplicated(["md5_pixel", "label"], keep="first")
t.loc[dup, ["label", "reason"]] = [None, "drop: exact duplicate within train"]

flags = pd.read_csv(ROOT / "outputs" / "repro" / "skin31_test_flags.csv")
te = pd.DataFrame({"split": "test", "src": flags.path, "md5_pixel": None,
                   "label": [canon[Path(p).parent.name.lower()] if s else None for p, s in zip(flags.path, flags.strict)],
                   "reason": ["kept (strict clean test)" if s else "drop: not in strict clean test" for s in flags.strict]})
man = pd.concat([t, te], ignore_index=True)

classes = sorted(c for c in train_names if c != DROP_CLASS)
for split in ("train", "test"):
    present = set(man[(man.split == split) & man.label.notna()].label)
    missing = set(classes) - present
    assert not missing, f"{split} lacks classes {missing}"

if DST.exists():
    shutil.rmtree(DST)
for r in man[man.label.notna()].itertuples():
    out = DST / r.split / r.label / Path(r.src).name
    out.parent.mkdir(parents=True, exist_ok=True)
    assert not out.exists(), out
    shutil.copy2(r.src, out)
    man.loc[r.Index, "dst"] = str(out.relative_to(DST)).replace("\\", "/")
man["src"] = man.src.map(lambda p: str(Path(p).relative_to(SRC)).replace("\\", "/"))
man.drop(columns="md5_pixel").to_csv(DST / "manifest.csv", index=False)

print(man.groupby(["split", "reason"]).size().to_string())
print("classes:", len(classes))
print(man[man.label.notna()].groupby("split").size().to_string())
zpath = ROOT / "data" / "skin31_relabel.zip"
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:
    for p in sorted(DST.rglob("*")):
        if p.is_file():
            z.write(p, p.relative_to(DST.parent))
print("zip:", zpath, round(zpath.stat().st_size / 1e6, 1), "MB",
      "sha256", hashlib.sha256(zpath.read_bytes()).hexdigest())
