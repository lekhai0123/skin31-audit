"""Build skin31_deconf (prereg/deconfound_analysis_plan.md): skin31_clean + dermoscopic BCC and melanoma
from HAM10000 added to the Basal Cell Carcinoma and Melanoma training folders.

Step 1 (--screen): eligibility and near-duplicate candidates for visual review.
Step 2 (--build):  sampling, dataset copy, manifest and zip. Reads prereg/deconf_excluded_ham.csv (may be empty).
"""
import argparse
import hashlib
import shutil
import zipfile
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
HAM = next((ROOT / "data" / "kaggle_cache").rglob("HAM10000_metadata.csv")).parent
CLEAN = ROOT / "data" / "skin31_clean"
DST = ROOT / "data" / "skin31_deconf"
OUT = ROOT / "outputs" / "deconf"
AUDIT = ROOT / "outputs" / "dup_check"
TARGET = {"bcc": ("Basal Cell Carcinoma", 334), "mel": ("Melanoma", 100)}
SEED = 20260930
MAX_HAMMING = 8


def ham_paths():
    files = {}
    for p in HAM.rglob("*.jpg"):
        files.setdefault(p.stem, p)
    return files


def eligible():
    meta = pd.read_csv(HAM / "HAM10000_metadata.csv")
    meta = meta[meta.dx.isin(TARGET)].copy()
    s31 = pd.read_csv(AUDIT / "hashes_skin31.csv")
    s31_ids = set(s31.path.map(lambda p: Path(p).stem))
    s31_les = set(pd.read_csv(AUDIT / "skin31_isic_lesions.csv").lesion_id.dropna())
    n0 = meta.groupby("dx").lesion_id.nunique().to_dict()
    meta = meta[~meta.image_id.isin(s31_ids) & ~meta.lesion_id.isin(s31_les)]
    # any lesion with an image already in skin31 is excluded entirely
    bad_les = set(pd.read_csv(HAM / "HAM10000_metadata.csv").query("image_id in @s31_ids").lesion_id)
    meta = meta[~meta.lesion_id.isin(bad_les)]
    one = meta.sort_values("image_id").groupby("lesion_id", as_index=False).first()
    print("lesions before / after skin31 exclusion:", n0, one.groupby("dx").size().to_dict())
    return one


def screen():
    OUT.mkdir(parents=True, exist_ok=True)
    one = eligible()
    files = ham_paths()
    one["path"] = one.image_id.map(lambda i: str(files[i]))
    one["phash"] = [int(str(imagehash.phash(Image.open(p).convert("RGB"))), 16) for p in one.path]
    # pixel MD5 against skin31, as a second check
    s31 = pd.read_csv(AUDIT / "hashes_skin31.csv")
    one["md5_pixel"] = [hashlib.md5(str(a.shape).encode() + a.tobytes()).hexdigest()
                        for a in (np.asarray(Image.open(p).convert("RGB")) for p in one.path)]
    one = one[~one.md5_pixel.isin(set(s31.md5_pixel))]
    one.to_csv(OUT / "ham_eligible.csv", index=False)

    refs = []
    d7 = pd.read_csv(AUDIT / "hashes_derm7pt.csv", dtype={"phash": "uint64"}).assign(ref="derm7pt")
    pad = pd.read_csv(AUDIT / "hashes_pad.csv", dtype={"phash": "uint64"}).assign(ref="pad")
    flags = pd.read_csv(ROOT / "outputs" / "repro" / "skin31_test_flags.csv")
    st = s31[s31.path.isin(set(flags[flags.strict].path))].assign(ref="skin31_test")
    st["phash"] = st.phash.astype("uint64")
    ref = pd.concat([d7, pad, st])[["path", "label", "ref", "phash"]].reset_index(drop=True)
    rp = ref.phash.to_numpy(np.uint64)
    for r in one.itertuples():
        x = rp ^ np.uint64(r.phash)
        d = np.array([bin(int(v)).count("1") for v in x])
        for j in np.flatnonzero(d <= MAX_HAMMING):
            refs.append({"image_id": r.image_id, "dx": r.dx, "ham_path": r.path, "hamming": int(d[j]),
                         "ref": ref.ref[j], "ref_path": ref.path[j], "ref_label": ref.label[j]})
    c = pd.DataFrame(refs, columns=["image_id", "dx", "ham_path", "hamming", "ref", "ref_path", "ref_label"])
    c.to_csv(OUT / "ham_candidates.csv", index=False)
    print("eligible:", one.groupby("dx").size().to_dict(), "| near-duplicate candidates:", len(c),
          c.groupby("ref").size().to_dict())


def build():
    one = pd.read_csv(OUT / "ham_eligible.csv")
    exc_path = ROOT / "prereg" / "deconf_excluded_ham.csv"
    excluded = set(pd.read_csv(exc_path).image_id) if exc_path.exists() else set()
    one = one[~one.image_id.isin(excluded)]
    rng = np.random.default_rng(SEED)
    picks = []
    for dx, (cls, n) in TARGET.items():
        pool = one[one.dx == dx].sort_values("image_id").reset_index(drop=True)
        k = min(n, len(pool))
        picks.append(pool.iloc[np.sort(rng.choice(len(pool), k, replace=False))].assign(target_class=cls))
        print(dx, "eligible", len(pool), "-> added", k)
    add = pd.concat(picks)
    if DST.exists():
        shutil.rmtree(DST)
    shutil.copytree(CLEAN, DST)
    for r in add.itertuples():
        shutil.copy2(r.path, DST / "train" / r.target_class / f"HAM_{r.image_id}.jpg")
    add[["image_id", "lesion_id", "dx", "dx_type", "target_class"]].to_csv(DST / "ham_added.csv", index=False)
    counts = {c: len(list((DST / "train" / c).iterdir())) for c in ("Basal Cell Carcinoma", "Melanoma")}
    print("train counts after:", counts, "| total train", sum(1 for _ in (DST / "train").rglob("*.jpg")))
    zpath = ROOT / "data" / "skin31_deconf.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:
        for p in sorted(DST.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(DST.parent))
    print("zip", round(zpath.stat().st_size / 1e6, 1), "MB sha256", hashlib.sha256(zpath.read_bytes()).hexdigest())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", action="store_true")
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args()
    if a.screen:
        screen()
    if a.build:
        build()
