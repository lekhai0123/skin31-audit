"""Source- and lesion-matched deconfounding control (MILK10k).

Arm B (clinical control):  skin31_clean + clinical close-up images of a lesion set S (BCC 327, melanoma 100).
Arm C (dermoscopic arm):   skin31_clean + dermoscopic images of the SAME lesion set S.

Eligibility (metadata only, fixed before any model is trained):
  - diagnosis BCC or MEL (MILK10k ground truth); diagnosis confirmed by histopathology;
  - melanoma metastasis excluded; no image of the lesion marked image_manipulation == "altered".
Selection: eligible lesions are put in a random order (seed 20260930); they are screened in that order and the
first N per diagnosis without a confirmed duplicate are taken. Duplicate screening: both images of the lesion vs the
31-class dataset (all), HAM10000 (all), Derm7pt (both modalities), PAD-UFES-20; exact MD5 of decoded pixels, and pHash
Hamming <= 8 candidates reviewed visually (decisions in prereg/milk_excluded_lesions.csv).

  --screen : order lesions, hash a buffer of candidates, write candidate pairs and contact sheets
  --build  : apply exclusions, select S, build data/skin31_milkB and data/skin31_milkC, zip both
"""
import argparse
import hashlib
import shutil
import zipfile
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
MILK = ROOT / "data" / "milk10k"
CLEAN = ROOT / "data" / "skin31_clean"
OUT = ROOT / "outputs" / "milk"
AUDIT = ROOT / "outputs" / "dup_check"
TARGET = {"BCC": ("Basal Cell Carcinoma", 327), "MEL": ("Melanoma", 100)}
SEED, MAX_HAM, COS_EXCLUDE = 20260930, 8, 0.80
BUFFER = {"BCC": 1.3, "MEL": 2.0}  # screened lesions per needed lesion


def eligible():
    m = pd.read_csv(MILK / "MILK10k_Training_Metadata.csv")
    g = pd.read_csv(MILK / "MILK10k_Training_GroundTruth.csv")
    s = pd.read_csv(MILK / "MILK10k_Training_Supplement.csv")
    g["dx"] = g.drop(columns="lesion_id").idxmax(axis=1)
    m = m.merge(g[["lesion_id", "dx"]], on="lesion_id").merge(s, on="isic_id", how="left")
    m["path"] = m.apply(lambda r: str(MILK / "MILK10k_Training_Input" / r.lesion_id / f"{r.isic_id}.jpg"), axis=1)
    m = m[m.dx.isin(TARGET)]
    bad = m[(m.diagnosis_confirm_type != "histopathology") | (m.image_manipulation == "altered")
            | m.diagnosis_full.str.contains("metastasis", case=False, na=False)].lesion_id
    m = m[~m.lesion_id.isin(set(bad))]
    les = m.groupby("lesion_id").filter(lambda d: set(d.image_type) == {"clinical: close-up", "dermoscopic"})
    return les


def ordered(les):
    rng = np.random.default_rng(SEED)
    out = []
    for dx in TARGET:
        ids = sorted(les[les.dx == dx].lesion_id.unique())
        out.append(pd.DataFrame({"lesion_id": [ids[i] for i in rng.permutation(len(ids))], "dx": dx}))
    o = pd.concat(out)
    o["rank"] = o.groupby("dx").cumcount()
    return o


def hashes(paths):
    rows = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        a = np.asarray(im)
        rows.append({"path": str(p), "md5_pixel": hashlib.md5(str(a.shape).encode() + a.tobytes()).hexdigest(),
                     "phash": int(str(imagehash.phash(im)), 16)})
    return pd.DataFrame(rows)


def reference():
    cache = OUT / "ref_hashes.csv"
    if cache.exists():
        return pd.read_csv(cache, dtype={"phash": "uint64"})
    parts = []
    for name in ("skin31", "derm7pt", "pad"):
        h = pd.read_csv(AUDIT / f"hashes_{name}.csv", dtype={"phash": "uint64"})
        parts.append(h[["path", "md5_pixel", "phash"]].assign(ref=name))
    ham_dir = next((ROOT / "data" / "kaggle_cache").rglob("HAM10000_metadata.csv")).parent
    seen, ham = set(), []
    for p in ham_dir.rglob("*.jpg"):
        if p.stem not in seen:
            seen.add(p.stem)
            ham.append(p)
    parts.append(hashes(ham).assign(ref="ham10000"))
    r = pd.concat(parts, ignore_index=True)
    r["phash"] = r.phash.astype("uint64")
    r.to_csv(cache, index=False)
    return r


def dino_cosine(a_paths, b_paths):
    import timm
    import torch
    m = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=224).cuda().eval()
    tf = timm.data.create_transform(**timm.data.resolve_data_config({"input_size": (3, 224, 224), "crop_pct": 1.0}, model=m))
    paths = sorted(set(a_paths) | set(b_paths))
    E = {}
    with torch.inference_mode():
        for i in range(0, len(paths), 64):
            b = paths[i:i + 64]
            x = torch.stack([tf(Image.open(p).convert("RGB")) for p in b]).cuda()
            for p, e in zip(b, torch.nn.functional.normalize(m(x).float(), dim=1).cpu().numpy()):
                E[p] = e
    return [float(E[a] @ E[b]) for a, b in zip(a_paths, b_paths)]


def screen():
    OUT.mkdir(parents=True, exist_ok=True)
    les = eligible()
    o = ordered(les)
    o.to_csv(OUT / "lesion_order.csv", index=False)
    print("eligible lesions:", o.groupby("dx").size().to_dict())
    take = pd.concat([o[(o.dx == dx) & (o["rank"] < int(n * BUFFER[dx]))] for dx, (_, n) in TARGET.items()])
    imgs = les[les.lesion_id.isin(take.lesion_id)]
    h = hashes(imgs.path).merge(imgs[["path", "lesion_id", "dx", "image_type"]], on="path")
    h["phash"] = h.phash.astype("uint64")
    h.to_csv(OUT / "milk_hashes.csv", index=False)
    ref = reference()
    rp = ref.phash.to_numpy(np.uint64)
    cand = []
    exact = h.merge(ref, on="md5_pixel", suffixes=("", "_ref"))
    for r in exact.itertuples():
        cand.append({"lesion_id": r.lesion_id, "dx": r.dx, "image_type": r.image_type, "milk_path": r.path,
                     "ref": r.ref, "ref_path": r.path_ref, "hamming": 0, "type": "md5_pixel"})
    for r in h.itertuples():
        d = np.array([bin(int(v)).count("1") for v in (rp ^ np.uint64(r.phash))])
        for j in np.flatnonzero(d <= MAX_HAM):
            cand.append({"lesion_id": r.lesion_id, "dx": r.dx, "image_type": r.image_type, "milk_path": r.path,
                         "ref": ref.ref[j], "ref_path": ref.path[j], "hamming": int(d[j]), "type": "phash"})
    c = pd.DataFrame(cand).drop_duplicates(["milk_path", "ref_path"]).sort_values("hamming").reset_index(drop=True)
    c["cosine"] = dino_cosine(c.milk_path, c.ref_path)
    c.to_csv(OUT / "milk_candidates.csv", index=False)
    ex = c[c.cosine >= COS_EXCLUDE].groupby("lesion_id").agg(dx=("dx", "first"), max_cosine=("cosine", "max"),
                                                              refs=("ref", lambda x: "|".join(sorted(set(x))))).reset_index()
    ex["reason"] = f"near-duplicate candidate (pHash<={MAX_HAM} and DINOv2 cosine>={COS_EXCLUDE})"
    ex.to_csv(ROOT / "prereg" / "milk_excluded_lesions.csv", index=False)
    print("excluded lesions:", ex.dx.value_counts().to_dict())
    print("screened images:", len(h), "| candidate pairs:", len(c), c.groupby(["ref", "type"]).size().to_dict() if len(c) else "")
    W, H = 220, 165
    for s in range(0, len(c), 8):
        ch = c.iloc[s:s + 8]
        sh = Image.new("RGB", (2 * W + 10, len(ch) * (H + 20)), "white")
        dr = ImageDraw.Draw(sh)
        for k, r in enumerate(ch.itertuples()):
            y = k * (H + 20)
            for col, p in enumerate([r.milk_path, r.ref_path]):
                im = Image.open(p).convert("RGB")
                im.thumbnail((W, H))
                sh.paste(im, (col * (W + 10), y + 18))
            dr.text((2, y + 2), f"#{s + k} d={r.hamming} {r.dx} {r.image_type[:8]} {r.lesion_id} vs {r.ref}", fill="black")
        sh.save(OUT / f"sheet_{s // 8:02d}.jpg", quality=80)


def build():
    les = eligible()
    o = pd.read_csv(OUT / "lesion_order.csv")
    exc_path = ROOT / "prereg" / "milk_excluded_lesions.csv"
    excluded = set(pd.read_csv(exc_path).lesion_id) if exc_path.exists() else set()
    sel = []
    for dx, (cls, n) in TARGET.items():
        pool = o[(o.dx == dx) & ~o.lesion_id.isin(excluded)].sort_values("rank")
        screened = set(pd.read_csv(OUT / "milk_hashes.csv").lesion_id)
        pool = pool[pool.lesion_id.isin(screened)]
        assert len(pool) >= n, f"not enough screened {dx}: {len(pool)}"
        sel.append(pool.head(n).assign(target_class=cls))
    sel = pd.concat(sel)
    sel.to_csv(ROOT / "prereg" / "milk_selected_lesions.csv", index=False)
    for arm, itype in (("B", "clinical: close-up"), ("C", "dermoscopic")):
        dst = ROOT / "data" / f"skin31_milk{arm}"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(CLEAN, dst)
        imgs = les[les.lesion_id.isin(sel.lesion_id) & (les.image_type == itype)].merge(sel[["lesion_id", "target_class"]], on="lesion_id")
        assert len(imgs) == len(sel)
        for r in imgs.itertuples():
            shutil.copy2(r.path, dst / "train" / r.target_class / f"MILK_{r.isic_id}.jpg")
        imgs[["lesion_id", "isic_id", "dx", "image_type", "target_class"]].to_csv(dst / "milk_added.csv", index=False)
        n_train = sum(1 for _ in (dst / "train").rglob("*.jpg"))
        z = ROOT / "data" / f"skin31_milk{arm}.zip"
        with zipfile.ZipFile(z, "w", zipfile.ZIP_STORED) as zf:
            for p in sorted(dst.rglob("*")):
                if p.is_file():
                    zf.write(p, p.relative_to(dst.parent))
        print(arm, itype, "added", len(imgs), "| train", n_train,
              {c: len(list((dst / "train" / c).iterdir())) for c in ("Basal Cell Carcinoma", "Melanoma")},
              "| zip", round(z.stat().st_size / 1e6), "MB sha256", hashlib.sha256(z.read_bytes()).hexdigest())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", action="store_true")
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args()
    if a.screen:
        screen()
    if a.build:
        build()
