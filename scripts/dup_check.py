"""Exact and near-duplicate check between external datasets and skin31 (plan section 3.5, task 0.15).

Exact:  MD5 of the file bytes, and MD5 of the decoded RGB array at native resolution
        (shape + pixel bytes), so identical content in a different container is caught.
Near:   64-bit pHash (imagehash.phash, hash_size=8), candidate pairs at Hamming <= 8.
        Every candidate pair must be reviewed visually (review.html).

Usage:
    python scripts/dup_check.py                     # skin31 vs Derm7pt
    python scripts/dup_check.py --pad path/to/PAD   # also PAD-UFES-20 (images/ folder)
"""
import argparse
import hashlib
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SKIN31 = ROOT / "data" / "Atlas dan ISIC2019 (31 classes)"
DERM7PT = ROOT / "data" / "release_v0"
OUT = ROOT / "outputs" / "dup_check"
MAX_HAMMING = 8
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def hash_image(path):
    raw = path.read_bytes()
    with Image.open(path) as im:
        rgb = im.convert("RGB")
        arr = np.asarray(rgb)
        ph = imagehash.phash(rgb, hash_size=8)
    pix = hashlib.md5(str(arr.shape).encode() + arr.tobytes()).hexdigest()
    return {
        "md5_file": hashlib.md5(raw).hexdigest(),
        "md5_pixel": pix,
        "phash": int(str(ph), 16),
        "height": arr.shape[0],
        "width": arr.shape[1],
    }


def hash_set(name, rows):
    """rows: list of dicts with at least 'path'. Cached to OUT/hashes_<name>.csv."""
    cache = OUT / f"hashes_{name}.csv"
    if cache.exists():
        df = pd.read_csv(cache, dtype={"phash": "uint64"})
        if len(df) == len(rows):
            return df
    out = []
    for i, r in enumerate(rows):
        out.append({**r, **hash_image(Path(r["path"]))})
        if (i + 1) % 500 == 0:
            print(f"  {name}: {i + 1}/{len(rows)}")
    df = pd.DataFrame(out)
    df["phash"] = df["phash"].astype("uint64")
    df.to_csv(cache, index=False)
    return df


def skin31_rows():
    rows = []
    for split in ("train", "test"):
        for cls_dir in sorted((SKIN31 / split).iterdir()):
            for p in sorted(cls_dir.iterdir()):
                if p.suffix.lower() in IMG_EXT:
                    rows.append({"path": str(p), "split": split, "label": cls_dir.name})
    return rows


def derm7pt_rows():
    meta = pd.read_csv(DERM7PT / "meta" / "meta.csv")
    rows = []
    for _, m in meta.iterrows():
        for mod in ("clinic", "derm"):
            rows.append({"path": str(DERM7PT / "images" / m[mod]), "split": mod,
                         "label": m["diagnosis"], "case_num": m["case_num"]})
    # 9 cases reuse the dermoscopic file as "clinic"; hash each file once
    return list({r["path"].lower(): r for r in rows}.values())


def pad_rows(pad_dir):
    pad_dir = Path(pad_dir).resolve()
    meta = pd.read_csv(pad_dir / "metadata.csv")
    files = {p.name: p for p in pad_dir.rglob("*") if p.suffix.lower() in IMG_EXT}
    return [{"path": str(files[m["img_id"]]), "split": "pad", "label": m["diagnostic"],
             "patient_id": m["patient_id"]} for _, m in meta.iterrows()]


def popcount64(x):
    x = x.copy()
    c = np.zeros(x.shape, dtype=np.uint8)
    for _ in range(8):
        c += np.unpackbits((x & np.uint64(0xFF)).astype(np.uint8)[..., None], axis=-1).sum(-1).astype(np.uint8)
        x >>= np.uint64(8)
    return c


def compare(src, tgt, tgt_name):
    pairs = []
    for key in ("md5_file", "md5_pixel"):
        m = tgt.merge(src, on=key, suffixes=("_tgt", "_src"))
        for _, r in m.iterrows():
            pairs.append({"type": key, "hamming": 0, "tgt_path": r["path_tgt"], "tgt_split": r["split_tgt"],
                          "tgt_label": r["label_tgt"], "src_path": r["path_src"],
                          "src_split": r["split_src"], "src_label": r["label_src"]})
    s = src["phash"].to_numpy(np.uint64)
    for i, t in enumerate(tgt["phash"].to_numpy(np.uint64)):
        d = popcount64(s ^ t)
        for j in np.flatnonzero(d <= MAX_HAMMING):
            pairs.append({"type": "phash", "hamming": int(d[j]),
                          "tgt_path": tgt["path"].iat[i], "tgt_split": tgt["split"].iat[i],
                          "tgt_label": tgt["label"].iat[i], "src_path": src["path"].iat[j],
                          "src_split": src["split"].iat[j], "src_label": src["label"].iat[j]})
    df = pd.DataFrame(pairs, columns=["type", "hamming", "tgt_path", "tgt_split", "tgt_label",
                                      "src_path", "src_split", "src_label"])
    df = df.sort_values(["hamming", "type"]).drop_duplicates(["tgt_path", "src_path", "type"])
    df["confirmed_duplicate"] = ""  # filled in by hand after visual review
    df.to_csv(OUT / f"candidates_{tgt_name}.csv", index=False)
    write_review(df, tgt_name)
    return df


def write_review(df, tgt_name):
    rows = []
    for k, r in enumerate(df.drop_duplicates(["tgt_path", "src_path"]).itertuples()):
        rows.append(
            f"<tr><td>{k}</td><td>{r.type}<br>d={r.hamming}</td>"
            f"<td><img src='{Path(r.tgt_path).as_uri()}'><br>{r.tgt_split} / {r.tgt_label}<br><small>{Path(r.tgt_path).name}</small></td>"
            f"<td><img src='{Path(r.src_path).as_uri()}'><br>{r.src_split} / {r.src_label}<br><small>{Path(r.src_path).name}</small></td></tr>")
    html = ("<html><meta charset='utf-8'><style>img{max-width:360px;max-height:280px}"
            "td{vertical-align:top;padding:6px;border-bottom:1px solid #ccc;font-family:sans-serif}</style>"
            f"<h3>{tgt_name} vs skin31: {len(rows)} candidate pairs (Hamming &le; {MAX_HAMMING})</h3>"
            "<table><tr><th>#</th><th>match</th><th>external</th><th>skin31</th></tr>"
            + "".join(rows) + "</table></html>")
    (OUT / f"review_{tgt_name}.html").write_text(html, encoding="utf-8")


def within_set(df, name):
    """Exact duplicates inside one set (reported, not removed)."""
    rep = []
    for key in ("md5_file", "md5_pixel"):
        g = df[df.duplicated(key, keep=False)].groupby(key)["path"].apply(list)
        rep.append(f"  {name} {key}: {len(g)} groups with identical content")
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pad", help="PAD-UFES-20 folder containing metadata.csv and images")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    src = hash_set("skin31", skin31_rows())
    targets = {"derm7pt": hash_set("derm7pt", derm7pt_rows())}
    if args.pad:
        targets["pad"] = hash_set("pad", pad_rows(args.pad))

    report = [f"skin31 images: {len(src)}"] + within_set(src, "skin31")
    for name, tgt in targets.items():
        df = compare(src, tgt, name)
        report.append(f"\n{name}: {len(tgt)} images")
        report += within_set(tgt, name)
        for t in ("md5_file", "md5_pixel", "phash"):
            report.append(f"  {t}: {(df['type'] == t).sum()} pairs")
        if len(df):
            report.append("  pHash distance histogram: "
                          + str(df[df.type == 'phash'].hamming.value_counts().sort_index().to_dict()))
            report.append("  pairs by external split x skin31 split:\n"
                          + df.groupby(["tgt_split", "src_split"]).size().to_string())
    text = "\n".join(report)
    (OUT / "report.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
