"""Blind review page for the 62 pHash candidate pairs between added HAM10000 images and Derm7pt / PAD-UFES-20 / benchmark
test images (intervention H). The earlier AI screening verdict is not shown. Output: outputs/deconf/review_ham/review.html"""
import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "deconf" / "review_ham"
OUT.mkdir(parents=True, exist_ok=True)
c = pd.read_csv(ROOT / "outputs" / "deconf" / "ham_candidates.csv")
rows = [{"pair": i, "set": r.ref, "ham": int(r.hamming), "tl": f"HAM10000 / {r.dx}", "rl": f"{r.ref} / {r.ref_label}",
         "tf": Path(r.ham_path).name, "rf": Path(r.ref_path).name,
         "timg": os.path.relpath(r.ham_path, OUT).replace("\\", "/").replace(" ", "%20"),
         "rimg": os.path.relpath(r.ref_path, OUT).replace("\\", "/").replace(" ", "%20")} for i, r in enumerate(c.itertuples())]
src = (ROOT / "outputs" / "dup_check" / "review_external" / "review.html").read_text(encoding="utf-8")
head, rest = src.split("const D=", 1)
tail = rest.split(";\n", 1)[1]
html = (head + "const D=" + json.dumps(rows, ensure_ascii=False) + ";\n" + tail)
html = html.replace("96 cặp ứng viên trùng: skin31 với Derm7pt/PAD", "62 cặp ứng viên trùng: ảnh HAM10000 thêm vào với Derm7pt/PAD/test")
html = html.replace("Trái: ảnh external · Phải: ảnh skin31", "Trái: ảnh HAM10000 · Phải: ảnh Derm7pt/PAD/test")
html = html.replace("'ext_review_v1'", "'ham_review_v1'").replace("external_review_rating.csv", "ham_review_rating.csv")
html = html.replace("pair,set,ext_file,skin31_file,rating,rater,date", "pair,ref,ham_file,ref_file,rating,rater,date")
(OUT / "review.html").write_text(html, encoding="utf-8")
missing = [r for r in rows for k in ("timg", "rimg") if not (OUT / r[k].replace("%20", " ")).exists()]
print(len(rows), "pairs ->", OUT / "review.html", "| missing images:", len(missing))
