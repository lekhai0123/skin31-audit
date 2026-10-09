"""Exact unit charts (one square = one image) for Figure 1b, as SVG for import into PowerPoint.
Physical size: 1 SVG unit = 1 mm; pitch 2.0 mm, square 1.7 mm, 20 columns, filled row by row from the top left.
Counts come from outputs/dup_check/skin31_isic_label_audit.csv (nevus folder: 294 agree, 78 actinic keratosis, 1 no
recorded diagnosis; seborrheic keratosis folder: 80 melanomas)."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "figure1_assets"
OUT.mkdir(parents=True, exist_ok=True)
AGREE, AK, MEL, NODX_STROKE = "#c9c8c1", "#eb6834", "#d03b3b", "#9a988f"
COLS, PITCH, SQ = 20, 2.0, 1.7

a = pd.read_csv(ROOT / "outputs/dup_check/skin31_isic_label_audit.csv")


def counts(folder):
    d = a[a.folder == folder]
    ak = int(((~d.agrees) & d.diagnosis_3.str.contains("actinic", case=False, na=False)).sum())
    mel = int(((~d.agrees) & d.diagnosis_3.str.contains("melanoma", case=False, na=False)).sum())
    nodx = int(((~d.agrees) & d.diagnosis_3.isna()).sum())
    return int(d.agrees.sum()), ak, mel, nodx


def waffle(name, seq):
    """seq: list of (fill, n, id); squares are placed row by row"""
    n = sum(k for _, k, _ in seq)
    rows = -(-n // COLS)
    w, h = COLS * PITCH, rows * PITCH
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}mm" height="{h}mm" viewBox="0 0 {w} {h}">']
    i = 0
    for fill, k, gid in seq:
        parts.append(f'<g id="{gid}">')
        for _ in range(k):
            r, c = divmod(i, COLS)
            x, y = c * PITCH + (PITCH - SQ) / 2, r * PITCH + (PITCH - SQ) / 2
            if fill is None:  # no recorded diagnosis: white square, grey outline
                parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{SQ}" height="{SQ}" rx="0.2" fill="#ffffff" stroke="{NODX_STROKE}" stroke-width="0.2"/>')
            else:
                parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{SQ}" height="{SQ}" rx="0.2" fill="{fill}"/>')
            i += 1
        parts.append("</g>")
    parts.append("</svg>")
    (OUT / name).write_text("\n".join(parts), encoding="utf-8")
    return n, rows, w, h


ag, ak, _, nodx = counts("nevus")
_, _, mel, _ = counts("seborrheic keratosis")
assert (ag, ak, nodx, mel) == (294, 78, 1, 80), (ag, ak, nodx, mel)
print("nevus", waffle("fig1b_waffle_nevus.svg", [(AGREE, ag, "agrees_with_ISIC_Archive"), (AK, ak, "actinic_keratosis"), (None, nodx, "no_recorded_diagnosis")]))
print("SK", waffle("fig1b_waffle_SK.svg", [(MEL, mel, "melanoma")]))
