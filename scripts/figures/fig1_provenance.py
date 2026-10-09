"""Fig. 1 (drawn at the manuscript text width). (a) Redistribution chain; (b) content of the seven ISIC-derived benchmark folders by ISIC Archive diagnosis;
(c) independence of the 305 ISIC-derived test images from the training partition. Data-driven, no patient images."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import AXIS, INK, INK2, MS, MUTED, ROOT, panel, save

AGREE = "#c9c8c1"      # label agrees with ISIC Archive
AK = "#eb6834"         # actinic keratosis filed elsewhere
MEL = "#d03b3b"        # melanoma filed elsewhere
NODX = "#f4f3ef"       # no recorded diagnosis

a = pd.read_csv(ROOT / "outputs/dup_check/skin31_isic_label_audit.csv")
order = ["nevus", "seborrheic keratosis", "actinic keratosis", "pigmented benign keratosis", "squamous cell carcinoma",
         "dermatofibroma", "vascular lesion"]
names = {"nevus": "Nevus", "seborrheic keratosis": "Seborrheic keratosis", "actinic keratosis": "Actinic keratosis",
         "pigmented benign keratosis": "Pigmented benign keratosis", "squamous cell carcinoma": "Squamous cell carcinoma",
         "dermatofibroma": "Dermatofibroma", "vascular lesion": "Vascular lesion"}


def comp(f):
    d = a[a.folder == f]
    ak = ((~d.agrees) & d.diagnosis_3.str.contains("actinic", case=False, na=False)).sum()
    mel = ((~d.agrees) & d.diagnosis_3.str.contains("melanoma", case=False, na=False)).sum()
    nodx = ((~d.agrees) & d.diagnosis_3.isna()).sum()
    return [int(d.agrees.sum()), int(ak), int(mel), int(nodx)]


C = np.array([comp(f) for f in order])
assert C[:, 1:].sum() == 159

fig = plt.figure(figsize=(MS - 0.4, 4.9))
gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], wspace=0.42, top=0.6, bottom=0.095)

# (a) chain ------------------------------------------------------------------
# solid node = examined directly; dashed node = not available (counts as reported in the original publication)
# the 155 images filed under two classes: checked in outputs/dup_check (upstream_compare.py)
ax = fig.add_axes([0.0, 0.7, 1.0, 0.3])
ax.set_xlim(-0.55, 4.55)
ax.set_ylim(-1.55, 1.15)
ax.axis("off")
stages = [("ISIC Archive", "diagnoses,\nlesion IDs", True),
          ("Kaggle compilation\n(9 classes)", "2,357 images", True),
          ("Published benchmark\n(31 classes, 2023)", "4,910 images\nas reported", False),
          ("Kaggle copy\n(31 classes)", "4,909 images", True),
          ("Derivative dataset\n(32 classes, 2026)", "5,034 images", True)]
ax.plot([0, 4], [0, 0], color=AXIS, lw=1.6, zorder=1)
for i, (t, sub, direct) in enumerate(stages):
    ax.scatter(i, 0, s=70, color="white", edgecolor=INK if i else MUTED, lw=1.6, zorder=3,
               linestyle="-" if direct else (0, (1.6, 1.2)))
    ax.text(i, 0.28, t, ha="center", va="bottom", fontsize=7, fontweight="bold", color=INK, linespacing=1.15)
    ax.text(i, -0.3, sub, ha="center", va="top", fontsize=7, color=INK2, linespacing=1.2)
edges = [(0.5, "155 images filed\nunder two classes"), (1.5, "class counts match;\nno melanoma folder"),
         (2.5, "1 atlas image\nfewer"), (3.5, "same images; splits\nmerged; + 124 new class")]
for x, t in edges:
    ax.text(x, 0.12, t, ha="center", va="bottom", fontsize=6.6, color=AK, style="italic", linespacing=1.15) if False else \
        ax.text(x, -0.95, t, ha="center", va="top", fontsize=6.8, color=AK, style="italic", linespacing=1.15)
ax.text(0.01, 0.98, "a", transform=ax.transAxes, fontweight="bold", fontsize=10, color=INK, va="top")

# (b) folder composition -------------------------------------------------------
ax = fig.add_subplot(gs[0, 0])
y = np.arange(len(order))[::-1]
left = np.zeros(len(order))
cols = [AGREE, AK, MEL, NODX]
labs = ["Agrees with ISIC Archive", "Actinic keratosis", "Melanoma", "No recorded diagnosis"]
for k in range(4):
    m = C[:, k] > 0
    ax.barh(y[m], C[m, k], left=left[m], color=cols[k], height=0.66, edgecolor="white" if k < 3 else MUTED, lw=0.6,
            hatch="////" if k == 3 else None, label=labs[k])
    left += C[:, k]
for yi, row, f in zip(y, C, order):
    tot = row.sum()
    extra = "  (all melanoma)" if f == "seborrheic keratosis" else ("  (78 AK, 1 no diagnosis)" if f == "nevus" else "")
    ax.text(tot + 6, yi, f"{tot}{extra}", va="center", fontsize=7, color=MEL if extra else INK2,
            fontweight="bold" if extra else "normal")
ax.set_yticks(y, [names[f] for f in order])
ax.set_xlim(0, 560)
ax.set_xlabel("Images in benchmark folder", fontsize=7.5)
ax.set_title("ISIC-derived folders by ISIC Archive diagnosis", loc="left", color=INK, fontsize=8)
ax.grid(axis="y", visible=False)
ax.legend(loc="lower right", fontsize=7, handlelength=1.2, borderaxespad=0.2)
panel(ax, "b")

# (c) test independence --------------------------------------------------------
ax = fig.add_subplot(gs[0, 1])
segs = [(32, MEL, "Same lesion, identical\ncopy in training"), (73, AK, "Same lesion, other\nimage in training"),
        (132, AGREE, "Lesion not in training"), (68, NODX, "No lesion ID")]
bottom = 0
for n, c, lab in segs:
    ax.bar(0, n, bottom=bottom, color=c, width=0.5, edgecolor="white" if c != NODX else MUTED, lw=0.6,
           hatch="////" if c == NODX else None)
    ax.text(0.33, bottom + n / 2, f"{lab} ({n})", va="center", fontsize=7, color=INK2, linespacing=1.15)
    bottom += n
ax.plot([-0.33, -0.33], [1, 104], color=INK, lw=1.2)
ax.text(-0.38, 52.5, "105 (34%)\nshare a lesion\nwith training", ha="right", va="center", fontsize=7, color=INK,
        fontweight="bold", linespacing=1.15)
ax.set_xlim(-1.3, 2.3)
ax.set_ylim(0, 305)
ax.set_xticks([])
ax.set_yticks([])
for s in ("left", "bottom"):
    ax.spines[s].set_visible(False)
ax.set_title("ISIC-derived test images (n = 305)", loc="left", color=INK, fontsize=8)
ax.grid(False)
panel(ax, "c")
save(fig, "old/draft_Figure_1_matplotlib")
