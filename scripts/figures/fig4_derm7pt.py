"""Fig. 3. Paired clinical and dermoscopic images of the same Derm7pt lesions (drawn at the manuscript text width).
(a) fraction predicted as an ISIC-derived class, by diagnosis; (b) recall, by diagnosis
(both: primary configuration, mean of 3 seeds; arrow = clinical -> dermoscopic view of the same lesions);
(c) P1 with 95% CI across models: with the six ISIC-derived outputs common to all models (solid) and, for the original
31-output models, with the seven outputs of the registered analysis, which include "seborrheic keratosis" (open)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from common import CLIN, CLIN_L, DERM, DERM_L, ERR, INK, INK2, MS, MUTED, NEUTRAL, ROOT, panel, save

FS, FS_S = 7.5, 7.0   # label and annotation sizes (pt) at final size

sec = pd.read_csv(ROOT / "outputs/derm7pt/secondary.csv")
s = sec[sec.model == "7_b0_kd_fusion"].groupby("dx")[["recall_clinic", "recall_derm", "isic_rate_clinic",
                                                     "isic_rate_derm", "n"]].mean()
order = ["NEV", "SEK", "DF", "VASC", "BCC", "MEL"]
names = {"MEL": "Melanoma", "BCC": "Basal cell carcinoma", "NEV": "Nevus", "SEK": "Seborrheic keratosis",
         "DF": "Dermatofibroma", "VASC": "Vascular lesion"}
src = {"NEV": "isic", "SEK": "isic", "DF": "isic", "VASC": "isic", "BCC": "atlas", "MEL": "both"}
s = s.loc[order]

ci = pd.read_csv(ROOT / "outputs/grouping_check/P1_ci_by_grouping.csv").set_index(["model", "grouping"])
label = {"1_b2_baseline": "B2 baseline", "2_b2_trainopt": "B2 + training opt.", "3_b2_fusion_teacher": "B2 fusion teacher",
         "4_b0_baseline": "B0 baseline", "5_b0_trainopt": "B0 + training opt.", "6_b0_kd_nonfusion": "B0 KD, non-fusion",
         "7_b0_kd_fusion": "B0 KD, fusion (primary)"}


def item(lab, model, orig=True):
    six = tuple(ci.loc[(model, "six"), ["P1", "ci_low", "ci_high"]])
    seven = tuple(ci.loc[(model, "seven"), ["P1", "ci_low", "ci_high"]]) if orig else None
    return lab, six, seven


groups = [("Original benchmark", [item(label[m], m) for m in label]),
          ("Released model", [item("Original authors' model", "authors")]),
          ("Corrected benchmark", [item(label[c], "corrected_" + c, orig=False) for c in ["4_b0_baseline", "5_b0_trainopt"]]),
          ("Exploratory", [item("ViT-S/16", "vit_s16")])]

fig = plt.figure(figsize=(MS, 6.1))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.45], hspace=0.5, wspace=0.07, left=0.31, right=0.98, top=0.92, bottom=0.08)
y = np.arange(len(order))[::-1]
for k, (cols, title, ltr) in enumerate([(("isic_rate_clinic", "isic_rate_derm"), "Predicted as ISIC-derived", "a"),
                                        (("recall_clinic", "recall_derm"), "Correct diagnosis (recall)", "b")]):
    ax = fig.add_subplot(gs[0, k])
    for yi, d in zip(y, order):
        band = {"isic": DERM_L, "atlas": CLIN_L, "both": "#f1efe9"}[src[d]]
        ax.axhspan(yi - 0.5, yi + 0.5, color=band, alpha=0.55, lw=0, zorder=0)
        c0, c1 = s.loc[d, cols[0]], s.loc[d, cols[1]]
        ax.annotate("", xy=(c1, yi), xytext=(c0, yi),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2, shrinkA=5, shrinkB=5, mutation_scale=8))
        ax.scatter(c0, yi, s=40, color=CLIN, edgecolor="white", lw=0.8, zorder=3)
        ax.scatter(c1, yi, s=40, color=DERM, marker="s", edgecolor="white", lw=0.8, zorder=3)
    ax.set_yticks(y, [f"{names[d]} (n = {int(s.loc[d, 'n'])})" for d in order] if k == 0 else [], fontsize=FS)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.5, len(order) - 0.5)
    ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"], fontsize=FS)
    ax.grid(axis="y", visible=False)
    panel(ax, ltr, title)
ax.text(1.0, -0.2, "Row colour: diagnosis is ISIC-derived (orange), atlas-derived (blue) or both (grey)", transform=ax.transAxes,
        fontsize=FS_S - 0.5, color=MUTED, ha="right")
fig.legend(handles=[Line2D([], [], marker="o", ls="", color=CLIN, label="clinical image"),
                    Line2D([], [], marker="s", ls="", color=DERM, label="dermoscopic image of the same lesion")],
           loc="upper center", bbox_to_anchor=(0.64, 1.0), ncol=2, fontsize=FS)

# (c) forest plot -------------------------------------------------------------------
ax = fig.add_subplot(gs[1, :])
rows, yy, ylab = [], [], []
pos = 0
for gname, items in groups:
    rows.append(("header", gname, pos))
    pos -= 1
    for it in items:
        rows.append(("item", it, pos))
        pos -= 1
    pos -= 0.45
DY = 0.24
for kind, it, yv in rows:
    if kind == "header":
        ax.text(-0.012, yv, it, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=FS, fontweight="bold",
                color=INK)
        continue
    lab, six, seven = it
    primary = "primary" in lab
    col = DERM if primary else (MUTED if lab.startswith("ViT") else INK2)
    y6 = yv - DY if seven else yv
    ax.plot([six[1], six[2]], [y6, y6], color=col, lw=1.8 if primary else 1.4, solid_capstyle="round")
    ax.scatter(six[0], y6, s=44 if primary else 30, color=col, marker="D" if lab.startswith("ViT") else "o",
               edgecolor="white", lw=0.7, zorder=3)
    ax.text(six[2] + 0.012, y6, f"{six[0]:.2f}", va="center", fontsize=FS_S, color=col)
    if seven:
        y7 = yv + DY
        ax.plot([seven[1], seven[2]], [y7, y7], color=MUTED, lw=0.9, solid_capstyle="round")
        ax.scatter(seven[0], y7, s=24, facecolor="white", edgecolor=MUTED, lw=1.0, marker="D" if lab.startswith("ViT") else "o", zorder=3)
    yy.append(yv)
    ylab.append(lab)
ax.set_yticks(yy, ylab, fontsize=FS)
ax.tick_params(axis="y", length=0)
for t in ax.get_yticklabels():
    if "primary" in t.get_text():
        t.set_color(DERM)
        t.set_fontweight("bold")
ax.axvline(0, color=MUTED, lw=0.8)
ax.set_xlim(-0.02, 0.8)
ax.set_ylim(min(yy) - 0.9, 0.9)
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8], ["0", "0.2", "0.4", "0.6", "0.8"], fontsize=FS)
ax.set_xlabel("P1: ISIC-derived prediction rate, dermoscopic minus clinical image of the same lesion (95% CI)", fontsize=FS)
ax.grid(axis="y", visible=False)
ax.legend(handles=[Line2D([], [], marker="o", color=INK2, lw=1.4, ms=5, label="six common outputs"),
                   Line2D([], [], marker="o", color=MUTED, mfc="white", lw=0.9, ms=4.5,
                          label="seven outputs (registered; original models)")],
          loc="center left", bbox_to_anchor=(0.01, 0.5), fontsize=FS_S, frameon=False, borderaxespad=0.2)
panel(ax, "c", "Modality dependence across models", x=-0.31)
save(fig, "Figure_3")
