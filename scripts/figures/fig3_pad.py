"""Fig. 2 (drawn at the manuscript text width). External testing on PAD-UFES-20 (six shared diagnoses).
(a) internal accuracy vs six-diagnosis balanced accuracy (restricted prediction), every model;
(b) fraction of PAD images predicted as an ISIC-derived class, by diagnosis, against the value expected if
    predictions followed the diagnosis (1 for diagnoses with an ISIC-derived class, 0 otherwise)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from common import CLIN, CLIN_L, DERM, DERM_L, ERR, INK, INK2, MUTED, MS, NEUTRAL, ROOT, panel, save

rep = pd.read_csv(ROOT / "outputs/repro/skin31_reproduction.csv").groupby("config").acc_fp32.mean() / 100
pad = pd.read_csv(ROOT / "outputs/pad/results.csv")
pad = pad[pad.cohort == "main"]
ba = pad[pad.metric == "bal_acc_6way"].set_index(["source", "config"])
b2 = ["1_b2_baseline", "2_b2_trainopt", "3_b2_fusion_teacher"]
b0 = ["4_b0_baseline", "5_b0_trainopt", "6_b0_kd_nonfusion", "7_b0_kd_fusion"]

fig, (a, b) = plt.subplots(1, 2, figsize=(MS, 3.35), gridspec_kw={"width_ratios": [1, 1.1], "wspace": 0.95})

# (a) -------------------------------------------------------------------------
a.axhspan(0, 1 / 6, color="#f3f2ee", zorder=0)
a.axhline(1 / 6, color=MUTED, lw=0.8, ls=(0, (3, 2)))
a.text(0.917, 1 / 6 - 0.012, "uniform-output\nreference (1/6)", fontsize=7, color=MUTED, ha="right", va="top")


def dot(x, r, marker, color, size=26, zorder=3):
    a.plot([x, x], [r.ci_low, r.ci_high], color=color, lw=1.1, alpha=0.55, solid_capstyle="round", zorder=zorder - 1)
    a.scatter(x, r["mean"], s=size, marker=marker, color=color, edgecolor="white", lw=0.7, zorder=zorder)


for c in b2:
    dot(rep[c], ba.loc[("orig", c)], "o", NEUTRAL)
for c in b0:
    dot(rep[c], ba.loc[("orig", c)], "s", INK2, size=22)
dot(0.8823, ba.loc[("authors", "authors")], "D", DERM, size=30, zorder=4)
a.annotate("authors' model", (0.8823, 0.303), xytext=(0.878, 0.388), fontsize=7, color=DERM,
           arrowprops=dict(arrowstyle="-", color=DERM, lw=0.6))
a.set_xlim(0.83, 0.92)
a.set_ylim(0.10, 0.42)
a.set_xticks([0.84, 0.86, 0.88, 0.90, 0.92], ["84%", "86%", "88%", "90%", "92%"])
a.set_xlabel("Internal benchmark accuracy", fontsize=7.5)
a.set_ylabel("PAD-UFES-20 balanced accuracy\n(six shared diagnoses)")
a.grid(axis="x", visible=False)
a.legend(handles=[Line2D([], [], marker="o", ls="", color=NEUTRAL, label="EfficientNet-B2"),
                  Line2D([], [], marker="s", ls="", color=INK2, label="EfficientNet-B0")],
         loc="lower left", bbox_to_anchor=(0.0, 0.23), fontsize=7, handletextpad=0.2, borderaxespad=0.1)
panel(a, "a", "Internal vs external")

# (b) -------------------------------------------------------------------------
order = ["ACK", "NEV", "SEK", "SCC", "BCC", "MEL"]
names = {"ACK": "Actinic keratosis", "NEV": "Nevus", "SEK": "Seborrheic keratosis", "SCC": "Squamous cell carcinoma",
         "BCC": "Basal cell carcinoma", "MEL": "Melanoma"}
isic_cls = {"ACK", "NEV", "SEK", "SCC"}
pm = pad[(pad.source == "orig") & (pad.config == "7_b0_kd_fusion")].set_index("metric")["mean"]
y = np.arange(len(order))[::-1]
b.axhspan(1.5, 5.5, color=DERM_L, alpha=0.45, zorder=0, lw=0)
b.axhspan(-0.5, 1.5, color=CLIN_L, alpha=0.6, zorder=0, lw=0)
b.text(1.0, 5.32, "ISIC-derived class", fontsize=7, color=DERM, ha="right", va="top", fontweight="bold")
b.text(1.0, 1.32, "Atlas-derived class", fontsize=7, color=CLIN, ha="right", va="top", fontweight="bold")
for yi, d in zip(y, order):
    obs, exp = pm[f"isic_rate_{d}"], 1.0 if d in isic_cls else 0.0
    col = DERM if d in isic_cls else CLIN
    if d != "MEL":  # melanoma has two outputs (atlas "melanoma", ISIC "seborrheic keratosis"): no single expectation
        b.annotate("", xy=(obs, yi), xytext=(exp, yi),
                   arrowprops=dict(arrowstyle="-|>", color=col, lw=1.1, alpha=0.7, shrinkA=4, shrinkB=4, mutation_scale=7))
        b.scatter(exp, yi, s=30, facecolor="white", edgecolor=col, lw=1.1, zorder=3)
    else:
        b.text(obs + 0.05, yi, "two outputs", fontsize=7, color=MUTED, va="center", style="italic")
    b.scatter(obs, yi, s=34, color=col, edgecolor="white", lw=0.7, zorder=4)
    b.text(obs, yi + 0.27, f"{obs:.2f}", fontsize=7, color=INK2, ha="center")
b.set_yticks(y, [names[d] for d in order])
b.set_xlim(-0.05, 1.05)
b.set_ylim(-0.5, 5.5)
b.set_xlabel("Fraction predicted as an ISIC-derived class")
b.grid(axis="y", visible=False)
b.legend(handles=[Line2D([], [], marker="o", ls="", mfc="white", mec=INK2, label="expected if routed by diagnosis"),
                  Line2D([], [], marker="o", ls="", color=INK2, label="observed (primary model)")],
         loc="upper center", bbox_to_anchor=(0.42, -0.2), ncol=2, fontsize=7, handletextpad=0.2)
panel(b, "b", "ISIC-derived output rate by diagnosis", x=-0.6)
save(fig, "Figure_2")
