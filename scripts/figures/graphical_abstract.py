"""OLD DRAFT (not submitted; the submitted graphical abstract was drawn by hand). Graphical abstract: 13 x 5 cm (w x h), >= 1328 x 531 px. Three data panels, no patient images."""
import matplotlib.pyplot as plt
import numpy as np
from common import AXIS, CLIN, DERM, INK, INK2, MUTED, OUT

AGREE, AK, MEL = "#c9c8c1", "#eb6834", "#d03b3b"
plt.rcParams.update({"font.size": 6.5})
fig = plt.figure(figsize=(13 / 2.54, 5 / 2.54))
fig.text(0.5, 0.975, "Label errors and source–modality confounding in a composite skin disease benchmark",
         ha="center", va="top", fontsize=7.6, fontweight="bold", color=INK)
L, W, B, H = [0.06, 0.395, 0.725], 0.235, 0.24, 0.47


def head(ax, n, title):
    ax.text(0, 1.1, f"{n}", transform=ax.transAxes, fontsize=8, fontweight="bold", color=INK, va="bottom")
    ax.text(0.09, 1.1, title, transform=ax.transAxes, fontsize=6.8, fontweight="bold", color=INK, va="bottom")


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=5.8, length=2)
    ax.grid(False)


# 1. labels: two benchmark folders by true diagnosis
ax = fig.add_axes([L[0] + 0.035, B, W - 0.035, H])
y = [1, 0]
ax.barh(y[0], 294, color=AGREE, height=0.55)
ax.barh(y[0], 79, left=294, color=AK, height=0.55)
ax.barh(y[1], 80, color=MEL, height=0.55)
ax.text(294 + 40, y[0] + 0.45, "AK", ha="center", fontsize=5.8, color=AK, fontweight="bold")
ax.text(84, y[1], "all melanoma", va="center", fontsize=5.8, color=MEL, fontweight="bold")
ax.set_yticks(y, ['"Nevus"', '"Seb. keratosis"'])
ax.set_xlim(0, 400)
ax.set_xticks([0, 200, 400])
ax.set_xlabel("images in folder", fontsize=5.8, color=INK2, labelpad=1)
ax.set_ylim(-0.6, 1.75)
clean(ax)
head(ax, "1", "Labels contradict biopsy")
ax.text(0.0, -0.36, "158 of 1,511 ISIC images; 34% of test\nimages share a lesion with training",
        transform=ax.transAxes, fontsize=5.6, color=INK2, va="top", linespacing=1.15)

# 2. same lesion, two views
ax = fig.add_axes([L[1], B, W, H])
x = [0, 1]
v = [0.282, 0.965]
ax.plot(x, v, color=MUTED, lw=1.2, zorder=1)
ax.scatter([0], [v[0]], s=36, color=CLIN, zorder=3, edgecolor="white", lw=0.8)
ax.scatter([1], [v[1]], s=36, color=DERM, marker="s", zorder=3, edgecolor="white", lw=0.8)
ax.text(0, v[0] - 0.13, f"{v[0]:.2f}", ha="center", fontsize=6, color=CLIN, fontweight="bold")
ax.text(1, v[1] + 0.06, f"{v[1]:.2f}", ha="center", fontsize=6, color=DERM, fontweight="bold")
ax.text(0.45, 0.42, "+68 pp", ha="left", fontsize=7, color=INK, fontweight="bold", rotation=0)
ax.set_xticks(x, ["clinical", "dermoscopic"])
ax.set_xlim(-0.4, 1.4)
ax.set_ylim(0, 1.15)
ax.set_yticks([0, 0.5, 1])
ax.set_ylabel("predicted as\nISIC class", fontsize=5.8, color=INK2, labelpad=1, linespacing=1.0)
clean(ax)
head(ax, "2", "Same lesion, two views")
ax.text(0.0, -0.36, "Derm7pt, 960 lesions: prediction follows\nthe source-linked modality",
        transform=ax.transAxes, fontsize=5.6, color=INK2, va="top", linespacing=1.15)

# 3. two failure modes
ax = fig.add_axes([L[2], B, W, H])
xs = np.arange(3)
gap = [0.610, 0.548, 0.345]    # P1 over all lesions, B0 + training opt.: original, corrected (A), + MILK10k dermoscopic (C)
nev = [None, 0.93, 0.76]
ax.bar(xs, gap, color=[MUTED, AGREE, DERM], width=0.6)
for xi, g in zip(xs, gap):
    ax.text(xi, g + 0.03, f"{g:.2f}", ha="center", fontsize=5.8, color=INK2)
ax.plot([1, 2], [nev[1], nev[2]], color=MEL, lw=1.2, marker="o", ms=3)
ax.text(2.35, nev[2] + 0.02, "nevi\nrecall", fontsize=5.4, color=MEL, va="center", linespacing=1.0)
ax.set_xticks(xs, ["original", "labels\ncorrected", "+ missing\nmodality"], fontsize=5.4)
ax.set_ylim(0, 1.15)
ax.set_yticks([0, 0.5, 1])
ax.set_ylabel("modality gap", fontsize=5.8, color=INK2, labelpad=1)
clean(ax)
head(ax, "3", "Two failure modes")
ax.text(0.0, -0.36, "Cleaning keeps the shortcut; adding the\nmodality reduces it but shifts errors",
        transform=ax.transAxes, fontsize=5.6, color=INK2, va="top", linespacing=1.15)
fig.savefig(OUT / "draft_graphical_abstract_matplotlib.png", dpi=300)
fig.savefig(OUT / "draft_graphical_abstract_matplotlib.pdf")
print("saved")
