"""Fig. 5 of the manuscript (saved as Figure_4 under the repository's earlier numbering; drawn at the manuscript text width). Training-data interventions on the corrected 30-class benchmark (Derm7pt, B0 + training opt., 3 seeds).
Arms: A corrected; H + HAM10000 dermoscopic BCC/MEL; B + MILK10k clinical; C + MILK10k dermoscopic of the same lesions.
(a) E1 dermoscopic recall of target classes; (b) E2 modality gap of target classes; (c) E3 modality gap of untreated
classes; (d) dermoscopic recall of untreated nevi. Brackets: matched contrast C - B."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import CLIN, DERM, ERR, INK, INK2, MS, NEUTRAL, OK, ROOT, panel, save

SEEDS = [42, 123, 2024]
names = np.array([p.name.lower() for p in sorted((ROOT / "data/skin31_clean/train").iterdir())])  # as pad_eval.class_names
man = pd.read_csv(ROOT / "prereg/derm7pt_manifest.csv")
nev = (man.cls == "NEV").to_numpy()
arms = [("A", "Corrected benchmark", ("deconf", "clean"), "#b9b7b0", None),
        ("R", "Relabelled (exploratory)", ("relabel_eval", "R"), "#a8662f", "....."),
        ("H", "+ HAM10000 dermoscopic", ("deconf", "deconf"), DERM, None),
        ("B", "+ MILK10k clinical", ("milk_eval", "B"), CLIN, None),
        ("C", "+ MILK10k dermoscopic", ("milk_eval", "C"), DERM, "////")]


def nev_recall(folder, arm):
    v = []
    for s in SEEDS:
        z = np.load(ROOT / "outputs" / folder / "logits" / f"derm7pt_{arm}_seed{s}.npz")
        v.append((names[z["derm"].argmax(1)][nev] == "nevus").mean())
    return float(np.mean(v))


dc = pd.read_csv(ROOT / "outputs/deconf/derm7pt_primary.csv").set_index("estimand")
mk = pd.read_csv(ROOT / "outputs/milk_eval/derm7pt_results.csv")
arm_rows = mk[mk.row.str.startswith("arm")].set_index("row")
cb = mk[mk.row == "C-B"].set_index("estimand")
rl = pd.read_csv(ROOT / "outputs/relabel_eval/derm7pt_results.csv").set_index("row")
E = {e: [dc.loc[e, "clean_mean"], rl.loc["arm R", e], dc.loc[e, "deconf_mean"], arm_rows.loc["arm B", e],
         arm_rows.loc["arm C", e]]
     for e in ["E1_derm_recall_target", "E2_P1_target", "E3_P1_control"]}
nv = [nev_recall(*a[2]) for a in arms]

# (letter, title, values, C-B estimand or None, colour of the C-B label)
panels = [("a", "Target classes\ndermoscopic recall  ↑", E["E1_derm_recall_target"], "E1_derm_recall_target", OK),
          ("b", "Target classes\nmodality gap  ↓", E["E2_P1_target"], "E2_P1_target", OK),
          ("c", "Untreated classes\nmodality gap", E["E3_P1_control"], "E3_P1_control", ERR),
          ("d", "Untreated nevi\ndermoscopic recall  ↑", nv, None, ERR)]

fig, axs2 = plt.subplots(2, 2, figsize=(MS - 0.65, 5.0), sharey=True, gridspec_kw={"wspace": 0.12, "hspace": 0.55})
axs = axs2.ravel()
y = np.arange(5)[::-1]
for ax, (ltr, title, vals, est, ccol) in zip(axs, panels):
    for yi, (code, lab, _, col, hatch), v in zip(y, arms, vals):
        ax.barh(yi, v, height=0.62, color=col, hatch=hatch, edgecolor="white", lw=0.6, zorder=2)
        ax.text(v + 0.025, yi, f"{v:.2f}", va="center", fontsize=8, color=INK2)
    # bracket between B and C, placed just right of the longer bar and its label
    xr = max(vals[3], vals[4]) + 0.25
    ax.plot([xr, xr + 0.03, xr + 0.03, xr], [y[3] + 0.12, y[3] + 0.12, y[4] - 0.12, y[4] - 0.12], color=INK, lw=0.7)
    if est:
        delta = cb.loc[est, "delta"]
    else:  # nevi: report the change of C relative to B as well, descriptively
        delta = vals[4] - vals[3]
    ax.text(xr + 0.06, (y[3] + y[4]) / 2, f"C−B\n{delta:+.2f}".replace("-", chr(0x2212)), fontsize=8, color=ccol, fontweight="bold",
            va="center", linespacing=1.05)
    ax.set_xlim(0, 1.35)
    ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"], fontsize=8)
    ax.set_ylim(-1.15, 4.55)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.text(-0.02, 1.05, ltr, transform=ax.transAxes, fontweight="bold", fontsize=10, color=INK, ha="right", va="bottom")
    ax.text(0.02, 1.05, title, transform=ax.transAxes, fontweight="bold", fontsize=8, color=INK, ha="left",
            va="bottom", linespacing=1.1)
for axl in (axs[0], axs[2]):
    axl.set_yticks(y, [f"{a[0]}   {a[1]}" for a in arms], fontsize=8)
    for t, a in zip(axl.get_yticklabels(), arms):
        t.set_color(INK2 if a[0] == "A" else a[3])
fig.text(0.98, 0.015, "↑ higher recall   ↓ smaller modality gap   C − B: same 427 lesions, dermoscopic vs clinical images added",
         ha="right", fontsize=7.5, color=INK2)
save(fig, "Figure_4")
