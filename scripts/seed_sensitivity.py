"""Post hoc, exploratory: leave-one-seed-out sensitivity of the mean contrasts between arms (Supplementary Table S15).
Each reduced estimate averages the two retained seeds of a seed-matched contrast (seeds 42, 123, 2024). Every per-seed
value is recomputed from the saved logits at full precision (no rounding before averaging); the estimand definitions
are those of vit_milk_eval.stats (E1-E3, P1, untreated -> melanoma, nevus and melanoma recall), derm7pt_overall.stats
(macro recall) and pad_eval.metrics (six-diagnosis balanced accuracy, registered rule)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import pad_eval as P  # noqa: E402
from derm7pt_overall import stats as overall  # noqa: E402
from vit_milk_eval import LABELS, stats  # noqa: E402

O = ROOT / "outputs"
OUT = O / "seed_sensitivity"
SEEDS = [42, 123, 2024]
D7 = {"A": O / "milk_eval/logits/derm7pt_A", "B": O / "milk_eval/logits/derm7pt_B", "C": O / "milk_eval/logits/derm7pt_C",
      "H": O / "deconf/logits/derm7pt_deconf", "R": O / "relabel_eval/logits/derm7pt_R",
      "vB": O / "vit_milk/logits/derm7pt_B", "vC": O / "vit_milk/logits/derm7pt_C"}
PAD = {"A": O / "milk_eval/logits/pad_A", "B": O / "milk_eval/logits/pad_B", "C": O / "milk_eval/logits/pad_C",
       "H": O / "deconf/logits/pad_deconf", "R": O / "relabel_eval/logits/pad_R",
       "vB": O / "vit_milk/logits/pad_B", "vC": O / "vit_milk/logits/pad_C"}
RA = r"$\rightarrow$"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    names = np.array(P.class_names("clean"))
    orig = P.class_names("orig")
    cls = pd.read_csv(ROOT / "prereg" / "derm7pt_manifest.csv").cls.to_numpy()
    pm_all = pd.read_csv(ROOT / "prereg" / "pad_manifest.csv")
    keep = (pm_all.exclude.fillna("") == "").to_numpy()  # outputs/pad logits cover all 2,298 images
    pm = pm_all[keep].reset_index(drop=True)
    y = pm.diagnostic.map({d: k for k, d in enumerate(P.PAD6)}).to_numpy()

    def d7(arm, s):
        z = np.load(f"{D7[arm]}_seed{s}.npz")
        pc, pd_ = z["clinic"].argmax(1), z["derm"].argmax(1)
        st = dict(zip(LABELS, stats(pc, pd_, cls, names)))
        st["macro_recall"] = overall(pd_, cls, names)[6]
        return st

    def pad(arm, s):
        return P.metrics(np.load(f"{PAD[arm]}_seed{s}.npy"), list(names), y)["bal_acc_6way"]

    def repair(cfg, s):
        L = O / "pad" / "logits"
        return (P.metrics(np.load(L / f"clean_{cfg}_seed{s}.npy")[keep], list(names), y)["bal_acc_6way"]
                - P.metrics(np.load(L / f"orig_{cfg}_seed{s}.npy")[keep], orig, y)["bal_acc_6way"])

    S = {(a, s): d7(a, s) for a in D7 for s in SEEDS}
    dd = lambda x, z, k: [S[(x, s)][k] - S[(z, s)][k] for s in SEEDS]
    dp = lambda x, z: [pad(x, s) - pad(z, s) for s in SEEDS]
    rows = [
        ("Repair (corrected $-$ original), B0 baseline", "PAD BA", [repair("4_b0_baseline", s) for s in SEEDS]),
        ("Repair (corrected $-$ original), B0 + training opt.", "PAD BA", [repair("5_b0_trainopt", s) for s in SEEDS]),
        ("R $-$ A", "MEL dermoscopic recall", dd("R", "A", "MEL_derm_recall")),
        ("R $-$ A", "P1", dd("R", "A", "P1_all")),
        ("R $-$ A", f"Untreated {RA} MEL", dd("R", "A", "untreated_derm_to_MEL")),
        ("R $-$ A", "PAD BA", dp("R", "A")),
        ("H $-$ A", "E1", dd("H", "A", "E1")),
        ("H $-$ A", "E2", dd("H", "A", "E2")),
        ("H $-$ A", "E3", dd("H", "A", "E3")),
        ("H $-$ A", "P1", dd("H", "A", "P1_all")),
        ("H $-$ A", "PAD BA", dp("H", "A")),
    ]
    for tag, (x, z) in (("EfficientNet-B0", ("C", "B")), ("ViT-S/16", ("vC", "vB"))):
        c = f"C $-$ B, {tag}"
        rows += [(c, "E1", dd(x, z, "E1")), (c, "E2", dd(x, z, "E2")), (c, "E3", dd(x, z, "E3")),
                 (c, f"Untreated {RA} MEL", dd(x, z, "untreated_derm_to_MEL")),
                 (c, "Nevus dermoscopic recall", dd(x, z, "NEV_derm_recall")),
                 (c, "Macro recall (dermoscopic)", dd(x, z, "macro_recall")), (c, "PAD BA", dp(x, z))]
    out = []
    for c, e, v in rows:
        v = np.array(v, dtype=float)
        loso = [np.delete(v, k).mean() for k in range(3)]
        out.append({"contrast": c, "endpoint": e, "all_seeds": v.mean(), "seed42": v[0], "seed123": v[1], "seed2024": v[2],
                    "without42": loso[0], "without123": loso[1], "without2024": loso[2],
                    "sign_all_seeds": "same" if (np.all(v > 0) or np.all(v < 0)) else "mixed",
                    "sign_all_loso": "same" if (np.all(np.array(loso) > 0) or np.all(np.array(loso) < 0)) else "mixed"})
    r = pd.DataFrame(out)
    r.to_csv(OUT / "leave_one_seed_out.csv", index=False)
    pd.set_option("display.width", 250)
    print(r.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
