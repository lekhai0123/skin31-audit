"""E-03 (exploratory): two extra training seeds (1 and 2) for arms B and C. Plan: prereg/e03_extra_seeds_plan.md.

logits : for each new seed, the Kaggle checkpoint is loaded with milk_eval.load
         (checkpoints_milk{B,C}/5_b0_trainopt/seed{s}); Derm7pt logits come from derm7pt_eval_clean.logits and
         PAD-UFES-20 logits from pad_eval.logits, exactly as in milk_eval.main. They are saved under outputs/e03/logits;
         the logits of the registered seeds (outputs/milk_eval/logits) are not touched.
stats  : per-seed endpoints with the functions that produced Supplementary Table S15 (vit_milk_eval.stats,
         derm7pt_overall.stats, pad_eval.metrics); C - B per seed for seeds 42, 123, 2024 (registered) and 1, 2 (new);
         five-seed mean, SD, 95% t interval over seeds (df = 4) and sign count.
--check-existing recomputes the three registered seeds only and checks them against Table S15
(outputs/seed_sensitivity/leave_one_seed_out.csv).
Usage (datn environment): python scripts/e03_eval.py [--check-existing]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import milk_eval as M  # noqa: E402
import pad_eval as P  # noqa: E402
from derm7pt_eval_clean import logits as d7_logits  # noqa: E402
from derm7pt_overall import stats as overall  # noqa: E402
from vit_milk_eval import LABELS, stats  # noqa: E402

O = ROOT / "outputs"
OLD = O / "milk_eval" / "logits"
NEW = O / "e03" / "logits"
RES = O / "e03"
OLD_SEEDS, NEW_SEEDS = [42, 123, 2024], [1, 2]
ENDPOINTS = [("E1", "E1"), ("E2", "E2"), ("E3", "E3"), ("Untreated -> MEL", "untreated_derm_to_MEL"),
             ("Nevus dermoscopic recall", "NEV_derm_recall"), ("Macro recall (dermoscopic)", "macro_recall"),
             ("PAD BA", "PAD BA")]
S15_NAMES = {"Untreated -> MEL": "Untreated $\\rightarrow$ MEL"}


def pad_inputs():
    pm = pd.read_csv(ROOT / "prereg" / "pad_manifest.csv")
    pm = pm[pm.exclude.fillna("") == ""].reset_index(drop=True)
    files = {p.name: p for p in P.PAD.rglob("*.png")}
    return [files[i] for i in pm.img_id], pm.diagnostic.map({d: k for k, d in enumerate(P.PAD6)}).to_numpy()


def make_logits(paths):
    man = pd.read_csv(ROOT / "prereg" / "derm7pt_manifest.csv")
    NEW.mkdir(parents=True, exist_ok=True)
    for arm in ("B", "C"):
        for s in NEW_SEEDS:
            f = NEW / f"derm7pt_{arm}_seed{s}.npz"
            if not f.exists():
                m = M.load(arm, s)
                np.savez(f, clinic=d7_logits(m, [M.D7 / p for p in man.clinic]),
                         derm=d7_logits(m, [M.D7 / p for p in man.derm]))
            g = NEW / f"pad_{arm}_seed{s}.npy"
            if not g.exists():
                np.save(g, P.logits(M.load(arm, s), paths, 30))


def endpoints(logit_dir, arm, s, names, cls, y):
    z = np.load(logit_dir / f"derm7pt_{arm}_seed{s}.npz")
    pc, pd_ = z["clinic"].argmax(1), z["derm"].argmax(1)
    st = dict(zip(LABELS, stats(pc, pd_, cls, names)))
    st["macro_recall"] = overall(pd_, cls, names)[6]
    st["PAD BA"] = P.metrics(np.load(logit_dir / f"pad_{arm}_seed{s}.npy"), list(names), y)["bal_acc_6way"]
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-existing", action="store_true")
    a = ap.parse_args()
    names = np.array(P.class_names("clean"))
    cls = pd.read_csv(ROOT / "prereg" / "derm7pt_manifest.csv").cls.to_numpy()
    paths, y = pad_inputs()
    seeds = [(s, OLD) for s in OLD_SEEDS]
    if not a.check_existing:
        make_logits(paths)
        seeds += [(s, NEW) for s in NEW_SEEDS]
    rows = []
    for label, key in ENDPOINTS:
        d = {s: endpoints(src, "C", s, names, cls, y)[key] - endpoints(src, "B", s, names, cls, y)[key]
             for s, src in seeds}
        rows.append({"endpoint": label, **{f"seed{s}": v for s, v in d.items()}})
    r = pd.DataFrame(rows)

    if a.check_existing:
        ref = pd.read_csv(O / "seed_sensitivity" / "leave_one_seed_out.csv")
        ref = ref[ref.contrast == "C $-$ B, EfficientNet-B0"].set_index("endpoint")
        worst = 0.0
        for row in r.itertuples():
            ref_row = ref.loc[S15_NAMES.get(row.endpoint, row.endpoint)]
            for s in OLD_SEEDS:
                worst = max(worst, abs(getattr(row, f"seed{s}") - ref_row[f"seed{s}"]))
        print(r.round(4).to_string(index=False))
        print(f"max |difference| from Table S15 over 3 seeds x {len(r)} endpoints: {worst:.2e}")
        assert worst < 1e-9
        return

    v = r[[f"seed{s}" for s, _ in seeds]].to_numpy(dtype=float)
    n = v.shape[1]
    r["mean_5"] = v.mean(1)
    r["sd_5"] = v.std(1, ddof=1)
    half = sps.t.ppf(0.975, n - 1) * r["sd_5"] / np.sqrt(n)
    r["t95_low"], r["t95_high"] = r["mean_5"] - half, r["mean_5"] + half
    r["mean_registered_3"] = v[:, :3].mean(1)
    r["n_same_sign_as_registered"] = [int((np.sign(row) == np.sign(m)).sum()) for row, m in zip(v, r["mean_registered_3"])]
    RES.mkdir(parents=True, exist_ok=True)
    r.to_csv(RES / "e03_c_minus_b_five_seeds.csv", index=False)
    pd.set_option("display.width", 250)
    print(r.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
