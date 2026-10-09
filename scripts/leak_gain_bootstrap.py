"""Exploratory: is the training-optimization gain on the ISIC part concentrated on test images whose lesion
also appears in train? Estimand per pair A->B: (gain on seen lesions) - (gain on unseen lesions), mean over 3 seeds.
Images are resampled with replacement within each group (seen / unseen), 2000 resamples."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "outputs" / "repro"
f = pd.read_csv(REPRO / "skin31_test_flags.csv")
isic = f.id.str.startswith("ISIC_") & ~f.disputed
seen = (isic & f.lesion_leak).to_numpy()
unseen = (isic & ~f.lesion_leak & ~f.exact_leak & f.lesion_id.notna()).to_numpy()
y = f.label.to_numpy()
print("seen:", seen.sum(), "unseen (with lesion_id):", unseen.sum())


def correct(cfg):
    return np.stack([np.load(REPRO / cfg / f"seed{s}" / "skin31_test_amp.npy").argmax(1) == y for s in (42, 123, 2024)])


PAIRS = [("1_b2_baseline", "2_b2_trainopt"), ("4_b0_baseline", "5_b0_trainopt"),
         ("5_b0_trainopt", "7_b0_kd_fusion"), ("2_b2_trainopt", "3_b2_fusion_teacher")]
rng = np.random.default_rng(20260930)
si, ui = np.flatnonzero(seen), np.flatnonzero(unseen)
rows = []
for a, b in PAIRS:
    d = (correct(b).astype(float) - correct(a)).mean(0)  # per-image gain, averaged over seeds
    g_seen, g_unseen = d[si].mean(), d[ui].mean()
    boots = [d[rng.choice(si, si.size)].mean() - d[rng.choice(ui, ui.size)].mean() for _ in range(2000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    rows.append({"pair": f"{a} -> {b}", "gain_seen_pp": 100 * g_seen, "gain_unseen_pp": 100 * g_unseen,
                 "diff_pp": 100 * (g_seen - g_unseen), "ci_low": 100 * lo, "ci_high": 100 * hi})
r = pd.DataFrame(rows).round(2)
r.to_csv(REPRO / "leak_gain_bootstrap.csv", index=False)
print(r.to_string(index=False))
