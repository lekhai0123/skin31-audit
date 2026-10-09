"""EXPLORATORY sensitivity analyses requested in review (04/10/2026). Saved logits only.
(1) PAD-UFES-20 six-diagnosis balanced accuracy when softmax probabilities of all outputs mapped to the same
    diagnosis are SUMMED before the argmax (the registered metric takes the argmax over mapped outputs first).
    Matters for the 31-class models, where melanoma has two outputs. Patient bootstrap for corrected - original.
(2) Strictly filtered test accuracy after also excluding test images whose most similar training pair was
    judged 'same' or 'uncertain' in the one-author visual review (similarity-enriched sample; descriptive).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from torchvision import datasets

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from pad_eval import PAD6, TARGET, patient_resamples  # noqa: E402

OUT = ROOT / "outputs" / "review_sensitivity"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [42, 123, 2024]
orig = [k.lower() for k, _ in sorted(json.loads((ROOT / "prereg/class_mapping.json").read_text())["class_to_idx"].items(),
                                     key=lambda kv: kv[1])]
clean = [p.name.lower() for p in sorted((ROOT / "data/skin31_clean/train").iterdir())]


def ba_summed(z, names, y):
    e = np.exp(z - z.max(1, keepdims=True))
    p = e / e.sum(1, keepdims=True)
    s = np.stack([p[:, [names.index(n) for n in TARGET[d] if n in names]].sum(1) for d in PAD6], 1)
    pred = s.argmax(1)
    return np.mean([(pred[y == k] == k).mean() for k in range(6)])


def ba_registered(z, names, y):
    cols, owner = [], []
    for k, d in enumerate(PAD6):
        for n in TARGET[d]:
            if n in names:
                cols.append(names.index(n))
                owner.append(k)
    pred = np.array(owner)[z[:, cols].argmax(1)]
    return np.mean([(pred[y == k] == k).mean() for k in range(6)])


pm = pd.read_csv(ROOT / "prereg/pad_manifest.csv")
main = (pm.exclude.fillna("") == "").to_numpy()
y = pm.diagnostic.map({d: k for k, d in enumerate(PAD6)}).to_numpy()[main]
pid = pm.patient_id.to_numpy()[main]
L = ROOT / "outputs/pad/logits"
models = {f"orig:{c}": ([L / f"orig_{c}_seed{s}.npy" for s in SEEDS], orig) for c in
          ["1_b2_baseline", "2_b2_trainopt", "3_b2_fusion_teacher", "4_b0_baseline", "5_b0_trainopt",
           "6_b0_kd_nonfusion", "7_b0_kd_fusion"]}
models["orig:authors"] = ([L / "authors_authors_seed0.npy"], orig)
for c in ["4_b0_baseline", "5_b0_trainopt"]:
    models[f"clean:{c}"] = ([L / f"clean_{c}_seed{s}.npy" for s in SEEDS], clean)
Z = {k: [np.load(f)[main] for f in fs] for k, (fs, _) in models.items()}
rows = []
for k, (_, names) in models.items():
    rows.append({"model": k, "BA_registered": np.mean([ba_registered(z, names, y) for z in Z[k]]),
                 "BA_summed": np.mean([ba_summed(z, names, y) for z in Z[k]])})
res, _ = patient_resamples(pid, y)
for c in ["4_b0_baseline", "5_b0_trainopt"]:
    for fn, lab in [(ba_registered, "registered"), (ba_summed, "summed")]:
        d = [fn(Z[f"clean:{c}"][i], clean, y) - fn(Z[f"orig:{c}"][i], orig, y) for i in range(3)]
        bs = [np.mean([fn(Z[f"clean:{c}"][i][r], clean, y[r]) - fn(Z[f"orig:{c}"][i][r], orig, y[r]) for i in range(3)])
              for r in res]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        rows.append({"model": f"corrected - original {c} ({lab})", "delta": np.mean(d), "per_seed": np.round(d, 4).tolist(),
                     "ci_low": lo, "ci_high": hi})
pad = pd.DataFrame(rows)
pad.to_csv(OUT / "pad_summed_probability.csv", index=False)
pd.set_option("display.width", 200)
print(pad.round(4).to_string(index=False))

# (2) atlas sensitivity on the strictly filtered test set
rat = pd.read_csv(ROOT / "outputs/embed_leak/review/rating_final_optionB.csv")
flag = rat[rat.visual_rating_author.isin(["same", "uncertain"])]
excl = set(zip(flag.test_class.str.lower(), flag.test_file))
test = datasets.ImageFolder(ROOT / "data/skin31_clean/test")
keys = [(Path(p).parent.name.lower(), Path(p).name) for p, _ in test.samples]
keep = np.array([k not in excl for k in keys])
ty = np.array(test.targets)
print("\nexcluded test images:", (~keep).sum(), "of", len(keep))
arms = {"A corrected": "outputs/milk_eval/logits/test_A", "H": "outputs/deconf/logits/test_deconf",
        "B": "outputs/milk_eval/logits/test_B", "C": "outputs/milk_eval/logits/test_C", "R relabel": "outputs/relabel_eval/logits/test_R"}
trow = []
for a, f in arms.items():
    acc = [np.load(ROOT / f"{f}_seed{s}.npy").argmax(1) == ty for s in SEEDS]
    trow.append({"model": a, "acc_strict_863": np.mean([x.mean() for x in acc]),
                 f"acc_excluding_{(~keep).sum()}": np.mean([x[keep].mean() for x in acc]),
                 "acc_on_excluded": np.mean([x[~keep].mean() for x in acc])})
# original configurations on the same strictly filtered images
flags = pd.read_csv(ROOT / "outputs/repro/skin31_test_flags.csv")
okeys = [(Path(p).parent.name.lower(), Path(p).name) for p in flags.path]
strict = flags.strict.to_numpy()
otarget = np.array([orig.index(Path(p).parent.name.lower()) for p in flags.path])
okeep = np.array([k not in excl for k in okeys]) & strict
for c in ["1_b2_baseline", "2_b2_trainopt", "4_b0_baseline", "5_b0_trainopt", "7_b0_kd_fusion"]:
    acc = [np.load(ROOT / f"outputs/repro/{c}/seed{s}/skin31_test_fp32.npy").argmax(1) == otarget for s in SEEDS]
    trow.append({"model": f"orig:{c}", "acc_strict_863": np.mean([x[strict].mean() for x in acc]),
                 f"acc_excluding_{(~keep).sum()}": np.mean([x[okeep].mean() for x in acc]),
                 "acc_on_excluded": np.mean([x[strict & ~okeep].mean() for x in acc])})
tt = pd.DataFrame(trow)
tt.to_csv(OUT / "atlas_exclusion_test_accuracy.csv", index=False)
print(tt.round(4).to_string(index=False))
