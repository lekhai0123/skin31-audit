"""Exploratory: overall diagnostic performance on Derm7pt for the matched MILK10k arms B and C (EfficientNet-B0 and
ViT-S/16). Argmax over all 30 outputs; a prediction is correct if it falls in the output mapped to the diagnosis
(Supplementary Table S3). Per-class recall, macro recall over the six diagnoses and accuracy, by view; C - B paired by
lesion and seed, lesion bootstrap (2,000 resamples) for fixed checkpoints."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import pad_eval as P  # noqa: E402

OUT = ROOT / "outputs" / "derm7pt_overall"
SEEDS, B, RNG_SEED = [42, 123, 2024], 2000, 20260930
DX = {"BCC": "basal cell carcinoma", "MEL": "melanoma", "NEV": "nevus", "SEK": "pigmented benign keratosis",
      "DF": "dermatofibroma", "VASC": "vascular lesion"}
MODELS = {"EfficientNet-B0": ROOT / "outputs" / "milk_eval" / "logits", "ViT-S/16": ROOT / "outputs" / "vit_milk" / "logits"}


def stats(pred, cls, names):
    ok = names[pred] == np.array([DX[c] for c in cls])
    rec = [ok[cls == d].mean() for d in DX]
    return np.array(rec + [np.mean(rec), ok.mean()])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    names = np.array(P.class_names("clean"))
    cls = pd.read_csv(ROOT / "prereg" / "derm7pt_manifest.csv").cls.to_numpy()
    labels = [f"recall_{d}" for d in DX] + ["macro_recall", "accuracy"]
    rows = []
    for model, d in MODELS.items():
        for view in ("derm", "clinic"):
            pr = {(a, s): np.load(d / f"derm7pt_{a}_seed{s}.npz")[view].argmax(1) for a in "BC" for s in SEEDS}
            v = {a: np.array([stats(pr[(a, s)], cls, names) for s in SEEDS]) for a in "BC"}
            diff = v["C"] - v["B"]
            rng = np.random.default_rng(RNG_SEED)
            bs = []
            while len(bs) < B:
                i = rng.integers(0, len(cls), len(cls))
                if any((cls[i] == c).sum() == 0 for c in DX):
                    continue
                bs.append(np.mean([stats(pr[("C", s)][i], cls[i], names) - stats(pr[("B", s)][i], cls[i], names)
                                   for s in SEEDS], axis=0))
            lo, hi = np.percentile(np.array(bs), [2.5, 97.5], axis=0)
            for k, lab in enumerate(labels):
                rows.append({"model": model, "view": view, "metric": lab, "B": v["B"][:, k].mean(), "C": v["C"][:, k].mean(),
                             "C_minus_B": diff[:, k].mean(), "per_seed": diff[:, k].round(4).tolist(),
                             "ci_low": lo[k], "ci_high": hi[k]})
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "derm7pt_overall.csv", index=False)
    print(r.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
