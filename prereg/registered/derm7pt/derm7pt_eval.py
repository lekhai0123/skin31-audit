"""Derm7pt evaluation, implementing prereg/derm7pt_analysis_plan.md exactly.

Refuses to run without --locked-commit (the GitHub commit that contains the analysis plan and manifests).
Usage: python scripts/derm7pt_eval.py --locked-commit <sha>
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from authors_model_check import load_authors_model  # noqa: E402
from reproduce import CONFIGS, SEEDS, load  # noqa: E402

PRE = ROOT / "prereg"
D7 = ROOT / "data" / "release_v0" / "images"
OUT = ROOT / "outputs" / "derm7pt"
EXPECTED_SHA = {
    "derm7pt_manifest.csv": "91b49892f6c9f7e66ac2d9a99bbfd70379303727c4e9b8c5110d480dbd2950f2",
    "class_mapping.json": "46893f03197d7e85075e9336788942d0366e317a54308122e876a9728ed788ea",
    "checkpoint_manifest.csv": "307e383bb8bdcf30a127f06d265099e1a658822978850bfdb6685303db392050",
}
ISIC = [0, 3, 18, 21, 25, 26, 30]
MEL_CLIN, MEL_DERM = 14, 25
CORRECT = {"MEL": {14, 25}, "BCC": {1}, "NEV": {18}, "VASC": {30}, "DF": {3}, "SEK": {21}}
PRIMARY = "7_b0_kd_fusion"
B, RNG_SEED = 2000, 20260930
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TF = None


def verify_inputs():
    for name, sha in EXPECTED_SHA.items():
        got = hashlib.sha256((PRE / name).read_bytes()).hexdigest()
        assert got == sha, f"{name} changed since the plan was locked"
    classes = json.loads((PRE / "class_mapping.json").read_text(encoding="utf-8"))["class_to_idx"]
    assert classes["Melanoma"] == 14 and classes["Seborrheic keratosis"] == 25 and classes["Nevus"] == 18
    ck = pd.read_csv(PRE / "checkpoint_manifest.csv")
    for r in ck.itertuples():
        assert hashlib.sha256((ROOT / r.path).read_bytes()).hexdigest() == r.sha256, r.path


@torch.inference_mode()
def logits(model, paths):
    from skin_disease.transforms import build_eval_transform
    tf = build_eval_transform(224)
    out = []
    for i in range(0, len(paths), 32):
        x = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i + 32]]).to(DEVICE)
        out.append(model(x).float().cpu()[:, :31])
    return torch.cat(out).numpy()


def run_inference(man):
    models = {(c, s): (lambda c=c, s=s: load(c, s)[0]) for c in CONFIGS for s in SEEDS}
    models[("authors", 0)] = load_authors_model
    for (c, s), make in models.items():
        f = OUT / "logits" / f"{c}_seed{s}.npz"
        if f.exists():
            continue
        f.parent.mkdir(parents=True, exist_ok=True)
        m = make()
        np.savez(f, clinic=logits(m, [D7 / p for p in man.clinic]), derm=logits(m, [D7 / p for p in man.derm]))
        print("done", c, s)


def estimands(pc, pd_, cls):
    """pc, pd_: argmax predictions (n,) for clinical and dermoscopic images of the same cases."""
    p1 = np.isin(pd_, ISIC).mean() - np.isin(pc, ISIC).mean()
    mel = cls == "MEL"
    p2 = ((pd_[mel] == MEL_DERM).mean() - (pc[mel] == MEL_DERM).mean()
          + (pc[mel] == MEL_CLIN).mean() - (pd_[mel] == MEL_CLIN).mean())
    return p1, p2


def bootstrap(preds, cls):
    """preds: list over seeds of (pc, pd). Per resample: estimand per seed, then mean over seeds."""
    rng = np.random.default_rng(RNG_SEED)
    n = len(cls)
    out = []
    while len(out) < B:
        idx = rng.integers(0, n, n)
        if (cls[idx] == "MEL").sum() == 0:
            continue
        out.append(np.mean([estimands(pc[idx], pd_[idx], cls[idx]) for pc, pd_ in preds], axis=0))
    return np.percentile(np.array(out), [2.5, 97.5], axis=0)


def verdict(per_seed, lo, hi):
    if all(v > 0 for v in per_seed) and lo > 0:
        return "supported"
    if all(v < 0 for v in per_seed) and hi < 0:
        return "reversed"
    return "no consistent evidence"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locked-commit", required=True)
    args = ap.parse_args()
    verify_inputs()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "LOCKED_COMMIT.txt").write_text(args.locked_commit)
    man = pd.read_csv(PRE / "derm7pt_manifest.csv")
    excluded = PRE / "derm7pt_excluded_cases.csv"  # from the visual duplicate review, if any
    if excluded.exists():
        man = man[~man.case_num.isin(pd.read_csv(excluded).case_num)]
    cls = man.cls.to_numpy()
    run_inference(man)

    rows, sec = [], []
    groups = {c: [(c, s) for s in SEEDS] for c in CONFIGS}
    groups["authors"] = [("authors", 0)]
    for g, keys in groups.items():
        preds = []
        for c, s in keys:
            z = np.load(OUT / "logits" / f"{c}_seed{s}.npz")
            preds.append((z["clinic"].argmax(1), z["derm"].argmax(1)))
        per_seed = [estimands(pc, pd_, cls) for pc, pd_ in preds]
        (lo1, lo2), (hi1, hi2) = bootstrap(preds, cls)
        for k, (name, lo, hi) in enumerate([("P1", lo1, hi1), ("P2", lo2, hi2)]):
            vals = [v[k] for v in per_seed]
            rows.append({"model": g, "estimand": name, "mean": np.mean(vals), "sd": np.std(vals, ddof=1) if len(vals) > 1 else np.nan,
                         "per_seed": vals, "ci_low": lo, "ci_high": hi,
                         "verdict": verdict(vals, lo, hi) if g == PRIMARY else "(robustness)"})
        for (c, s), (pc, pd_) in zip(keys, preds):  # secondary S1, S2
            for dx in CORRECT:
                m = cls == dx
                sec.append({"model": c, "seed": s, "dx": dx,
                            "recall_clinic": np.isin(pc[m], list(CORRECT[dx])).mean(),
                            "recall_derm": np.isin(pd_[m], list(CORRECT[dx])).mean(),
                            "isic_rate_clinic": np.isin(pc[m], ISIC).mean(),
                            "isic_rate_derm": np.isin(pd_[m], ISIC).mean(),
                            "same_pred": (pc[m] == pd_[m]).mean(), "n": int(m.sum())})
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "primary.csv", index=False)
    pd.DataFrame(sec).to_csv(OUT / "secondary.csv", index=False)
    print(r.drop(columns="per_seed").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
