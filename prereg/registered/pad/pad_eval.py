"""PAD-UFES-20 evaluation implementing prereg/pad_analysis_plan.md.

Refuses to run without --locked-commit. Retrained (corrected-data) models are evaluated if their checkpoints
exist in checkpoints_clean/<config>/seed<k>/*.pth; otherwise they are skipped and can be added by rerunning.
Usage: python scripts/pad_eval.py --locked-commit <osf id>
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from authors_model_check import load_authors_model  # noqa: E402
from reproduce import CONFIGS, DEVICE, SEEDS, load  # noqa: E402
from skin_disease.transforms import build_eval_transform  # noqa: E402

PRE = ROOT / "prereg"
PAD = ROOT / "data" / "archive"
OUT = ROOT / "outputs" / "pad"
CLEAN_CKPT = ROOT / "checkpoints_clean"
CLEAN_CONFIGS = ["4_b0_baseline", "5_b0_trainopt"]
PAD6 = ["ACK", "BCC", "MEL", "NEV", "SEK", "SCC"]
TARGET = {"ACK": ["actinic keratosis"], "BCC": ["basal cell carcinoma"],
          "MEL": ["melanoma", "seborrheic keratosis"], "NEV": ["nevus"],
          "SEK": ["pigmented benign keratosis"], "SCC": ["squamous cell carcinoma"]}
ISIC_NAMES = {"actinic keratosis", "dermatofibroma", "nevus", "pigmented benign keratosis",
              "seborrheic keratosis", "squamous cell carcinoma", "vascular lesion"}
B, RNG_SEED = 2000, 20260930


def class_names(kind):
    if kind == "clean":
        return [p.name.lower() for p in sorted((ROOT / "data" / "skin31_clean" / "train").iterdir())]
    m = json.loads((PRE / "class_mapping.json").read_text(encoding="utf-8"))["class_to_idx"]
    return [k.lower() for k, _ in sorted(m.items(), key=lambda kv: kv[1])]


def load_clean(cfg, seed):
    path = next((CLEAN_CKPT / cfg / f"seed{seed}").glob("*.pth"))
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=30)
    missing, unexpected = model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=False)
    assert not missing and not unexpected, path
    return model.to(DEVICE).eval()


@torch.inference_mode()
def logits(model, paths, n_out):
    tf = build_eval_transform(224)
    out = []
    for i in range(0, len(paths), 32):
        x = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i + 32]]).to(DEVICE)
        out.append(model(x).float().cpu()[:, :n_out])
    return torch.cat(out).numpy()


def metrics(z, names, y):
    """z: logits (n, n_out); y: PAD class index 0..5 (PAD6 order)."""
    cols, owner = [], []
    for k, dx in enumerate(PAD6):
        for nm in TARGET[dx]:
            if nm in names:
                cols.append(names.index(nm))
                owner.append(k)
    owner = np.array(owner)
    zr = z[:, cols]
    pred6 = owner[zr.argmax(1)]
    pr = np.exp(zr - zr.max(1, keepdims=True))
    pr /= pr.sum(1, keepdims=True)
    score = np.stack([pr[:, owner == k].sum(1) for k in range(6)], 1)
    rec = np.array([(pred6[y == k] == k).mean() for k in range(6)])
    full = z.argmax(1)
    ok_full = np.array([names[p] in TARGET[PAD6[t]] for p, t in zip(full, y)])
    rec_full = np.array([ok_full[y == k].mean() for k in range(6)])
    auc = np.array([roc_auc_score(y == k, score[:, k]) for k in range(6)])
    isic = np.array([names[p] in ISIC_NAMES for p in full])
    return {"bal_acc_6way": rec.mean(), "macro_auroc_6way": auc.mean(), "bal_acc_full": rec_full.mean(),
            "bal_acc_5way_noSEK": rec[[0, 1, 2, 3, 5]].mean(), "isic_rate": isic.mean(),
            **{f"recall_{d}": r for d, r in zip(PAD6, rec)}, **{f"auroc_{d}": a for d, a in zip(PAD6, auc)},
            **{f"isic_rate_{d}": isic[y == k].mean() for k, d in enumerate(PAD6)}}


def patient_resamples(pid, y):
    rng = np.random.default_rng(RNG_SEED)
    groups = pd.Series(np.arange(len(pid))).groupby(pid).apply(list)
    keys, out, rejected = groups.index.to_numpy(), [], 0
    while len(out) < B:
        idx = np.concatenate(groups.loc[rng.choice(keys, keys.size)].to_numpy())
        if np.unique(y[idx]).size < 6:
            rejected += 1
            continue
        out.append(idx)
    return out, rejected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locked-commit", required=True)
    args = ap.parse_args()
    sums = dict(l.split(" *")[::-1] for l in (PRE / "SHA256SUMS_pad.txt").read_text().split("\n") if l.strip())
    for name, sha in sums.items():
        assert hashlib.sha256((PRE / name).read_bytes()).hexdigest() == sha, f"{name} changed"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "LOCKED_COMMIT.txt").write_text(args.locked_commit)

    man = pd.read_csv(PRE / "pad_manifest.csv")
    files = {p.name: p for p in PAD.rglob("*.png")}
    paths = [files[i] for i in man.img_id]
    y_all = man.diagnostic.map({d: k for k, d in enumerate(PAD6)}).to_numpy()
    cohorts = {"main": (man.exclude.fillna("") == "").to_numpy(), "all2298": np.ones(len(man), bool)}

    models = {}
    for c in CONFIGS:
        for s in SEEDS:
            models[("orig", c, s)] = (lambda c=c, s=s: load(c, s)[0], "orig", 31)
    models[("authors", "authors", 0)] = (load_authors_model, "orig", 31)
    for c in CLEAN_CONFIGS:
        for s in SEEDS:
            if list((CLEAN_CKPT / c / f"seed{s}").glob("*.pth")):
                models[("clean", c, s)] = (lambda c=c, s=s: load_clean(c, s), "clean", 30)
    for key, (make, kind, n_out) in models.items():
        f = OUT / "logits" / f"{key[0]}_{key[1]}_seed{key[2]}.npy"
        if not f.exists():
            f.parent.mkdir(parents=True, exist_ok=True)
            np.save(f, logits(make(), paths, n_out))
            print("inferred", key)

    rows, per_seed = [], {}
    for cname, mask in cohorts.items():
        y, pid = y_all[mask], man.patient_id.to_numpy()[mask]
        res, rejected = patient_resamples(pid, y)
        groups = {}
        for (src, c, s), (_, kind, _) in models.items():
            groups.setdefault((src, c, kind), []).append(s)
        for (src, c, kind), seeds in groups.items():
            names = class_names(kind)
            zs = {s: np.load(OUT / "logits" / f"{src}_{c}_seed{s}.npy")[mask] for s in seeds}
            point = {s: metrics(zs[s], names, y) for s in seeds}
            per_seed[(cname, src, c)] = point
            boot = {m: [] for m in ("bal_acc_6way", "macro_auroc_6way")}
            for idx in res:
                bm = [metrics(zs[s][idx], names, y[idx]) for s in seeds]
                for m in boot:
                    boot[m].append(np.mean([b[m] for b in bm]))
            for m, v in point[seeds[0]].items():
                vals = [point[s][m] for s in seeds]
                row = {"cohort": cname, "source": src, "config": c, "metric": m, "mean": np.mean(vals),
                       "sd": np.std(vals, ddof=1) if len(vals) > 1 else np.nan}
                if m in boot:
                    row["ci_low"], row["ci_high"] = np.percentile(boot[m], [2.5, 97.5])
                rows.append(row)
            print(cname, src, c, "done")
        # Q2 contrasts: corrected minus original, paired on patient resamples and seed number
        for c in CLEAN_CONFIGS:
            if ("clean", c, "clean") not in groups:
                continue
            seeds = groups[("clean", c, "clean")]
            zc = {s: np.load(OUT / "logits" / f"clean_{c}_seed{s}.npy")[mask] for s in seeds}
            zo = {s: np.load(OUT / "logits" / f"orig_{c}_seed{s}.npy")[mask] for s in seeds}
            for m in ("bal_acc_6way", "macro_auroc_6way"):
                d_seed = [metrics(zc[s], class_names("clean"), y)[m] - metrics(zo[s], class_names("orig"), y)[m] for s in seeds]
                d_boot = [np.mean([metrics(zc[s][i], class_names("clean"), y[i])[m] - metrics(zo[s][i], class_names("orig"), y[i])[m]
                                   for s in seeds]) for i in res]
                lo, hi = np.percentile(d_boot, [2.5, 97.5])
                v = ("improved" if all(d > 0 for d in d_seed) and lo > 0 else
                     "worse" if all(d < 0 for d in d_seed) and hi < 0 else "no consistent evidence of a difference")
                rows.append({"cohort": cname, "source": "corrected-original", "config": c, "metric": f"delta_{m}",
                             "mean": np.mean(d_seed), "per_seed": d_seed, "ci_low": lo, "ci_high": hi, "verdict": v})
        print(cname, "rejected resamples:", rejected)
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "results.csv", index=False)
    show = r[r.metric.isin(["bal_acc_6way", "macro_auroc_6way", "isic_rate", "bal_acc_full"]) | r.metric.str.startswith("delta")]
    print(show.drop(columns=[c for c in ("per_seed",) if c in show]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
