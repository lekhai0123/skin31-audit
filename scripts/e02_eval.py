"""E-02 evaluation (plan: prereg/e02_balanced_view_plan.md, Sections 5 and 6).

predict : FP32 predictions of every final-epoch checkpoint on both views of its fold's test cases
          (resize to 224 x 224 and ImageNet normalisation, as for every other evaluation in the paper)
analyze : estimands per arm and seed, COUPLED - BALANCED contrasts, case bootstrap, decision rules, wording

Usage (datn environment):
  python scripts/e02_eval.py predict
  python scripts/e02_eval.py analyze [--pred-dir DIR --out-dir DIR]
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "outputs" / "e02" / "runs"
DATA = ROOT / "data" / "e02"
PRED = ROOT / "outputs" / "e02" / "predictions"
RES = ROOT / "outputs" / "e02" / "results"
FOLDS_CSV = ROOT / "prereg" / "e02_folds.csv"
CLASSES = ["BCC", "DF", "MEL", "NEV", "SEK", "VASC"]
G, T = ["NEV", "SEK", "VASC", "DF"], ["BCC", "MEL"]
SEEDS, ARMS, K = [42, 123, 2024], ["COUPLED", "BALANCED"], 5
B, RNG_SEED = 2000, 20260930
CKPT = "final_b0_training_optimization_final.pth"
NAMES = ["P1", "E2", "E3", "E1", "clin_recall_T", "BA_clinic", "BA_derm"]
EXPECT = {"P1": "C>B", "E2": "C>B", "E3": "C>B", "E1": "B>C"}  # COUPLED vs BALANCED
PRIMARY = ["P1", "E2"]


# ----------------------------------------------------------------------------------------------- predict
def predict():
    import timm
    import torch
    from PIL import Image
    from torchvision import transforms

    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PRED.mkdir(parents=True, exist_ok=True)
    pat = re.compile(r"^(\d{4})_(clinic|derm)_")
    for arm in ARMS:
        for f in range(K):
            files = sorted(p for p in (DATA / f"fold{f}" / arm / "test").rglob("*") if p.is_file())
            for seed in SEEDS:
                run = RUNS / arm / f"fold{f}" / f"seed{seed}"
                good = [a for a in sorted(run.glob("attempt*")) if (a / CKPT).exists()
                        and pd.read_csv(a / "experiment_summary.csv").status.iloc[0] == "ok"]
                assert good, f"no finished run for {arm} fold{f} seed{seed}"
                att = good[0]
                classes = json.load(open(att / "class_names_skin31.json", encoding="utf-8"))["class_names"]
                assert classes == CLASSES, classes
                m = timm.create_model("efficientnet_b0", pretrained=False, num_classes=len(CLASSES))
                m.load_state_dict(torch.load(att / CKPT, map_location="cpu", weights_only=True), strict=True)
                m = m.to(dev).eval()
                rows = []
                with torch.inference_mode():
                    for i in range(0, len(files), 32):
                        batch = files[i:i + 32]
                        x = torch.stack([tf(Image.open(p).convert("RGB")) for p in batch]).to(dev)
                        z = m(x).float().cpu().numpy()
                        for p, zz in zip(batch, z):
                            mm = pat.match(p.name)
                            rows.append(dict(case_num=int(mm.group(1)), view=mm.group(2), cls=p.parent.name,
                                             pred=CLASSES[int(zz.argmax())],
                                             **{f"logit_{c}": float(v) for c, v in zip(CLASSES, zz)}))
                pd.DataFrame(rows).to_csv(PRED / f"{arm}_fold{f}_seed{seed}.csv", index=False)
                print("predicted", arm, f, seed, len(rows), flush=True)


# ----------------------------------------------------------------------------------------------- analyze
def load_predictions(pred_dir):
    folds = pd.read_csv(FOLDS_CSV)
    out = {}
    for arm in ARMS:
        for seed in SEEDS:
            d = pd.concat([pd.read_csv(pred_dir / f"{arm}_fold{f}_seed{seed}.csv").assign(fold=f) for f in range(K)])
            # every case exactly once per view, from the fold in which it was a test case
            chk = d.merge(folds, on="case_num", suffixes=("", "_ref"))
            assert (chk.fold == chk.fold_ref).all() and (chk.cls == chk.cls_ref).all()
            w = d.pivot(index="case_num", columns="view", values="pred").reindex(folds.case_num)
            assert w.notna().all().all() and len(w) == 960
            out[(arm, seed)] = w
    return folds.set_index("case_num").cls.reindex(out[(ARMS[0], SEEDS[0])].index).to_numpy(), out


def indicators(w, cls):
    pc, pd_ = w["clinic"].to_numpy(), w["derm"].to_numpy()
    return (np.isin(pc, G).astype(float), np.isin(pd_, G).astype(float),
            (pc == cls).astype(float), (pd_ == cls).astype(float), pc, pd_)


def estimands(ind, cls, idx):
    gc, gd, cc, cd, pc, pd_ = (a[idx] for a in ind)
    y = cls[idx]
    diff = gd - gc
    t, g = np.isin(y, T), np.isin(y, G)
    e1 = np.mean([cd[y == k].mean() for k in T])
    c1 = np.mean([cc[y == k].mean() for k in T])
    ba_c = np.mean([cc[y == k].mean() for k in CLASSES])
    ba_d = np.mean([cd[y == k].mean() for k in CLASSES])
    return np.array([diff.mean(), diff[t].mean(), diff[g].mean(), e1, c1, ba_c, ba_d])


def verdict(per_seed, lo, hi, expect_positive):
    s = 1 if expect_positive else -1
    if all(np.sign(v) == s for v in per_seed) and (lo > 0 if s > 0 else hi < 0):
        return "supported (expected direction)"
    if all(np.sign(v) == -s for v in per_seed) and (hi < 0 if s > 0 else lo > 0):
        return "opposite direction"
    return "no consistent evidence of a difference"


def analyze(pred_dir, out_dir):
    cls, W = load_predictions(pred_dir)
    n = len(cls)
    ind = {k: indicators(w, cls) for k, w in W.items()}
    allidx = np.arange(n)
    est = {k: estimands(v, cls, allidx) for k, v in ind.items()}
    rows = [dict(arm=a, seed=s, estimand=nm, value=est[(a, s)][j]) for (a, s) in est for j, nm in enumerate(NAMES)]
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "e02_estimands.csv", index=False)

    d_seed = np.array([est[("COUPLED", s)] - est[("BALANCED", s)] for s in SEEDS])  # seeds x estimands
    rng = np.random.default_rng(RNG_SEED)
    boots = []
    while len(boots) < B:
        i = rng.integers(0, n, n)
        if any((cls[i] == k).sum() == 0 for k in CLASSES):
            continue
        boots.append(np.mean([estimands(ind[("COUPLED", s)], cls, i) - estimands(ind[("BALANCED", s)], cls, i)
                              for s in SEEDS], axis=0))
    lo, hi = np.percentile(np.array(boots), [2.5, 97.5], axis=0)
    crow = []
    for j, nm in enumerate(NAMES):
        r = dict(estimand=nm, contrast="COUPLED - BALANCED", mean=d_seed[:, j].mean(),
                 **{f"seed{s}": d_seed[k, j] for k, s in enumerate(SEEDS)}, ci_low=lo[j], ci_high=hi[j],
                 role="primary" if nm in PRIMARY else ("secondary" if nm in EXPECT else "descriptive"))
        if nm in EXPECT:
            r["verdict"] = verdict(d_seed[:, j], lo[j], hi[j], expect_positive=(EXPECT[nm] == "C>B"))
        crow.append(r)
    con = pd.DataFrame(crow)
    con.to_csv(out_dir / "e02_contrasts.csv", index=False)

    p1c = np.mean([est[("COUPLED", s)][0] for s in SEEDS])
    p1b = np.mean([est[("BALANCED", s)][0] for s in SEEDS])
    share = p1b / p1c
    sup = all(con.set_index("estimand").loc[e, "verdict"].startswith("supported") for e in PRIMARY)
    if sup and share <= 0.25:
        wording = ("In a controlled six-diagnosis experiment on Derm7pt, coupling the training view to the diagnosis "
                   "was sufficient to produce paired-view routing, and balancing the view within each diagnosis "
                   "removed most of it.")
    elif sup and share < 0.5:
        wording = "...coupling increased paired-view routing; balancing the view reduced it but did not remove it."
    elif sup:
        wording = "...coupling increased paired-view routing, but most of it remained when the view was balanced."
    else:
        wording = ("No consistent evidence that coupling the training view to the diagnosis increases paired-view "
                   "routing in this setting.")
    summary = (f"P1 COUPLED (mean of seeds) = {p1c:.3f}; P1 BALANCED = {p1b:.3f}; residual share = {share:.3f}\n"
               f"primary contrasts supported: {sup}\nwording: {wording}\n"
               "scope: models trained and tested on Derm7pt with six diagnoses; this does not establish how much of "
               "the routing of the benchmark-trained models is due to coupling.\n")
    (out_dir / "e02_summary.txt").write_text(summary, encoding="utf-8")
    print(con.round(3).to_string(index=False))
    print(summary)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["predict", "analyze"])
    ap.add_argument("--pred-dir", type=Path, default=PRED)
    ap.add_argument("--out-dir", type=Path, default=RES)
    a = ap.parse_args()
    predict() if a.step == "predict" else analyze(a.pred_dir, a.out_dir)


if __name__ == "__main__":
    main()
