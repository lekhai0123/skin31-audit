"""E-02 (balanced-view reference) data preparation: folds, view assignment per arm and image folders.
No model is trained or evaluated here. Plan: prereg/e02_balanced_view_plan.md.

Outputs
  prereg/e02_folds.csv            case -> fold (five folds, stratified by diagnosis, RNG seed 20260930)
  prereg/e02_arm_assignment.csv   fold, arm, training case -> view (clinic or derm)
  data/e02/fold{f}/{ARM}/train/{class}/...   one image per training case, the view given by the assignment
  data/e02/fold{f}/{ARM}/test/{class}/...    both views of every test case of the fold
Images are hard links to data/release_v0/images (Derm7pt is not copied or redistributed); file names are
"{case:04d}_{view}_{original name}".
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
MAN = ROOT / "prereg" / "derm7pt_manifest.csv"
IMG = ROOT / "data" / "release_v0" / "images"
DATA = ROOT / "data" / "e02"
FOLDS_CSV = ROOT / "prereg" / "e02_folds.csv"
ASSIGN_CSV = ROOT / "prereg" / "e02_arm_assignment.csv"
SEED, K = 20260930, 5
CLASSES = ["BCC", "DF", "MEL", "NEV", "SEK", "VASC"]
TARGET = {"BCC", "MEL"}
ARMS = ["COUPLED", "BALANCED"]


def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        os.link(src, dst)


def main():
    man = pd.read_csv(MAN)
    assert len(man) == 960 and sorted(man.cls.unique()) == CLASSES and man.case_num.is_unique

    # folds
    fold = np.full(len(man), -1)
    for f, (_, te) in enumerate(StratifiedKFold(K, shuffle=True, random_state=SEED).split(np.zeros(len(man)), man.cls)):
        fold[te] = f
    assert (fold >= 0).all()
    folds = man[["case_num", "cls"]].assign(fold=fold)

    # view assignment
    rows = []
    for f in range(K):
        tr = folds[folds.fold != f]
        for r in tr.itertuples():
            rows.append((f, "COUPLED", r.case_num, r.cls, "clinic" if r.cls in TARGET else "derm"))
        rng = np.random.default_rng(SEED + f)
        for c in CLASSES:
            cases = rng.permutation(tr[tr.cls == c].case_num.to_numpy())
            n_clin = len(cases) // 2
            rows += [(f, "BALANCED", int(cn), c, "clinic" if i < n_clin else "derm") for i, cn in enumerate(cases)]
    assign = pd.DataFrame(rows, columns=["fold", "arm", "case_num", "cls", "view"])
    assign = assign.sort_values(["fold", "arm", "case_num"]).reset_index(drop=True)

    # checks required by the plan
    for f in range(K):
        te = set(folds.case_num[folds.fold == f])
        for arm in ARMS:
            a = assign[(assign.fold == f) & (assign.arm == arm)]
            assert not (set(a.case_num) & te), "test case in training"
            assert set(a.case_num) == set(folds.case_num[folds.fold != f])
        cc = assign[(assign.fold == f) & (assign.arm == "COUPLED")].cls.value_counts().sort_index()
        cb = assign[(assign.fold == f) & (assign.arm == "BALANCED")].cls.value_counts().sort_index()
        assert cc.equals(cb), "class counts differ between arms"
        b = assign[(assign.fold == f) & (assign.arm == "BALANCED")]
        for c in CLASSES:
            n = (b.cls == c).sum()
            assert ((b.cls == c) & (b.view == "clinic")).sum() == n // 2
    assert folds.groupby("case_num").size().eq(1).all()

    folds.to_csv(FOLDS_CSV, index=False)
    assign.to_csv(ASSIGN_CSV, index=False)

    # image folders (hard links)
    paths = man.set_index("case_num")[["clinic", "derm"]]
    for f in range(K):
        test_cases = folds[folds.fold == f]
        for arm in ARMS:
            base = DATA / f"fold{f}" / arm
            for r in assign[(assign.fold == f) & (assign.arm == arm)].itertuples():
                rel = paths.loc[r.case_num, r.view]
                link(IMG / rel, base / "train" / r.cls / f"{r.case_num:04d}_{r.view}_{Path(rel).name}")
            for r in test_cases.itertuples():
                for view in ("clinic", "derm"):
                    rel = paths.loc[r.case_num, view]
                    link(IMG / rel, base / "test" / r.cls / f"{r.case_num:04d}_{view}_{Path(rel).name}")
    # summary
    for f in range(K):
        n_te = (folds.fold == f).sum()
        n_tr = {arm: sum(1 for _ in (DATA / f"fold{f}" / arm / "train").rglob("*") if _.is_file()) for arm in ARMS}
        n_test_files = sum(1 for _ in (DATA / f"fold{f}" / "COUPLED" / "test").rglob("*") if _.is_file())
        print(f"fold {f}: test cases {n_te} ({n_test_files} images) | train images {n_tr}")
    print("wrote", FOLDS_CSV.name, "and", ASSIGN_CSV.name)


if __name__ == "__main__":
    main()
