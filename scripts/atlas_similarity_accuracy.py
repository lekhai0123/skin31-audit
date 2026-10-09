"""Recompute Supplementary Table S2 from saved FP32 predictions, without images or inference.

Usage: python scripts/atlas_similarity_accuracy.py [--output PATH]
Legacy AMP-derived values are retained separately as accuracy_by_similarity_historical_AMP.csv.
"""
import argparse
import csv
import json
from pathlib import Path, PurePosixPath

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [42, 123, 2024]
MODELS = ["1_b2_baseline", "2_b2_trainopt", "4_b0_baseline", "5_b0_trainopt", "7_b0_kd_fusion"]
FIELDS = ["", "all", "Atlas >=0.93", "Atlas 0.90-0.93", "Atlas 0.85-0.90", "Atlas <0.85", "ISIC all"]


def read(rel):
    with (ROOT / rel).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def key(value):
    p = PurePosixPath(str(value).replace("\\", "/"))
    return p.parent.name.lower(), p.name


def summary(correct, keys, similarities):
    cosine = np.array([similarities[k] for k in keys])
    atlas = np.array([not filename.startswith("ISIC_") for _, filename in keys])
    masks = [np.ones(len(keys), dtype=bool), atlas & (cosine >= 0.93),
             atlas & (cosine >= 0.90) & (cosine < 0.93),
             atlas & (cosine >= 0.85) & (cosine < 0.90), atlas & (cosine < 0.85), ~atlas]
    return {field: float(correct[:, mask].mean() * 100) for field, mask in zip(FIELDS[1:], masks)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/embed_leak/accuracy_by_similarity.csv")
    args = parser.parse_args()
    flags = read("outputs/repro/skin31_test_flags.csv")
    strict = np.array([r["strict"].lower() == "true" for r in flags])
    rows = [r for r in flags if r["strict"].lower() == "true"]
    strict_keys = [key(r["path"]) for r in rows]
    truth = np.array([folder for folder, _ in strict_keys])
    assert len(rows) == 863 and len(set(strict_keys)) == 863
    mapping = json.loads((ROOT / "prereg/class_mapping.json").read_text(encoding="utf-8"))["class_to_idx"]
    original_names = np.array([name.lower() for name, _ in sorted(mapping.items(), key=lambda kv: kv[1])])
    clean_names = np.array([name for name in original_names if name != "seborrheic keratosis"])
    similarities = {key(r["test"]): float(r["cosine"]) for r in read("outputs/embed_leak/top_match_per_test_image.csv")}
    results = []
    for model in MODELS:
        correct = np.stack([original_names[np.load(ROOT / f"outputs/repro/{model}/seed{s}/skin31_test_fp32.npy").argmax(1)[strict]] == truth for s in SEEDS])
        results.append({"": model, **summary(correct, strict_keys, similarities)})
    test_manifest = [r for r in read("lists/skin31_clean/manifest.csv") if r["split"] == "test" and r["dst"]]
    # This is ImageFolder's class/filename order. No source images are needed.
    clean_keys = [key(r["dst"]) for r in sorted(test_manifest, key=lambda r: r["dst"])]
    assert len(clean_keys) == 863 and set(clean_keys) == set(strict_keys)
    index = {k: i for i, k in enumerate(clean_keys)}
    order = np.array([index[k] for k in strict_keys])
    for model in ["4_b0_baseline", "5_b0_trainopt"]:
        correct = np.stack([clean_names[np.load(ROOT / f"outputs/clean_eval/{model}_seed{s}_fp32.npy").argmax(1)[order]] == truth for s in SEEDS])
        results.append({"": "clean " + model, **summary(correct, strict_keys, similarities)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(results)
    print(f"Recomputed {len(results)} FP32 rows from cached predictions: {args.output}")


if __name__ == "__main__":
    main()
