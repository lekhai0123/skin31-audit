# Audit of a composite 31-class skin disease benchmark

Code, image lists and results for:

> Khai-Thanh Le, Pham Trong Huynh. *Same Lesion, Different Answer: Label Errors and Source–Modality Confounding in a Composite Skin Disease Benchmark.* (manuscript)

**No images are included.** All images must be obtained from their original sources (see *Data*). This repository contains identifiers, hashes, corrected label lists, analysis code, model predictions and checkpoint hashes.

Reported metrics are computed from archived checkpoints. `scripts/reproduce.py` reproduces internal benchmark inference; the external and intervention scripts evaluate their respective cohorts. The release includes cached logits and checkpoint SHA256 hashes, so core statistical results can also be verified without the original images or model weights.

## Contents

| Folder | Content |
|---|---|
| `scripts/` | Audit, evaluation and figure scripts (see table below) |
| `code/skin_disease/` | Model constructors and transforms required for inference; legacy general training utilities are also included |
| `scripts/figures/` | Scripts that produce the paper figures |
| `prereg/` | Analysis plans exactly as registered on OSF, with the manifests and hashes they reference |
| `lists/` | Corrected benchmark (`skin31_clean`) and intervention datasets (`skin31_deconf` = arm H, `skin31_milkB`, `skin31_milkC`, `skin31_relabel` = arm R): per-image manifest with the reason for every change, class counts before/after, and the added HAM10000/MILK10k images |
| `outputs/` | Results tables and model outputs (logits) behind every number in the paper |
| `checkpoints/` | SHA256 hashes of all model checkpoints, including arm R and the ViT-S/16 arms B and C (weights are not included) |
| `graphical_abstract/` | Sources and licences of the ISIC Archive images used in the graphical abstract |
| `notebooks/` | Archived Kaggle working template and a separately labelled portable adaptation (see `notebooks/README.md`) |

## Pre-registrations

| Analysis | OSF | Evaluation script |
|---|---|---|
| Derm7pt paired-lesion test | [osf.io/jzsr9](https://osf.io/jzsr9) (replaces osf.io/643ht, see below) | `scripts/derm7pt_eval.py` |
| PAD-UFES-20 external testing | [osf.io/hn7yk](https://osf.io/hn7yk) | `scripts/pad_eval.py` |
| HAM10000 intervention (arm H) | [osf.io/pe8q6](https://osf.io/pe8q6) | `scripts/deconf_eval.py` |
| MILK10k matched control (arms B, C) | [osf.io/5ksja](https://osf.io/5ksja) | `scripts/milk_eval.py` |

The Derm7pt plan was registered as osf.io/643ht on 30 September 2026. That registration was withdrawn by mistake on 9 October 2026 while its embargo was being ended; the byte-identical files were re-deposited as osf.io/jzsr9 (its description records the replacement). Registered files that predate the replacement, such as the PAD and HAM10000 plans, still cite osf.io/643ht and are left unchanged.

Files in `prereg/` and the evaluation scripts above preserve the registered versions. Complete, unmodified local registration bundles and their original SHA256 lists are also in `prereg/registered/`. PAD was registered after an exploratory evaluation; arm R, ViT, overall diagnostic performance and later sensitivity analyses are exploratory. Deviations and timing are reported in the Supplementary material.

## Reproduce from cached predictions (no images or checkpoints)

Use Python 3.12. Create a virtual environment and install `requirements-cached.txt`, then run from the repository root:

```bash
python scripts/verify_cached_results.py --bootstrap --out verification
```

This loads the numeric prediction arrays with `allow_pickle=False`, checks their shape and finiteness, validates the unchanged local registration bundles, and recomputes the paired E1/E2/E3 estimates and 2,000 lesion-bootstrap CIs for R−A, H−A and C−B on B0 and ViT. It checks them against the archived result tables. It does not download images, load model weights or retrain models. `lists/class_mapping_clean.json` records the 30-output order independently of an image folder.

## Reproduce inference or training

Install `requirements.txt` in a separate environment matching the recorded evaluation versions. Obtain the datasets from the sources below and use the released manifests/build scripts to reconstruct the partitions. For inference, obtain the exact checkpoints listed in `checkpoints/` and put them at the paths recorded in those manifests. Weights are not bundled; until a checkpoint download is supplied, the release supports cached statistical verification but not independent image-to-logit reproduction from a fresh clone. The authors' released model is expected at `ref_efficientskindis/Weights.h5`; see the source article's code repository.

The inference scripts use local source images even when caches are present, and some intervention inference functions require CUDA. They are not the image-free entry point above. Historical `--locked-commit` values are recorded in `outputs/*/LOCKED_COMMIT.txt`; do not overwrite the original provenance when running new analyses.

For training, read `notebooks/README.md`: the archived notebook needs private module/path configuration, while `training_kaggle_portable.ipynb` is an adaptation prepared after the experiments. It has been checked for Python syntax but has not been GPU retrained during preflight. Keep exploratory additions separate from the registered analyses.

The general training helpers in `code/skin_disease/experiment.py` and `datasets.py` are legacy utilities, not the executable record of the reported runs. Their defaults and RNG/loader sequencing must not be assumed to match the archived notebook. Use the documented notebook protocol for reproduction; the supplied checkpoint hashes and logits identify the actual runs.

## Scripts

| Script | Purpose |
|---|---|
| `dup_check.py` | Exact and near-duplicate detection (pixel MD5, pHash) within the benchmark and against Derm7pt / PAD-UFES-20 |
| `upstream_compare.py` | Pixel identity with the Kaggle nine-class ISIC compilation |
| `isic_label_audit.py` | Benchmark labels vs ISIC Archive diagnoses |
| `isic_lesion_leakage.py` | Lesion-level train–test leakage (ISIC `lesion_id`) |
| `authors_model_check.py` | Original authors' released model on the public partitions |
| `reproduce.py` | Recompute benchmark metrics of the 21 checkpoints |
| `label_error_impact.py`, `leak_gain_bootstrap.py` | Effect of label errors and leakage on internal accuracy |
| `embed_leak_check.py` | DINOv2 similarity of atlas test images to training images |
| `atlas_similarity_accuracy.py` | Descriptive atlas accuracy by similarity from FP32 cached logits; historical AMP table retained separately |
| `isic_unknown_id_sensitivity.py` | ISIC internal accuracy/leakage sensitivity excluding images without lesion IDs, from cached outputs |
| `build_clean_skin31.py`, `build_deconf_skin31.py`, `build_milk_arms.py` | Build the corrected and intervention datasets |
| `derm7pt_eval.py`, `derm7pt_eval_clean.py`, `pad_eval.py`, `deconf_eval.py`, `milk_eval.py`, `vit_eval.py` | Evaluations |
| `build_relabel_skin31.py`, `relabel_eval.py`, `relabel_identity_check.py` | Relabelling arm R (exploratory) |
| `vit_milk_eval.py` | Arms B and C repeated with ViT-S/16 (exploratory; Supplementary Table S13) |
| `intervention_tradeoffs.py`, `per_class_recall.py` | Trade-offs, per-seed values, per-class recall, paired test accuracy (Supplementary Tables S7–S10, S17) |
| `derm7pt_overall.py` | Per-diagnosis recall, macro recall and accuracy on Derm7pt for arms B and C (Supplementary Table S14) |
| `internal_accuracy.py` | Internal benchmark accuracy of every model from saved logits (Supplementary Table S4) |
| `seed_sensitivity.py` | Leave-one-seed-out sensitivity of mean contrasts (post hoc; Supplementary Table S15) |
| `e3_decomposition.py` | Change in E3 split into its dermoscopic and clinical components (Supplementary Table S16) |
| `grouping_check.py` | Modality gap of original and corrected models on common outputs (Supplementary Section S10) |
| `exploratory_sensitivity.py`, `review_sensitivity.py` | Probability-mass, biopsy-confirmed, summed-probability and atlas-exclusion sensitivity analyses |
| `ga_image_candidates.py` | Search for openly licensed ISIC images for the graphical abstract |
| `check_mendeley32.py` | Check of the 2026 derivative dataset |
| `make_external_review.py`, `make_ham_review.py` | Blind visual review pages for duplicate candidates |
| `build_bib.py`, `figures/*.py` | Paper references and figures |

## Visual reviews

The image-free descriptive sensitivity tables can be regenerated separately:

```bash
python scripts/atlas_similarity_accuracy.py
python scripts/isic_unknown_id_sensitivity.py
```

The atlas helper uses the canonical FP32 caches. `outputs/embed_leak/accuracy_by_similarity_historical_AMP.csv` preserves the earlier AMP table for provenance; use `accuracy_by_similarity.csv` for the current paper.

All candidate duplicate pairs between the benchmark and the external test sets (96 pairs) and between the added HAM10000 images and Derm7pt / PAD-UFES-20 / benchmark test images (62 pairs) were reviewed blind by an author: `outputs/dup_check/review_external/external_review_rating.csv`, `outputs/deconf/review_ham/ham_review_rating.csv`. MILK10k candidates were excluded by an objective rule (DINOv2 cosine ≥ 0.80; `prereg/milk_excluded_lesions.csv`). The atlas similarity review (one author, descriptive) is in `outputs/embed_leak/review/rating_final_optionB.csv`.

## Data

Place the datasets under `data/` as follows (paths used by the scripts):

| Dataset | Source | Licence | Path |
|---|---|---|---|
| 31-class benchmark (public copy) | Kaggle `kelixo25/31-classes-of-skin-disease` | CC BY-SA 4.0 | `data/Atlas dan ISIC2019 (31 classes)/{train,test}/<class>/` |
| Nine-class ISIC compilation | Kaggle `nodoubttome/skin-cancer9-classesisic` | see source | `data/kaggle_cache/` |
| Derm7pt | [derm.cs.sfu.ca](https://derm.cs.sfu.ca) | CC BY-NC-ND 4.0; see source access terms | `data/release_v0/` |
| PAD-UFES-20 | [Mendeley Data](https://data.mendeley.com/datasets/zr7vgbcyr2/1) | CC BY 4.0 | `data/archive/` |
| HAM10000 | [Harvard Dataverse](https://doi.org/10.7910/DVN/DBW86T) | CC BY-NC 4.0 | `data/kaggle_cache/` |
| MILK10k | [ISIC Archive](https://doi.org/10.34970/648456) | CC BY-NC | `data/milk10k/` |
| 32-class derivative | [Mendeley Data](https://doi.org/10.17632/pgd42j3h5c.2) | CC BY 4.0 | `data/mendeley32/` |

ISIC diagnoses are queried from the ISIC Archive API (`https://api.isic-archive.com/api/v2/images/<ISIC_ID>/`).
Paths in the released CSV files are relative to the project root.

## Environment

Evaluation: Python 3.12, PyTorch 2.10.0, torchvision 0.25.0, timm 1.0.26, Pillow 11.3.0, NumPy 2.0.2 (see `requirements.txt`). Training: Kaggle notebook in `notebooks/` (NVIDIA T4, PyTorch 2.10.0 + CUDA 12.8). The training set of every model is the concatenation of seven fixed augmented views of each image, sampled with inverse-class-frequency weights (see Supplementary Section S13 of the paper).

## Licence

Code: MIT (`LICENSE`). Lists, manifests and results: see `LICENSE-DATA`.

## Citation

Archived release: https://doi.org/10.5281/zenodo.23268495 . Development repository: https://github.com/lekhai0123/skin31-audit . See `CITATION.cff`.
