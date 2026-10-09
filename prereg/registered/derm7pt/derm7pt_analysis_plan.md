# Derm7pt analysis plan (pre-specified)

**Status:** written 30/09/2026, before any model has been run on Derm7pt. Committed to GitHub before evaluation; the commit timestamp is the lock. After the lock, no estimand, mapping, preprocessing or decision rule below is changed. Anything added later is labelled exploratory.

## 1. Question

Do models trained on the 31-class dataset assign a lesion to a class according to its **imaging modality** rather than its **diagnosis**?

Background (established before this plan, see `docs/audit_findings.md`): the 7 ISIC-derived classes of the training set are dermoscopic and the 24 Atlas-derived classes are clinical photographs. The class named "Seborrheic keratosis" (index 25) consists entirely of dermoscopic melanoma, so the models have one clinical melanoma class ("Melanoma", index 14) and one dermoscopic melanoma class (index 25). Derm7pt provides a clinical and a dermoscopic image of **the same lesion**, so modality varies while diagnosis is fixed.

## 2. Data

- `derm7pt_manifest.csv` (SHA256 91b49892f6c9f7e66ac2d9a99bbfd70379303727c4e9b8c5110d480dbd2950f2): 960 cases with both images; the 9 cases whose "clinic" image is a copy of the dermoscopic one, and the lentigo, melanosis and miscellaneous cases, are excluded.
- Diagnosis mapping: all melanoma subtypes → MEL (249); all nevus subtypes → NEV (575); BCC (42); seborrheic keratosis → SEK (45); vascular lesion → VASC (29); dermatofibroma → DF (20).
- Duplicate rule: a case is removed only if one of its images is confirmed by visual review to be an exact or near-duplicate of a training/test image of the 31-class dataset (candidates in `outputs/dup_check/review_derm7pt.html`; no exact MD5 matches were found). The review uses images only, not model outputs. Removed cases are listed.
- Unit of analysis and resampling: the case (lesion). Cases are treated as independent (no patient identifier).

## 3. Models

- The 21 checkpoints in `checkpoint_manifest.csv` (SHA256 307e383bb8bdcf30a127f06d265099e1a658822978850bfdb6685303db392050), 7 configurations × 3 seeds.
- The EfficientSkinDis authors' released model (`Weights.h5`, SHA256 dba4c28ba2ba926c6c27f28a42de92e9a079d52e936771c2a503b69b4d362d19), logits 0–30.
- **Primary configuration:** fusion-distilled B0 (`7_b0_kd_fusion`), all 3 seeds. The other 6 configurations and the authors' model are reported in full as robustness checks.
- Models retrained on a corrected dataset, if any, are evaluated later with this same plan.

## 4. Inference

PIL `.convert("RGB")`; `Resize((224, 224))` bilinear, no aspect preservation, no crop; `ToTensor`; ImageNet normalization; `eval()`, `inference_mode()`, FP32, no TTA. Pre-softmax logits over all 31 outputs are saved. Class order is asserted against `class_mapping.json` (SHA256 46893f03197d7e85075e9336788942d0366e317a54308122e876a9728ed788ea).

## 5. Estimands

Index sets: ISIC-derived outputs I = {0 AK, 3 DF, 18 NEV, 21 PBK, 25 "SK", 26 SCC, 30 VASC}; Atlas-derived outputs A = the other 24. Prediction = argmax over 31 logits.

**P1 (primary). Modality shift in source group.**
r_m = fraction of cases whose prediction is in I, for modality m ∈ {clinical, dermoscopic}.
**Δ1 = r_dermoscopic − r_clinical**, paired over the same cases.
Uses no diagnosis labels. A model that classifies by diagnosis alone gives Δ1 ≈ 0.

**P2 (primary). Melanoma routed by modality.** On MEL cases only:
**Δ2 = [P(pred = 25 | derm) − P(pred = 25 | clinic)] + [P(pred = 14 | clinic) − P(pred = 14 | derm)]**, paired.
Under a modality shortcut, clinical melanoma goes to 14 and dermoscopic melanoma to 25, so Δ2 > 0. Both 14 and 25 count as melanoma when scoring correctness (secondary S1).

**Uncertainty:** 2000 paired bootstrap resamples of cases, RNG seed 20260930; percentile 95% CI. Within each resample the estimand is computed per seed and averaged across the 3 seeds. Seed-to-seed SD is reported separately.

**Decision rule for the primary configuration (applied to P1 and P2 separately):**
- **Supported:** estimand > 0 in all 3 seeds and the CI lower bound > 0. Wording: *"predictions were modality-dependent"*; report the size with CI.
- **Reversed:** < 0 in all 3 seeds and CI upper bound < 0.
- **Otherwise:** *"no consistent evidence of modality-dependent predictions"*. Never "no effect".

The claim "consistent with a modality shortcut" is written only if both P1 and P2 are supported. Magnitude is always reported; no threshold is attached to the word "large".

## 6. Secondary (descriptive, no decision rule)

- **S1.** Recall per diagnosis × modality. Correct outputs: MEL → {14, 25}; BCC → {1}; NEV → {18}; VASC → {30}; DF → {3}; SEK → {21} (the dataset has no genuine seborrheic keratosis class; PBK is the nearest ISIC category). Also reported: the share of SEK cases predicted as 25.
- **S2.** Per-case agreement: fraction of cases whose clinical and dermoscopic predictions are the same class.
- **S3.** P1 and P2 for all other configurations and the authors' model.
- **S4.** Sensitivity: NEV excluding Reed/Spitz and blue nevi (`nevus_atypical` in the manifest).
- **S5.** P1 restricted to each diagnosis.

## 7. Known limitations (stated in advance)

- Derm7pt clinical images are close-up photographs from an atlas, not smartphone images; modality is not the only difference from the training data.
- Derm7pt melanomas are mostly thin or in situ, a different disease spectrum from the training melanoma classes.
- No patient identifier; cases may not be independent.
- Derm7pt diagnoses are taken as reference; nevi may not all be histopathologically confirmed.
