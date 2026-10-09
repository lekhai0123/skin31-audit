# PAD-UFES-20 analysis plan (pre-specified)

**Status:** written 01/10/2026 and registered on OSF before evaluation. Already inspected on PAD before this plan: only the fusion-distilled B0, seed 42, on the uncleaned cohort with the earlier 6-class mapping (SEK → skin31 index 25). That run is reported as exploratory. No other checkpoint and none of the retrained models has been evaluated on PAD. After registration, nothing below is changed; later additions are labelled exploratory.

Companion registration: Derm7pt analysis plan (osf.io/643ht), which found modality-dependent predictions.

## 1. Questions

- **Q1.** How do models trained on the 31-class dataset perform on smartphone clinical images from a different source (PAD-UFES-20)?
- **Q2.** Does retraining on the corrected dataset (label errors fixed, the mislabeled "Seborrheic keratosis" class removed, leakage removed) change PAD performance relative to the original training data?
- **Q3 (descriptive).** Which output groups do PAD images go to: ISIC-derived (dermoscopic) or Atlas-derived (clinical) classes?

## 2. Data

- `pad_manifest.csv` (SHA256 in `SHA256SUMS.txt`): all 2298 images with patient ID, diagnosis, SHA256 and an exclusion column.
- **Analysis cohort:** 2269 images, 1371 patients. Excluded: 26 images forming 13 pairs of pixel-identical images with conflicting diagnoses (both images), and the second copy of 3 pixel-identical pairs with the same diagnosis.
- Diagnoses: BCC 833, ACK 723, NEV 243, SEK 234, SCC 184, MEL 52 (36 patients; MEL estimates will be unstable).
- No PAD image was found to duplicate a training or test image of the 31-class dataset (MD5 and pHash with visual review of all 49 candidates).
- **Sensitivity cohort:** the original 2298 images.

## 3. Models

- The 21 original checkpoints (`checkpoint_manifest.csv`), 7 configurations × 3 seeds, 31 outputs.
- The EfficientSkinDis authors' released model (`Weights.h5`), logits 0–30.
- Retrained models on the corrected dataset (30 outputs): EfficientNet-B0 baseline and EfficientNet-B0 + training optimization, seeds 42, 123, 2024, same code and hyperparameters as the originals, trained on Kaggle. Their checkpoints' SHA256 are recorded before evaluation.

## 4. Inference

As in the Derm7pt plan: RGB, `Resize((224, 224))` bilinear without aspect preservation, ImageNet normalization, `eval()`, FP32, no TTA; logits saved.

## 5. Label mapping (by class name, so it applies to 31- and 30-output models)

| PAD | Correct model output(s) |
|---|---|
| ACK | Actinic keratosis |
| BCC | Basal Cell Carcinoma |
| MEL | Melanoma, and "Seborrheic keratosis" where that output exists (it contains only dermoscopic melanoma) |
| NEV | Nevus |
| SEK | Pigmented Benign Keratosis (ISIC's benign keratosis category includes seborrheic keratosis) |
| SCC | Squamous cell carcinoma |

## 6. Metrics

**Primary.**
- **6-way balanced accuracy.** Prediction = argmax over the outputs listed in §5 only (6 or 7 logits); outputs mapped to the same PAD class count as that class.
- **6-way macro AUROC.** Softmax over the same logits; a PAD class's score is the sum of its outputs' probabilities; one-vs-rest AUROC per class, unweighted mean.

**Secondary.**
- Full-label-space balanced accuracy: argmax over all outputs; any output outside §5 counts as wrong.
- Per-class recall and AUROC.
- **Q3:** fraction of images whose full-label-space prediction is an ISIC-derived output (Actinic keratosis, Dermatofibroma, Nevus, Pigmented Benign Keratosis, "Seborrheic keratosis" if present, Squamous cell carcinoma, Vascular lesion), overall and by diagnosis. Reference: 4 of the 6 PAD diagnoses (ACK, NEV, SEK, SCC) correspond to ISIC-derived outputs.
- Sensitivity: 5-way metrics excluding SEK; all metrics on the 2298-image cohort.

**Chance levels, always reported alongside:** 6-way balanced accuracy 1/6; AUROC 0.5.

## 7. Uncertainty and decision rules

- Seed SD reported separately from sampling uncertainty.
- **Patient-level bootstrap:** 2000 resamples of `patient_id`, all images of a sampled patient included; RNG seed 20260930; percentile 95% CI; within each resample the metric is computed per seed and averaged over seeds. A resample lacking any of the 6 diagnoses is discarded and redrawn; the number discarded is reported.
- **Q2 contrast (per configuration, B0 baseline and B0 + training optimization):** Δ = metric(corrected) − metric(original), paired on the same patient resamples, seeds matched by seed number.
  - **Improved:** Δ > 0 for all 3 seeds and CI lower bound > 0.
  - **Worse:** Δ < 0 for all 3 seeds and CI upper bound < 0.
  - Otherwise: *"no consistent evidence of a difference"*; never "no effect".
- No multiplicity adjustment; CIs are interpreted individually. Primary configuration for Q1 wording: fusion-distilled B0 (as in the Derm7pt plan); for Q2: B0 + training optimization.

## 8. Known limitations

- PAD differs from the training data in modality, device, population, prevalence and diagnostic confirmation at once; the analysis does not isolate one factor.
- Most ACK, NEV and SEK in PAD are not biopsy-confirmed; SCC includes Bowen's disease.
- MEL has 52 images from 36 patients.
- Seed 42 of the fusion-distilled B0 was inspected before this plan (on the uncleaned cohort, with the old SEK mapping).
