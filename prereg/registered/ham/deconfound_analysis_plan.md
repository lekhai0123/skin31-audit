# Deconfounding experiment: analysis plan (pre-specified)

**Status:** written 01/10/2026, before the deconfounded models are trained. Registered on OSF before evaluation; after registration nothing below is changed, and later additions are labelled exploratory.

Companion registrations: Derm7pt plan (osf.io/643ht), PAD plan (osf.io/hn7yk).

## 1. Question

In the 31-class dataset every class has a single imaging modality (ISIC-derived classes: dermoscopic; Atlas-derived classes: clinical). Models trained on it, including models retrained on the label-corrected version, make modality-dependent predictions on Derm7pt (same lesion, two modalities). **Does adding the missing modality to selected classes reduce modality dependence for those classes, and only for those classes?**

## 2. Intervention

- **Base data:** the corrected dataset `skin31_clean` (3806 train / 863 test, 30 classes).
- **Target classes:** Basal Cell Carcinoma and Melanoma (Atlas-derived, clinical only).
- **Added images:** dermoscopic BCC and melanoma images from HAM10000, added to the training folders of these two classes. Number added per class equals that class's existing training count (BCC 334, Melanoma 100), so each target class becomes about half clinical and half dermoscopic. If fewer eligible lesions exist, all eligible lesions are used and the number is reported.
- **Eligibility:** one image per HAM10000 lesion (the lowest `image_id`); lesion and image not present anywhere in the 31-class dataset (by ISIC ID and by ISIC `lesion_id`); not a near-duplicate of any Derm7pt, PAD-UFES-20 or 31-class test image (pHash Hamming ≤ 8 candidates reviewed visually; confirmed duplicates excluded). Sampling among eligible lesions: random, seed 20260930.
- **Unchanged:** all other classes, the test set (863 images; no HAM image is added to test), code, environment and hyperparameters.
- **Model:** EfficientNet-B0 + training optimization (label smoothing 0.1, CutMix α=1.0, SAM ρ=0.10, AdamW, cosine, lr 2.5e-4, 15 epochs), seeds 42, 123, 2024, trained with the same Kaggle notebook and environment as the corrected models.
- **Comparison model:** the corrected-data models of the same configuration and seeds (`checkpoints_clean/5_b0_trainopt`).

**Data as built (before training):** `scripts/build_deconf_skin31.py`. Eligible HAM10000 lesions after exclusions: BCC 327, melanoma 614 (no HAM10000 BCC or melanoma lesion occurs in the 31-class dataset). 62 near-duplicate candidates (29 Derm7pt, 27 PAD, 6 test) were all reviewed visually; none was a duplicate, none excluded. Added: **BCC 327** (all eligible; target 334), **melanoma 100**. Training counts: Basal Cell Carcinoma 334 → 661, Melanoma 100 → 200; total training images 4233; test unchanged (863). The list of added images is `deconf_ham_added.csv`; dataset archive `skin31_deconf.zip`, SHA256 59fc826ac126178d0dc0f3e5037a7e516f0a5fe76b246f42fcc829f7fc0ef1a3. Evaluation code: `deconf_eval.py`.

## 3. Evaluation data

- Derm7pt: `derm7pt_manifest.csv` (960 cases, clinical and dermoscopic image of the same lesion), as in osf.io/643ht.
- PAD-UFES-20: the 2269-image cohort of osf.io/hn7yk.
- The 863-image corrected test set.
- Inference as in the earlier plans (Resize 224×224, ImageNet normalization, FP32, no TTA). Outputs are resolved by class name.

## 4. Primary estimands (Derm7pt; deconfounded minus corrected, seeds matched, paired over cases)

Target diagnoses T = {BCC, MEL}; control diagnoses N = {NEV, SEK, VASC, DF}. Correct outputs: BCC → Basal Cell Carcinoma; MEL → Melanoma.

- **E1. Dermoscopic recall of target diagnoses.** Macro mean over BCC and MEL of recall on the dermoscopic images. Expected under the confounding hypothesis: Δ > 0.
- **E2. Modality shift on target diagnoses.** P1 restricted to T: fraction of dermoscopic images predicted as an ISIC-derived class minus the same fraction for clinical images. Expected: Δ < 0.
- **E3. Specificity control.** P1 restricted to N. Expected: no consistent change, since nothing was added to these classes.

ISIC-derived outputs: Actinic keratosis, Dermatofibroma, Nevus, Pigmented Benign Keratosis, Squamous cell carcinoma, Vascular lesion.

**Uncertainty:** 2000 paired bootstrap resamples of cases, RNG seed 20260930, percentile 95% CI; within a resample the difference is computed per seed and averaged over seeds.

**Decision rules (each estimand separately):**
- **Supported in the expected direction:** Δ has the expected sign in all 3 seeds and the CI excludes 0 on that side.
- **Opposite direction:** opposite sign in all 3 seeds and CI excludes 0 on the other side.
- Otherwise: *"no consistent evidence of a change"*.

**Wording:** *"Adding the missing modality reduced modality dependence for the target classes"* only if E1 and E2 are both supported. *"…specifically"* only if, in addition, E3 is not supported in the direction of E2 or its point estimate is less than half of E2's in absolute value. Magnitudes are always reported.

## 5. Secondary (descriptive)

- Clinical-image recall of BCC and MEL on Derm7pt (checks that clinical performance is not traded away).
- P1 and P2 over all Derm7pt cases, as defined in osf.io/643ht.
- PAD-UFES-20: 6-way balanced accuracy, macro AUROC, per-class recall and ISIC-output rate, deconfounded vs corrected, with patient-level bootstrap CI (2000 resamples), as in osf.io/hn7yk.
- Accuracy and balanced accuracy on the 863-image test set.

## 6. Known limitations

- HAM10000 images come from the ISIC ecosystem; Derm7pt dermoscopic images come from a different atlas, so a gain on Derm7pt reflects transfer across dermoscopic sources as well as modality.
- Only two classes are deconfounded; ISIC-derived classes still have no clinical images.
- Added images increase the size of the two target classes; the class-balanced sampler of the training code limits, but does not remove, this effect.
- One configuration, three seeds.
