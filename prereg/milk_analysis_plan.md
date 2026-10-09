# Source- and lesion-matched deconfounding control (MILK10k): analysis plan (pre-specified)

**Status:** written 02/10/2026, after the data sets were built and before any model is trained on them. Registered on OSF before training. After registration nothing below is changed; later additions are labelled exploratory.

Companion registrations: Derm7pt plan (osf.io/643ht), PAD plan (osf.io/hn7yk), first deconfounding experiment with HAM10000 (osf.io/pe8q6).

## 1. Question

In the first deconfounding experiment (osf.io/pe8q6), adding dermoscopic BCC and melanoma images to the clinical-only BCC and Melanoma classes increased dermoscopic recall and reduced modality dependence for those classes on Derm7pt. That intervention changed two things at once: the modality available to the classes and the number of new, distinct images. (Class exposure during training is unchanged because the training code samples classes with inverse-frequency weights.)

**Primary question:** Does adding dermoscopic images reduce modality dependence for the target classes more than adding the same number of new clinical images of the same lesions from the same source?

## 2. Arms

All arms: EfficientNet-B0 + training optimization (label smoothing 0.1, CutMix α=1.0, SAM ρ=0.10, AdamW, cosine, lr 2.5e-4, 15 epochs), seeds 42, 123, 2024, same Kaggle notebook and environment as the earlier corrected and deconfounded models; test set unchanged (863 images).

| Arm | Training data | Checkpoints |
|---|---|---|
| A | corrected dataset `skin31_clean` (3806 images) | existing (`checkpoints_clean/5_b0_trainopt`) |
| B | A + **clinical close-up** images of lesion set S | to be trained |
| C | A + **dermoscopic** images of the **same** lesion set S | to be trained |

Lesion set S: 327 BCC and 100 melanoma lesions from the MILK10k training set (ISIC; CC-BY-NC), the same counts as osf.io/pe8q6. Each lesion contributes one image to B and the other image to C, so B and C match in diagnosis, lesion, source and number of added images, and differ in modality. After adding, Basal Cell Carcinoma has 661 and Melanoma 200 training images in both arms (4233 total).

## 3. Selection of S (done before registration, metadata and image hashes only)

- Eligible: MILK10k ground-truth diagnosis BCC or MEL; diagnosis confirmed by histopathology; not "Melanoma metastasis"; no image of the lesion marked `image_manipulation = altered`; both images present. Eligible: BCC 2429, MEL 383 lesions.
- Order: eligible lesions randomly permuted per diagnosis (seed 20260930); screened in that order until N lesions without a duplicate candidate are found (screened: BCC 425, MEL 200).
- Duplicate screening: both images of each screened lesion against the 31-class dataset (all images), HAM10000 (all images), Derm7pt (both modalities) and PAD-UFES-20: exact MD5 of decoded pixels (none found), and pHash Hamming ≤ 8 pairs (562). A lesion is excluded if any of its candidate pairs has DINOv2 (ViT-S/14) cosine similarity ≥ 0.80 (objective, conservative rule; no visual adjudication). Excluded: 49 MEL (mostly near-identical to HAM10000 images, cosine up to 0.99), 5 BCC.
- Files: `milk_selected_lesions.csv` (S), `milk_excluded_lesions.csv`, `milk_lesion_order.csv`, `build_milk_arms.py`; archives `skin31_milkB.zip` (SHA256 203b06ea5850411b96a347debaf0a139d5df2c3150112eae41a0d46b823c10f8) and `skin31_milkC.zip` (SHA256 4b7b6eea9fb1a13eab2e45d74e551f14e81d5aed8b283a92c39b27d2667d3443).

## 4. Estimands (Derm7pt, 960 cases, as in osf.io/pe8q6)

Target diagnoses T = {BCC, MEL}; control diagnoses N = {NEV, SEK, VASC, DF}. ISIC-derived outputs: Actinic keratosis, Dermatofibroma, Nevus, Pigmented Benign Keratosis, Squamous cell carcinoma, Vascular lesion. Outputs resolved by class name.

- **E1 (primary):** macro mean over BCC and MEL of recall on dermoscopic images (BCC → Basal Cell Carcinoma; MEL → Melanoma).
- **E2 (primary):** modality shift on T: fraction of dermoscopic images predicted as an ISIC-derived class minus the same fraction for clinical images.
- **E3 (secondary):** the same shift on N.
- Descriptive: clinical-image recall on T; modality shift over all cases.

**Primary contrast: C − B**, seed-matched, paired over cases. Expected under the hypothesis that modality, not added image diversity, drives the effect: E1 C − B > 0; E2 C − B < 0.
**Secondary contrasts:** B − A and C − A for all estimands (descriptive); E3 for C − B.

**Uncertainty:** 2000 paired bootstrap resamples of cases (RNG seed 20260930); within each resample, per-seed differences averaged over seeds; percentile 95% CI. Resamples lacking any of the six diagnoses are redrawn.

**Decision rule (E1, E2 separately):** supported in the expected direction if the difference has the expected sign in all 3 seeds and the CI excludes 0 on that side; opposite direction if the opposite holds; otherwise "no consistent evidence of a difference".

**Wording:**
- Both E1 and E2 supported: *"adding dermoscopic images reduced modality dependence for the target classes beyond the effect of adding an equal number of new clinical images of the same lesions."*
- Only E1 supported: *"…improved dermoscopic recognition of the target classes beyond…"*, without a claim about modality dependence.
- Neither: the effect of the first experiment cannot be attributed to modality rather than added images.

## 5. Secondary (descriptive)

- PAD-UFES-20 (2269-image cohort of osf.io/hn7yk): 6-way balanced accuracy and macro AUROC, C − B with patient-level bootstrap CI; ISIC-output rate per arm.
- Accuracy on the 863-image corrected test set per arm.

## 6. Known limitations (stated in advance)

- **E2 may move for a reason other than dermoscopic learning:** MILK10k clinical images are close-ups, closer in framing to Derm7pt clinical images than the Atlas photographs are. Arm B may therefore change the clinical side of E2. E1 is not affected by this and is listed first.
- MILK10k and HAM10000 dermoscopic images are both ISIC-ecosystem images; Derm7pt dermoscopic images come from a different atlas.
- Only two classes; one configuration; three seeds.
