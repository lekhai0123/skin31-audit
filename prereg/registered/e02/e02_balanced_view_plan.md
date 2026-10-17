# Balanced-view reference on Derm7pt (E-02): analysis plan — fixed before training; OSF registration pending

**Status:** the plan and lock files were fixed on 17/10/2026, before any EfficientNet model was trained on the arms below. The SHA-256 of this plan and of the locked files are in `prereg/e02_lock_sha256.txt`. A pilot with frozen features was run before this plan was written (Section 7).

OSF was unavailable for scheduled maintenance from 16 to 19 October 2026, so training started before registration. The unchanged plan will be registered on OSF when it is available, with the same SHA-256 values. Nothing below is changed after training starts; later additions are labelled exploratory. Because registration follows the start of training, results are reported as following a plan fixed in advance, not as pre-registered.

Companion registrations: Derm7pt (osf.io/643ht, replaced by osf.io/jzsr9), PAD-UFES-20 (osf.io/hn7yk), HAM10000 intervention (osf.io/pe8q6), MILK10k control (osf.io/5ksja).

## 1. Question

In the benchmark, every class comes from a single source, so the imaging view of the training images is coupled to the diagnosis: basal cell carcinoma and melanoma are represented only by clinical atlas images, and the other diagnoses of Derm7pt map to classes represented only by dermoscopic ISIC images. On paired views of the same lesions, predictions shift with the view (P1). This shift may arise from the coupling of view and diagnosis in the training data, from differences in the diagnostic information that the two views show, or from both.

**Primary question:** with the same lesions, the same number of training images and the same training recipe, does coupling the training view to the diagnosis produce larger paired-view routing than balancing the view within each diagnosis?

## 2. Data and folds

- Derm7pt, the 960 cases with both a clinical and a dermoscopic image used in osf.io/643ht (`prereg/derm7pt_manifest.csv`, with image SHA-256): NEV 575, MEL 249, SEK 45, BCC 42, VASC 29, DF 20. Six-class task (BCC, MEL, NEV, SEK, VASC, DF).
- Five-fold cross-validation at case level, stratified by diagnosis (scikit-learn `StratifiedKFold`, shuffled, random state 20260930): both images of a case are always in the same fold, and every case is a test case exactly once (192 test cases per fold). The fold of every case is saved before any training (`prereg/e02_folds.csv`).
- Derm7pt has no patient identifiers; cases are treated as independent, as in osf.io/643ht.

## 3. Arms (built per fold from the training cases only)

| Arm | Training images |
|---|---|
| COUPLED | BCC and MEL cases contribute their clinical image; NEV, SEK, VASC and DF cases contribute their dermoscopic image (mirrors the benchmark) |
| BALANCED | Every case contributes one image; within each diagnosis, half of the training cases (rounded down) contribute the clinical image and the rest the dermoscopic image, chosen at random (RNG seed 20260930 + fold index) |

COUPLED and BALANCED contain the same cases, the same number of images per class and the same number of optimisation steps; they differ only in which view represents each case. The view assignment of every case in every fold is saved before training (`e02_arm_assignment.csv`). Test images of a fold are never in its training set; both views of every test case are predicted.

## 4. Training

EfficientNet-B0 (ImageNet-pretrained `timm` weights) with the training-optimisation recipe of the 30-output models (manuscript Supplementary Section S13): seven fixed views per training image; inverse-class-frequency sampling; AdamW; learning rate 2.5 × 10⁻⁴ with cosine schedule; weight decay 10⁻⁶; label smoothing 0.1; CutMix (α = 1); sharpness-aware minimisation (ρ = 0.10); batch size 32; 15 epochs; 224 × 224 pixels; final-epoch weights. No validation set, no checkpoint selection and no tuning. Seeds 42, 123 and 2024. Total: 2 arms × 5 folds × 3 seeds = 30 models.

Code: the released portable training notebook (code release v1.0.0, `notebooks/training_kaggle_portable.ipynb`). Its training functions, recipe constants and B0 training-optimisation configuration were checked definition by definition against the notebook used for the benchmark models and are identical; the only differences, in `build_model` and `run_experiment`, are input checks that do not apply to EfficientNet-B0 without distillation. For local execution the notebook is converted to a script by `scripts/e02_make_runtime.py`, which:
1. drops the line `pip install timm==1.0.26` (the environment already has timm 1.0.26);
2. defines `display` as `print`;
3. selects the B0 training-optimisation experiment;
4. replaces the two flip lambdas by named functions with the same body;
5. places module-level statements under `if __name__ == "__main__":`, so that Windows data-loader workers do not re-run the script.

Training functions are not changed. `SKIN31_ROOT` points to `data/e02/fold{f}/{ARM}` (subfolders `train` and `test`).

Runs are made locally (NVIDIA GTX 1080 Ti; PyTorch 2.10.0, torchvision 0.25.0, timm 1.0.26), whereas the benchmark models were trained on Kaggle (NVIDIA T4). A run that fails for a technical reason (for example, a data-loader worker that cannot start) is restarted with the same seed, up to three attempts (`scripts/e02_run_training.py`). Predictions are recomputed in FP32 from the final-epoch weights (`scripts/e02_eval.py`).

## 5. Estimands

Out-of-fold predictions for both images of every case, one set per seed. Group G = {NEV, SEK, VASC, DF} (the diagnoses represented by dermoscopic images in COUPLED); targets T = {BCC, MEL}.

- **P1 (primary):** fraction of dermoscopic images predicted as a diagnosis in G minus the same fraction for clinical images, over all 960 cases.
- **E2 (primary):** the same difference over the T cases.
- **E3 (secondary):** the same difference over the G cases.
- **E1 (secondary):** recall on dermoscopic images of BCC and MEL, averaged over the two diagnoses with equal weight.
- **Descriptive:** recall on clinical images of BCC and MEL (equal-weight mean); six-class balanced accuracy for each view; residual share = P1(BALANCED) / P1(COUPLED), each averaged over seeds.

**Primary contrast: COUPLED − BALANCED**, seed-matched and paired over cases. Expected: P1 > 0 and E2 > 0. Secondary: E3 COUPLED − BALANCED > 0; E1 BALANCED − COUPLED > 0.

**Uncertainty:** 2,000 paired bootstrap resamples of cases (RNG seed 20260930); within each resample the per-seed differences are averaged over seeds; percentile 95% intervals; resamples lacking any of the six diagnoses are redrawn. The intervals describe the sampling of cases for fixed models, not variation between training runs; per-seed values are reported.

**Decision rule (P1 and E2 separately):** supported in the expected direction if the difference has the expected sign in all three seeds and the interval excludes 0 on that side; opposite direction if the reverse holds; otherwise "no consistent evidence of a difference". This rule governs the wording; it is not a significance test.

**Residual share (descriptive, thresholds fixed in advance):** ≤ 0.25, coupling accounts for most of the paired-view routing in this setting; between 0.25 and 0.5, balancing reduces the routing but a substantial part remains; ≥ 0.5, most of the routing remains when the view is balanced.

**Wording:**
- P1 and E2 supported, residual share ≤ 0.25: *"In a controlled six-diagnosis experiment on Derm7pt, coupling the training view to the diagnosis was sufficient to produce paired-view routing, and balancing the view within each diagnosis removed most of it."*
- P1 and E2 supported, residual share between 0.25 and 0.5: *"…coupling increased paired-view routing; balancing the view reduced it but did not remove it."*
- P1 and E2 supported, residual share ≥ 0.5: *"…coupling increased paired-view routing, but most of it remained when the view was balanced."*
- Otherwise: *"No consistent evidence that coupling the training view to the diagnosis increases paired-view routing in this setting."*
- In every case: the result concerns models trained and tested on Derm7pt with six diagnoses; it does not establish how much of the routing of the benchmark-trained models is due to coupling.

## 6. Known limitations (stated in advance)

- Training and test images come from the same dataset, so there is no domain shift, unlike for the benchmark-trained models.
- Six diagnoses; small classes (DF 20, VASC 29 cases) give wide intervals for per-class quantities.
- Cases are treated as independent because patient identifiers are unavailable.
- Three seeds; hardware differs from that of the benchmark models.

## 7. Pilot (disclosed)

Before this plan was written, a pilot used frozen DINOv2 ViT-S/14 features of the same 960 cases with a logistic regression (five-fold cross-validation repeated ten times; `scripts/pilot_e02_probe.py`; `outputs/pilot_e02/pilot_results.csv`). P1 was 0.939 [0.927, 0.952] for COUPLED and −0.016 [−0.036, 0.004] for BALANCED (residual share −0.017). The residual-share thresholds of Section 5 were stated before the pilot was run. The pilot informed the decision to run this experiment; arms, estimands and decision rules above were fixed before any EfficientNet training. The pilot is not part of the reported evidence.

## 8. Files locked at registration

The SHA-256 of each file below is listed in `prereg/e02_lock_sha256.txt` and copied into the registration:
- `prereg/derm7pt_manifest.csv` (the 960 cases, as in osf.io/643ht)
- `prereg/e02_folds.csv` (case → fold) and `prereg/e02_arm_assignment.csv` (fold, arm, case → view)
- `scripts/build_e02_arms.py` (folds, view assignment and image folders; the images are hard links to the local Derm7pt copy, which is not redistributed)
- `scripts/e02_make_runtime.py` and the generated `outputs/e02/runtime/e02_train.py`
- `scripts/e02_run_training.py` (runs, restarts after technical failures, run log with the registration status)
- `scripts/e02_eval.py` (FP32 predictions, estimands, bootstrap, decision rules, wording)
- `notebooks/training_kaggle_portable.ipynb` of code release v1.0.0

Before registration, the pipeline was run end to end once on a six-class toy set of benchmark images (no Derm7pt image), and the evaluation code was run on synthetic predictions. No E-02 model was trained.
