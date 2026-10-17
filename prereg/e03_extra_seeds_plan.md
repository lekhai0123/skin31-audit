# Extra training seeds for the matched contrast C − B (E-03): analysis plan (exploratory)

**Status:** written on 17/10/2026, before any model with the extra seeds was trained. Exploratory and post hoc relative to the MILK10k plan (osf.io/5ksja), which specified seeds 42, 123 and 2024; the registered three-seed estimates in the manuscript are not changed. This plan and its analysis code are fixed by SHA-256 (`prereg/e03_lock_sha256.txt`) before training.

## 1. Question

How much does the matched contrast C − B vary between independent training runs? The confidence intervals in the manuscript describe the sampling of test lesions for fixed checkpoints and do not include variation between training runs.

## 2. Runs

- Arms B and C of osf.io/5ksja: the corrected benchmark plus the clinical (B) or dermoscopic (C) images of the same 427 MILK10k lesions; Kaggle datasets `skin31-milkb` and `skin31-milkc`, unchanged.
- EfficientNet-B0 with the training-optimisation recipe, trained with the same Kaggle notebook as seeds 42, 123 and 2024. Only the seed is changed, to **1** and **2**. The seeds were fixed before training and were not selected.
- Only the B0 training-optimisation configuration is run. Every configuration is re-seeded at its start and receives freshly built data loaders, so omitting the baseline configuration does not change it.
- Final-epoch weights; no selection. Both arms of a seed are trained in the same Kaggle environment; the library versions printed by the notebook are recorded. A run that fails for a technical reason is restarted with the same seed.
- Checkpoints are stored as `checkpoints_milk{B,C}/5_b0_trainopt/seed{1,2}/`, with the layout of the existing seeds.

## 3. Endpoints and analysis

- The endpoints of Supplementary Table S15 for C − B (EfficientNet-B0): E1, E2, E3, untreated dermoscopic lesions predicted as melanoma, nevus dermoscopic recall, macro dermoscopic recall over the six Derm7pt diagnoses, and PAD-UFES-20 six-diagnosis balanced accuracy.
- Computed locally with `scripts/e03_eval.py`. The script loads the checkpoints with `milk_eval.load`, computes logits with `derm7pt_eval_clean.logits` and `pad_eval.logits` as in `milk_eval.py`, and computes the endpoints with the functions that produced Table S15 (`vit_milk_eval.stats`, `derm7pt_overall.stats`, `pad_eval.metrics`). Applied to the three registered seeds, it reproduces Table S15 exactly (largest difference 8.3 × 10⁻¹⁷).
- Per-seed C − B for all five seeds; five-seed mean and SD; 95% interval over seeds from the t distribution with 4 degrees of freedom, as an estimate that includes variation between training runs; number of seeds with the same sign as the registered three-seed mean.

## 4. Reporting

- All five seeds are reported in the Supplementary Information, whatever the results. The three-seed estimates and intervals in the main text are unchanged.
- If a new seed has the opposite sign for E1 or E2, the main-text statement that the contrast was consistent across seeds is qualified to report the five-seed result.

## 5. Limitations

- Five runs give a wide t interval.
- The Kaggle environment may differ from that of the first three seeds; the versions are reported.
