# Training templates and the experimental protocol

`training_kaggle.ipynb` is the archived working template, with outputs stripped. Its historical imports point to two Kaggle datasets containing private architecture modules. Its default group is knowledge distillation and its teacher path is user specific. The template therefore requires configuration; it is not a one-click replay of all 48 trained checkpoints.

`training_kaggle_portable.ipynb` is a separately labelled adaptation prepared after the experiments. It imports the released `code/skin_disease` package, supports EfficientNet-B2 original/dual-level fusion and timm models, and defaults to the B0 group. Architecture variants outside the audit are rejected. It pins timm 1.0.26, skips unrelated HAM initialisation for Skin31, removes unused coarse mapping and disables the historical PAD-inference cell during training. No GPU training was performed during the release preflight, so this is a portable template rather than a claim that retraining will reproduce the archived weights bit for bit.

Attach the release folder as a Kaggle dataset containing `code/skin_disease`. Obtain the source images under their original access conditions and rebuild the derived datasets with the builder scripts and their released manifests. Configure these environment variables in an initial code cell, before executing the portable template:

```python
import os
os.environ["SKIN31_ROOT"] = "/kaggle/input/YOUR-DATASET/skin31_milkB"  # change per arm
os.environ["TRAINING_SEED"] = "42"  # repeat in clean sessions for 123 and 2024
os.environ["FINAL_GROUP"] = "b0"
```

The explicit placeholder above is a path to your attached dataset; it is not a missing image download bundled with this repository. Check the displayed raw counts before training:

| Dataset / arm | Classes | Raw train | Test |
|---|---:|---:|---:|
| Original benchmark | 31 | 3,915 | 994 |
| Corrected / A | 30 | 3,806 | 863 |
| Relabel / R | 30 | 3,870 | 863 |
| HAM10000 / H | 30 | 4,233 | 863 |
| MILK10k clinical / B | 30 | 4,233 | 863 |
| MILK10k dermoscopic / C | 30 | 4,233 | 863 |

For B0 interventions, filter the experiment list after the FINAL configuration cell and before the run cell:

```python
experiments = [e for e in experiments if e["exp_name"] == "FINAL_B0_Training_Optimization"]
```

For ViT original/B/C experiments, override the list at the same position:

```python
RUN_NAME = "final_vit"
experiments = [{
    "exp_name": "FINAL_ViT_S16_Baseline",
    "model_name": "vit_small_patch16_224.augreg_in21k_ft_in1k",
    "model_variant": "original", "optimizer_name": "adamw",
    "scheduler_name": "cosine", "lr_value": 1e-4,
    "label_smoothing": 0.0, "use_cutmix": False, "use_sam": False,
    "drop_rate": 0.0,
}]
```

The explicit timm weight identifier documents the ImageNet-21k then ImageNet-1k pretrained weights used for ViT. Keep timm 1.0.26 to avoid silently changing the default pretrained variant. The portable template rejects missing or mismatched KD teacher checkpoints and verifies that train/test class indices agree.

Training settings: 224 × 224 input, batch 32, 15 fixed epochs, weight decay 1e-6, seven fixed views per raw training image, inverse-frequency class-balanced sampling with replacement, and the final epoch checkpoint. B0 baseline uses AdamW/cosine at 5e-4. Training optimisation uses 2.5e-4, label smoothing 0.1, CutMix alpha 1 and SAM rho 0.10. The original training notebook prints test metrics during training, but the final protocol uses the last epoch; do not choose checkpoints or hyperparameters using the external test sets.

Each clean session writes to `/kaggle/working/seed<seed>`. Keep its `class_names_skin31.json` and final checkpoint. The repository's checkpoint manifests specify the expected destination and SHA256 of the actual archived experiment files. Fresh training can vary with hardware/library state; the hashes identify the reported checkpoints, not a guarantee of deterministic regeneration.

For B2 and distillation experiments, configure the group and the archived teacher explicitly using the paper's Supplementary training section. The portable KD group requires `KD_TEACHER_PATH`; distinguish the original B2 teacher from the fusion B2 teacher and set the teacher variant accordingly. The default historical KD cell must not be assumed to reproduce both KD arms without that change.
