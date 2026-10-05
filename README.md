# SSR-ALC

Code for **SSR-ALC: Semi-supervised representation learning for anatomical landmark classification of endoscopy images**
Mokarrameh Einlou, Omid Yaghoobian, Khan A. Wahid. *Biomedical Signal Processing and Control* 123 (2026) 110672.
[doi:10.1016/j.bspc.2026.110672](https://doi.org/10.1016/j.bspc.2026.110672)

SSR-ALC trains a convolutional autoencoder on labeled **and** unlabeled endoscopy images, with a
classification loss on the labeled images shaping the latent space (Eq. 3 of the paper).
**LATE-SMOTE** oversamples minority classes by interpolating in that latent space instead of pixel space.
A small MLP classifies the latent vectors. On the six anatomical landmarks of HyperKvasir, the 512-d model
reaches a macro-F1 of **97.29 ± 2.20 %** (5-fold CV), including **97.12 %** on the Ileum class. The original
dataset has only 9 labeled Ileum images.

## Contents

| Paper item | Script | Config |
|---|---|---|
| Unsupervised AE (needed by Exp. 1 and 3) | `train_only_ae.py` | `configs/only_ae.yaml` |
| Ablation Exp. 1–5, Tables 9–10, Figs. 4–5 | `exp1_train.py` … `exp5_train.py` | `configs/exp{1..5}_default.yaml` |
| Hyperparameter grid search, Table 6 | `hpo_train.py`, `visualization/hpo_vis.py` | `configs/hpo.yaml` |
| Cross-fold check of selected (α, β, γ), Table 7 | `hpo_train.py` | `configs/hpo_crossfold.yaml` |
| Ileum sensitivity, Fig. 9 | `exp4_train.py` (once per N), `analysis/ileum_sensitivity_summary.py` | `configs/ileum_sensitivity.yaml` |
| FixMatch baseline, Tables 14–15 | `fixmatch_train.py` | `configs/fixmatch.yaml` |
| Paired t-test (Sec. 5.9.3) | `analysis/paired_ttest.py` | – |
| Kvasir v2 transfer, Table 13 | `train_kvasir.py` | `configs/kvasir.yaml` |
| SSIM, Table 11 | `analysis/ssim_eval.py` | – |
| Latent clustering / t-SNE, Table 12, Fig. 8 | `analysis/manual_style_analysis.py --tsne --cluster_metrics`, `analysis/only_ae_cluster_metrics.py` | – |
| Training curves, Figs. 4–5 | `analysis/manual_style_analysis.py` | – |
| Parameter count, Table 16 | `analysis/param_count.py` | – |

```
engine/        training loops (one per experiment) and models (engine/models/)
options/       YAML + command-line options (command line overrides YAML)
utils/         data loading, cross-validation splits
splits/        every split and image list used in the paper
scripts/       data preparation, full reproduction script, SLURM example
analysis/      result tables, statistics, plots
visualization/ plotting helpers
```

## Installation

Python 3.9+ (the experiments ran on Python 3.11, CUDA 12.2, one NVIDIA A100 80 GB).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install `torch`/`torchvision` for your CUDA version following <https://pytorch.org> if the default wheel does not fit your system.

## Data

1. **HyperKvasir** (CC BY 4.0): download the labeled and unlabeled images from <https://datasets.simula.no/hyper-kvasir/>.
   We use the six anatomical-landmark classes (cecum, ileum, retroflex-rectum → `rectum`, pylorus,
   retroflex-stomach → `stomach`, z-line) and a fixed subset of 25,700 unlabeled images.
2. **Additional Ileum images:** 56 Ileum images from [The Gastrolab Image Gallery](https://www.gastrolab.net/), collected following
   F. Sedighipour Chafjiri, *Anatomical classification of the gastrointestinal tract using ensemble transfer learning*,
   University of Saskatchewan, 2023 ([link](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=xZDlXb8AAAAJ&citation_for_view=xZDlXb8AAAAJ:2osOgNQ5qMEC)).
   They are listed in `splits/gastrolab_ileum_extra.csv` under the file names used in our experiments (`0.png` … `55.png`).
   Because the images were renamed, please contact the corresponding author (see the paper) to obtain the exact set.
3. **Kvasir v2** (only for Table 13): <https://datasets.simula.no/kvasir/>.

Build the folder layout the scripts expect (copies or symlinks only the images listed in `splits/`):

```bash
python scripts/prepare_data.py \
    --hyperkvasir_labeled   /data/hyper-kvasir/labeled-images \
    --hyperkvasir_unlabeled /data/hyper-kvasir/unlabeled-images \
    --gastrolab_ileum       /data/gastrolab_ileum \
    --kvasir                /data/kvasir-dataset-v2 \
    --out dataset
```

```
dataset/
  labeled/<class>/…            HyperKvasir images + the 56 GastroLab images in labeled/ileum/
  labeled/extra_ileum/…        the same 56 images, used only as train-only extras
  unlabeled/{train,val,test}/  22,500 / 1,400 / 1,800 images
  kvasir/labeled/<class>/…
```

### Splits

| File | Used for |
|---|---|
| `hyperkvasir_5fold_expanded.csv` | Main experiments, FixMatch. 4,160 images, 65 Ileum. The 56 GastroLab images are distributed across folds; the original images keep their original folds |
| `hyperkvasir_5fold_original.csv` | Ileum sensitivity analysis and HPO. 4,104 images, 9 Ileum. N extra Ileum images are added to the **training** set only |
| `hyperkvasir_5fold_no_ileum.csv` | Exp. 5 (Ileum removed) |
| `hyperkvasir_unlabeled_subset.csv` | The fixed unlabeled subset (`train`/`val`/`test`) |
| `gastrolab_ileum_extra.csv` | The 56 additional Ileum images and their fold in the expanded split |
| `kvasir_v2_10fold.csv` | Kvasir v2, stratified 10-fold |

In every fold, the held-out fold is the test set. The remaining folds are split 80/20 (stratified, seed 42) into training and validation sets.
Use either the expanded split **or** the original split with train-only extras, never both. The code refuses
to add extra images that are byte-identical to images already in a split.

## Reproducing the paper

Every script takes a YAML config, and any option can be overridden on the command line (`--data_path`, `--folds 0 1`, `--aes convAE`, …).

```bash
# Unsupervised autoencoders (used by Exp. 1 and 3)
python train_only_ae.py --config configs/only_ae.yaml --data_path dataset --save_models

# Ablation study (Table 9): each config runs the 512-, 1024- and 2048-d models over 5 folds
python exp4_train.py --config configs/exp4_default.yaml --data_path dataset --save_models

# FixMatch and the paired t-test
python fixmatch_train.py --config configs/fixmatch.yaml --data_path dataset
python analysis/paired_ttest.py --run_a checkpoints/EXP_4/0/simpleNN_convAE --run_b checkpoints/FIXMATCH/fixmatch_resnet18/0
```

`scripts/reproduce_all.sh` runs everything in order. `scripts/slurm_example.sh` is a template for SLURM clusters.
Outputs go under `checkpoints/<name>/`, with one folder per model and fold, e.g. `checkpoints/EXP_4/0/simpleNN_convAE/fold_2/`. Each fold writes a per-epoch log (`<k>.csv`) and test metrics (`<k>_test.csv`),
plus the best weights when `--save_models` is set. Fold `k` in the code is fold `k+1` in the paper.

Training takes roughly 300 s per epoch for SSR-ALC on one A100, about 7–8 h per fold with early stopping.

### Results reported in the paper (HyperKvasir, 5-fold CV, macro-F1 %)

Mean ± standard deviation over the five test folds (Table 9 of the paper).

| | 512-d | 1024-d | 2048-d |
|---|---|---|---|
| Exp. 1: unsupervised AE | 85.35 ± 1.72 | 81.07 ± 1.50 | 81.94 ± 4.29 |
| Exp. 2: + classifier feedback | 42.73 ± 9.51 | 55.77 ± 8.34 | 58.40 ± 8.71 |
| Exp. 3: + LATE-SMOTE | 88.37 ± 2.21 | 85.16 ± 1.85 | 89.47 ± 0.67 |
| **Exp. 4: SSR-ALC** | **97.29 ± 2.20** | 95.79 ± 2.73 | 96.53 ± 1.52 |
| Exp. 5: SSR-ALC, no Ileum | 97.04 ± 1.29 | 98.87 ± 0.59 | 99.19 ± 0.32 |
| FixMatch (ResNet-18) | 90.92 ± 1.41 | | |

## Implementation notes

These details are in the code but not spelled out in the paper:

- **LATE-SMOTE** uses `imblearn.SMOTE(sampling_strategy="minority")`, so only the smallest class is oversampled.
  Synthetic latent vectors are regenerated every epoch. `n_min` below is the minority-class count among the epoch's labeled features.
  `k_neighbors` differs by engine:
  - Exp. 4 / sensitivity: `min(10, n_min − 1, 1 + ⌊√n_min⌋)`
  - Exp. 3: `min(10, n_min − 1)`
  - HPO: `min(2, n_min − 1)`
- The classification term inside the autoencoder loss (Exp. 2, 4, 5, HPO) uses class-balanced cross-entropy weights.
  The classifier itself is trained with class-balanced cross-entropy in Exp. 1 and 2. In Exp. 3, 4 and HPO it uses unweighted
  cross-entropy on the SMOTE-balanced features, and in Exp. 5 unweighted cross-entropy.
- Which model is evaluated on the test fold: Exp. 1–4, HPO and FixMatch reload the weights from the epoch with the best
  validation macro-F1. Exp. 5 and Kvasir evaluate the model from the last training epoch.
- In `engine_exp_5.py` and `engine_kvasir_no_un.py`, α follows a schedule: each time the classifier's
  learning rate is reduced, α is set to `1 + 9 · (1 − exp(−epoch/100))`, with epoch counted from 1.
- Kvasir (Eq. 4): the code computes `alpha · CE + gamma · labeled reconstruction`, and `beta` is unused.
- The unsupervised autoencoder (`train_only_ae.py`) uses a learning rate of 1e-3. All joint models use 1e-4.
- LR schedulers: `ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-6)` on the AE validation loss and on the validation macro-F1.
  Early stopping: patience 20 (min. improvement 0.001) on validation macro-F1. Exceptions: `train_only_ae.py` stops on the
  validation reconstruction loss and `train_kvasir.py` on validation accuracy.

## Citation

```bibtex
@article{einlou2026ssralc,
  title   = {{SSR-ALC}: Semi-supervised representation learning for anatomical landmark classification of endoscopy images},
  author  = {Einlou, Mokarrameh and Yaghoobian, Omid and Wahid, Khan A.},
  journal = {Biomedical Signal Processing and Control},
  volume  = {123},
  pages   = {110672},
  year    = {2026},
  doi     = {10.1016/j.bspc.2026.110672}
}
```

## License and acknowledgements

Code: MIT (see `LICENSE`). HyperKvasir is CC BY 4.0. Check the terms of GastroLab and Kvasir before redistributing their images.
`visualization/vis.py` includes a confusion-matrix function adapted from mlxtend (BSD-3-Clause, © Sebastian Raschka).

This work was supported in part by the Natural Sciences and Engineering Research Council of Canada (NSERC) and the New Frontiers in Research Fund (NFRF).
