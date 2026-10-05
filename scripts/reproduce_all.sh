#!/usr/bin/env bash
# Runs every experiment in the paper, in dependency order.
# Usage: bash scripts/reproduce_all.sh /path/to/dataset
# Each step is independent; comment out what you don't need.
# Runtime on one A100 (80 GB): ~7-8 h per fold for the joint models.
set -euo pipefail
DATA=${1:-dataset}

# 0) Unsupervised AE pretraining (needed by Experiments 1 and 3)
python train_only_ae.py --config configs/only_ae.yaml --data_path "$DATA" --save_models

# 1) Ablation study, Table 9 / 10 (5-fold CV, three latent sizes each)
python exp1_train.py --config configs/exp1_default.yaml --data_path "$DATA" --save_models
python exp2_train.py --config configs/exp2_default.yaml --data_path "$DATA" --save_models
python exp3_train.py --config configs/exp3_default.yaml --data_path "$DATA" --save_models
python exp4_train.py --config configs/exp4_default.yaml --data_path "$DATA" --save_models
python exp5_train.py --config configs/exp5_default.yaml --data_path "$DATA" --save_models

# 2) Hyperparameter grid search, Table 6 (fold 5 only)
python hpo_train.py --config configs/hpo.yaml --data_path "$DATA" --save_models
# Cross-fold check of a selected configuration, Table 7 (edit search_space in the config)
python hpo_train.py --config configs/hpo_crossfold.yaml --data_path "$DATA" --save_models

# 3) Ileum sensitivity analysis, Fig. 9
for N in 0 10 20 30 40 50 56; do
  python exp4_train.py --config configs/ileum_sensitivity.yaml --data_path "$DATA" --extra_img_n "$N"
done
python analysis/ileum_sensitivity_summary.py --run_dir checkpoints/ileum_sensitivity

# 4) FixMatch baseline, Tables 14 / 15, and the paired t-test
python fixmatch_train.py --config configs/fixmatch.yaml --data_path "$DATA"
python analysis/paired_ttest.py --run_a checkpoints/EXP_4/0/simpleNN_convAE --run_b checkpoints/FIXMATCH/fixmatch_resnet18/0

# 5) Kvasir v2 transfer, Table 13 (needs the Experiment 4 weights and the Kvasir v2 images)
# python train_kvasir.py --config configs/kvasir.yaml --data_path /path/to/kvasir
