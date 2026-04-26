#!/usr/bin/env bash

# Just a place to put commands so it's easy to copy-paste and modify for different runs. 
# Not meant to be a polished script.

model_dir=/mnt/chromatin/home/ms7280/high-dim-stats/final-proj/SpliceVI-HDS/models/preprocessv2_baseline_20260413_202607
train_h5mu=/mnt/chromatin/home/ms7280/high-dim-stats/final-proj/SpliceVI-HDS/data/processed/splicevi_custom_input_train70.h5mu
test_h5mu=/mnt/chromatin/home/ms7280/high-dim-stats/final-proj/SpliceVI-HDS/data/processed/splicevi_custom_input_test30.h5mu
bash scripts/run_staged_eval.sh \
    --mode full \
    --model-dir $model_dir \
    --train-h5mu $train_h5mu \
    --test-h5mu $test_h5mu \
    --batch-key None \
    [--impute-filter-boundary-psi] \
    [--skip-precheck] \
    [--min-atse-count <int>] \
    [--use-wandb --wandb-project <name> --wandb-group <name>]


python eval_splicevi.py \
    --train_mdata_path data/processed/splicevi_custom_input_train70.h5mu \
    --test_mdata_path data/processed/splicevi_custom_input_test30.h5mu \
    --model_dir models/custom_baseline_20260419_084951 \
    --batch_key seq_batch \
    --fig_dir logs/eval_runs/preprocessv2_eval \
    --impute_batch_size 512 \
    --umap_top_n_celltypes 15 \
    --umap_obs_keys class subclass cluster \
    --cross_fold_splits both \
    --cross_fold_targets class subclass cluster mouse.id tissue_celltype tissue \
    --cross_fold_k 5 \
    --cross_fold_classifiers logreg \
    --cross_fold_metrics accuracy f1_weighted precision_weighted recall_weighted \
    --evals umap clustering train_eval test_eval cross_fold_classification age_r2_heatmap \
    --min_atse_count 15

python train_splicevi.py \
    --train_mdata_path data/processed/splicevi_custom_input_train70.h5mu \
  --model_dir models/custom_baseline_$(date +"%Y%m%d_%H%M%S") \
  --batch_key None \
  --max_epochs 25 \
  --use_wandb --wandb_project splicevi-training
