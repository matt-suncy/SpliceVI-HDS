#!/usr/bin/env bash
set -euo pipefail

# End-to-end local pipeline:
# 1) Build model-ready .h5mu from tables in data/
# 2) Validate schema compatibility with SpliceVI
# 3) Create paper-aligned external train/test split (70/30 stratified)
# 4) Start training with train_splicevi.py on external-train split

# CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
ENV_NAME="${ENV_NAME:-splicevi-env}"
MAX_CELLS="${MAX_CELLS:--1}"
MAX_EXPR_FEATURES="${MAX_EXPR_FEATURES:--1}"
MAX_SPLICING_FEATURES="${MAX_SPLICING_FEATURES:--1}"
ATSE_GROUPING_MODE="${ATSE_GROUPING_MODE:-both_anchors}"
MASK_ATSE_THRESHOLD="${MASK_ATSE_THRESHOLD:-0}"
BATCH_KEY="${BATCH_KEY:-seq_batch}"
SPLIT_SEED="${SPLIT_SEED:-42}"
SPLIT_TEST_FRAC="${SPLIT_TEST_FRAC:-0.3}"
SPLIT_AGE_COL="${SPLIT_AGE_COL:-age_days}"
SPLIT_CELLTYPE_COL="${SPLIT_CELLTYPE_COL:-class}"
SPLIT_MIN_STRATUM_SIZE="${SPLIT_MIN_STRATUM_SIZE:-5}"

FULL_H5MU="data/processed/splicevi_custom_input.h5mu"
TRAIN_H5MU="data/processed/splicevi_custom_input_train70.h5mu"
TEST_H5MU="data/processed/splicevi_custom_input_test30.h5mu"

# source "${CONDA_BASE}/etc/profile.d/conda.sh"

eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

mkdir -p data/processed models logs

echo "Building SpliceVI Mudata object from input tables..."

python scripts/build_splicevi_mudata.py \
  --expr-matrix data/Tasic2018_MO_VIS_core.individual.expr.mat.txt \
  --splicing-matrix data/MO_VIS_core.individual.cass.mat.txt \
  --metadata-csvs data/MO_sample_metadata.csv data/VIS_sample_metadata.csv \
  --expr-group-map data/MO_VIS_core.individual2group.expr.conf \
  --as-group-map data/MO_VIS_core.individual2group.as.conf \
  --output-h5mu "${FULL_H5MU}" \
  --atse-grouping-mode "${ATSE_GROUPING_MODE}" \
  --mask-atse-threshold "${MASK_ATSE_THRESHOLD}" \
  --max-cells "${MAX_CELLS}" \
  --max-expr-features "${MAX_EXPR_FEATURES}" \
  --max-splicing-features "${MAX_SPLICING_FEATURES}"

echo "Validating SpliceVI Mudata object..."

python scripts/validate_splicevi_mudata.py \
  --h5mu "${FULL_H5MU}" \
  --atse-grouping-mode "${ATSE_GROUPING_MODE}" \
  --mask-atse-threshold "${MASK_ATSE_THRESHOLD}"

echo "Creating external train/test split (paper-aligned: 70/30 stratified)..."

python scripts/create_test_split.py \
  --train-path "${FULL_H5MU}" \
  --output-train-path "${TRAIN_H5MU}" \
  --output-test-path "${TEST_H5MU}" \
  --test-frac "${SPLIT_TEST_FRAC}" \
  --seed "${SPLIT_SEED}" \
  --stratify-age-col "${SPLIT_AGE_COL}" \
  --stratify-celltype-col "${SPLIT_CELLTYPE_COL}" \
  --min-stratum-size "${SPLIT_MIN_STRATUM_SIZE}"

echo "Validating external TRAIN split..."
python scripts/validate_splicevi_mudata.py \
  --h5mu "${TRAIN_H5MU}" \
  --atse-grouping-mode "${ATSE_GROUPING_MODE}" \
  --mask-atse-threshold "${MASK_ATSE_THRESHOLD}"

echo "Validating external TEST split..."
python scripts/validate_splicevi_mudata.py \
  --h5mu "${TEST_H5MU}" \
  --atse-grouping-mode "${ATSE_GROUPING_MODE}" \
  --mask-atse-threshold "${MASK_ATSE_THRESHOLD}"

echo "Starting training..."

python train_splicevi.py \
  --train_mdata_path "${TRAIN_H5MU}" \
  --model_dir models/custom_baseline_$(date +"%Y%m%d_%H%M%S") \
  --batch_key "${BATCH_KEY}" \
  --max_epochs 25

echo "Pipeline complete."
echo "  Full dataset : ${FULL_H5MU}"
echo "  Train split  : ${TRAIN_H5MU}"
echo "  Test split   : ${TEST_H5MU}"
echo "Check model directory for training outputs."
