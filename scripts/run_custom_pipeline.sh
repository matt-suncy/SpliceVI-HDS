#!/usr/bin/env bash
set -euo pipefail

# End-to-end local pipeline:
# 1) Build model-ready .h5mu from tables in data/
# 2) Validate schema compatibility with SpliceVI
# 3) Start training with train_splicevi.py

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
ENV_NAME="${ENV_NAME:-splicevi-env}"
MAX_CELLS="${MAX_CELLS:--1}"
MAX_EXPR_FEATURES="${MAX_EXPR_FEATURES:--1}"
MAX_SPLICING_FEATURES="${MAX_SPLICING_FEATURES:--1}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

mkdir -p data/processed models logs

echo "Building SpliceVI Mudata object from input tables..."

python scripts/build_splicevi_mudata.py \
  --expr-matrix data/Tasic2018_MO_VIS_core.individual.expr.mat.txt \
  --splicing-matrix data/MO_VIS_core.individual.cass.mat.txt \
  --metadata-csvs data/MO_sample_metadata.csv data/VIS_sample_metadata.csv \
  --expr-group-map data/MO_VIS_core.individual2group.expr.conf \
  --as-group-map data/MO_VIS_core.individual2group.as.conf \
  --output-h5mu data/processed/splicevi_custom_input.h5mu \
  --max-cells "${MAX_CELLS}" \
  --max-expr-features "${MAX_EXPR_FEATURES}" \
  --max-splicing-features "${MAX_SPLICING_FEATURES}"

echo "Validating SpliceVI Mudata object..."

python scripts/validate_splicevi_mudata.py \
  --h5mu data/processed/splicevi_custom_input.h5mu

echo "Starting training..."

python train_splicevi.py \
  --train_mdata_path data/processed/splicevi_custom_input.h5mu \
  --model_dir models/custom_baseline_$(date +"%Y%m%d_%H%M%S") \
  --batch_key None \
  --max_epochs 1 # Just 1 epoch for testing; increase for real training

echo "Pipeline complete. Check model directory for outputs."
