#!/usr/bin/env bash
set -euo pipefail

# Staged SpliceVI evaluation runner for retrained checkpoints.
# Stage 1 (smoke): quick compatibility check (masked_impute if masked inputs are provided; otherwise test_eval).
# Stage 2 (full): full evaluation suite with the same model/data wiring.

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_staged_eval.sh \
    --mode smoke|full \
    --model-dir <path> \
    --train-h5mu <path> \
    --test-h5mu <path> \
    [--batch-key <obs_col_or_None>] \
    [--mapping-csv <path>] \
    [--output-base <dir>] \
    [--eval <block>]... \
    [--umap-obs-key <obs_key>]... \
    [--cross-fold-target <obs_key>]... \
    [--cross-fold-classifier logreg|rf|ridge]... \
    [--cross-fold-metric accuracy|f1_weighted|precision_weighted|recall_weighted|r2|mae|rmse]... \
    [--cross-fold-splits train|test|both] \
    [--cross-fold-k <int>] \
    [--age-target-col <obs_key>] \
    [--impute-batch-size <int>] \
    [--umap-top-n-celltypes <int>] \
    [--masked-h5mu <path>]... \
    [--masked-resampled] \
    [--impute-filter-boundary-psi] \
    [--skip-precheck] \
    [--min-atse-count <int>] \
    [--use-wandb --wandb-project <name> --wandb-group <name>] \
    [--wandb-run-name <name>] \
    [--wandb-run-name-prefix <prefix>]

Examples:
  # Smoke run with masked imputation
  bash scripts/run_staged_eval.sh \
    --mode smoke \
    --model-dir models/custom_baseline_20260409_120000 \
    --train-h5mu data/processed/train.h5mu \
    --test-h5mu data/processed/test.h5mu \
    --masked-h5mu data/processed/test_masked_25.h5mu \
    --masked-resampled

  # Full run after smoke passes
  bash scripts/run_staged_eval.sh \
    --mode full \
    --model-dir models/custom_baseline_20260409_120000 \
    --train-h5mu data/processed/train.h5mu \
    --test-h5mu data/processed/test.h5mu \
    --masked-h5mu data/processed/test_masked_25.h5mu \
    --masked-h5mu data/processed/test_masked_50.h5mu \
    --masked-resampled
EOF
}

MODE=""
MODEL_DIR=""
TRAIN_H5MU=""
TEST_H5MU=""
BATCH_KEY="None"
MAPPING_CSV=""
OUTPUT_BASE="logs/eval_runs"
MIN_ATSE_COUNT=15
MASKED_RESAMPLED=false
IMPUTE_FILTER_BOUNDARY_PSI=false
AGE_TARGET_COL="age_days"
IMPUTE_BATCH_SIZE=512
UMAP_TOP_N_CELLTYPES=15
RUN_SCRIPT_PATH="eval_splicevi.py"
USE_WANDB=false
WANDB_PROJECT=""
WANDB_GROUP=""
WANDB_ENTITY=""
WANDB_RUN_NAME=""
WANDB_RUN_NAME_PREFIX="staged_eval"
WANDB_LOG_FREQ=1000
SKIP_PRECHECK=false
CROSS_FOLD_SPLITS_OVERRIDE=""
CROSS_FOLD_K=5

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
ENV_NAME="${ENV_NAME:-splicevi-env}"

MASKED_H5MU_PATHS=()
EVALS=()
UMAP_OBS_KEYS=()
CROSS_FOLD_TARGETS=()
CROSS_FOLD_CLASSIFIERS=()
CROSS_FOLD_METRICS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --model-dir)
      MODEL_DIR="$2"
      shift 2
      ;;
    --train-h5mu)
      TRAIN_H5MU="$2"
      shift 2
      ;;
    --test-h5mu)
      TEST_H5MU="$2"
      shift 2
      ;;
    --batch-key)
      BATCH_KEY="$2"
      shift 2
      ;;
    --mapping-csv)
      MAPPING_CSV="$2"
      shift 2
      ;;
    --output-base)
      OUTPUT_BASE="$2"
      shift 2
      ;;
    --eval)
      EVALS+=("$2")
      shift 2
      ;;
    --umap-obs-key)
      UMAP_OBS_KEYS+=("$2")
      shift 2
      ;;
    --cross-fold-target)
      CROSS_FOLD_TARGETS+=("$2")
      shift 2
      ;;
    --cross-fold-classifier)
      CROSS_FOLD_CLASSIFIERS+=("$2")
      shift 2
      ;;
    --cross-fold-metric)
      CROSS_FOLD_METRICS+=("$2")
      shift 2
      ;;
    --cross-fold-splits)
      CROSS_FOLD_SPLITS_OVERRIDE="$2"
      shift 2
      ;;
    --cross-fold-k)
      CROSS_FOLD_K="$2"
      shift 2
      ;;
    --age-target-col)
      AGE_TARGET_COL="$2"
      shift 2
      ;;
    --impute-batch-size)
      IMPUTE_BATCH_SIZE="$2"
      shift 2
      ;;
    --umap-top-n-celltypes)
      UMAP_TOP_N_CELLTYPES="$2"
      shift 2
      ;;
    --masked-h5mu)
      MASKED_H5MU_PATHS+=("$2")
      shift 2
      ;;
    --masked-resampled)
      MASKED_RESAMPLED=true
      shift
      ;;
    --impute-filter-boundary-psi)
      IMPUTE_FILTER_BOUNDARY_PSI=true
      shift
      ;;
    --skip-precheck)
      SKIP_PRECHECK=true
      shift
      ;;
    --min-atse-count)
      MIN_ATSE_COUNT="$2"
      shift 2
      ;;
    --use-wandb)
      USE_WANDB=true
      shift
      ;;
    --wandb-project)
      WANDB_PROJECT="$2"
      shift 2
      ;;
    --wandb-group)
      WANDB_GROUP="$2"
      shift 2
      ;;
    --wandb-entity)
      WANDB_ENTITY="$2"
      shift 2
      ;;
    --wandb-run-name)
      WANDB_RUN_NAME="$2"
      shift 2
      ;;
    --wandb-run-name-prefix)
      WANDB_RUN_NAME_PREFIX="$2"
      shift 2
      ;;
    --wandb-log-freq)
      WANDB_LOG_FREQ="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$MODE" || -z "$MODEL_DIR" || -z "$TRAIN_H5MU" || -z "$TEST_H5MU" ]]; then
  echo "Missing required arguments." >&2
  usage
  exit 1
fi

if [[ "$MODE" != "smoke" && "$MODE" != "full" ]]; then
  echo "--mode must be one of: smoke, full" >&2
  exit 1
fi

if [[ -n "$CROSS_FOLD_SPLITS_OVERRIDE" && "$CROSS_FOLD_SPLITS_OVERRIDE" != "train" && "$CROSS_FOLD_SPLITS_OVERRIDE" != "test" && "$CROSS_FOLD_SPLITS_OVERRIDE" != "both" ]]; then
  echo "--cross-fold-splits must be one of: train, test, both" >&2
  exit 1
fi

if [[ "$USE_WANDB" == true && -z "$WANDB_PROJECT" ]]; then
  echo "--wandb-project is required when --use-wandb is set." >&2
  exit 1
fi

# Accept either a model directory or a direct path to any .pt file.
if [[ -f "$MODEL_DIR" && "$MODEL_DIR" == *.pt ]]; then
  MODEL_DIR="$(dirname "$MODEL_DIR")"
fi

for p in "$MODEL_DIR" "$TRAIN_H5MU" "$TEST_H5MU"; do
  if [[ ! -e "$p" ]]; then
    echo "Missing required path: $p" >&2
    exit 1
  fi
done

if [[ -n "$MAPPING_CSV" && ! -e "$MAPPING_CSV" ]]; then
  echo "Mapping CSV not found: $MAPPING_CSV" >&2
  exit 1
fi

for p in "${MASKED_H5MU_PATHS[@]}"; do
  if [[ ! -e "$p" ]]; then
    echo "Masked file not found: $p" >&2
    exit 1
  fi
done

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

if [[ "$SKIP_PRECHECK" == true ]]; then
  echo "[PRECHECK] Skipped by flag (--skip-precheck)."
else
  echo "[PRECHECK] Validating TRAIN MuData: ${TRAIN_H5MU}"
  python scripts/validate_splicevi_mudata.py --h5mu "$TRAIN_H5MU"

  echo "[PRECHECK] Validating TEST MuData: ${TEST_H5MU}"
  python scripts/validate_splicevi_mudata.py --h5mu "$TEST_H5MU"
fi

TS="$(date +"%Y%m%d_%H%M%S")"
MODEL_BASENAME="$(basename "$MODEL_DIR")"
RUN_NAME="eval_${MODE}_${MODEL_BASENAME}_${TS}"
RUN_DIR="${OUTPUT_BASE}/${RUN_NAME}"
FIG_DIR="${RUN_DIR}/figures"
mkdir -p "$FIG_DIR"

if [[ "$MODE" == "smoke" ]]; then
  if [[ ${#EVALS[@]} -eq 0 ]]; then
    if [[ ${#MASKED_H5MU_PATHS[@]} -gt 0 ]]; then
      EVALS=(masked_impute)
    else
      EVALS=(test_eval)
    fi
  fi
  CROSS_FOLD_SPLITS="train"
else
  if [[ ${#EVALS[@]} -eq 0 ]]; then
    EVALS=(umap clustering train_eval test_eval cross_fold_classification age_r2_heatmap)
  fi
  CROSS_FOLD_SPLITS="both"
fi

if [[ -n "$CROSS_FOLD_SPLITS_OVERRIDE" ]]; then
  CROSS_FOLD_SPLITS="$CROSS_FOLD_SPLITS_OVERRIDE"
fi

if [[ ${#MASKED_H5MU_PATHS[@]} -gt 0 ]]; then
  _has_masked=false
  for _eval in "${EVALS[@]}"; do
    if [[ "$_eval" == "masked_impute" ]]; then
      _has_masked=true
      break
    fi
  done
  if [[ "$_has_masked" == false ]]; then
    EVALS+=(masked_impute)
  fi
fi

if [[ ${#UMAP_OBS_KEYS[@]} -eq 0 ]]; then
  UMAP_OBS_KEYS=(class subclass age_days)
fi
if [[ ${#CROSS_FOLD_TARGETS[@]} -eq 0 ]]; then
  CROSS_FOLD_TARGETS=(class subclass age_days)
fi
if [[ ${#CROSS_FOLD_CLASSIFIERS[@]} -eq 0 ]]; then
  CROSS_FOLD_CLASSIFIERS=(logreg)
fi
if [[ ${#CROSS_FOLD_METRICS[@]} -eq 0 ]]; then
  CROSS_FOLD_METRICS=(accuracy f1_weighted precision_weighted recall_weighted)
fi

CMD=(
  python "$RUN_SCRIPT_PATH"
  --train_mdata_path "$TRAIN_H5MU"
  --test_mdata_path "$TEST_H5MU"
  --model_dir "$MODEL_DIR"
  --batch_key "$BATCH_KEY"
  --fig_dir "$FIG_DIR"
  --age_target_col "$AGE_TARGET_COL"
  --impute_batch_size "$IMPUTE_BATCH_SIZE"
  --umap_top_n_celltypes "$UMAP_TOP_N_CELLTYPES"
  --umap_obs_keys "${UMAP_OBS_KEYS[@]}"
  --cross_fold_splits "$CROSS_FOLD_SPLITS"
  --cross_fold_targets "${CROSS_FOLD_TARGETS[@]}"
  --cross_fold_k "$CROSS_FOLD_K"
  --cross_fold_classifiers "${CROSS_FOLD_CLASSIFIERS[@]}"
  --cross_fold_metrics "${CROSS_FOLD_METRICS[@]}"
  --evals "${EVALS[@]}"
  --min_atse_count "$MIN_ATSE_COUNT"
)

if [[ -n "$MAPPING_CSV" ]]; then
  CMD+=(--mapping_csv "$MAPPING_CSV")
fi
if [[ ${#MASKED_H5MU_PATHS[@]} -gt 0 ]]; then
  CMD+=(--masked_test_mdata_paths "${MASKED_H5MU_PATHS[@]}")
fi
if [[ "$MASKED_RESAMPLED" == true ]]; then
  CMD+=(--masked_test_mdata_is_resampled)
fi
if [[ "$IMPUTE_FILTER_BOUNDARY_PSI" == true ]]; then
  CMD+=(--impute_filter_boundary_psi)
fi

if [[ "$USE_WANDB" == true ]]; then
  CMD+=(--use_wandb --wandb_project "$WANDB_PROJECT" --wandb_log_freq "$WANDB_LOG_FREQ")
  if [[ -z "$WANDB_RUN_NAME" ]]; then
    WANDB_RUN_NAME="${WANDB_RUN_NAME_PREFIX}_${MODEL_BASENAME}"
  fi
  CMD+=(--wandb_run_name "$WANDB_RUN_NAME")
  if [[ -n "$WANDB_GROUP" ]]; then
    CMD+=(--wandb_group "$WANDB_GROUP")
  fi
  if [[ -n "$WANDB_ENTITY" ]]; then
    CMD+=(--wandb_entity "$WANDB_ENTITY")
  fi
fi

echo "=============================================================="
echo "[RUN] Stage               : ${MODE}"
echo "[RUN] Model               : ${MODEL_DIR}"
echo "[RUN] Train MuData        : ${TRAIN_H5MU}"
echo "[RUN] Test MuData         : ${TEST_H5MU}"
echo "[RUN] Mapping CSV         : ${MAPPING_CSV:-"(none)"}"
echo "[RUN] Output run dir      : ${RUN_DIR}"
echo "[RUN] Figure dir          : ${FIG_DIR}"
echo "[RUN] Age target col      : ${AGE_TARGET_COL}"
echo "[RUN] Evals               : ${EVALS[*]}"
echo "[RUN] UMAP obs keys       : ${UMAP_OBS_KEYS[*]}"
echo "[RUN] Cross-fold targets  : ${CROSS_FOLD_TARGETS[*]}"
echo "[RUN] Cross-fold splits   : ${CROSS_FOLD_SPLITS}"
echo "[RUN] Masked files        : ${#MASKED_H5MU_PATHS[@]}"
echo "=============================================================="

printf '%q ' "${CMD[@]}" | tee "${RUN_DIR}/launch_command.sh"
echo | tee -a "${RUN_DIR}/launch_command.sh"

"${CMD[@]}" 2>&1 | tee "${RUN_DIR}/eval.log"

echo "[DONE] Evaluation finished. See ${RUN_DIR}"
