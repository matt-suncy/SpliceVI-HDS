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
    [--output-base <dir>] \
    [--masked-h5mu <path>]... \
    [--masked-resampled] \
    [--impute-filter-boundary-psi] \
    [--skip-precheck] \
    [--min-atse-count <int>] \
    [--use-wandb --wandb-project <name> --wandb-group <name>]

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
BATCH_KEY="seq_batch"
OUTPUT_BASE="logs/eval_runs"
MIN_ATSE_COUNT=15
MASKED_RESAMPLED=false
IMPUTE_FILTER_BOUNDARY_PSI=false
USE_WANDB=false
WANDB_PROJECT=""
WANDB_GROUP=""
WANDB_ENTITY=""
WANDB_LOG_FREQ=1000
SKIP_PRECHECK=false

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
ENV_NAME="${ENV_NAME:-splicevi-env}"

MASKED_H5MU_PATHS=()

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
    --output-base)
      OUTPUT_BASE="$2"
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

if [[ "$USE_WANDB" == true && -z "$WANDB_PROJECT" ]]; then
  echo "--wandb-project is required when --use-wandb is set." >&2
  exit 1
fi

# Accept either a model directory or a direct path to model.pt.
if [[ -f "$MODEL_DIR" && "$(basename "$MODEL_DIR")" == "model.pt" ]]; then
  MODEL_DIR="$(dirname "$MODEL_DIR")"
fi

for p in "$MODEL_DIR" "$TRAIN_H5MU" "$TEST_H5MU"; do
  if [[ ! -e "$p" ]]; then
    echo "Missing required path: $p" >&2
    exit 1
  fi
done

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
  if [[ ${#MASKED_H5MU_PATHS[@]} -gt 0 ]]; then
    EVALS=(masked_impute)
  else
    EVALS=(test_eval)
  fi
  CROSS_FOLD_SPLITS="train"
else
  EVALS=(umap clustering train_eval test_eval cross_fold_classification age_r2_heatmap)
  if [[ ${#MASKED_H5MU_PATHS[@]} -gt 0 ]]; then
    EVALS+=(masked_impute)
  fi
  CROSS_FOLD_SPLITS="both"
fi

mapfile -t _KEY_LINES < <(
  python - "$TRAIN_H5MU" <<'PY'
import sys
import mudata as mu

path = sys.argv[1]
m = mu.read_h5mu(path, backed="r")
cols = list(m["rna"].obs.columns)

preferred_umap = [
    "age_days",
    "broad_cell_type",
    "medium_cell_type",
    "class",
    "subclass",
    "cluster",
    "cell_type",
    "tissue",
]
preferred_cross = [
    "broad_cell_type",
    "medium_cell_type",
    "mouse.id",
    "tissue_celltype",
    "tissue",
    "class",
    "subclass",
    "cluster",
    "cell_type",
    "batch",
]

umap = [k for k in preferred_umap if k in cols]
if not umap and cols:
    umap = [cols[0]]

cross = [k for k in preferred_cross if k in cols]
if not cross:
    if "mouse.id" in cols:
        cross = ["mouse.id"]
    elif umap:
        cross = [umap[0]]
    elif cols:
        cross = [cols[0]]

print("UMAP=" + " ".join(umap[:3]))
print("CROSS=" + " ".join(cross))
PY
)

UMAP_OBS_KEYS=()
CROSS_FOLD_TARGETS=()
for _line in "${_KEY_LINES[@]}"; do
  case "$_line" in
    UMAP=*)
      _vals="${_line#UMAP=}"
      if [[ -n "$_vals" ]]; then
        read -r -a UMAP_OBS_KEYS <<< "$_vals"
      fi
      ;;
    CROSS=*)
      _vals="${_line#CROSS=}"
      if [[ -n "$_vals" ]]; then
        read -r -a CROSS_FOLD_TARGETS <<< "$_vals"
      fi
      ;;
  esac
done

if [[ ${#UMAP_OBS_KEYS[@]} -eq 0 ]]; then
  UMAP_OBS_KEYS=(group_highlighted)
fi
if [[ ${#CROSS_FOLD_TARGETS[@]} -eq 0 ]]; then
  CROSS_FOLD_TARGETS=(mouse.id)
fi

CROSS_FOLD_CLASSIFIERS=(logreg)
CROSS_FOLD_METRICS=(accuracy f1_weighted precision_weighted recall_weighted)

CMD=(
  python eval_splicevi.py
  --train_mdata_path "$TRAIN_H5MU"
  --test_mdata_path "$TEST_H5MU"
  --model_dir "$MODEL_DIR"
  --batch_key "$BATCH_KEY"
  --fig_dir "$FIG_DIR"
  --impute_batch_size 512
  --umap_top_n_celltypes 15
  --umap_obs_keys "${UMAP_OBS_KEYS[@]}"
  --cross_fold_splits "$CROSS_FOLD_SPLITS"
  --cross_fold_targets "${CROSS_FOLD_TARGETS[@]}"
  --cross_fold_k 5
  --cross_fold_classifiers "${CROSS_FOLD_CLASSIFIERS[@]}"
  --cross_fold_metrics "${CROSS_FOLD_METRICS[@]}"
  --evals "${EVALS[@]}"
  --min_atse_count "$MIN_ATSE_COUNT"
)

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
echo "[RUN] Output run dir      : ${RUN_DIR}"
echo "[RUN] Figure dir          : ${FIG_DIR}"
echo "[RUN] Evals               : ${EVALS[*]}"
echo "[RUN] UMAP obs keys       : ${UMAP_OBS_KEYS[*]}"
echo "[RUN] Cross-fold targets  : ${CROSS_FOLD_TARGETS[*]}"
echo "[RUN] Masked files        : ${#MASKED_H5MU_PATHS[@]}"
echo "=============================================================="

printf '%q ' "${CMD[@]}" | tee "${RUN_DIR}/launch_command.sh"
echo | tee -a "${RUN_DIR}/launch_command.sh"

"${CMD[@]}" 2>&1 | tee "${RUN_DIR}/eval.log"

echo "[DONE] Evaluation finished. See ${RUN_DIR}"
