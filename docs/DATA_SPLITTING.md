# Data Splitting in the Custom SpliceVI Pipeline

This document defines how data is split in the custom workflow and how it maps to the SpliceVI paper logic.

## Goals

The split policy is designed to match the paper's two-level split behavior:

1. External split on the full dataset: 70% train, 30% test.
2. External split is stratified on age and cell type.
3. Internal split during training: 90% train, 10% validation within external-train.

In this repository, external stratification uses:

1. Age column: `age_numeric`.
2. Cell-type column: `class`.

## Where Splitting Happens

External split script:

1. `scripts/create_test_split.py`

Pipeline wiring:

1. `scripts/run_custom_pipeline.sh`

Training entrypoint (internal split handled by model defaults):

1. `train_splicevi.py`

## Data Flow

1. Build full MuData from raw tables:
   - `scripts/build_splicevi_mudata.py`
   - Output: `data/processed/splicevi_custom_input.h5mu`
2. Validate full MuData schema:
   - `scripts/validate_splicevi_mudata.py`
3. Create external stratified split:
   - Input: full MuData
   - Outputs:
     - `data/processed/splicevi_custom_input_train70.h5mu`
     - `data/processed/splicevi_custom_input_test30.h5mu`
4. Validate both split artifacts.
5. Train on external-train file only.
6. Evaluate with train/test pair from the same split run.

## External Split Algorithm

Implemented in `scripts/create_test_split.py`.

### Inputs

1. Full MuData path (`--train-path`).
2. Output paths (`--output-train-path`, `--output-test-path`).
3. Split parameters:
   - `--test-frac` (default `0.3`)
   - `--seed` (default `42`)
   - `--stratify-age-col` (default `age_numeric`)
   - `--stratify-celltype-col` (default `class`)
   - `--min-stratum-size` (default `5`)

### Steps

1. Read full MuData.
2. Build stratum key per cell: `age_numeric || class`.
3. Fill missing stratification values with `unknown`.
4. Collapse rare strata (count < `min_stratum_size`) to `other`.
5. Perform deterministic split with `train_test_split(..., stratify=strata, random_state=seed)`.
6. Create complementary train/test MuData files using split indices.
7. Add split audit metadata columns (unless disabled):
   - `_split_role` (`train` or `test`)
   - `_split_seed`
   - `_split_source_file`
   - `_split_strata_key`
8. Assert there is no cell-ID overlap between train and test outputs.

## Internal Train/Validation Split

No separate validation file is created by default.

1. `train_splicevi.py` trains on external-train MuData.
2. `SPLICEVI.train()` is called with default `train_size=None`, `validation_size=None`.
3. scvi `DataSplitter` default behavior applies: effective internal 90/10 train/validation.

This matches the paper's internal split logic while keeping external test fully held out.

## Effective Proportions

Given full dataset size $N$:

1. External train: $0.70N$
2. External test: $0.30N$
3. Internal train (inside external train): $0.90 \times 0.70N = 0.63N$
4. Internal val (inside external train): $0.10 \times 0.70N = 0.07N$

## Reproducibility

Control knobs:

1. `--seed` controls deterministic external split.
2. `--test-frac` controls global train/test ratio.
3. Stratification columns are explicit CLI parameters.
4. Split metadata columns are persisted in outputs for audit.

Recommended practice:

1. Keep split seed fixed across comparable experiments.
2. Save the exact split command in logs.
3. Use train/test outputs generated from the same source file and split run.

## Leakage Prevention Rules

1. Never train on the full unsplit file when using paper-aligned evaluation.
2. Train only on `*_train70.h5mu`.
3. Evaluate on `*_test30.h5mu` from the same split operation.
4. Ensure no overlap in cell IDs between train and test outputs.

## Validation Checklist

Before training/evaluation:

1. `train_n + test_n == full_n`.
2. `train_obs_names ∩ test_obs_names == ∅`.
3. `rna.var_names` are identical in train and test.
4. `splicing.var_names` are identical in train and test.
5. Split ratio is close to 70/30 (integer rounding expected).
6. Age and class distributions are reasonably preserved between full/train/test.

## Commands

### Create split artifacts

```bash
python scripts/create_test_split.py \
  --train-path data/processed/splicevi_custom_input.h5mu \
  --output-train-path data/processed/splicevi_custom_input_train70.h5mu \
  --output-test-path data/processed/splicevi_custom_input_test30.h5mu \
  --test-frac 0.3 \
  --seed 42 \
  --stratify-age-col age_numeric \
  --stratify-celltype-col class \
  --min-stratum-size 5
```

### Run full custom pipeline with paper-aligned split

```bash
bash scripts/run_custom_pipeline.sh
```

Environment overrides for split behavior:

```bash
SPLIT_SEED=42 \
SPLIT_TEST_FRAC=0.3 \
SPLIT_AGE_COL=age_numeric \
SPLIT_CELLTYPE_COL=class \
SPLIT_MIN_STRATUM_SIZE=5 \
bash scripts/run_custom_pipeline.sh
```

## Failure Modes

1. Missing `age_numeric` or `class` columns:
   - Split fails with explicit column error.
2. Too many rare strata:
   - Rare strata collapse to `other` before splitting.
3. Stratified split still impossible:
   - Script logs reason and falls back to deterministic random split unless stratification is disabled explicitly.
4. Accidental train/test leakage:
   - Script raises error if overlap is detected.
