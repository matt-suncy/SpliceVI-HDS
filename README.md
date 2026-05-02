# Analysis and Evaluation of SpliceVI Joint Embeddings on Mouse Neuronal Data

Multimodal VAE for joint modeling of alternative splicing (PSI) and gene expression from single-cell data. Built as a fork of [SpliceVI](https://github.com/daklab/SpliceVI).

---

## Installation

### 1. Create and activate the environment

```bash
conda create -n splicevi-env python=3.12
conda activate splicevi-env
```

### 2. Install the package

```bash
git clone https://github.com/daklab/SpliceVI.git
cd SpliceVI
pip install -e .
```

This installs all dependencies automatically from `pyproject.toml`. For W&B logging, additionally run `pip install wandb`.

---

## Repository Structure

```
SpliceVI/
├── src/splicevi/
│   ├── splicevi.py          # SPLICEVI model class for GE+AS (training, inference, DE, DS)
│   ├── splicevae.py         # VAE module for GE+AS (encoder/decoder architecture + all mixers)
│   ├── partialvae.py        # AS Missingness-Aware Partial VAE module
│   ├── eddisplice.py        # EDDISPLICE model class for single-modality AS-only VAE using PARTIALVAE
│   └── __init__.py          # Package exports
├── train_splicevi.py            # Training entry point (CLI argument parsing, model setup, fit)
├── eval_splicevi.py             # Evaluation entry point (UMAPs, metrics, imputation)
├── slurm_train_splicevi.sh      # SLURM job script — takes modality_weights as $1, edit hyperparameters here
├── submit_train_jobs.sh         # Submits one SLURM job per mixer variant (all 7 at once)
├── eval_splicevi.sh             # SLURM job script for evaluation
├── scripts/
│   ├── build_splicevi_mudata.py        # Build .h5mu from raw expression/splicing tables
│   ├── validate_splicevi_mudata.py     # Check required layers/fields are present
│   ├── create_test_split.py            # Stratified 70/30 train/test split
│   ├── multinomial_resampling_masking.py  # Generate masked test files for imputation eval
│   └── run_staged_eval.sh              # Smoke → full evaluation runner
├── pyproject.toml           # Package configuration and dependencies
└── requirements.txt         # Full conda environment freeze (reference only)
```

---

## SpliceVI Model

The core PyTorch module `SPLICEVAE` in `src/splicevi/splicevae.py` contains the full dual-encoder/dual-decoder architecture, including all latent mixers. See [docs/splicevae_model.md](docs/splicevae_model.md) for detailed parameter descriptions.

---

### Joint Latent Mixing

The `modality_weights` parameter selects how the expression and splicing posteriors are
combined into a shared latent `z`. See [docs/latent_mixers.md](docs/latent_mixers.md) for
full details on each mixer's implementation, parameters, and warmup behaviour.

| `modality_weights` | Class | Learnable | Description |
|---|---|---|---|
| `equal` | *(built-in)* | No | Weighted average with uniform weights |
| `universal` | *(built-in)* | Yes | Single scalar weight shared across all cells |
| `cell` | *(built-in)* | Yes | Per-cell weight vector `(n_obs, 2)` |
| `concatenate` | *(built-in)* | No | Concatenates both latents; doubles the effective latent dimension |
| `sum` | `SumMixer` | No | Elementwise sum of both latent vectors |
| `product` | `ProductMixer` | No | Elementwise product of both latent vectors |
| `gating` | `GatingMixer` | Yes | Dimension-wise sigmoid gate from concatenated means; same gate applied to mean and variance |
| `cross_attention` | `CrossAttentionMixer(reverse=False)` | Yes | Per-dimension attention: AS queries GE |
| `cross_attention_reverse` | `CrossAttentionMixer(reverse=True)` | Yes | Per-dimension attention: GE queries AS |
| `mlp` | `MLPMixer` | Yes | Two-layer MLP on concatenated latents; separate networks for mean and log-variance |


---

## Usage

### Training

**Single mixer variant** — edit hyperparameters in `slurm_train_splicevi.sh`, then pass
the desired `modality_weights` value as the first argument:

```bash
sbatch slurm_train_splicevi.sh cross_attention
sbatch slurm_train_splicevi.sh equal
```

Model output lands in `models/splicevi_<modality_weights>_<timestamp>/`.
W&B run name and group are derived from the same argument.

**All mixer variants at once** — `submit_train_jobs.sh` submits one SLURM job per variant:

```bash
bash submit_train_jobs.sh
# submits: concatenate, sum, product, cross_attention, cross_attention_reverse, gating, mlp
```

**Without SLURM** (interactive / local):

```bash
python train_splicevi.py \
  --train_mdata_path data/processed/splicevi_custom_input_train70.h5mu \
  --model_dir models/my_run \
  --modality_weights cross_attention
# full option list:
python train_splicevi.py --help
```

### Evaluation

```bash
sbatch eval_splicevi.sh
# or directly:
python eval_splicevi.py --help
```

## Custom Data Workflow (Tables -> .h5mu -> Train)

If your data starts as separate expression/splicing/metadata tables in `data/`, use:

1. Build a model-ready MuData file:

```bash
python scripts/build_splicevi_mudata.py \
  --expr-matrix data/Tasic2018_MO_VIS_core.individual.expr.mat.txt \
  --splicing-matrix data/MO_VIS_core.individual.cass.mat.txt \
  --metadata-csvs data/MO_sample_metadata.csv data/VIS_sample_metadata.csv \
  --expr-group-map data/MO_VIS_core.individual2group.expr.conf \
  --as-group-map data/MO_VIS_core.individual2group.as.conf \
  --atse-grouping-mode both_anchors \
  --mask-atse-threshold 0 \
  --output-h5mu data/processed/splicevi_custom_input.h5mu
```

For a quick smoke test on a subset, add:

```bash
  --max-cells 512 --max-expr-features 5000 --max-splicing-features 5000
```

2. Validate that the generated `.h5mu` has all required fields/layers:

```bash
python scripts/validate_splicevi_mudata.py \
  --h5mu data/processed/splicevi_custom_input.h5mu \
  --atse-grouping-mode both_anchors \
  --mask-atse-threshold 0
```

3. Create paper-aligned external split artifacts (70/30 stratified on `age_days` + `class`):

```bash
python scripts/create_test_split.py \
  --train-path data/processed/splicevi_custom_input.h5mu \
  --output-train-path data/processed/splicevi_custom_input_train70.h5mu \
  --output-test-path data/processed/splicevi_custom_input_test30.h5mu \
  --test-frac 0.3 \
  --seed 42 \
  --stratify-age-col age_days \
  --stratify-celltype-col class
```

4. Train using the external-train split file:

```bash
python train_splicevi.py \
  --train_mdata_path data/processed/splicevi_custom_input_train70.h5mu \
  --model_dir models/custom_baseline_run \
  --batch_key seq_batch
```

5. (Optional, for imputation eval) Generate masked TEST artifacts:

```bash
python scripts/multinomial_resampling_masking.py \
  --input-test-h5mu data/processed/splicevi_custom_input_test30.h5mu \
  --output-dir data/processed/masked_impute \
  --mask-fracs 0.25 0.50 \
  --mode resampled
```

This writes files like:
- `RESAMPLED_25_PERCENT_<test_stem>.h5mu`
- `RESAMPLED_50_PERCENT_<test_stem>.h5mu`

Use `--mode legacy` (or `--mode both`) if you need the legacy masked layers
(`junc_ratio_masked_original`, `junc_ratio_masked_bin_mask`).

### How masked imputation evaluation is computed

`eval_splicevi.py` evaluates imputation only when `masked_impute` is in `--evals`
and masked files are passed via `--masked_test_mdata_paths`.

For each masked file:

1. Load masked MuData and enforce feature compatibility with training features.
2. Run model inference with `get_normalized_splicing(...)` to predict PSI for all
   cell-junction pairs.
3. Build the evaluation mask:
   - **Resampled mode** (`--masked_test_mdata_is_resampled`):
     - ground truth = `splicing.layers['junc_ratio_original']`
     - include entries where PSI `> 0`
     - if `--impute_filter_boundary_psi` is set, further require PSI `< 1`
     - if `--min_atse_count != -1`, further require
       `splicing.layers['cell_by_cluster_matrix_original'] >= min_atse_count`
   - **Legacy mode**:
     - ground truth = `splicing.layers['junc_ratio_masked_original']`
     - eval mask = `splicing.layers['junc_ratio_masked_bin_mask']`
4. Extract `(ground_truth, prediction)` pairs only at nonzero entries of the final
   mask and compute metrics.

Reported metrics per masked file:
- `pearson`, `spearman`
- `l1_mean`, `l1_median`, `l1_p90`
- `pred_min`, `pred_max`
- `smape`, `cosine_sim`, `minmax_ratio`, `rmse`
- `n_eval_entries` and run settings (`impute_batch_size`,
  `impute_filter_boundary_psi`, `min_atse_count`)

Artifacts:
- Console logs under each eval run directory
- `figures/imputation_metrics.csv` (one row per masked file, including zero-entry
  cases as NaN metrics)

One-command helper:

```bash
bash scripts/run_custom_pipeline.sh
```

Detailed split behavior and reproducibility checklist:

- See [docs/DATA_SPLITTING.md](docs/DATA_SPLITTING.md)

## Retrained Model Evaluation (Smoke -> Full)

Use the staged evaluator to run a fast compatibility check first, then a full evaluation sweep with the same model and data wiring.

Smoke run (recommended first):

```bash
bash scripts/run_staged_eval.sh \
  --mode smoke \
  --model-dir models/custom_baseline_run \
  --train-h5mu data/processed/train_splicevi_input.h5mu \
  --test-h5mu data/processed/test_splicevi_input.h5mu \
  --masked-h5mu data/processed/masked_impute/RESAMPLED_25_PERCENT_test_splicevi_input.h5mu \
  --masked-resampled
```

Full run (after smoke passes):

```bash
bash scripts/run_staged_eval.sh \
  --mode full \
  --model-dir models/custom_baseline_run \
  --train-h5mu data/processed/train_splicevi_input.h5mu \
  --test-h5mu data/processed/test_splicevi_input.h5mu \
  --masked-h5mu data/processed/masked_impute/RESAMPLED_25_PERCENT_test_splicevi_input.h5mu \
  --masked-h5mu data/processed/masked_impute/RESAMPLED_50_PERCENT_test_splicevi_input.h5mu \
  --masked-resampled
```

What this runner does:

- Validates both train/test `.h5mu` files with `scripts/validate_splicevi_mudata.py` before evaluation.
- In `smoke` mode, runs `masked_impute` if masked files are provided; otherwise falls back to `test_eval`.
- In `full` mode, runs `umap`, `clustering`, `train_eval`, `test_eval`, `cross_fold_classification`, `age_r2_heatmap`, and optionally `masked_impute`.
- Writes run artifacts to `logs/eval_runs/eval_<mode>_<model>_<timestamp>/` including `eval.log` and `launch_command.sh` for reproducibility.

### Required Output Schema for `train_splicevi.py`

The builder writes a `.h5mu` with:

- Modalities: `rna` and `splicing`
- `rna.layers['length_norm']`
- `rna.obsm['X_library_size']`
- `rna.var['modality'] == 'Gene_Expression'`
- `splicing.layers['junc_ratio']`
- `splicing.layers['cell_by_junction_matrix']`
- `splicing.layers['cell_by_cluster_matrix']`
- `splicing.layers['psi_mask']`
- `splicing.var['modality'] == 'Splicing'`
- `splicing.var['event_id']` (ATSE grouping)
- `obs['donor_id']`
- `obs['age_days']` (numeric; mirrored to `age_numeric` for compatibility)

### Notes on Splicing Layer Construction

The builder derives required splicing layers directly from the event IDs in
`MO_VIS_core.individual.cass.mat.txt`:

- Each event row is first expanded into two intermediate junction rows:
  - upstream junction `(upstream_end, cassette_start)`
  - downstream junction `(cassette_end, downstream_start)`
- Intermediate rows are then collapsed to unique genomic junction features using
  key `(event_type, gene_id, junction_side, junction_start, junction_end)`.
  This removes duplicate per-row junction artifacts while preserving gene-aware
  coordinates and junction-side identity.
- Junction counts are inferred from event support fields in `[inc/exc]` and PSI:
  - upstream count per cell: `round(PSI * inc_support)`
  - downstream count per cell: `round(PSI * exc_support)`
- For collapsed junctions, counts and support are aggregated across contributing
  intermediate rows; `junc_ratio` is recomputed as
  `aggregated_counts / aggregated_support` and clipped to `[0, 1]`.
- ATSE counts are computed by summing junction counts within grouping key
  `splicing.var['event_id']`, controlled by `--atse-grouping-mode`:
  - `both_anchors` (default)
  - `upstream_only`
  - `downstream_only`
- Binary mask is derived from ATSE counts (not PSI missingness):
  - `psi_mask = 1` when `ATSE_count > --mask-atse-threshold`
  - `psi_mask = 0` otherwise

This is now the default and only supported count/mask construction mode.

Future additions:
- [ ] Add `tutorial.ipynb` for a walkthrough of model setup, training, and application to other datasets.
- [ ] Add trained models to Hugging Face
---

## References

> Vaidyanathan S, Isaev K, Zweig A, Knowles DA. *Robust Integration of Sparse Single-Cell Alternative Splicing and Gene Expression Data with SpliceVI*. bioRxiv 2025.11.26.690853. https://doi.org/10.1101/2025.11.26.690853

