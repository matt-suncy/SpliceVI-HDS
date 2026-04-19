# Fork of SpliceVI, module of Matthew and Ellena's final project

Multimodal VAE for joint modeling of alternative splicing (PSI) and gene expression from single-cell data. Built on [scvi-tools](https://github.com/scverse/scvi-tools).

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

After installation, `from splicevi import SPLICEVI` works from any script or notebook.

> **Exact environment:** `requirements.txt` contains a full freeze of the development environment for reproducibility reference, but has machine-specific paths and is not intended as a direct install source.

---

## Repository structure

```
SpliceVI/
├── src/splicevi/
│   ├── splicevi.py          # SPLICEVI model class for GE+AS (training, inference, DE, DS)
│   ├── splicevae.py         # VAE module for GE+AS (encoder/decoder architecture)
│   ├── partialvae.py        # AS Missingness-Aware Partial VAE module
│   ├── eddisplice.py        # EDDISPLICE model class for single-modality AS-only VAE using PARTIALVAE
│   └── __init__.py          # Package exports
├── train_splicevi.py        # Training entry point
├── eval_splicevi.py         # Evaluation entry point (UMAPs, metrics, imputation)
├── train_splicevi.sh        # SLURM job script for training
├── eval_splicevi.sh         # SLURM job script for evaluation
├── pyproject.toml           # Package configuration
└── requirements.txt         # Full conda environment freeze
```

---

## Usage

Configure paths and hyperparameters at the top of the shell scripts, then submit:

```bash
sbatch train_splicevi.sh
sbatch eval_splicevi.sh
```

Or run directly:

```bash
python train_splicevi.py --help
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

3. Create paper-aligned external split artifacts (70/30 stratified on `age_numeric` + `class`):

```bash
python scripts/create_test_split.py \
  --train-path data/processed/splicevi_custom_input.h5mu \
  --output-train-path data/processed/splicevi_custom_input_train70.h5mu \
  --output-test-path data/processed/splicevi_custom_input_test30.h5mu \
  --test-frac 0.3 \
  --seed 42 \
  --stratify-age-col age_numeric \
  --stratify-celltype-col class
```

4. Train using the external-train split file:

```bash
python train_splicevi.py \
  --train_mdata_path data/processed/splicevi_custom_input_train70.h5mu \
  --model_dir models/custom_baseline_run \
  --batch_key seq_batch
```

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
  --masked-h5mu data/processed/test_masked_25.h5mu \
  --masked-resampled
```

Full run (after smoke passes):

```bash
bash scripts/run_staged_eval.sh \
  --mode full \
  --model-dir models/custom_baseline_run \
  --train-h5mu data/processed/train_splicevi_input.h5mu \
  --test-h5mu data/processed/test_splicevi_input.h5mu \
  --masked-h5mu data/processed/test_masked_25.h5mu \
  --masked-h5mu data/processed/test_masked_50.h5mu \
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

### Notes on Splicing Layer Construction

The builder derives required splicing layers directly from the event IDs in
`MO_VIS_core.individual.cass.mat.txt`:

- Each event row is expanded into two junction features:
  - upstream junction `(upstream_end, cassette_start)`
  - downstream junction `(cassette_end, downstream_start)`
- Junction counts are inferred from event support fields in `[inc/exc]` and PSI:
  - upstream count per cell: `round(PSI * inc_support)`
  - downstream count per cell: `round(PSI * exc_support)`
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

## Citation

If you use SpliceVI in your work, please cite:

> Vaidyanathan S, Isaev K, Zweig A, Knowles DA. *Robust Integration of Sparse Single-Cell Alternative Splicing and Gene Expression Data with SpliceVI*. bioRxiv 2025.11.26.690853. https://doi.org/10.1101/2025.11.26.690853

```bibtex
@article{splicevi,
  title   = {Robust Integration of Sparse Single-Cell Alternative Splicing and Gene Expression Data with {SpliceVI}},
  author  = {Vaidyanathan, Smriti and Isaev, Keren and Zweig, Aaron and Knowles, David A},
  journal = {bioRxiv},
  year    = {2025},
  doi     = {10.1101/2025.11.26.690853}
}
```
