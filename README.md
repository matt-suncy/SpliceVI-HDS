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
  --output-h5mu data/processed/splicevi_custom_input.h5mu
```

For a quick smoke test on a subset, add:

```bash
  --max-cells 512 --max-expr-features 5000 --max-splicing-features 5000
```

2. Validate that the generated `.h5mu` has all required fields/layers:

```bash
python scripts/validate_splicevi_mudata.py \
  --h5mu data/processed/splicevi_custom_input.h5mu
```

3. Train using the generated file:

```bash
python train_splicevi.py \
  --train_mdata_path data/processed/splicevi_custom_input.h5mu \
  --model_dir models/custom_baseline_run \
  --batch_key None
```

One-command helper:

```bash
bash scripts/run_custom_pipeline.sh
```

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

### Notes on Pseudo-counts

If raw splicing count matrices are not available, the builder derives pseudo junction
counts from PSI values using `--pseudo-depth` and then computes ATSE totals by summing
junction pseudo-counts within each `event_id` group.

This enables training with ratio-only splicing inputs, but raw-count-based layers are
preferred when available.

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
