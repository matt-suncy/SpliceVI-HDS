# Joint Embedding of Alternative Splicing and Gene Expression Using Partial VAEs with Mouse Neuronal Data

Multimodal VAE for joint modeling of alternative splicing (PSI) and gene expression from single-cell data. Built as a fork of [SpliceVI](https://github.com/daklab/SpliceVI).

This fork extends the original SpliceVI model with various alternative latent mixing strategies — including cross-attention, gating, and MLP-based mixers — for combining the expression and splicing posterior distributions into a shared latent space. The primary goal is to compare these mixing strategies on mouse neuronal single-cell data and evaluate their effect on representation quality and cell-type classification.

---

## Installation

### 1. Create and activate the environment

```bash
conda create -n splicevi-env python=3.12
conda activate splicevi-env
```

### 2. Install the package

```bash
git clone https://github.com/matt-suncy/SpliceVI-HDS.git
git switch submission
cd SpliceVI-HDS
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
│   └── run_staged_eval.sh              # Evaluation runner (from smoke test to full eval)
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
| `concatenate` | *(built-in)* | No | Concatenates both latents; doubles the effective latent dimension |
| `sum` | `SumMixer` | No | Elementwise sum of both latent vectors |
| `product` | `ProductMixer` | No | Elementwise product of both latent vectors |
| `gating` | `GatingMixer` | Yes | Dimension-wise sigmoid gate from concatenated means; same gate applied to mean and variance |
| `cross_attention` | `CrossAttentionMixer(reverse=False)` | Yes | Per-dimension attention: AS queries GE |
| `cross_attention_reverse` | `CrossAttentionMixer(reverse=True)` | Yes | Per-dimension attention: GE queries AS |
| `mlp` | `MLPMixer` | Yes | Two-layer MLP on concatenated latents; separate networks for mean and log-variance |


---

## Training

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

## Custom Data Preprocessing Workflow

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

One-command helper:

```bash
bash scripts/run_custom_pipeline.sh
```

Detailed split behavior and reproducibility checklist:

- See [docs/data_splitting.md](docs/data_splitting.md)

## Evaluation

Use the staged evaluator to run a fast compatibility check first, then a full evaluation sweep with the same model and data wiring.

Smoke run (optional but recommended):

```bash
bash scripts/run_staged_eval.sh \
  --mode smoke \
  --model-dir models/custom_baseline_run \
  --train-h5mu data/processed/splicevi_custom_input_train70.h5mu \
  --test-h5mu data/processed/splicevi_custom_input_test30.h5mu \
```

Full run (after smoke passes):

```bash
bash scripts/run_staged_eval.sh \
  --mode full \
  --model-dir models/custom_baseline_run \
  --train-h5mu data/processed/splicevi_custom_input_train70.h5mu \
  --test-h5mu data/processed/splicevi_custom_input_test30.h5mu
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
- `obs['age_days']` (numeric; mirrored to `age_numeric` for compatibility)

---

## References

> Vaidyanathan S, Isaev K, Zweig A, Knowles DA. *Robust Integration of Sparse Single-Cell Alternative Splicing and Gene Expression Data with SpliceVI*. bioRxiv 2025.11.26.690853. https://doi.org/10.1101/2025.11.26.690853

