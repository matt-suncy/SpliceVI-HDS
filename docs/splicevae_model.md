# SPLICEVAE Model Parameters

`SPLICEVAE` (`src/splicevi/splicevae.py`) is the core PyTorch module. It contains:

1. **Latent mixer classes** (lines 21–145) — see [latent_mixers.md](latent_mixers.md)
2. **`LibrarySizeEncoder`** — encodes per-cell library size from expression input
3. **`DecoderSplice`** — decodes latent `z` to splicing usage probabilities via sigmoid
4. **`SPLICEVAE`** — the full dual-encoder/dual-decoder VAE

---

## Data dimensions

| Parameter | Type | Description |
|---|---|---|
| `n_input_genes` | int | Number of gene expression features |
| `n_input_junctions` | int | Number of splicing junction features |
| `n_batch` | int | Number of batch categories (0 = no batch correction) |
| `n_obs` | int | Number of cells (required when `modality_weights="cell"`) |

---

## Architecture

| Parameter | Choices | Default | Description |
|---|---|---|---|
| `splicing_encoder_architecture` | `vanilla`, `partial` | `vanilla` | `vanilla`: standard scvi `Encoder`. `partial`: `PartialEncoderEDDIFaster` (missingness-aware) |
| `splicing_decoder_architecture` | `vanilla`, `linear` | `vanilla` | `vanilla`: nonlinear `DecoderSplice`. `linear`: `LinearDecoder` |
| `expression_architecture` | `vanilla`, `linear` | `vanilla` | `vanilla`: nonlinear `DecoderSCVI`. `linear`: `LinearDecoderSCVI` |
| `n_hidden` | int | auto | Hidden layer width. Defaults to `min(128, sqrt(n_input_junctions))` |
| `n_latent` | int | auto | Latent dimensionality. Defaults to `sqrt(n_hidden)`. Doubled internally for `concatenate` mixing |
| `n_layers_encoder` | int | 2 | Encoder depth (also controls post-pooling MLP depth in partial encoder) |
| `n_layers_decoder` | int | 2 | Decoder depth (unused by linear decoders) |
| `dropout_rate` | float | 0.1 | Dropout for all MLP stacks |
| `use_batch_norm` | `encoder`, `decoder`, `none`, `both` | `none` | Where to apply BatchNorm |
| `use_layer_norm` | `encoder`, `decoder`, `none`, `both` | `both` | Where to apply LayerNorm |
| `latent_distribution` | `normal`, `ln` | `normal` | Posterior family. `ln` applies a softmax transform before decoding |
| `deeply_inject_covariates` | bool | False | Inject covariates at every decoder layer |
| `encode_covariates` | bool | False | Concatenate continuous covariates to encoder inputs |

---

## Expression likelihood

| Parameter | Choices | Default | Description |
|---|---|---|---|
| `gene_likelihood` | `zinb`, `nb`, `poisson` | `zinb` | Reconstruction likelihood for gene expression |
| `gene_dispersion` | `gene`, `gene-batch`, `gene-label`, `gene-cell` | `gene` | Dispersion parameterisation |
| `use_size_factor_key` | bool | False | Use provided library size factors instead of learning them |

---

## Splicing likelihood

| Parameter | Choices | Default | Description |
|---|---|---|---|
| `splicing_loss_type` | `binomial`, `beta_binomial`, `dirichlet_multinomial` | `dirichlet_multinomial` | Reconstruction loss for splicing |
| `splicing_concentration` | float or None | None | Scalar concentration override for beta-binomial |
| `dm_concentration` | `atse`, `scalar` | `atse` | For Dirichlet-multinomial: per-ATSE or scalar concentration |

---

## Partial encoder knobs

Only active when `splicing_encoder_architecture="partial"`.

| Parameter | Default | Description |
|---|---|---|
| `code_dim` | 16 | Dimensionality of per-junction embedding codes before pooling |
| `h_hidden_dim` | 64 | Hidden width of the per-junction h-subnetwork |
| `encoder_hidden_dim` | 128 | Hidden width of the post-pooling MLP |
| `pool_mode` | `mean` | Aggregation over observed junctions: `mean` or `sum` |
| `max_nobs` | -1 | Cap on scatter-chunk size (-1 disables chunking) |

---

## Modality mixing

| Parameter | Choices | Default | Description |
|---|---|---|---|
| `modality_weights` | see [latent_mixers.md](latent_mixers.md) | `equal` | How to combine expression and splicing posteriors |
| `modality_penalty` | `Jeffreys`, `MMD`, `None` | `None` | Alignment penalty between the two per-modality posteriors |
