# Latent Mixing in SPLICEVAE

`SPLICEVAE` encodes expression and splicing separately into two posterior distributions
`q(z | x_ge)` and `q(z | x_as)`. The `modality_weights` parameter selects how these are
combined into a single shared latent `z` before decoding.

For the learned mixers (`gating`, `cross_attention`, `cross_attention_reverse`, `mlp`),
mixing is done in **parameter space** — directly on the posterior statistics `mu` and
`var` — rather than on reparameterised samples. Mixer classes are defined at the top of
`src/splicevi/splicevae.py`.

---

## Overview

| `modality_weights` | Class | Learnable | Description |
|---|---|---|---|
| `equal` | *(built-in)* | No | Weighted average with uniform weights `[0.5, 0.5]` |
| `universal` | *(built-in)* | Yes | Single scalar weight shared across all cells, learned as `nn.Parameter` |
| `cell` | *(built-in)* | Yes | Per-cell weight vector `(n_obs, 2)`, learned as `nn.Parameter` |
| `concatenate` | *(built-in)* | No | Concatenates both latents; doubles the effective latent dimension |
| `sum` | `SumMixer` | No | Elementwise sum of both latent vectors |
| `product` | `ProductMixer` | No | Elementwise product of both latent vectors |
| `gating` | `GatingMixer` | Yes | Dimension-wise sigmoid gate from concatenated means; same gate applied to mean and variance |
| `cross_attention` | `CrossAttentionMixer(reverse=False)` | Yes | Per-dimension attention: AS queries GE |
| `cross_attention_reverse` | `CrossAttentionMixer(reverse=True)` | Yes | Per-dimension attention: GE queries AS |
| `mlp` | `MLPMixer` | Yes | Two-layer MLP on concatenated latents; separate networks for mean and log-variance |

---

## `GatingMixer`

A single linear gate network produces a per-dimension blend weight from the concatenation
of both means. The same gate is applied to both mean and variance.

```
gate   = sigmoid( Linear(2Z → Z)( [mu_as | mu_ge] ) )
mu_out = gate * mu_as + (1 - gate) * mu_ge
v_out  = gate * v_as  + (1 - gate) * v_ge          
```

Parameters: `n_latent` (int) — latent dimensionality.

---

## `CrossAttentionMixer`

Each latent dimension is treated as a single token (`embed_dim=1`, `num_heads=1`,
`batch_first=True`). The attention mechanism operates per batch element over the
`n_latent`-length sequence. `forward()` is used for reparameterised sample mixing;
`mix_params()` is used during training and operates on posterior statistics directly.

### Constructor parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_latent` | int | — | Latent dimensionality (sequence length for the attention) |
| `reverse` | bool | `False` | `False`: AS queries GE — `Attention(Q=z_as, K=z_ge, V=z_ge)`. `True`: GE queries AS — `Attention(Q=z_ge, K=z_as, V=z_as)` |
| `shared_var_attn` | bool | `False` | Whether mean and variance share one attention head or use separate heads (see below) |

### `shared_var_attn=False` (default — separate heads)

Two independent `nn.MultiheadAttention` modules are created:

- `self.attn` — used for the mean. Q, K, V all come from `mu_as` / `mu_ge`.
- `self.attn_v` — used for the log-variance. Q, K, and V all come from
  `log(v_as)` / `log(v_ge)`, so the head operates entirely in log-variance space.

This allows the model to learn independent attention patterns for location (mean) and
uncertainty (variance). It doubles the number of attention parameters relative to the
shared-head variant.

```
# mean (non-reverse)
Q, K, V = mu_as, mu_ge, mu_ge  →  self.attn  →  mu_out

# variance (non-reverse)
Q, K, V = log(v_as), log(v_ge), log(v_ge)  →  self.attn_v  →  log_v_out  →  exp().clamp(1e-6)
```

### `shared_var_attn=True` (shared head)

Only `self.attn` is created. The variance pass reuses the mean head, passing Q and K
from the means but substituting V with `log(variance)`. This ties the variance routing
to the mean attention pattern.

```
# mean (non-reverse)
Q, K, V = mu_as, mu_ge, mu_ge      →  self.attn  →  mu_out

# variance (non-reverse) — same head, different V
Q, K, V = mu_as, mu_ge, log(v_ge)  →  self.attn  →  log_v_out  →  exp().clamp(1e-6)
```

### Instantiation in `SPLICEVAE`

`cross_attention` and `cross_attention_reverse` are instantiated at lines ~617/623 of
`splicevae.py`. The `shared_var_attn` flag is not currently exposed as a CLI argument;
it defaults to `False` and can be changed by editing the instantiation site directly.

---

## `MLPMixer`

Concatenates both latent vectors and passes them through a two-layer MLP. Two completely
separate networks are used for mean and log-variance.

```
# mean
[mu_as | mu_ge]        →  Linear(2Z, 128) → ReLU → Linear(128, Z)  →  mu_out

# log-variance
[log(v_as) | log(v_ge)]  →  Linear(2Z, 128) → ReLU → Linear(128, Z)  →  log_v_out  →  exp().clamp(1e-6)
```

Parameters: `n_latent` (int), `n_hidden` (int, default 128).

---

## Warmup behaviour

During KL warmup (`n_epochs_kl_warmup` epochs), the learned mixers
(`gating`, `cross_attention`, `cross_attention_reverse`, `mlp`) receive only the
splicing latent as input (expression branch is gated off via `cross_gate`). This
prevents the expression encoder from dominating early training before the splicing
encoder has converged. After warmup, `cross_gate` is opened and both branches
contribute to the mixed latent.

The simple mixers (`equal`, `universal`, `cell`, `concatenate`, `sum`, `product`) are
not affected by warmup gating.
