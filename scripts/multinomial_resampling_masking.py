#!/usr/bin/env python
"""Generate masked TEST MuData files for imputation evaluation.

Supports:
1) Resampled mode (for --masked_test_mdata_is_resampled):
   - writes junc_ratio_original
   - writes cell_by_cluster_matrix_original
   - perturbs junc_ratio via binomial resampling and optional entry masking
2) Legacy masked mode:
   - writes junc_ratio_masked_original + junc_ratio_masked_bin_mask
   - zeroes masked entries in junc_ratio
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mudata as mu
import numpy as np
from scipy import sparse


def _as_csr(mat):
    if sparse.isspmatrix_csr(mat):
        return mat.copy()
    return sparse.csr_matrix(mat)


def _drop_positions_csr(mat_csr: sparse.csr_matrix, drop_rows: np.ndarray, drop_cols: np.ndarray):
    """Remove specific coordinates from a CSR matrix (set them to zero)."""
    if drop_rows.size == 0:
        return mat_csr.copy()
    coo = mat_csr.tocoo(copy=True)
    width = np.int64(mat_csr.shape[1])
    drop_idx = np.unique(drop_rows.astype(np.int64) * width + drop_cols.astype(np.int64))
    coo_idx = coo.row.astype(np.int64) * width + coo.col.astype(np.int64)
    keep = ~np.isin(coo_idx, drop_idx, assume_unique=False)
    out = sparse.csr_matrix(
        (coo.data[keep], (coo.row[keep], coo.col[keep])),
        shape=mat_csr.shape,
    )
    out.eliminate_zeros()
    return out


def _mask_from_positions(shape, rows: np.ndarray, cols: np.ndarray):
    data = np.ones(rows.size, dtype=np.float32)
    return sparse.csr_matrix((data, (rows, cols)), shape=shape)


def _build_output_mudata(m_in):
    m_out = mu.MuData(
        {
            "rna": m_in["rna"].copy(),
            "splicing": m_in["splicing"].copy(),
        }
    )
    m_out.obs = m_in.obs.copy()
    return m_out


def generate_masked(
    input_test_h5mu: str,
    output_dir: str,
    mask_fracs: list[float],
    mode: str,
    seed: int,
    overwrite: bool,
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[LOAD] {input_test_h5mu}")
    m = mu.read_h5mu(input_test_h5mu)
    ad = m["splicing"]

    junc_ratio = _as_csr(ad.layers["junc_ratio"])
    atse_counts = _as_csr(ad.layers["cell_by_cluster_matrix"])
    psi_mask = _as_csr(ad.layers["psi_mask"])

    rows, cols = junc_ratio.nonzero()
    nnz = rows.size
    if nnz == 0:
        raise ValueError("Input junc_ratio has no non-zero entries; cannot build masked data.")

    print(f"[INFO] n_cells={ad.n_obs}, n_junctions={ad.n_vars}, junc_ratio nnz={nnz}")

    for frac in mask_fracs:
        if frac <= 0 or frac >= 1:
            raise ValueError(f"mask_frac must be in (0,1), got {frac}")
        pct = int(round(frac * 100))
        rng = np.random.default_rng(seed + pct)
        selected = rng.random(nnz) < frac
        sel_rows = rows[selected]
        sel_cols = cols[selected]
        print(f"[MASK] frac={frac:.3f} ({pct}%) -> selected {selected.sum()}/{nnz} entries")

        if mode in {"resampled", "both"}:
            m_res = _build_output_mudata(m)
            ad_res = m_res["splicing"]

            # Preserve originals for resampled-eval mode.
            ad_res.layers["junc_ratio_original"] = junc_ratio.copy()
            ad_res.layers["cell_by_cluster_matrix_original"] = atse_counts.copy()

            # Binomial resampling of PSI using ATSE counts as trials where available.
            res = junc_ratio.copy()
            count_vals = np.asarray(atse_counts[rows, cols]).ravel().astype(np.int64)
            count_vals = np.maximum(count_vals, 1)
            p = np.clip(junc_ratio.data.astype(np.float64), 0.0, 1.0)
            sampled = rng.binomial(count_vals, p)
            res.data = (sampled / count_vals).astype(np.float32)

            # Optional masking of a subset of originally non-zero entries.
            if selected.any():
                # Set selected PSI entries to zero.
                drop_idx = np.flatnonzero(selected)
                res.data[drop_idx] = 0.0
                res.eliminate_zeros()

                # Keep setup layers consistent with masked positions.
                counts_masked = _drop_positions_csr(atse_counts, sel_rows, sel_cols)
                psi_mask_masked = _drop_positions_csr(psi_mask, sel_rows, sel_cols)
            else:
                counts_masked = atse_counts.copy()
                psi_mask_masked = psi_mask.copy()

            ad_res.layers["junc_ratio"] = res
            ad_res.layers["cell_by_cluster_matrix"] = counts_masked
            ad_res.layers["psi_mask"] = psi_mask_masked

            out_path = out_dir / f"RESAMPLED_{pct}_PERCENT_{Path(input_test_h5mu).stem}.h5mu"
            if out_path.exists() and not overwrite:
                raise FileExistsError(f"{out_path} exists. Use --overwrite to replace.")
            print(f"[SAVE] {out_path}")
            m_res.write_h5mu(str(out_path))

        if mode in {"legacy", "both"}:
            m_legacy = _build_output_mudata(m)
            ad_leg = m_legacy["splicing"]

            masked_input = _drop_positions_csr(junc_ratio, sel_rows, sel_cols)
            counts_masked = _drop_positions_csr(atse_counts, sel_rows, sel_cols)
            psi_mask_masked = _drop_positions_csr(psi_mask, sel_rows, sel_cols)

            gt_masked = sparse.csr_matrix(
                (junc_ratio.data[selected].astype(np.float32), (sel_rows, sel_cols)),
                shape=junc_ratio.shape,
            )
            bin_mask = _mask_from_positions(junc_ratio.shape, sel_rows, sel_cols)

            ad_leg.layers["junc_ratio"] = masked_input
            ad_leg.layers["cell_by_cluster_matrix"] = counts_masked
            ad_leg.layers["psi_mask"] = psi_mask_masked
            ad_leg.layers["junc_ratio_masked_original"] = gt_masked
            ad_leg.layers["junc_ratio_masked_bin_mask"] = bin_mask

            out_path = out_dir / f"MASKED_{pct}_PERCENT_{Path(input_test_h5mu).stem}.h5mu"
            if out_path.exists() and not overwrite:
                raise FileExistsError(f"{out_path} exists. Use --overwrite to replace.")
            print(f"[SAVE] {out_path}")
            m_legacy.write_h5mu(str(out_path))

    print("[DONE] Masked/resampled files generated.")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate masked TEST MuData files for eval_splicevi.py masked_impute mode. "
            "Can produce resampled mode files, legacy masked mode files, or both."
        )
    )
    parser.add_argument("--input-test-h5mu", type=str, required=True, help="Input TEST .h5mu file.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for output masked files.")
    parser.add_argument(
        "--mask-fracs",
        type=float,
        nargs="+",
        default=[0.25, 0.50],
        help="Mask fractions to generate (default: 0.25 0.50).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["resampled", "legacy", "both"],
        default="resampled",
        help="Output format mode (default: resampled).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed base.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    args = parser.parse_args()

    generate_masked(
        input_test_h5mu=args.input_test_h5mu,
        output_dir=args.output_dir,
        mask_fracs=list(args.mask_fracs),
        mode=args.mode,
        seed=args.seed,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
