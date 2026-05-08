#!/usr/bin/env python
"""Validate a MuData file against SpliceVI training requirements."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import mudata as mu
import numpy as np
import pandas as pd
from scipy import sparse


REQUIRED_SPLICE_LAYERS = [
    "junc_ratio",
    "cell_by_junction_matrix",
    "cell_by_cluster_matrix",
    "psi_mask",
]


def _to_csr(layer: Any) -> sparse.csr_matrix:
    if sparse.issparse(layer):
        return layer.tocsr()
    return sparse.csr_matrix(np.asarray(layer))


def _validate_event_id_format(event_ids: pd.Series, grouping_mode: str) -> None:
    expected_parts = {
        "both_anchors": 4,
        "upstream_only": 3,
        "downstream_only": 3,
    }[grouping_mode]

    bad_examples = []
    for value in event_ids.astype(str):
        parts = value.split("|")
        if len(parts) != expected_parts:
            bad_examples.append(value)
            if len(bad_examples) >= 5:
                break
            continue
        if not parts[-1].isdigit() or (expected_parts == 4 and not parts[-2].isdigit()):
            bad_examples.append(value)
            if len(bad_examples) >= 5:
                break

    if bad_examples:
        raise ValueError(
            "splicing.var['event_id'] format does not match grouping mode "
            f"'{grouping_mode}'. Example values: {bad_examples}"
        )


def _parse_memberships(raw: str) -> list[str]:
    return [tok for tok in str(raw).split(";;") if tok]


def _build_j2e_from_var(sp_var: pd.DataFrame, grouping_mode: str) -> tuple[sparse.csr_matrix, int]:
    if "event_id_memberships" not in sp_var.columns:
        event_ids = sp_var["event_id"].astype(str)
        _validate_event_id_format(event_ids, grouping_mode)
        event_codes, event_uniques = pd.factorize(event_ids, sort=True)
        j2e = sparse.coo_matrix(
            (
                np.ones_like(event_codes, dtype=np.int8),
                (np.arange(len(event_codes), dtype=np.int32), event_codes.astype(np.int32)),
            ),
            shape=(len(event_codes), len(event_uniques)),
        ).tocsr()
        return j2e, int(len(event_uniques))

    row_idx: list[int] = []
    col_idx: list[int] = []
    event_to_col: dict[str, int] = {}
    membership_counts: list[int] = []

    for row_i, raw in enumerate(sp_var["event_id_memberships"].astype(str).tolist()):
        row_tokens = list(dict.fromkeys(_parse_memberships(raw)))
        if len(row_tokens) == 0:
            primary = str(sp_var["event_id"].iloc[row_i]) if "event_id" in sp_var.columns else ""
            row_tokens = [primary] if primary else []
        _validate_event_id_format(pd.Series(row_tokens, dtype=str), grouping_mode)
        membership_counts.append(len(row_tokens))
        for tok in row_tokens:
            col_j = event_to_col.setdefault(tok, len(event_to_col))
            row_idx.append(row_i)
            col_idx.append(col_j)

    if "event_id_membership_count" in sp_var.columns:
        declared = pd.to_numeric(sp_var["event_id_membership_count"], errors="coerce")
        if declared.isna().any():
            raise ValueError("splicing.var['event_id_membership_count'] has non-numeric values.")
        expected = np.asarray(membership_counts, dtype=np.int32)
        if not np.array_equal(declared.to_numpy(dtype=np.int32), expected):
            raise ValueError(
                "splicing.var['event_id_membership_count'] is inconsistent with "
                "splicing.var['event_id_memberships']."
            )

    j2e = sparse.coo_matrix(
        (
            np.ones(len(row_idx), dtype=np.int8),
            (np.asarray(row_idx, dtype=np.int32), np.asarray(col_idx, dtype=np.int32)),
        ),
        shape=(sp_var.shape[0], len(event_to_col)),
    ).tocsr()
    if j2e.nnz:
        j2e.data = np.ones_like(j2e.data, dtype=np.int8)
    return j2e, int(len(event_to_col))


def validate(path: Path, grouping_mode: str, mask_atse_threshold: int) -> None:
    # Backed mode keeps validation lightweight for large .h5mu files.
    mdata = mu.read_h5mu(path, backed="r")

    if "rna" not in mdata.mod or "splicing" not in mdata.mod:
        raise ValueError("MuData must contain modalities 'rna' and 'splicing'.")

    rna = mdata["rna"]
    sp = mdata["splicing"]
    layers_obj = sp.layers

    if layers_obj is None:
        raise ValueError("Splicing modality has no layers.")
    layers = cast(dict[str, Any], layers_obj)

    if rna.n_obs != sp.n_obs:
        raise ValueError(f"Cell mismatch: rna={rna.n_obs}, splicing={sp.n_obs}")

    missing_layers = [k for k in REQUIRED_SPLICE_LAYERS if k not in layers]
    if missing_layers:
        raise ValueError(f"Missing required splicing layers: {missing_layers}")

    if "modality" not in rna.var.columns:
        raise ValueError("Missing rna.var['modality']")
    if "modality" not in sp.var.columns:
        raise ValueError("Missing splicing.var['modality']")
    if "event_id" not in sp.var.columns:
        raise ValueError("Missing splicing.var['event_id']")
    if "junction_id" not in sp.var.columns:
        raise ValueError("Missing splicing.var['junction_id']")
    required_junction_key_cols = [
        "event_type",
        "gene_id",
        "junction_side",
        "junction_start",
        "junction_end",
    ]
    missing_junction_key_cols = [c for c in required_junction_key_cols if c not in sp.var.columns]
    if missing_junction_key_cols:
        raise ValueError(
            "Missing required splicing.var columns for junction dedup validation: "
            f"{missing_junction_key_cols}"
        )

    if "donor_id" not in mdata.obs.columns:
        raise ValueError("Missing mdata.obs['donor_id']")
    if "X_library_size" not in rna.obsm:
        raise ValueError("Missing rna.obsm['X_library_size']")

    if len(set(rna.obs_names)) != rna.n_obs:
        raise ValueError("Duplicate rna.obs_names detected")
    if len(set(sp.obs_names)) != sp.n_obs:
        raise ValueError("Duplicate splicing.obs_names detected")

    if not np.array_equal(rna.obs_names.to_numpy(), sp.obs_names.to_numpy()):
        raise ValueError("RNA and splicing obs_names ordering mismatch")

    ratio = _to_csr(sp.layers["junc_ratio"]).astype(np.float32)
    junc_counts = _to_csr(sp.layers["cell_by_junction_matrix"])
    atse_counts = _to_csr(sp.layers["cell_by_cluster_matrix"])
    psi_mask = _to_csr(sp.layers["psi_mask"])

    if ratio.shape != (sp.n_obs, sp.n_vars):
        raise ValueError(
            "splicing.layers['junc_ratio'] has wrong shape: "
            f"{ratio.shape}, expected {(sp.n_obs, sp.n_vars)}"
        )
    if junc_counts.shape != (sp.n_obs, sp.n_vars):
        raise ValueError(
            "splicing.layers['cell_by_junction_matrix'] has wrong shape: "
            f"{junc_counts.shape}, expected {(sp.n_obs, sp.n_vars)}"
        )
    if psi_mask.shape != (sp.n_obs, sp.n_vars):
        raise ValueError(
            "splicing.layers['psi_mask'] has wrong shape: "
            f"{psi_mask.shape}, expected {(sp.n_obs, sp.n_vars)}"
        )

    dedup_keys = sp.var[required_junction_key_cols].astype(str)
    dup_mask = dedup_keys.duplicated(keep=False).to_numpy()
    if dup_mask.any():
        dup_examples = dedup_keys.loc[dup_mask].head(5).to_dict(orient="records")
        raise ValueError(
            "Duplicate deduplicated junction keys found in splicing.var. "
            f"Example duplicates: {dup_examples}"
        )

    if ratio.nnz:
        if not np.isfinite(ratio.data).all():
            raise ValueError("junc_ratio contains non-finite values.")
        ratio_min = float(np.min(ratio.data))
        ratio_max = float(np.max(ratio.data))
        if ratio_min < 0.0 or ratio_max > 1.0:
            raise ValueError(
                f"junc_ratio values out of [0,1] range: min={ratio_min}, max={ratio_max}."
            )

    expected_atse_shape = (sp.n_obs, sp.n_vars)
    if atse_counts.shape != expected_atse_shape:
        raise ValueError(
            "splicing.layers['cell_by_cluster_matrix'] has wrong shape: "
            f"{atse_counts.shape}, expected {expected_atse_shape}"
        )

    j2e, n_event_groups = _build_j2e_from_var(sp.var, grouping_mode)

    if junc_counts.nnz and not np.allclose(junc_counts.data, np.rint(junc_counts.data)):
        raise ValueError("cell_by_junction_matrix must contain integer-valued counts.")
    if atse_counts.nnz and not np.allclose(atse_counts.data, np.rint(atse_counts.data)):
        raise ValueError("cell_by_cluster_matrix must contain integer-valued counts.")
    if junc_counts.nnz and np.min(junc_counts.data) < 0:
        raise ValueError("cell_by_junction_matrix contains negative values.")
    if atse_counts.nnz and np.min(atse_counts.data) < 0:
        raise ValueError("cell_by_cluster_matrix contains negative values.")

    if psi_mask.nnz:
        unique_mask_values = np.unique(psi_mask.data)
        if not np.all(np.isin(unique_mask_values, [0, 1])):
            raise ValueError(f"psi_mask must be binary; found values {unique_mask_values.tolist()}")

    recomputed_event_cluster = (junc_counts @ j2e).astype(np.int32)
    recomputed_cluster = (recomputed_event_cluster @ j2e.T).astype(np.int32)
    diff_cluster = (recomputed_cluster != atse_counts)
    if diff_cluster.nnz != 0:
        raise ValueError(
            "cell_by_cluster_matrix is inconsistent with cell_by_junction_matrix and event_id mapping. "
            f"Mismatched entries: {diff_cluster.nnz}"
        )

    event_observed = (recomputed_event_cluster > int(mask_atse_threshold)).astype(np.int8)
    expected_mask = (event_observed @ j2e.T).astype(np.int8).tocsr()
    if expected_mask.nnz:
        expected_mask.data = np.ones_like(expected_mask.data, dtype=np.int8)

    diff_mask = (expected_mask != psi_mask)
    if diff_mask.nnz != 0:
        raise ValueError(
            "psi_mask is inconsistent with ATSE-derived rule mask=(cell_by_cluster_matrix > threshold). "
            f"Threshold={mask_atse_threshold}, mismatched entries: {diff_mask.nnz}"
        )

    n_genes = int((rna.var["modality"] == "Gene_Expression").sum())
    n_junctions = int((sp.var["modality"] == "Splicing").sum())
    n_events = n_event_groups

    print("MuData validation passed")
    print(f"File: {path}")
    print(f"Cells: {mdata.n_obs}")
    print(f"Genes (modality==Gene_Expression): {n_genes}")
    print(f"Junctions (modality==Splicing): {n_junctions}")
    print(f"ATSE groups (event_id unique): {n_events}")
    print(f"ATSE grouping mode: {grouping_mode}")
    print(f"Mask threshold (ATSE > threshold): {mask_atse_threshold}")
    print(f"Splicing layers: {list(layers)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate SpliceVI MuData input.")
    parser.add_argument(
        "--h5mu",
        default="data/processed/splicevi_custom_input.h5mu",
        help="Path to .h5mu file.",
    )
    parser.add_argument(
        "--atse-grouping-mode",
        type=str,
        default="both_anchors",
        choices=["both_anchors", "upstream_only", "downstream_only"],
        help="Expected event_id grouping key format used at build time.",
    )
    parser.add_argument(
        "--mask-atse-threshold",
        type=int,
        default=0,
        help="Expected threshold for psi_mask rule: mask = (ATSE > threshold).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    validate(
        Path(args.h5mu),
        grouping_mode=args.atse_grouping_mode,
        mask_atse_threshold=args.mask_atse_threshold,
    )
