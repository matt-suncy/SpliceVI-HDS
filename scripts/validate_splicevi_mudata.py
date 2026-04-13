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

    _validate_event_id_format(sp.var["event_id"], grouping_mode)

    junc_counts = _to_csr(sp.layers["cell_by_junction_matrix"])
    atse_counts = _to_csr(sp.layers["cell_by_cluster_matrix"])
    psi_mask = _to_csr(sp.layers["psi_mask"])

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

    event_codes, event_uniques = pd.factorize(sp.var["event_id"].astype(str), sort=True)
    expected_atse_shape = (sp.n_obs, sp.n_vars)
    if atse_counts.shape != expected_atse_shape:
        raise ValueError(
            "splicing.layers['cell_by_cluster_matrix'] has wrong shape: "
            f"{atse_counts.shape}, expected {expected_atse_shape}"
        )

    j2e = sparse.coo_matrix(
        (
            np.ones_like(event_codes, dtype=np.int8),
            (np.arange(len(event_codes), dtype=np.int32), event_codes.astype(np.int32)),
        ),
        shape=(len(event_codes), len(event_uniques)),
    ).tocsr()

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
    n_events = int(sp.var["event_id"].nunique())

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
