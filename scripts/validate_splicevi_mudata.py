#!/usr/bin/env python
"""Validate a MuData file against SpliceVI training requirements."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import mudata as mu
import numpy as np


REQUIRED_SPLICE_LAYERS = [
    "junc_ratio",
    "cell_by_junction_matrix",
    "cell_by_cluster_matrix",
    "psi_mask",
]


def validate(path: Path) -> None:
    mdata = mu.read_h5mu(path)

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

    n_genes = int((rna.var["modality"] == "Gene_Expression").sum())
    n_junctions = int((sp.var["modality"] == "Splicing").sum())
    n_events = int(sp.var["event_id"].nunique())

    print("MuData validation passed")
    print(f"File: {path}")
    print(f"Cells: {mdata.n_obs}")
    print(f"Genes (modality==Gene_Expression): {n_genes}")
    print(f"Junctions (modality==Splicing): {n_junctions}")
    print(f"ATSE groups (event_id unique): {n_events}")
    print(f"Splicing layers: {list(layers)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate SpliceVI MuData input.")
    parser.add_argument(
        "--h5mu",
        default="data/processed/splicevi_custom_input.h5mu",
        help="Path to .h5mu file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    validate(Path(args.h5mu))
