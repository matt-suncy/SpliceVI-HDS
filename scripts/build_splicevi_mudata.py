#!/usr/bin/env python
"""Build a SpliceVI-ready MuData object from tabular expression/splicing inputs.

This script converts separate expression, splicing, and metadata files into a
single .h5mu file that satisfies train_splicevi.py expectations.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import anndata as ad
import mudata as mu
import numpy as np
import pandas as pd
from scipy import sparse


def _canonicalize_sample_id(sample_id: str) -> str:
    """Normalize sample identifiers across expression/splicing/metadata files."""
    sid = str(sample_id).strip()
    sid = sid.replace("\\", "/")
    sid = sid.split("/")[-1]
    sid = re.sub(r"\.(expr\.txt|countit|txt)$", "", sid)
    return sid


def _derive_event_group(junction_id: str) -> str:
    """Derive ATSE grouping key from a junction identifier.

    Example:
    CA-...-170683[INC][40/1][DNT] -> CA-...-170683
    """
    return str(junction_id).split("[")[0]


def _get_header_columns(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return handle.readline().rstrip("\n").split("\t")


def _sample_column_map(path: Path, id_col: str, name_col: str) -> dict[str, str]:
    """Map canonical sample_id -> original column name for a matrix file."""
    cols = _get_header_columns(path)
    out = {}
    for c in cols:
        if c in {id_col, name_col}:
            continue
        out[_canonicalize_sample_id(c)] = c
    return out


def _read_matrix(
    path: Path,
    id_col: str,
    name_col: str,
    selected_samples: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Read a tab-delimited matrix with feature rows and sample columns."""
    usecols = None
    if selected_samples is not None:
        sample_map = _sample_column_map(path, id_col=id_col, name_col=name_col)
        missing = [s for s in selected_samples if s not in sample_map]
        if missing:
            raise ValueError(f"{path} is missing selected samples; example: {missing[:5]}")
        usecols = [id_col, name_col] + [sample_map[s] for s in selected_samples]

    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, usecols=usecols)
    if id_col not in df.columns or name_col not in df.columns:
        raise ValueError(f"{path} must contain columns '{id_col}' and '{name_col}'.")

    sample_cols = [c for c in df.columns if c not in {id_col, name_col}]
    if not sample_cols:
        raise ValueError(f"{path} has no sample columns.")

    renamed = {c: _canonicalize_sample_id(c) for c in sample_cols}
    df = df.rename(columns=renamed)

    # Preserve original column order after canonicalization.
    canonical_cols = [renamed[c] for c in sample_cols]
    return df, canonical_cols


def _load_group_map(path: Path | None, key_name: str) -> pd.Series | None:
    if path is None:
        return None
    gdf = pd.read_csv(path, sep="\t", header=None, names=["sample", key_name], dtype=str)
    if gdf.empty:
        return None
    gdf["sample"] = gdf["sample"].map(_canonicalize_sample_id)
    return gdf.drop_duplicates("sample").set_index("sample")[key_name]


def _make_obs(
    metadata_path: Path,
    shared_samples: list[str],
    expr_group_map: pd.Series | None,
    as_group_map: pd.Series | None,
) -> pd.DataFrame:
    try:
        obs = pd.read_csv(metadata_path, dtype=str)
    except UnicodeDecodeError:
        obs = pd.read_csv(metadata_path, dtype=str, encoding="latin1")
    if "seq_name" not in obs.columns:
        raise ValueError("metadata file must contain a 'seq_name' column.")

    obs["_sample_id"] = obs["seq_name"].map(_canonicalize_sample_id)
    obs = obs.drop_duplicates("_sample_id").set_index("_sample_id")

    missing = [s for s in shared_samples if s not in obs.index]
    if missing:
        raise ValueError(
            f"{len(missing)} shared samples are missing from metadata seq_name; example: {missing[:5]}"
        )

    obs = obs.loc[shared_samples].copy()

    if "donor_id" not in obs.columns:
        if "donor" in obs.columns:
            obs["donor_id"] = obs["donor"]
        else:
            obs["donor_id"] = obs.index

    if expr_group_map is not None:
        obs["expr_group"] = obs.index.to_series().map(expr_group_map).fillna("unknown")
    if as_group_map is not None:
        obs["as_group"] = obs.index.to_series().map(as_group_map).fillna("unknown")

    obs.index.name = None
    return obs


def _unique_feature_index(primary: pd.Series, secondary: pd.Series, prefix: str) -> pd.Index:
    """Create deterministic unique feature names."""
    primary = primary.fillna("").astype(str)
    secondary = secondary.fillna("").astype(str)
    out = []
    seen = {}
    for i, (p, s) in enumerate(zip(primary, secondary), start=1):
        base = p if p else (s if s else f"{prefix}_{i}")
        n = seen.get(base, 0)
        seen[base] = n + 1
        out.append(base if n == 0 else f"{base}__dup{n}")
    return pd.Index(out)


def build(args: argparse.Namespace) -> None:
    expr_path = Path(args.expr_matrix)
    sp_path = Path(args.splicing_matrix)

    expr_map = _sample_column_map(expr_path, id_col="gene_id", name_col="NAME")
    sp_map = _sample_column_map(sp_path, id_col="#event_id", name_col="NAME")
    shared_samples = [s for s in expr_map if s in sp_map]
    if not shared_samples:
        raise ValueError("No overlapping sample IDs were found between expression and splicing matrices.")

    if args.max_cells > 0:
        shared_samples = shared_samples[: args.max_cells]

    expr_df, expr_cols = _read_matrix(
        expr_path,
        id_col="gene_id",
        name_col="NAME",
        selected_samples=shared_samples,
    )
    sp_df, sp_cols = _read_matrix(
        sp_path,
        id_col="#event_id",
        name_col="NAME",
        selected_samples=shared_samples,
    )

    expr_group_map = _load_group_map(Path(args.expr_group_map), "expr_group") if args.expr_group_map else None
    as_group_map = _load_group_map(Path(args.as_group_map), "as_group") if args.as_group_map else None

    shared_samples = [s for s in expr_cols if s in set(sp_cols)]
    if not shared_samples:
        raise ValueError("No overlapping sample IDs were found between expression and splicing matrices.")

    if args.max_expr_features > 0:
        expr_df = expr_df.head(args.max_expr_features).copy()
    if args.max_splicing_features > 0:
        sp_df = sp_df.head(args.max_splicing_features).copy()

    obs = _make_obs(Path(args.metadata_csv), shared_samples, expr_group_map, as_group_map)

    # Align matrices to shared sample ordering and coerce to numeric.
    expr_vals = expr_df[shared_samples].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    sp_vals_df = sp_df[shared_samples].replace("", np.nan)
    sp_vals = sp_vals_df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    psi_mask = (~np.isnan(sp_vals)).astype(np.int8)
    sp_vals = np.nan_to_num(sp_vals, nan=0.0)
    sp_vals = np.clip(sp_vals, 0.0, 1.0)

    if args.min_cells_per_feature > 0:
        expr_keep = (expr_vals > 0).sum(axis=1) >= args.min_cells_per_feature
        sp_keep = psi_mask.sum(axis=1) >= args.min_cells_per_feature
        expr_df = expr_df.loc[expr_keep].reset_index(drop=True)
        sp_df = sp_df.loc[sp_keep].reset_index(drop=True)
        expr_vals = expr_vals[expr_keep]
        sp_vals = sp_vals[sp_keep]
        psi_mask = psi_mask[sp_keep]

    # Build RNA modality.
    rna_var = pd.DataFrame(
        {
            "gene_id": expr_df["gene_id"].astype(str).to_numpy(),
            "gene_name": expr_df["NAME"].astype(str).to_numpy(),
            "modality": "Gene_Expression",
        },
        index=_unique_feature_index(expr_df["gene_id"], expr_df["NAME"], "gene"),
    )
    rna_x = sparse.csr_matrix(expr_vals.T)
    rna_obs = obs.copy()
    rna_obs.index = shared_samples

    rna = ad.AnnData(X=rna_x, obs=rna_obs, var=rna_var)
    rna.layers["length_norm"] = rna_x.copy()
    libsize = np.asarray(rna_x.sum(axis=1)).ravel().astype(np.float32)
    rna.obsm["X_library_size"] = libsize.reshape(-1, 1)

    # Build splicing modality.
    junc_ids = sp_df["#event_id"].astype(str)
    event_groups = junc_ids.map(_derive_event_group)

    sp_var = pd.DataFrame(
        {
            "junction_id": junc_ids.to_numpy(),
            "gene_name": sp_df["NAME"].astype(str).to_numpy(),
            "event_id": event_groups.to_numpy(),
            "modality": "Splicing",
        },
        index=_unique_feature_index(junc_ids, sp_df["NAME"], "junction"),
    )

    ratio_csr = sparse.csr_matrix(sp_vals.T)
    mask_csr = sparse.csr_matrix(psi_mask.T)

    # Pseudo-count fallback: build junc counts from PSI when raw counts are unavailable.
    pseudo_depth = int(args.pseudo_depth)
    junc_counts_dense = np.rint(sp_vals * pseudo_depth).astype(np.int32)
    observed = psi_mask.astype(bool)
    junc_counts_dense[observed & (junc_counts_dense == 0)] = 1
    junc_counts_dense[~observed] = 0
    junc_counts_csr = sparse.csr_matrix(junc_counts_dense.T)

    event_codes, event_uniques = pd.factorize(event_groups, sort=True)
    j2e = sparse.coo_matrix(
        (
            np.ones_like(event_codes, dtype=np.int8),
            (np.arange(len(event_codes), dtype=np.int32), event_codes.astype(np.int32)),
        ),
        shape=(len(event_codes), len(event_uniques)),
    ).tocsr()
    cluster_counts = (junc_counts_csr @ j2e).astype(np.int32)

    sp_obs = obs.copy()
    sp_obs.index = shared_samples
    sp = ad.AnnData(X=ratio_csr, obs=sp_obs, var=sp_var)
    sp.layers["junc_ratio"] = ratio_csr.copy()
    sp.layers["cell_by_junction_matrix"] = junc_counts_csr
    sp.layers["cell_by_cluster_matrix"] = cluster_counts
    sp.layers["psi_mask"] = mask_csr

    mdata = mu.MuData({"rna": rna, "splicing": sp})
    mdata.obs = obs.copy()

    out_path = Path(args.output_h5mu)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mdata.write_h5mu(str(out_path))

    print("Built MuData successfully")
    print(f"Output: {out_path}")
    print(f"Cells: {mdata.n_obs}")
    print(f"Genes: {mdata['rna'].n_vars}")
    print(f"Junctions: {mdata['splicing'].n_vars}")
    print(f"ATSE groups: {len(event_uniques)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a SpliceVI-ready .h5mu from custom tables.")
    parser.add_argument(
        "--expr-matrix",
        default="data/Tasic2018_MO_VIS_core.individual.expr.mat.txt",
        help="Tab-delimited expression matrix path.",
    )
    parser.add_argument(
        "--splicing-matrix",
        default="data/MO_VIS_core.individual.cass.mat.txt",
        help="Tab-delimited splicing matrix path.",
    )
    parser.add_argument(
        "--metadata-csv",
        default="data/MO_sample_metadata.csv",
        help="Metadata CSV path containing seq_name and donor metadata.",
    )
    parser.add_argument(
        "--expr-group-map",
        default="data/MO_VIS_core.individual2group.expr.conf",
        help="Optional sample-to-group mapping for expression (TSV, no header).",
    )
    parser.add_argument(
        "--as-group-map",
        default="data/MO_VIS_core.individual2group.as.conf",
        help="Optional sample-to-group mapping for splicing (TSV, no header).",
    )
    parser.add_argument(
        "--output-h5mu",
        default="data/processed/splicevi_custom_input.h5mu",
        help="Output .h5mu path.",
    )
    parser.add_argument(
        "--pseudo-depth",
        type=int,
        default=50,
        help="Pseudo read depth used to convert PSI to pseudo junction counts.",
    )
    parser.add_argument(
        "--min-cells-per-feature",
        type=int,
        default=1,
        help="Drop features observed in fewer than this many cells.",
    )
    parser.add_argument(
        "--max-cells",
        type=int,
        default=-1,
        help="If >0, keep only the first N shared cells (for smoke tests).",
    )
    parser.add_argument(
        "--max-expr-features",
        type=int,
        default=-1,
        help="If >0, keep only the first N expression features.",
    )
    parser.add_argument(
        "--max-splicing-features",
        type=int,
        default=-1,
        help="If >0, keep only the first N splicing features.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
