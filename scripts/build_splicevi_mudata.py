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


EVENT_ID_RE = re.compile(
    r"^(?P<event_type>[^-\[]+)-(?P<gene_id>[^-\[]+)-(?P<upstream_end>\d+)-"
    r"(?P<cassette_start>\d+)-(?P<cassette_end>\d+)-(?P<downstream_start>\d+)"
    r"(?:\[(?P<tag1>[^\]]+)\])(?:\[(?P<tag2>[^\]]+)\])(?:\[[^\]]+\])*$"
)


def _canonicalize_sample_id(sample_id: str) -> str:
    """Normalize sample identifiers across expression/splicing/metadata files."""
    sid = str(sample_id).strip()
    sid = sid.replace("\\", "/")
    sid = sid.split("/")[-1]
    sid = re.sub(r"\.(expr\.txt|countit|txt)$", "", sid)
    return sid


def _parse_event_id(event_id: str) -> dict[str, int | str]:
    """Parse event_id into coordinates and support counts.

    Expected format:
    CA-gene-upstream_end-cassette_start-cassette_end-downstream_start
    [inclusion_tag][inclusion_reads/exclusion_reads][optional_tags...]
    """
    raw = str(event_id)
    match = EVENT_ID_RE.match(raw)
    if match is None:
        raise ValueError(f"Malformed #event_id: {raw}")

    count_match = re.match(r"^(\d+)/(\d+)$", match.group("tag2"))
    if count_match is None:
        raise ValueError(
            "Malformed #event_id count token for "
            f"'{raw}'; expected second bracket token like '[6/2]'."
        )

    return {
        "raw_event_id": raw,
        "event_type": match.group("event_type"),
        "gene_id": match.group("gene_id"),
        "upstream_end": int(match.group("upstream_end")),
        "cassette_start": int(match.group("cassette_start")),
        "cassette_end": int(match.group("cassette_end")),
        "downstream_start": int(match.group("downstream_start")),
        "inclusion_tag": match.group("tag1"),
        "inc_support": int(count_match.group(1)),
        "exc_support": int(count_match.group(2)),
    }


def _derive_event_group(parsed: dict[str, int | str], grouping_mode: str) -> str:
    """Build event grouping key based on selected anchor matching mode."""
    event_type = str(parsed["event_type"])
    gene_id = str(parsed["gene_id"])
    upstream_end = int(parsed["upstream_end"])
    downstream_start = int(parsed["downstream_start"])

    if grouping_mode == "both_anchors":
        return f"{event_type}|{gene_id}|{upstream_end}|{downstream_start}"
    if grouping_mode == "upstream_only":
        return f"{event_type}|{gene_id}|{upstream_end}"
    if grouping_mode == "downstream_only":
        return f"{event_type}|{gene_id}|{downstream_start}"

    raise ValueError(
        "Invalid --atse-grouping-mode. Expected one of: "
        "both_anchors, upstream_only, downstream_only"
    )


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


def _read_metadata_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype=str)
    except UnicodeDecodeError:
        return pd.read_csv(path, dtype=str, encoding="latin1")


def _make_obs(
    metadata_paths: list[Path],
    shared_samples: list[str],
    expr_group_map: pd.Series | None,
    as_group_map: pd.Series | None,
) -> pd.DataFrame:
    if not metadata_paths:
        raise ValueError("At least one metadata CSV path must be provided.")

    frames = [_read_metadata_csv(p).rename(columns=lambda c: str(c).strip()) for p in metadata_paths]
    obs = pd.concat(frames, axis=0, ignore_index=True)
    duplicated_cols = pd.Index(obs.columns)[pd.Index(obs.columns).duplicated()].tolist()
    if duplicated_cols:
        raise ValueError(f"metadata has duplicate columns after normalization: {duplicated_cols}")
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

    age_col = "age_days" if "age_days" in obs.columns else None
    if age_col is None:
        for c in ("age_numeric", "age_day", "age", "age_in_days"):
            if c in obs.columns:
                obs["age_days"] = obs[c]
                age_col = c
                break
    if age_col is None:
        raise ValueError(
            "metadata must contain 'age_days' (or one of: age_numeric, age_day, age, age_in_days)."
        )

    age_days = pd.to_numeric(obs["age_days"], errors="coerce")
    if age_days.isna().all():
        raise ValueError(
            f"metadata age column '{age_col}' could not be parsed into numeric age values."
        )
    obs["age_days"] = age_days.astype(np.float32)

    # Keep backward compatibility for existing split/eval scripts that may still expect age_numeric.
    if "age_numeric" not in obs.columns:
        obs["age_numeric"] = obs["age_days"]

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


def _compact_unique_join(values: pd.Series, max_items: int = 5) -> str:
    uniq = pd.Index(values.astype(str)).drop_duplicates().tolist()
    if len(uniq) <= max_items:
        return ";;".join(uniq)
    return ";;".join(uniq[:max_items]) + f";;...(+{len(uniq) - max_items})"


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

    metadata_paths = [Path(p) for p in args.metadata_csvs]
    obs = _make_obs(metadata_paths, shared_samples, expr_group_map, as_group_map)

    # Align matrices to shared sample ordering and coerce to numeric.
    expr_vals = expr_df[shared_samples].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    sp_vals_df = sp_df[shared_samples].replace("", np.nan)
    sp_vals = sp_vals_df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    obs_mask = (~np.isnan(sp_vals)).astype(np.int8)
    sp_vals = np.nan_to_num(sp_vals, nan=0.0)
    sp_vals = np.clip(sp_vals, 0.0, 1.0)

    if args.min_cells_per_feature > 0:
        expr_keep = (expr_vals > 0).sum(axis=1) >= args.min_cells_per_feature
        sp_keep = obs_mask.sum(axis=1) >= args.min_cells_per_feature
        expr_df = expr_df.loc[expr_keep].reset_index(drop=True)
        sp_df = sp_df.loc[sp_keep].reset_index(drop=True)
        expr_vals = expr_vals[expr_keep]
        sp_vals = sp_vals[sp_keep]
        obs_mask = obs_mask[sp_keep]

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
    # Keep both locations for compatibility with SpliceVI/scvi variants.
    rna.obs["X_library_size"] = libsize
    rna.obsm["X_library_size"] = libsize.reshape(-1, 1)

    if args.junction_dedup_mode != "enabled":
        raise ValueError("Only junction dedup mode 'enabled' is supported.")

    # Build splicing modality from event-level PSI and event_id-derived metadata.
    # We keep two-junction expansion as intermediate representation and then collapse
    # duplicate genomic junctions.
    raw_event_ids = sp_df["#event_id"].astype(str).to_numpy()
    gene_names = sp_df["NAME"].astype(str).to_numpy()
    parsed_events = [_parse_event_id(eid) for eid in raw_event_ids]

    n_events = len(parsed_events)
    if n_events == 0:
        raise ValueError("No splicing rows available after filtering.")

    n_junctions_expanded = n_events * 2
    expanded_vals = np.repeat(sp_vals, 2, axis=0)
    junction_support = np.zeros(n_junctions_expanded, dtype=np.float32)

    junc_ids: list[str] = []
    gene_id_list: list[str] = []
    event_groups_list: list[str] = []
    gene_name_list: list[str] = []
    event_type_list: list[str] = []
    source_event_id_list: list[str] = []
    junction_side_list: list[str] = []
    junction_start_list: list[int] = []
    junction_end_list: list[int] = []
    upstream_end_list: list[int] = []
    cassette_start_list: list[int] = []
    cassette_end_list: list[int] = []
    downstream_start_list: list[int] = []
    inc_support_list: list[int] = []
    exc_support_list: list[int] = []

    for i, (parsed, gene_name) in enumerate(zip(parsed_events, gene_names, strict=True)):
        event_group = _derive_event_group(parsed, args.atse_grouping_mode)
        raw_event = str(parsed["raw_event_id"])
        event_type = str(parsed["event_type"])
        gene_id = str(parsed["gene_id"])
        upstream_end = int(parsed["upstream_end"])
        cassette_start = int(parsed["cassette_start"])
        cassette_end = int(parsed["cassette_end"])
        downstream_start = int(parsed["downstream_start"])
        inc_support = int(parsed["inc_support"])
        exc_support = int(parsed["exc_support"])

        # Upstream inclusion junction: (upstream_end, cassette_start)
        junc_ids.append(f"{raw_event}|JUP:{upstream_end}-{cassette_start}")
        gene_id_list.append(gene_id)
        event_groups_list.append(event_group)
        gene_name_list.append(str(gene_name))
        event_type_list.append(event_type)
        source_event_id_list.append(raw_event)
        junction_side_list.append("upstream")
        junction_start_list.append(upstream_end)
        junction_end_list.append(cassette_start)
        upstream_end_list.append(upstream_end)
        cassette_start_list.append(cassette_start)
        cassette_end_list.append(cassette_end)
        downstream_start_list.append(downstream_start)
        inc_support_list.append(inc_support)
        exc_support_list.append(exc_support)
        junction_support[2 * i] = float(inc_support)

        # Downstream inclusion junction: (cassette_end, downstream_start)
        junc_ids.append(f"{raw_event}|JDN:{cassette_end}-{downstream_start}")
        gene_id_list.append(gene_id)
        event_groups_list.append(event_group)
        gene_name_list.append(str(gene_name))
        event_type_list.append(event_type)
        source_event_id_list.append(raw_event)
        junction_side_list.append("downstream")
        junction_start_list.append(cassette_end)
        junction_end_list.append(downstream_start)
        upstream_end_list.append(upstream_end)
        cassette_start_list.append(cassette_start)
        cassette_end_list.append(cassette_end)
        downstream_start_list.append(downstream_start)
        inc_support_list.append(inc_support)
        exc_support_list.append(exc_support)
        junction_support[2 * i + 1] = float(exc_support)

    expanded_df = pd.DataFrame(
        {
            "source_event_id": np.asarray(source_event_id_list, dtype=object),
            "event_id": np.asarray(event_groups_list, dtype=object),
            "event_type": np.asarray(event_type_list, dtype=object),
            "gene_id": np.asarray(gene_id_list, dtype=object),
            "gene_name": np.asarray(gene_name_list, dtype=object),
            "junction_side": np.asarray(junction_side_list, dtype=object),
            "junction_start": np.asarray(junction_start_list, dtype=np.int32),
            "junction_end": np.asarray(junction_end_list, dtype=np.int32),
            "support": junction_support.astype(np.float32),
            "expanded_junction_id": np.asarray(junc_ids, dtype=object),
            "upstream_end": np.asarray(upstream_end_list, dtype=np.int32),
            "cassette_start": np.asarray(cassette_start_list, dtype=np.int32),
            "cassette_end": np.asarray(cassette_end_list, dtype=np.int32),
            "downstream_start": np.asarray(downstream_start_list, dtype=np.int32),
            "inc_support": np.asarray(inc_support_list, dtype=np.int32),
            "exc_support": np.asarray(exc_support_list, dtype=np.int32),
            "expanded_row_index": np.arange(n_junctions_expanded, dtype=np.int32),
        }
    )

    dedup_key_cols = [
        "event_type",
        "gene_id",
        "junction_side",
        "junction_start",
        "junction_end",
    ]
    dedup_keys = pd.MultiIndex.from_frame(expanded_df[dedup_key_cols])
    expanded_to_unique_codes, unique_dedup_keys = pd.factorize(dedup_keys, sort=False)
    n_junctions_unique = int(len(unique_dedup_keys))
    expanded_df["unique_junction_index"] = expanded_to_unique_codes.astype(np.int32)

    expanded_to_unique = sparse.coo_matrix(
        (
            np.ones(n_junctions_expanded, dtype=np.int8),
            (
                np.arange(n_junctions_expanded, dtype=np.int32),
                expanded_to_unique_codes.astype(np.int32),
            ),
        ),
        shape=(n_junctions_expanded, n_junctions_unique),
    ).tocsr()

    junc_counts_expanded_dense = np.rint(expanded_vals * junction_support[:, None]).astype(np.int32)
    junc_counts_expanded_dense = np.clip(junc_counts_expanded_dense, 0, None)
    junc_counts_expanded_csr = sparse.csr_matrix(junc_counts_expanded_dense.T)
    junc_counts_csr = (junc_counts_expanded_csr @ expanded_to_unique).astype(np.int32)

    support_sum = np.bincount(
        expanded_to_unique_codes,
        weights=junction_support.astype(np.float64),
        minlength=n_junctions_unique,
    ).astype(np.float32)

    ratio_csr = junc_counts_csr.astype(np.float32).tocsr()
    inv_support = np.zeros_like(support_sum, dtype=np.float32)
    valid_support = support_sum > 0
    inv_support[valid_support] = 1.0 / support_sum[valid_support]
    ratio_csr = (ratio_csr @ sparse.diags(inv_support, offsets=0, format="csr")).tocsr()
    if ratio_csr.nnz:
        ratio_csr.data = np.clip(ratio_csr.data, 0.0, 1.0)

    event_codes, event_uniques = pd.factorize(expanded_df["event_id"].astype(str), sort=True)
    j2e = sparse.coo_matrix(
        (
            np.ones(n_junctions_expanded, dtype=np.int8),
            (
                expanded_to_unique_codes.astype(np.int32),
                event_codes.astype(np.int32),
            ),
        ),
        shape=(n_junctions_unique, len(event_uniques)),
    ).tocsr()
    if j2e.nnz:
        j2e.data = np.ones_like(j2e.data, dtype=np.int8)
    event_cluster_counts = (junc_counts_csr @ j2e).astype(np.int32)
    cluster_counts = (event_cluster_counts @ j2e.T).astype(np.int32)

    event_observed = (event_cluster_counts > int(args.mask_atse_threshold)).astype(np.int8)
    mask_csr = (event_observed @ j2e.T).astype(np.int8).tocsr()
    if mask_csr.nnz > 0:
        mask_csr.data = np.ones_like(mask_csr.data, dtype=np.int8)

    unique_meta = (
        expanded_df.groupby("unique_junction_index", sort=False)
        .agg(
            event_type=("event_type", "first"),
            gene_id=("gene_id", "first"),
            gene_name=("gene_name", "first"),
            junction_side=("junction_side", "first"),
            junction_start=("junction_start", "first"),
            junction_end=("junction_end", "first"),
            support_sum=("support", "sum"),
            source_event_count=("source_event_id", "nunique"),
            source_event_examples=("source_event_id", _compact_unique_join),
            event_id_primary=("event_id", "first"),
            event_id_memberships=(
                "event_id",
                lambda s: ";;".join(pd.Index(s.astype(str)).drop_duplicates().tolist()),
            ),
            event_id_membership_count=("event_id", "nunique"),
            upstream_end=("upstream_end", "first"),
            cassette_start=("cassette_start", "first"),
            cassette_end=("cassette_end", "first"),
            downstream_start=("downstream_start", "first"),
        )
        .reset_index(drop=True)
    )

    junction_id_series = (
        unique_meta["event_type"].astype(str)
        + "|"
        + unique_meta["gene_id"].astype(str)
        + "|"
        + unique_meta["junction_side"].astype(str)
        + "|"
        + unique_meta["junction_start"].astype(str)
        + "-"
        + unique_meta["junction_end"].astype(str)
    )
    if junction_id_series.duplicated().any():
        dup_examples = junction_id_series[junction_id_series.duplicated()].head(5).tolist()
        raise ValueError(f"Duplicate junction_id values after deduplication: {dup_examples}")

    sp_var = pd.DataFrame(
        {
            "junction_id": junction_id_series.to_numpy(),
            "event_type": unique_meta["event_type"].astype(str).to_numpy(),
            "gene_id": unique_meta["gene_id"].astype(str).to_numpy(),
            "gene_name": unique_meta["gene_name"].astype(str).to_numpy(),
            "event_id": unique_meta["event_id_primary"].astype(str).to_numpy(),
            "event_id_memberships": unique_meta["event_id_memberships"].astype(str).to_numpy(),
            "event_id_membership_count": unique_meta["event_id_membership_count"].astype(np.int32).to_numpy(),
            "source_event_count": unique_meta["source_event_count"].astype(np.int32).to_numpy(),
            "source_event_examples": unique_meta["source_event_examples"].astype(str).to_numpy(),
            "junction_side": unique_meta["junction_side"].astype(str).to_numpy(),
            "junction_start": unique_meta["junction_start"].astype(np.int32).to_numpy(),
            "junction_end": unique_meta["junction_end"].astype(np.int32).to_numpy(),
            "junction_support_sum": unique_meta["support_sum"].astype(np.float32).to_numpy(),
            "upstream_end": unique_meta["upstream_end"].astype(np.int32).to_numpy(),
            "cassette_start": unique_meta["cassette_start"].astype(np.int32).to_numpy(),
            "cassette_end": unique_meta["cassette_end"].astype(np.int32).to_numpy(),
            "downstream_start": unique_meta["downstream_start"].astype(np.int32).to_numpy(),
            "modality": "Splicing",
        },
        index=pd.Index(junction_id_series.to_numpy(), dtype=str),
    )

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
    print(f"Event rows: {n_events}")
    print(f"Expanded junction rows (pre-dedup): {n_junctions_expanded}")
    print(f"Unique junction rows (post-dedup): {n_junctions_unique}")
    reduction_pct = (
        (1.0 - (float(n_junctions_unique) / float(n_junctions_expanded))) * 100.0
        if n_junctions_expanded > 0
        else 0.0
    )
    print(f"Junction dedup reduction: {reduction_pct:.2f}%")
    print(f"ATSE groups: {len(event_uniques)}")
    print(f"ATSE grouping mode: {args.atse_grouping_mode}")
    print(f"Mask threshold (ATSE > threshold): {args.mask_atse_threshold}")


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
        default=None,
        help="Deprecated single metadata CSV path; use --metadata-csvs instead.",
    )
    parser.add_argument(
        "--metadata-csvs",
        nargs="+",
        default=["data/MO_sample_metadata.csv", "data/VIS_sample_metadata.csv"],
        help="One or more metadata CSV files containing seq_name and donor metadata.",
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
        "--atse-grouping-mode",
        type=str,
        default="both_anchors",
        choices=["both_anchors", "upstream_only", "downstream_only"],
        help=(
            "How to build event groups for ATSE counts and mask derivation: "
            "both_anchors (default), upstream_only, or downstream_only."
        ),
    )
    parser.add_argument(
        "--mask-atse-threshold",
        type=int,
        default=0,
        help="Binary mask rule: mask=1 when ATSE count > threshold.",
    )
    parser.add_argument(
        "--junction-dedup-mode",
        type=str,
        default="enabled",
        choices=["enabled"],
        help=(
            "Junction feature construction mode. "
            "Only 'enabled' is supported; expanded rows are deduplicated to unique junctions."
        ),
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
    args = parser.parse_args()
    if args.metadata_csv and args.metadata_csvs == ["data/MO_sample_metadata.csv", "data/VIS_sample_metadata.csv"]:
        args.metadata_csvs = [args.metadata_csv]
    return args


if __name__ == "__main__":
    build(parse_args())
