#!/usr/bin/env python
"""Create paper-aligned external train/test splits from a full MuData file.

Default behavior matches the paper split policy:
- 70/30 external train/test split (``--test-frac 0.3``)
- Stratified by ``age_numeric`` and ``class``
- Deterministic with a fixed seed
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mudata as mu
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _derive_default_train_path(test_path: Path) -> Path:
    return test_path.with_name(f"{test_path.stem}_train70{test_path.suffix}")


def _build_strata(
    obs: pd.DataFrame,
    age_col: str,
    celltype_col: str,
    min_stratum_size: int,
    missing_label: str,
    rare_label: str,
) -> pd.Series:
    missing_cols = [c for c in [age_col, celltype_col] if c not in obs.columns]
    if missing_cols:
        raise ValueError(
            "Cannot stratify split because required obs columns are missing: "
            f"{missing_cols}. Available sample: {list(obs.columns)[:20]}"
        )

    age = obs[age_col].astype("string").fillna(missing_label)
    celltype = obs[celltype_col].astype("string").fillna(missing_label)
    strata = age + "||" + celltype

    if min_stratum_size > 1:
        counts = strata.value_counts()
        rare = counts[counts < min_stratum_size].index
        if len(rare) > 0:
            print(
                f"[STRATA] Collapsing {len(rare)} rare strata (<{min_stratum_size} cells) "
                f"into '{rare_label}'."
            )
            strata = strata.where(~strata.isin(rare), other=rare_label)

    return strata.astype(str)


def _subset_mudata(mdata, indices: np.ndarray):
    modalities = {name: mdata.mod[name][indices, :].copy() for name in mdata.mod.keys()}
    out = mu.MuData(modalities)
    out.obs = mdata.obs.iloc[indices].copy()
    return out


def _annotate_split_metadata(
    mdata,
    split_role: str,
    seed: int,
    source_path: str,
    strata: pd.Series,
):
    obs = mdata.obs
    obs["_split_role"] = split_role
    obs["_split_seed"] = str(seed)
    obs["_split_source_file"] = source_path
    obs["_split_strata_key"] = strata.reindex(obs.index).astype("string").fillna("unknown").to_numpy()

    split_cols = ["_split_role", "_split_seed", "_split_source_file", "_split_strata_key"]
    for mod_name in mdata.mod.keys():
        for col in split_cols:
            mdata.mod[mod_name].obs[col] = obs[col].reindex(mdata.mod[mod_name].obs.index).to_numpy()


def create_train_test_split(
    train_path: str,
    output_train_path: str,
    output_test_path: str,
    test_frac: float = 0.3,
    seed: int = 42,
    stratify_age_col: str = "age_numeric",
    stratify_celltype_col: str = "class",
    min_stratum_size: int = 5,
    missing_label: str = "unknown",
    rare_label: str = "other",
    disable_stratify: bool = False,
    add_split_metadata: bool = True,
):
    if test_frac <= 0.0 or test_frac >= 1.0:
        raise ValueError(f"test_frac must be in (0, 1), got {test_frac}.")

    print(f"[LOAD] Loading full MuData from {train_path}...")
    mdata = mu.read_h5mu(train_path)
    print("[INFO] Full MuData shapes:")
    print(f"       RNA: n_obs={mdata['rna'].n_obs}, n_vars={mdata['rna'].n_vars}")
    print(f"       Splicing: n_obs={mdata['splicing'].n_obs}, n_vars={mdata['splicing'].n_vars}")

    n_cells = int(mdata.n_obs)
    indices = np.arange(n_cells, dtype=np.int64)

    strata = None
    if disable_stratify:
        print("[STRATA] Stratification disabled; using random split only.")
    else:
        strata = _build_strata(
            mdata.obs,
            age_col=stratify_age_col,
            celltype_col=stratify_celltype_col,
            min_stratum_size=min_stratum_size,
            missing_label=missing_label,
            rare_label=rare_label,
        )
        if int(strata.nunique()) < 2:
            print("[STRATA] Only one stratum found after preprocessing; falling back to random split.")
            strata = None
        else:
            print(f"[STRATA] Using {int(strata.nunique())} strata for splitting.")

    try:
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_frac,
            random_state=seed,
            shuffle=True,
            stratify=None if strata is None else strata.to_numpy(),
        )
    except ValueError as exc:
        if strata is None:
            raise
        print(f"[STRATA] Stratified split failed ({exc}); retrying random split.")
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_frac,
            random_state=seed,
            shuffle=True,
            stratify=None,
        )

    train_idx = np.sort(train_idx)
    test_idx = np.sort(test_idx)
    train_n = int(train_idx.size)
    test_n = int(test_idx.size)

    print(f"[SPLIT] Train cells: {train_n} ({(train_n / n_cells) * 100:.2f}%)")
    print(f"[SPLIT] Test  cells: {test_n} ({(test_n / n_cells) * 100:.2f}%)")

    mdata_train = _subset_mudata(mdata, train_idx)
    mdata_test = _subset_mudata(mdata, test_idx)

    strata_for_metadata = (
        pd.Series("unstratified", index=mdata.obs.index, dtype="string")
        if strata is None
        else strata
    )
    if add_split_metadata:
        _annotate_split_metadata(
            mdata_train,
            split_role="train",
            seed=seed,
            source_path=train_path,
            strata=strata_for_metadata,
        )
        _annotate_split_metadata(
            mdata_test,
            split_role="test",
            seed=seed,
            source_path=train_path,
            strata=strata_for_metadata,
        )

    train_out = Path(output_train_path)
    test_out = Path(output_test_path)
    train_out.parent.mkdir(parents=True, exist_ok=True)
    test_out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[SAVE] Writing TRAIN split to {train_out}...")
    mdata_train.write_h5mu(str(train_out))
    print(f"[SAVE] Writing TEST split to {test_out}...")
    mdata_test.write_h5mu(str(test_out))

    overlap = mdata_train.obs_names.intersection(mdata_test.obs_names)
    if len(overlap) != 0:
        raise RuntimeError(f"Split overlap detected: {len(overlap)} duplicated cell IDs.")

    print("[DONE] Train/test split created successfully.")
    print(f"       TRAIN RNA: n_obs={mdata_train['rna'].n_obs}, n_vars={mdata_train['rna'].n_vars}")
    print(f"       TEST  RNA: n_obs={mdata_test['rna'].n_obs}, n_vars={mdata_test['rna'].n_vars}")
    print(f"       TRAIN splice: n_obs={mdata_train['splicing'].n_obs}, n_vars={mdata_train['splicing'].n_vars}")
    print(f"       TEST  splice: n_obs={mdata_test['splicing'].n_obs}, n_vars={mdata_test['splicing'].n_vars}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Create complementary train/test MuData splits from a full .h5mu file. "
            "Defaults follow paper-like external split settings (70/30 stratified)."
        )
    )
    parser.add_argument(
        "--train-path",
        type=str,
        required=True,
        help="Path to full MuData file to split.",
    )
    parser.add_argument(
        "--output-train-path",
        type=str,
        default=None,
        help="Path to save TRAIN split MuData file.",
    )
    parser.add_argument(
        "--output-test-path",
        type=str,
        default=None,
        help="Path to save TEST split MuData file.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help=(
            "Deprecated alias for --output-test-path. If used without --output-train-path, "
            "a default TRAIN path '<output-path stem>_train70.h5mu' is created."
        ),
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.3,
        help="Fraction of cells for test split (default: 0.3).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--stratify-age-col",
        type=str,
        default="age_numeric",
        help="obs column used for age stratification (default: age_numeric).",
    )
    parser.add_argument(
        "--stratify-celltype-col",
        type=str,
        default="class",
        help="obs column used for cell-type stratification (default: class).",
    )
    parser.add_argument(
        "--min-stratum-size",
        type=int,
        default=5,
        help="Collapse strata with fewer than this many cells (default: 5).",
    )
    parser.add_argument(
        "--missing-label",
        type=str,
        default="unknown",
        help="Fill label for missing stratification values (default: unknown).",
    )
    parser.add_argument(
        "--rare-label",
        type=str,
        default="other",
        help="Label used when collapsing rare strata (default: other).",
    )
    parser.add_argument(
        "--disable-stratify",
        action="store_true",
        help="Disable stratification and use random split only.",
    )
    parser.add_argument(
        "--no-split-metadata",
        action="store_true",
        help="Do not add split audit metadata columns to obs.",
    )

    args = parser.parse_args()

    output_test_path = args.output_test_path or args.output_path
    if output_test_path is None:
        raise ValueError("Provide --output-test-path (or deprecated --output-path).")

    output_train_path = args.output_train_path
    if output_train_path is None:
        output_train_path = str(_derive_default_train_path(Path(output_test_path)))
        print(f"[PATH] --output-train-path not provided. Using: {output_train_path}")

    create_train_test_split(
        train_path=args.train_path,
        output_train_path=output_train_path,
        output_test_path=output_test_path,
        test_frac=args.test_frac,
        seed=args.seed,
        stratify_age_col=args.stratify_age_col,
        stratify_celltype_col=args.stratify_celltype_col,
        min_stratum_size=args.min_stratum_size,
        missing_label=args.missing_label,
        rare_label=args.rare_label,
        disable_stratify=args.disable_stratify,
        add_split_metadata=(not args.no_split_metadata),
    )
