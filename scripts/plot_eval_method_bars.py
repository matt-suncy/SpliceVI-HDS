#!/usr/bin/env python
"""Unified plotting for cross-fold eval results across mixing methods.

Produces:
1) Classification F1 grouped bars (GE/AS/joint per method, per target task)
2) Regression (age_days) grouped bars with 95% CI error bars
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SPACE_ORDER = ["expression", "splicing", "joint"]
SPACE_LABELS = {"expression": "GE", "splicing": "AS", "joint": "joint"}

METHOD_ORDER = [
    "equal",
    "sum",
    "product",
    "concatenate",
    "cross_attention",
    "gating",
    "mlp",
]


def infer_method_name(model_dir: str) -> str:
    base = Path(str(model_dir)).name
    if base == "equal":
        return "equal"
    if base.startswith("splicevi_"):
        rest = base[len("splicevi_") :]
        for k in ["cross_attention", "concatenate", "gating", "product", "sum", "mlp"]:
            if rest.startswith(k):
                return k
    return base


def collect_runs(summary_tsv: str, run_overrides: List[str]) -> Dict[str, Path]:
    runs: Dict[str, Path] = {}
    s = pd.read_csv(summary_tsv, sep="\t")
    s = s[s["status"] == "SUCCESS"].copy()
    for _, row in s.iterrows():
        method = infer_method_name(row["model_dir"])
        runs[method] = Path(row["run_dir"])

    for item in run_overrides:
        if "=" not in item:
            raise ValueError(f"--run must be method=run_dir, got: {item}")
        method, run_dir = item.split("=", 1)
        runs[method.strip()] = Path(run_dir.strip())
    return runs


def load_crossfold(method_runs: Dict[str, Path]) -> pd.DataFrame:
    all_df = []
    for method, run_dir in method_runs.items():
        p = run_dir / "figures" / "cross_fold_classification_results.csv"
        if not p.exists():
            print(f"[WARN] Missing cross-fold results for {method}: {p}")
            continue
        df = pd.read_csv(p)
        df["method"] = method
        df["run_dir"] = str(run_dir)
        all_df.append(df)
    if not all_df:
        raise FileNotFoundError("No cross_fold_classification_results.csv found for provided runs.")
    out = pd.concat(all_df, ignore_index=True)
    out["space"] = out["space"].astype(str)
    out["target"] = out["target"].astype(str)
    out["metric"] = out["metric"].astype(str)
    out["classifier"] = out["classifier"].astype(str)
    out["split"] = out["split"].astype(str)
    out["target_type"] = out.get("target_type", "unknown").astype(str)
    return out


def ordered_methods(values: List[str]) -> List[str]:
    vset = list(dict.fromkeys(values))
    ordered = [m for m in METHOD_ORDER if m in vset]
    tail = [m for m in vset if m not in ordered]
    return ordered + tail


def plot_grouped_bars(
    df: pd.DataFrame,
    methods: List[str],
    title: str,
    ylabel: str,
    output_path: Path,
    with_ci: bool = False,
):
    spaces = [s for s in SPACE_ORDER if s in df["space"].unique()]
    if not spaces:
        raise ValueError("No expected spaces found in data.")

    width = 0.22
    x = np.arange(len(methods))
    offsets = np.linspace(-(len(spaces) - 1) * width / 2, (len(spaces) - 1) * width / 2, len(spaces))

    fig, ax = plt.subplots(figsize=(max(10, 1.2 * len(methods)), 5))
    for off, space in zip(offsets, spaces):
        sub = (
            df[df["space"] == space]
            .set_index("method")
            .reindex(methods)
            .reset_index()
        )
        y = sub["mean"].to_numpy(dtype=float)
        err = None
        if with_ci:
            n = sub["n_folds"].fillna(1).replace(0, 1).to_numpy(dtype=float)
            std = sub["std"].fillna(0.0).to_numpy(dtype=float)
            err = 1.96 * std / np.sqrt(n)
        ax.bar(
            x + off,
            y,
            width=width,
            label=SPACE_LABELS.get(space, space),
            yerr=err,
            capsize=3 if with_ci else 0,
            alpha=0.9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title="Embedding")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot grouped eval bars across mixing methods from cross-fold CSV outputs."
    )
    parser.add_argument("--summary-tsv", type=str, required=True, help="Batch summary TSV (e.g. non_equal summary.tsv).")
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Additional method=run_dir entry (e.g., equal=logs/eval_runs/eval_full_equal_... ).",
    )
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Which split to plot (default: test).")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory for plots/csv.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    method_runs = collect_runs(args.summary_tsv, args.run)
    df = load_crossfold(method_runs)

    df = df[df["split"] == args.split].copy()
    if df.empty:
        raise ValueError(f"No rows found for split={args.split}.")

    methods = ordered_methods(df["method"].tolist())

    # Classification: F1 weighted for each target task.
    cls = df[
        (df["target_type"] == "classification")
        & (df["classifier"] == "logreg")
        & (df["metric"] == "f1_weighted")
    ].copy()
    if cls.empty:
        print("[WARN] No classification F1 rows found.")
    else:
        cls_targets = sorted(cls["target"].unique().tolist())
        for target in cls_targets:
            plot_df = cls[cls["target"] == target].copy()
            out = out_dir / f"classification_f1_{args.split}_{target}.png"
            plot_grouped_bars(
                plot_df,
                methods=methods,
                title=f"Classification F1 ({target}) — {args.split}",
                ylabel="F1 weighted",
                output_path=out,
                with_ci=False,
            )
            print(f"[SAVE] {out}")

    # Regression: age_days ridge, with confidence (95% CI from fold std).
    reg = df[
        (df["target"] == "age_days")
        & (df["target_type"] == "regression")
        & (df["classifier"] == "ridge")
    ].copy()
    if reg.empty:
        print("[WARN] No age_days ridge rows found.")
    else:
        # Main requested view: R2 with confidence.
        reg_r2 = reg[reg["metric"] == "r2"].copy()
        if not reg_r2.empty:
            out = out_dir / f"regression_age_days_r2_with_ci_{args.split}.png"
            plot_grouped_bars(
                reg_r2,
                methods=methods,
                title=f"Regression R2 (age_days, ridge) — {args.split}",
                ylabel="R2",
                output_path=out,
                with_ci=True,
            )
            print(f"[SAVE] {out}")

        # Also provide MAE/RMSE companion plots with CI.
        for metric, label in [("mae", "MAE"), ("rmse", "RMSE")]:
            mdf = reg[reg["metric"] == metric].copy()
            if mdf.empty:
                continue
            out = out_dir / f"regression_age_days_{metric}_with_ci_{args.split}.png"
            plot_grouped_bars(
                mdf,
                methods=methods,
                title=f"Regression {label} (age_days, ridge) — {args.split}",
                ylabel=label,
                output_path=out,
                with_ci=True,
            )
            print(f"[SAVE] {out}")

    # Save merged table for reproducibility.
    merged_csv = out_dir / f"merged_cross_fold_{args.split}.csv"
    df.to_csv(merged_csv, index=False)
    print(f"[SAVE] {merged_csv}")


if __name__ == "__main__":
    main()
