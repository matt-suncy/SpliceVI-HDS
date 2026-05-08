#!/usr/bin/env python
"""
test_mixers.py

Smoke-test training for any modality_weights strategy.
Switch strategies with --modality_weights:

    python scripts/test_mixers.py --modality_weights gating
    python scripts/test_mixers.py --modality_weights cross_attention
    python scripts/test_mixers.py --modality_weights mlp
    python scripts/test_mixers.py --modality_weights sum
    python scripts/test_mixers.py --modality_weights product

Sweep all new strategies in one go:

    python scripts/test_mixers.py --sweep

Regression-check existing strategies:

    python scripts/test_mixers.py --modality_weights equal
    python scripts/test_mixers.py --modality_weights concatenate
"""

import argparse
import sys
import math
import mudata as mu
from splicevi import SPLICEVI

# ---------------------------------------------------------------------------
# Config — tweak these for faster/slower runs
# ---------------------------------------------------------------------------
DATA_PATH   = "data/processed/splicevi_custom_input.h5mu"
MAX_EPOCHS  = 3          # keep low for smoke tests; raise for real checks
BATCH_SIZE  = 128
N_LATENT    = 10         # small latent for speed

NEW_STRATEGIES      = ["sum", "product", "gating", "cross_attention", "mlp"]
EXISTING_STRATEGIES = ["equal", "universal", "concatenate"]
ALL_STRATEGIES      = NEW_STRATEGIES + EXISTING_STRATEGIES

# Shared model kwargs (mirror defaults from train_splicevi.sh)
MODEL_KWARGS = dict(
    splicing_loss_type="dirichlet_multinomial",
    splicing_encoder_architecture="partial",
    n_latent=N_LATENT,
)

TRAIN_KWARGS = dict(
    max_epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    n_epochs_kl_warmup=0,   # skip warmup for smoke tests
    check_val_every_n_epoch=1,
)
# ---------------------------------------------------------------------------


def load_mdata(path: str):
    print(f"[DATA] Loading {path} ...")
    mdata = mu.read_h5mu(path, backed="r")
    mdata.obs.rename(columns={"donor_id": "mouse.id"}, inplace=True)
    mdata.mod["rna"].obs.rename(columns={"donor_id": "mouse.id"}, inplace=True)
    mdata.mod["splicing"].obs.rename(columns={"donor_id": "mouse.id"}, inplace=True)
    print(f"[DATA] {mdata['rna'].n_obs} cells, "
          f"{mdata['rna'].n_vars} genes, "
          f"{mdata['splicing'].n_vars} junctions")
    return mdata


def run_one(mdata, modality_weights: str) -> float:
    """Train for MAX_EPOCHS and return the final train loss. Raises on NaN."""
    print(f"\n{'='*60}")
    print(f"  modality_weights = {modality_weights!r}")
    print(f"{'='*60}")

    n_genes     = int((mdata["rna"].var["modality"] == "Gene_Expression").sum())
    n_junctions = int((mdata["splicing"].var["modality"] == "Splicing").sum())

    SPLICEVI.setup_mudata(
        mdata,
        batch_key=None,
        size_factor_key="X_library_size",
        rna_layer="length_norm",
        junc_ratio_layer="junc_ratio",
        atse_counts_layer="cell_by_cluster_matrix",
        junc_counts_layer="cell_by_junction_matrix",
        psi_mask_layer="psi_mask",
        modalities={"rna_layer": "rna", "junc_ratio_layer": "splicing"},
    )

    model = SPLICEVI(
        mdata,
        n_genes=n_genes,
        n_junctions=n_junctions,
        modality_weights=modality_weights,
        **MODEL_KWARGS,
    )

    model.train(**TRAIN_KWARGS)

    # Pull the final train loss from the trainer history
    history = model.history
    train_loss = float(history["train_loss_epoch"].iloc[-1])

    if math.isnan(train_loss):
        raise ValueError(f"NaN loss after training with modality_weights={modality_weights!r}")

    print(f"[OK] modality_weights={modality_weights!r}  final train loss = {train_loss:.4f}")
    return train_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--modality_weights",
        default="gating",
        choices=ALL_STRATEGIES,
        help="Latent combination strategy to test (default: gating)",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run all NEW strategies sequentially and report a summary table",
    )
    parser.add_argument(
        "--sweep_all",
        action="store_true",
        help="Like --sweep but includes existing strategies too (regression check)",
    )
    parser.add_argument(
        "--data_path",
        default=DATA_PATH,
        help=f"Path to .h5mu file (default: {DATA_PATH})",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=MAX_EPOCHS,
        help=f"Epochs per run (default: {MAX_EPOCHS})",
    )
    args = parser.parse_args()

    TRAIN_KWARGS["max_epochs"] = args.max_epochs

    mdata = load_mdata(args.data_path)

    if args.sweep_all:
        strategies = ALL_STRATEGIES
    elif args.sweep:
        strategies = NEW_STRATEGIES
    else:
        strategies = [args.modality_weights]

    results = {}
    failures = {}

    for strategy in strategies:
        try:
            loss = run_one(mdata, strategy)
            results[strategy] = loss
        except Exception as exc:
            failures[strategy] = str(exc)
            print(f"[FAIL] {strategy}: {exc}", file=sys.stderr)

    # Summary
    if len(strategies) > 1:
        print(f"\n{'='*60}")
        print("  Summary")
        print(f"{'='*60}")
        print(f"  {'strategy':<20}  {'loss':>10}  status")
        print(f"  {'-'*20}  {'-'*10}  ------")
        for s in strategies:
            if s in results:
                print(f"  {s:<20}  {results[s]:>10.4f}  OK")
            else:
                print(f"  {s:<20}  {'—':>10}  FAIL: {failures[s]}")

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
