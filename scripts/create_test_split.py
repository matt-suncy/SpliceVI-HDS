#!/usr/bin/env python
"""
Create a proper test split from training MuData with matching dimensions.

Ensures test MuData has the same genes/features as the training data
so that the trained model can make predictions on it.
"""

import argparse
import numpy as np
import mudata as mu


def create_test_split(train_path, output_path, test_frac=0.2, seed=42):
    """
    Load training MuData, split into smaller test MuData with same genes.
    
    Args:
        train_path: Path to full training MuData
        output_path: Path to save test split
        test_frac: Fraction of cells to use for test (0-1)
        seed: Random seed for reproducibility
    """
    print(f"[LOAD] Loading training MuData from {train_path}...")
    # Load into memory to allow subsetting and copying
    mdata = mu.read_h5mu(train_path)
    
    print(f"[INFO] Training MuData shapes:")
    print(f"       RNA: n_obs={mdata['rna'].n_obs}, n_vars={mdata['rna'].n_vars}")
    print(f"       Splicing: n_obs={mdata['splicing'].n_obs}, n_vars={mdata['splicing'].n_vars}")
    
    # Randomly select test cells
    np.random.seed(seed)
    n_cells = mdata['rna'].n_obs
    n_test = max(1, int(n_cells * test_frac))
    test_indices = np.random.choice(n_cells, size=n_test, replace=False)
    
    print(f"[SELECT] Selecting {n_test} cells ({test_frac*100:.1f}%) for test...")
    
    # Subset all modalities with same cell indices
    mdata_test = mu.MuData({
        'rna': mdata['rna'][test_indices, :].copy(),
        'splicing': mdata['splicing'][test_indices, :].copy()
    })
    
    print(f"[SAVE] Saving test split to {output_path}...")
    mdata_test.write_h5mu(output_path)
    
    print(f"[DONE] Test split created:")
    print(f"       RNA: n_obs={mdata_test['rna'].n_obs}, n_vars={mdata_test['rna'].n_vars}")
    print(f"       Splicing: n_obs={mdata_test['splicing'].n_obs}, n_vars={mdata_test['splicing'].n_vars}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create test split from training MuData with matching gene dimensions."
    )
    parser.add_argument(
        "--train-path",
        type=str,
        required=True,
        help="Path to training MuData file",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save test split MuData file",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.2,
        help="Fraction of cells to use for test set (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    
    args = parser.parse_args()
    create_test_split(args.train_path, args.output_path, args.test_frac, args.seed)
