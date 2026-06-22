"""
Converts your dataset into the PyTorch tensor format expected by CCO.

Supported inputs
----------------
- NumPy arrays (X, Y)
- Pandas DataFrame (features + label column)
- CSV file path (features + label column)
- KEEL .dat file path (ARFF-like format used by imbalanced-learning benchmarks)

Output
------
Writes two files to `out_dir`:
    X  — float32 tensor of shape (n_samples, n_features)
    Y  — float32 tensor of shape (n_samples,) with values 0 and 1

Usage (CLI)
-----------
    python prepare_data.py --csv path/to/data.csv --label_col target --out data/my_dataset/
    python prepare_data.py --dat path/to/data.dat --out data/my_dataset/
    python prepare_data.py --dat path/to/data.dat --majority_label negative --out data/my_dataset/

Usage (API)
-----------
    from prepare_data import prepare_data
    prepare_data(X=my_array, Y=my_labels, out_dir="data/my_dataset/")
    prepare_data(dat_path="data/glass/glass2.dat", out_dir="data/glass2/")
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch


def _load_keel_dat(dat_path):
    """Parse a KEEL .dat file into (X_array float32, Y_array of raw label strings)."""
    attributes = []
    input_cols = []
    output_col = None
    in_data = False
    rows = []

    with open(dat_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            lower = line.lower()
            if lower.startswith('@attribute'):
                parts = line.split(None, 2)
                attributes.append(parts[1])
            elif lower.startswith('@inputs'):
                input_cols = [s.strip() for s in line.split(None, 1)[1].split(',')]
            elif lower.startswith('@outputs'):
                output_col = line.split(None, 1)[1].strip()
            elif lower.startswith('@data'):
                in_data = True
            elif in_data:
                rows.append(line)

    if not attributes:
        raise ValueError(f"No @attribute declarations found in {dat_path}")

    if output_col is None:
        output_col = attributes[-1] if not input_cols else next(
            a for a in attributes if a not in input_cols
        )

    df = pd.DataFrame(
        [[v.strip() for v in row.split(',')] for row in rows],
        columns=attributes,
    )

    feature_cols = [c for c in attributes if c != output_col]
    X = df[feature_cols].apply(pd.to_numeric).values.astype(np.float32)
    Y = df[output_col].values

    return X, Y


def prepare_data(out_dir, X=None, Y=None, df=None, csv_path=None, dat_path=None,
                 label_col=-1, majority_label=None):
    """
    Convert data to CCO-ready PyTorch tensors and save to out_dir.

    Parameters
    ----------
    out_dir : str
        Directory where X and Y tensor files will be written.
        Created automatically if it does not exist.
    X : array-like, optional
        Feature matrix of shape (n_samples, n_features).
        Provide together with Y when passing arrays directly.
    Y : array-like, optional
        Label vector of shape (n_samples,).
        Must contain exactly two unique values.
    df : pd.DataFrame, optional
        DataFrame containing features and a label column.
        Use label_col to identify the label.
    csv_path : str, optional
        Path to a CSV file. Loaded into a DataFrame, then treated
        the same as df.
    label_col : int or str, optional
        Column name or index for the label when using df or csv_path.
        Defaults to -1 (last column).
    majority_label : scalar, optional
        Which of the two class values should become 0 (majority).
        The other value becomes 1 (minority / positive class).
        If None, the more frequent class is automatically chosen as majority.

    Returns
    -------
    X_tensor : torch.Tensor  shape (n_samples, n_features), float32
    Y_tensor : torch.Tensor  shape (n_samples,), float32, values {0, 1}
    """
    # --- resolve input source ---
    if dat_path is not None:
        X, Y = _load_keel_dat(dat_path)
    elif csv_path is not None:
        df = pd.read_csv(csv_path, header=0)

    if df is not None:
        if isinstance(label_col, int) and not isinstance(label_col, bool):
            label_col = df.columns[label_col]
        Y = df[label_col].values
        X = df.drop(columns=[label_col]).values
    elif X is not None and Y is not None:
        X = np.array(X)
        Y = np.array(Y)
    else:
        raise ValueError("Provide either (X, Y), df, csv_path, or dat_path.")

    X = X.astype(np.float32)
    Y = Y.flatten()

    # --- validate binary labels ---
    unique_labels = np.unique(Y)
    if len(unique_labels) != 2:
        raise ValueError(
            f"Expected exactly 2 unique class values, found {len(unique_labels)}: {unique_labels}. "
            "CCO is configured for binary classification only."
        )

    # --- remap labels to 0 / 1 ---
    if majority_label is None:
        counts = {lbl: np.sum(Y == lbl) for lbl in unique_labels}
        majority_label = max(counts, key=counts.get)

    minority_label = unique_labels[unique_labels != majority_label][0]
    Y_binary = np.where(Y == majority_label, 0, 1).astype(np.float32)

    # --- report ---
    n_majority = int(np.sum(Y_binary == 0))
    n_minority = int(np.sum(Y_binary == 1))
    ir = n_majority / n_minority if n_minority > 0 else float('inf')
    print(f"Samples      : {len(Y_binary)}")
    print(f"Features     : {X.shape[1]}")
    print(f"Majority (0) : {n_majority}  [original label: {majority_label}]")
    print(f"Minority (1) : {n_minority}  [original label: {minority_label}]")
    print(f"Imbalance ratio: {ir:.2f}:1")

    # --- save ---
    os.makedirs(out_dir, exist_ok=True)
    X_tensor = torch.tensor(X)
    Y_tensor = torch.tensor(Y_binary)
    torch.save(X_tensor, os.path.join(out_dir, "X"))
    torch.save(Y_tensor, os.path.join(out_dir, "Y"))
    print(f"\nSaved to '{out_dir}'")
    print(f"  X : {list(X_tensor.shape)}  dtype={X_tensor.dtype}")
    print(f"  Y : {list(Y_tensor.shape)}  dtype={Y_tensor.dtype}")

    return X_tensor, Y_tensor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for CCO experiments.")
    parser.add_argument("--dat", type=str, help="Path to KEEL .dat file.")
    parser.add_argument("--csv", type=str, help="Path to CSV file.")
    parser.add_argument("--label_col", default=-1,
                        help="Label column name or index (default: last column).")
    parser.add_argument("--majority_label", default=None,
                        help="Original label value to treat as majority (class 0). "
                             "Auto-detected if omitted.")
    parser.add_argument("--out", type=str, required=True,
                        help="Output directory for X and Y tensor files.")
    args = parser.parse_args()

    # try to parse label_col as int, fall back to string
    try:
        label_col = int(args.label_col)
    except (ValueError, TypeError):
        label_col = args.label_col

    majority_label = args.majority_label
    if majority_label is not None:
        try:
            majority_label = int(majority_label)
        except ValueError:
            try:
                majority_label = float(majority_label)
            except ValueError:
                pass  # keep as string

    prepare_data(
        out_dir=args.out,
        dat_path=args.dat,
        csv_path=args.csv,
        label_col=label_col,
        majority_label=majority_label,
    )