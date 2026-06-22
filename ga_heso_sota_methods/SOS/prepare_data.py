"""
Converts a project dataset into the file structure expected by SOS.

SOS expects per-fold files under tabular_datasets/:
    {dataset_name}_exp{k}.npz   — numpy archive with 'train' and 'test' arrays
                                   shape (n, n_features + 1), last col = label
    {dataset_name}_exp{k}.json  — column metadata (types, min/max, i2s)

Data is NOT pre-normalised here — SOS's GeneralTransformer does that internally
(continuous columns → [0,1], categorical label → one-hot).

Returns image_size: the GeneralTransformer output dimension, which equals
    n_continuous_features + n_label_classes
This must be set as config.data.image_size before calling run_lib.train().

Usage (API)
-----------
    from prepare_data import prepare_sos_data
    data = get_yeast_3_data()          # (X_df, y_series) from base_functions
    image_size = prepare_sos_data(data, 'yeast3')
"""

import json
import os

import numpy as np
from sklearn.model_selection import StratifiedKFold

_SOS_PATH = os.path.dirname(os.path.abspath(__file__))
_TABULAR_DIR = os.path.join(_SOS_PATH, 'tabular_datasets')


def prepare_sos_data(data, dataset_name, n_splits=5, random_state=42):
    """
    Parameters
    ----------
    data         : tuple (X_df, y_series) as returned by get_*_data() helpers
    dataset_name : base name for the output files, e.g. 'yeast3'

    Returns
    -------
    image_size : int — transformed feature dimension to set as config.data.image_size
    """
    X_df, y_series = data
    X = X_df.values.astype(float)
    y = y_series.values

    n_features = X.shape[1]
    unique_labels = sorted(set(y.tolist()))
    n_classes = len(unique_labels)

    # Build JSON metadata — all feature columns are continuous, label is categorical.
    # min/max are computed on the full dataset so GeneralTransformer normalises
    # consistently regardless of which fold is active.
    columns = []
    for i, col_name in enumerate(X_df.columns):
        columns.append({
            "name": str(col_name),
            "type": "continuous",
            "min": float(X[:, i].min()),
            "max": float(X[:, i].max()),
        })
    columns.append({
        "name": "label",
        "type": "categorical",
        "size": n_classes,
        "i2s": [int(v) for v in unique_labels],
    })
    meta = {"columns": columns, "problem_type": "binary_classification"}

    # GeneralTransformer output dim:
    #   continuous columns → 1 col each
    #   categorical label  → n_classes cols (one-hot)
    image_size = n_features + n_classes

    os.makedirs(_TABULAR_DIR, exist_ok=True)

    skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    for k, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # SOS expects label as last column; raw (non-normalised) values
        train_arr = np.column_stack([X_train, y_train])
        test_arr = np.column_stack([X_test, y_test])

        np.savez(os.path.join(_TABULAR_DIR, f'{dataset_name}_exp{k}.npz'),
                 train=train_arr, test=test_arr)
        with open(os.path.join(_TABULAR_DIR, f'{dataset_name}_exp{k}.json'), 'w') as f:
            json.dump(meta, f, indent=4)

        print(f"exp{k}: train={len(y_train)}  test={len(y_test)}  "
              f"minority_train={int((y_train == unique_labels[-1]).sum())}  "
              f"minority_test={int((y_test == unique_labels[-1]).sum())}")

    print(f"\nData saved to: {_TABULAR_DIR}")
    print(f"Features: {n_features}  |  image_size={image_size}")
    return image_size