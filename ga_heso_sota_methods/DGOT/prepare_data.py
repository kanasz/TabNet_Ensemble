"""
Converts a project dataset into the folder structure expected by DGOT.

DGOT expects (per fold expK):
    datasets/{name}/DGOT/expK/xtrain.npy   shape (n_train, 1, n_features)  normalised [-1,1]
    datasets/{name}/DGOT/expK/ytrain.npy   shape (n_train,)
    datasets/{name}/TEST/expK/xtest.npy    shape (n_test,  n_features)     normalised [-1,1]
    datasets/{name}/TEST/expK/ytest.npy    shape (n_test,)

Normalisation is fitted on the train fold only (no data leakage).

Usage (API)
-----------
    from prepare_data import prepare_dgot_data
    data = get_yeast_3_data()          # (X_df, y_series) from base_functions
    prepare_dgot_data(data, 'yeast3')  # writes into ./datasets/yeast3/
"""

import os
import numpy as np
from sklearn.model_selection import StratifiedKFold


def prepare_dgot_data(data, dataset_name, base_dir=None, n_splits=5, random_state=42):
    """
    Parameters
    ----------
    data         : tuple (X_df, y_series) as returned by get_*_data() helpers
    dataset_name : folder name under datasets/ (e.g. 'yeast3')
    base_dir     : root of DGOT directory; defaults to the directory of this file
    """
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    X_df, y_series = data
    X = X_df.values.astype(float)
    y = y_series.values.astype(float)

    skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)

    for k, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        dgot_dir = os.path.join(base_dir, 'datasets', dataset_name, 'DGOT', f'exp{k}')
        test_dir = os.path.join(base_dir, 'datasets', dataset_name, 'TEST',  f'exp{k}')
        os.makedirs(dgot_dir, exist_ok=True)
        os.makedirs(test_dir,  exist_ok=True)

        X_train_raw = X[train_idx]
        X_test_raw  = X[test_idx]

        # normalisation fitted on train fold only — no test statistics leak in
        dmin = np.min(X_train_raw, axis=0)
        dmax = np.max(X_train_raw, axis=0)
        X_train_norm = ((X_train_raw - dmin) / (dmax - dmin + 1e-8)) * 2 - 1
        X_test_norm  = ((X_test_raw  - dmin) / (dmax - dmin + 1e-8)) * 2 - 1

        y_train = y[train_idx]
        y_test  = y[test_idx]

        # DGOT training format: (n, 1, features)
        np.save(os.path.join(dgot_dir, 'xtrain.npy'), X_train_norm[:, None, :])
        np.save(os.path.join(dgot_dir, 'ytrain.npy'), y_train)

        # TEST format: (n, features)
        np.save(os.path.join(test_dir, 'xtest.npy'), X_test_norm)
        np.save(os.path.join(test_dir, 'ytest.npy'), y_test)

        print(f"exp{k}: train={len(y_train)}  test={len(y_test)}  "
              f"minority_train={int(y_train.sum())}  minority_test={int(y_test.sum())}")

    print(f"\nData saved to: {os.path.join(base_dir, 'datasets', dataset_name)}")
    print(f"Features: {X.shape[1]}  |  Use --feature_len={X.shape[1]} when running train.py")