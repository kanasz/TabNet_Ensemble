"""
Converts a project dataset into the folder structure expected by DGOT.

DGOT expects (per fold expK):
    datasets/{name}/DGOT/expK/xtrain.npy   shape (n_train, 1, n_features)  normalised [-1,1]
    datasets/{name}/DGOT/expK/ytrain.npy   shape (n_train,)
    datasets/{name}/TEST/expK/xtest.npy    shape (n_test,  n_features)     normalised [-1,1]
    datasets/{name}/TEST/expK/ytest.npy    shape (n_test,)

Numerical columns are normalised to [-1, 1] using per-fold min-max (no leakage).
Categorical columns are one-hot encoded using a per-fold encoder (no leakage).
Encoded features are concatenated: [numericals | one-hot categoricals].

Usage (API)
-----------
    from prepare_data import prepare_dgot_data
    data = get_yeast_3_data()          # (X_df, y_series) from base_functions
    feat_len = prepare_dgot_data(data, 'yeast3')  # writes into ./datasets/yeast3/
"""

import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder


def prepare_dgot_data(data, dataset_name, base_dir=None, n_splits=5, random_state=42,
                      numerical_cols=None, categorical_cols=None):
    """
    Parameters
    ----------
    data             : tuple (X_df, y_series) as returned by get_*_data() helpers
    dataset_name     : folder name under datasets/ (e.g. 'yeast3')
    base_dir         : root of DGOT directory; defaults to the directory of this file
    numerical_cols   : list of column names to normalise to [-1, 1];
                       auto-detected from dtype if None
    categorical_cols : list of column names to one-hot encode;
                       auto-detected from dtype if None

    Returns
    -------
    encoded_feature_len : int — total feature width after encoding
                          (pass this to GaDGOTTuner as feature_len)
    """
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    X_df, y_series = data

    if numerical_cols is None:
        numerical_cols = list(X_df.select_dtypes(include='number').columns)
    if categorical_cols is None:
        # derive from columns not listed as numerical — avoids dtype misdetection
        # on KEEL files where float columns may load as object dtype
        categorical_cols = [c for c in X_df.columns if c not in numerical_cols]

    X_num = X_df[numerical_cols].values.astype(float) if numerical_cols else np.empty((len(X_df), 0))
    X_cat = X_df[categorical_cols].values              if categorical_cols else np.empty((len(X_df), 0))
    y     = y_series.values.astype(float)

    skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)

    encoded_feature_len = None

    for k, (train_idx, test_idx) in enumerate(skf.split(X_df.values, y)):
        dgot_dir = os.path.join(base_dir, 'datasets', dataset_name, 'DGOT', f'exp{k}')
        test_dir = os.path.join(base_dir, 'datasets', dataset_name, 'TEST',  f'exp{k}')
        os.makedirs(dgot_dir, exist_ok=True)
        os.makedirs(test_dir,  exist_ok=True)

        # numerical: min-max → [-1, 1], fit on train fold only
        if X_num.shape[1] > 0:
            X_num_train = X_num[train_idx]
            X_num_test  = X_num[test_idx]
            dmin = np.min(X_num_train, axis=0)
            dmax = np.max(X_num_train, axis=0)
            X_num_train = ((X_num_train - dmin) / (dmax - dmin + 1e-8)) * 2 - 1
            X_num_test  = ((X_num_test  - dmin) / (dmax - dmin + 1e-8)) * 2 - 1
        else:
            X_num_train = np.empty((len(train_idx), 0))
            X_num_test  = np.empty((len(test_idx),  0))

        # categorical: one-hot encode, fit on train fold only
        if X_cat.shape[1] > 0:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            X_cat_train = ohe.fit_transform(X_cat[train_idx])
            X_cat_test  = ohe.transform(X_cat[test_idx])
        else:
            X_cat_train = np.empty((len(train_idx), 0))
            X_cat_test  = np.empty((len(test_idx),  0))

        X_train_norm = np.concatenate([X_num_train, X_cat_train], axis=1)
        X_test_norm  = np.concatenate([X_num_test,  X_cat_test],  axis=1)

        if encoded_feature_len is None:
            encoded_feature_len = X_train_norm.shape[1]

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
    print(f"Features after encoding: {encoded_feature_len}  |  "
          f"Use --feature_len={encoded_feature_len} when running train.py")

    return encoded_feature_len