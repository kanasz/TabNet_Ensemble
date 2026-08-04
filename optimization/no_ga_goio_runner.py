"""
Fixed-baseline (no GA) runner for GOIO — reusable across datasets.

GOIO is imported and run exactly as published (main.py / MLVAE / CLDM
untouched, aside from CLDM/evaluation.py additively reporting gmean/AUC and
returning per-sample predictions, for consistency with how CCO/DGOT/SOS are
reported elsewhere in this project). run_goio_baseline() drives GOIO's own
5-fold-CV pipeline once with the paper's default hyperparameters and
aggregates the 5 folds' results — no hyperparameter search.

NOTE: MLVAE/main.py hardcodes 'cuda:0' in a few spots internally (not gated
by args.device), so this requires an actual GPU.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GOIO_PATH = os.path.join(_PROJECT_ROOT, 'ga_heso_sota_methods', 'GOIO')

# GOIO's own modules use relative paths ('./data/...', './ckpt_.../',
# './synthetic/...') and import each other as top-level packages
# (MLVAE.main, CLDM.main, ...) — same pattern as DGOT's tuner.
os.chdir(_GOIO_PATH)
sys.path.insert(0, _GOIO_PATH)

from ga_heso_sota_methods.GOIO.data.dataprocessing import Default_processing
from ga_heso_sota_methods.GOIO.MLVAE.main import main as train_mlvae
from ga_heso_sota_methods.GOIO.CLDM.main import main as train_cldm
from ga_heso_sota_methods.GOIO.CLDM.sample import main as sample_cldm
from ga_heso_sota_methods.GOIO.CLDM.evaluation import main as eval_cldm

# Default_processing() always builds exactly 5 StratifiedKFold folds
# internally (hardcoded n_splits=5) — not a free parameter here.
N_FOLDS = 5


def _prepare_csv(data, dataset_name, categorical_cols=None):
    """Write (X_df, y_series) into the .csv format GOIO's own Default_processing()
    expects: data/datasets/{name}/{name}.csv, all numeric columns, label last, header row present.

    Default_processing() does `pd.read_csv(...).values.astype(float)`, so every
    column must be float-castable — categorical columns (e.g. abalone's 'Sex')
    can't be written as raw strings. They're label-encoded to small consecutive
    integers (0, 1, 2, ...) instead of one-hot: GOIO's own data_description()
    auto-detects "categorical" columns by exactly this pattern (low cardinality,
    consecutive non-negative integers — see the bundled abalone_15 example,
    whose categorical column is 0.0/1.0/2.0), and its tokenizer expects a single
    integer category index per column, not one-hot vectors."""
    x_df, y_series = data
    x_df = x_df.copy()
    categorical_cols = categorical_cols or []

    for col in x_df.columns:
        if col in categorical_cols:
            x_df[col] = pd.factorize(x_df[col].astype(str).str.strip())[0].astype(float)
        else:
            x_df[col] = x_df[col].astype(float)

    out_dir = os.path.join(_GOIO_PATH, 'data', 'datasets', dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    df = x_df
    df['label'] = y_series.values
    df.to_csv(os.path.join(out_dir, f'{dataset_name}.csv'), index=False)


def _build_args(dataset_name, exp, device, dist, proto, kld, condition, max_beta, min_beta, lambd, threshold):
    return argparse.Namespace(
        dataname=dataset_name,
        exp=exp,
        threshold=threshold,
        dist=dist,
        proto=proto,
        kld=kld,
        condition=condition,
        max_beta=max_beta,
        min_beta=min_beta,
        lambd=lambd,
        save_path=f'synthetic/{dataset_name}/exp{exp}/CLDM',
        device=device,
        gpu=0,
    )


def _aggregate(fold_csvs):
    """Each fold's CLDM/evaluation.py output has 50 rows (5 classifiers x 10 repetitions) plus an appended mean row
    and an appended std row. Pull the per-fold mean row, then average across folds."""
    fold_means = []
    for path in fold_csvs:
        df = pd.read_csv(path, index_col=0)
        fold_means.append(df.iloc[-2])  # the appended "means" row

    fold_means_df = pd.DataFrame(fold_means)
    summary = pd.DataFrame({
        'mean_across_folds': fold_means_df.mean(axis=0),
        'std_across_folds': fold_means_df.std(axis=0),
    })
    return fold_means_df, summary


def run_goio_baseline(data, dataset_name, results_file, categorical_cols=None, dist=1, proto=1, kld=1,
                      condition=True, max_beta=1e-2, min_beta=1e-5, lambd=0.7, threshold=0.2, seed=2024):
    """categorical_cols: names of any true categorical feature columns in
    data[0] (e.g. ['Sex'] for abalone) — leave None for purely numeric
    datasets (yeast, wine, glass, ecoli)."""

    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    print(f"Preparing {dataset_name} data for GOIO...")
    _prepare_csv(data, dataset_name, categorical_cols=categorical_cols)

    split_args = argparse.Namespace(dataname=dataset_name)
    Default_processing(split_args)

    fold_csvs = []
    true_values = []
    predicted_values = []
    fold_geometric_means = []

    for exp in range(N_FOLDS):
        print(f"\n=== {dataset_name} exp{exp}: training MLVAE ===")
        args = _build_args(dataset_name, exp, device, dist, proto, kld, condition,
                           max_beta, min_beta, lambd, threshold)
        train_mlvae(args)

        print(f"=== {dataset_name} exp{exp}: training CLDM ===")
        train_cldm(args)

        print(f"=== {dataset_name} exp{exp}: sampling synthetic data ===")
        sample_cldm(args)

        print(f"=== {dataset_name} exp{exp}: evaluating ===")
        rf_true, rf_pred, fold_gmean = eval_cldm(args)
        true_values.append(rf_true)
        predicted_values.append(rf_pred)
        fold_geometric_means.append(fold_gmean)

        fold_csvs.append(os.path.join(args.save_path, f'{dataset_name}.csv'))

    print("\n=== Aggregating across folds ===")
    fold_means_df, summary = _aggregate(fold_csvs)
    print(fold_means_df)
    print(summary)

    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    csv_out_path = results_file + '_summary.csv'
    summary.to_csv(csv_out_path)
    print(f"\nSaved fold-aggregated summary to {csv_out_path}")

    result = {
        'fitness': float(np.mean(fold_geometric_means)),
        'true_values': true_values,
        'predicted_values': predicted_values,
    }
    with open(results_file + '.txt', 'w') as f:
        f.write(str(result))
    print(f"Saved DGOT/SOS-style result file to {results_file}.txt")
    print(f"evaluated fitness: {result['fitness']:.6f}")

    return result
