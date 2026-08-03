"""
Fixed-baseline (no GA) run of GOIO on yeast3 — pilot/test version.

GOIO is imported and run exactly as published (main.py / MLVAE / CLDM
untouched, aside from adding gmean to CLDM/evaluation.py's reported metrics
for consistency with how the project already reports CCO/DGOT/SOS). This
script just drives its own 5-fold-CV pipeline once with the paper's default
hyperparameters, then aggregates the 5 folds' results — no hyperparameter
search.

NOTE: MLVAE/main.py hardcodes 'cuda:0' in a few spots (not gated by
args.device), so this requires an actual GPU — it will not run on a CPU-only
machine even though args.device is set below.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

from base_functions import get_yeast_3_data

seed = 2024  # matches GOIO's own main.py seed, kept identical for consistency
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_GOIO_PATH = os.path.join(_PROJECT_ROOT, 'ga_heso_sota_methods', 'GOIO')

# Computed before the chdir below (which points GOIO's own relative paths at
# its own directory) so the final result file lands next to the other
# methods' results, same convention as prediction_yeast_dgot.py / _sos.py.
_RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'GOIO_yeast_3')

# GOIO's own modules use relative paths ('./data/...', './ckpt_.../',
# './synthetic/...') and import each other as top-level packages
# (MLVAE.main, CLDM.main, ...) — same pattern as DGOT's tuner.
os.chdir(_GOIO_PATH)
sys.path.insert(0, _GOIO_PATH)

from data.dataprocessing import Default_processing          # noqa: E402
from MLVAE.main import main as train_MLVAE                  # noqa: E402
from CLDM.main import main as train_CLDM                    # noqa: E402
from CLDM.sample import main as sample_CLDM                 # noqa: E402
from CLDM.evaluation import main as eval_CLDM                # noqa: E402

DATANAME = 'yeast3'
N_FOLDS = 5


def _prepare_csv(dataname):
    """Write the project's yeast data into the .csv format GOIO's own
    Default_processing() expects: data/datasets/{name}/{name}.csv, all
    numeric columns, label last, header row present (contents unused —
    only .values is read)."""
    X_df, y_series = get_yeast_3_data()
    out_dir = os.path.join(_GOIO_PATH, 'data', 'datasets', dataname)
    os.makedirs(out_dir, exist_ok=True)

    df = X_df.copy()
    df['label'] = y_series.values
    df.to_csv(os.path.join(out_dir, f'{dataname}.csv'), index=False)


def _build_args(exp, device):
    """Mirrors GOIO's own get_args() defaults (utils.py) — the paper's
    published hyperparameters, no tuning."""
    return argparse.Namespace(
        dataname=DATANAME,
        exp=exp,
        threshold=0.2,
        dist=1,
        proto=1,
        kld=1,
        condition=True,       # get_args() uses action='store_false' -> default True
        max_beta=1e-2,
        min_beta=1e-5,
        lambd=0.7,
        save_path=f'synthetic/{DATANAME}/exp{exp}/CLDM',
        device=device,
        gpu=0,
    )


def _aggregate(fold_csvs):
    """Each fold's CLDM/evaluation.py output has 50 rows (5 classifiers x
    10 repetitions) plus an appended mean row and an appended std row.
    Pull the per-fold mean row, then average across folds — same pattern
    as the project's CCO/DGOT/SOS tuners averaging per-fold gmean."""
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


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    print(f"Preparing {DATANAME} data for GOIO...")
    _prepare_csv(DATANAME)

    split_args = argparse.Namespace(dataname=DATANAME)
    Default_processing(split_args)

    fold_csvs = []
    true_values = []
    predicted_values = []
    fold_gmeans = []

    for exp in range(N_FOLDS):
        print(f"\n=== {DATANAME} exp{exp}: training MLVAE ===")
        args = _build_args(exp, device)
        train_MLVAE(args)

        print(f"=== {DATANAME} exp{exp}: training CLDM ===")
        train_CLDM(args)

        print(f"=== {DATANAME} exp{exp}: sampling synthetic data ===")
        sample_CLDM(args)

        print(f"=== {DATANAME} exp{exp}: evaluating ===")
        rf_true, rf_pred, fold_gmean = eval_CLDM(args)
        true_values.append(rf_true)
        predicted_values.append(rf_pred)
        fold_gmeans.append(fold_gmean)

        fold_csvs.append(os.path.join(args.save_path, f'{DATANAME}.csv'))

    print("\n=== Aggregating across folds ===")
    fold_means_df, summary = _aggregate(fold_csvs)
    print(fold_means_df)
    print(summary)

    csv_out_path = os.path.join(_GOIO_PATH, 'synthetic', DATANAME, f'{DATANAME}_summary.csv')
    summary.to_csv(csv_out_path)
    print(f"\nSaved fold-aggregated summary to {csv_out_path}")

    # Same {'fitness', 'true_values', 'predicted_values'} .txt convention
    # used by DGOT/SOS's on_stop() — fitness = mean G-mean across the 5
    # folds (each fold's G-mean itself already averaged over that fold's
    # 5 classifiers x 10 repetitions); true/predicted_values are the
    # RandomForestClassifier's per-sample predictions per fold.
    result = {
        'fitness': float(np.mean(fold_gmeans)),
        'true_values': true_values,
        'predicted_values': predicted_values,
    }
    os.makedirs(os.path.dirname(_RESULTS_FILE), exist_ok=True)
    with open(_RESULTS_FILE + '.txt', 'w') as f:
        f.write(str(result))
    print(f"Saved DGOT/SOS-style result file to {_RESULTS_FILE}.txt")
    print(f"evaluated fitness: {result['fitness']:.6f}")