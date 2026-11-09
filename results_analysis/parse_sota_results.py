"""
Aggregates the per-fold results of the original-protocol SOTA baselines into a
G-mean and a ROC AUC summary table, matching the $mean%_{std%}$ cell format
used in results_aggregation/aggregated_real_data_gmeans.csv:
  results_aggregation/aggregated_sota_geometric_means.csv
  results_aggregation/aggregated_sota_roc_auc.csv

Only these baselines are read, each from
  predictions/{dataset_folder}/results/{method}/{method}_{dataset}_folds.csv
(one row per CV fold, with 'gmean' and 'auc' columns):
  - cco_original   (optimization/no_ga_cco_original_runner.py)
  - dgot_baseline  (optimization/no_ga_dgot_runner.py)
  - sos_original   (optimization/no_ga_sos_original_runner.py)
  - goio_original  (optimization/no_ga_goio_original_runner.py)
  - mlos_original  (optimization/no_ga_mlos_original_runner.py)

The multi-classifier runners (dgot_baseline, goio_original, sos_original,
mlos_original) already store each fold's mean over 5 classifiers x 10 repeats
in the _folds.csv; AUCs there come from predict_proba / decision_function.
"""

import os

import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_METHODS = ['cco_original', 'dgot_baseline', 'sos_original', 'goio_original', 'mlos_original']
_FOLDS_SUFFIX = '_folds.csv'
_OUTPUT_FILES = {
    'gmean': 'aggregated_sota_geometric_means.csv',
    'auc': 'aggregated_sota_roc_auc.csv',
}


def _format_cell(values):
    values = [v for v in values if not np.isnan(v)]
    if not values:
        return ''
    mean_pct = np.mean(values) * 100
    std_pct = np.std(values) * 100
    return f'${mean_pct:.1f}_{{{std_pct:.0f}}}$'


def _find_fold_csvs():
    found = []
    predictions_dir = os.path.join(_PROJECT_ROOT, 'predictions')
    if not os.path.isdir(predictions_dir):
        return found
    for ds_folder in os.listdir(predictions_dir):
        for method in _METHODS:
            method_dir = os.path.join(predictions_dir, ds_folder, 'results', method)
            if not os.path.isdir(method_dir):
                continue
            prefix = f'{method}_'
            for fname in os.listdir(method_dir):
                if fname.startswith(prefix) and fname.endswith(_FOLDS_SUFFIX):
                    dataset = fname[len(prefix):-len(_FOLDS_SUFFIX)]
                    found.append((method, dataset, os.path.join(method_dir, fname)))
    return found


def build_summary(metric='gmean'):
    """metric: 'gmean' or 'auc'."""
    if metric not in _OUTPUT_FILES:
        raise ValueError(f"metric must be one of {sorted(_OUTPUT_FILES)}")
    all_files = _find_fold_csvs()
    if not all_files:
        print(f"No *{_FOLDS_SUFFIX} files found under predictions/*/results/{{{','.join(_METHODS)}}}/ yet.")
        return pd.DataFrame()

    rows = {}
    methods_seen = set()
    for method, dataset, path in all_files:
        methods_seen.add(method)
        try:
            cell = _format_cell(pd.read_csv(path, index_col=0)[metric].tolist())
        except Exception as exc:
            print(f"Failed to parse {path}: {exc}")
            cell = ''
        rows.setdefault(dataset, {})[method] = cell

    method_order = [m for m in _METHODS if m in methods_seen]
    df = pd.DataFrame.from_dict(rows, orient='index', columns=method_order)
    df.index.name = 'dataset'
    df = df.sort_index()
    return df


if __name__ == '__main__':
    out_dir = os.path.join(_PROJECT_ROOT, 'results_aggregation')
    for metric, file_name in _OUTPUT_FILES.items():
        summary = build_summary(metric)
        if summary.empty:
            continue
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, file_name)
        summary.to_csv(out_path)
        print(f"\n=== {metric} ===")
        print(summary)
        print(f"Saved summary to {out_path}")
