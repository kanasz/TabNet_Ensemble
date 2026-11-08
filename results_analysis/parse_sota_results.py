"""
Parses the .txt result files written by the CCO/DGOT/SOS/GOIO baseline
tuners/runners and aggregates them into a G-mean and a ROC AUC summary table,
matching the $mean%_{std%}$ cell format used in
results_aggregation/aggregated_real_data_gmeans.csv:
  results_aggregation/aggregated_sota_geometric_means.csv
  results_aggregation/aggregated_sota_roc_auc.csv

Each .txt is a str()-dumped {'fitness', 'true_values', 'predicted_values'}
dict (see ga_cco_tuner.py / ga_dgot_tuner.py / ga_sos_tuner.py / no_ga_goio_runner.py
on_stop/run_goio_baseline), with true_values/predicted_values holding one
array per CV fold. Two directory conventions exist in this repo:
  - results/{cco,dgot,sos}/{method}_{dataset}.txt
  - predictions/{dataset_folder}/results/{goio,dgot_baseline,cco_original,sos_original,goio_original}/{method}_{dataset}.txt

dgot_baseline (optimization/no_ga_dgot_runner.py) evaluates 5 classifiers x 10
repeats per fold; its .txt holds only RandomForest's last repeat, so its cell
is built from the per-fold values in {method}_{dataset}_folds.csv instead.

goio (optimization/no_ga_goio_runner.py) likewise evaluates 5 classifiers x 10
repeats per fold, and its .txt holds only RandomForest's last repeat. Its cell
therefore comes from goio_{dataset}_summary.csv (mean/std across the 5 folds,
AUC from predict_proba), so goio and dgot_baseline are aggregated the same way.
That file stores the sample std (ddof=1); it is rescaled to the population std
used by every other cell. Falls back to the .txt if the summary is missing.

goio_original (optimization/no_ga_goio_original_runner.py) is read from its
_folds.csv the same way (mean over 5 classifiers x 10 repeats per fold, AUC
from predict_proba / SVC decision_function).

AUC caveat: the .txt files hold only hard 0/1 predictions, so for cco / dgot /
sos the AUC computed from them is not a curve-based AUC (with hard labels it
equals balanced accuracy). dgot_baseline's and goio's AUCs are real ones,
computed from predict_proba scores.
"""

import os

import numpy as np
import pandas as pd
from imbalanced_ensemble.metrics import geometric_mean_score
from sklearn.metrics import roc_auc_score

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TOP_LEVEL_METHODS = ['cco', 'dgot', 'sos']
_PREDICTIONS_FOLDER_METHODS = ['goio', 'dgot_baseline', 'cco_original', 'sos_original', 'goio_original']
# Methods whose per-fold G-means are read from {method}_{dataset}_folds.csv.
_FOLD_CSV_METHODS = {'dgot_baseline', 'cco_original', 'sos_original', 'goio_original'}
# Methods whose mean/std are read from {method}_{dataset}_summary.csv.
_SUMMARY_CSV_METHODS = {'goio'}
_N_FOLDS = 5
# .txt files in a method folder that are not result files: cco_original also
# keeps upstream CCO's raw log as {method}_{dataset}_upstream_TEST.txt.
_AUXILIARY_TXT_SUFFIXES = ('_upstream_TEST.txt',)
_METHOD_COLUMN_ORDER = ['cco', 'cco_original', 'dgot', 'dgot_baseline', 'sos', 'sos_original', 'goio', 'goio_original']
_OUTPUT_FILES = {
    'gmean': 'aggregated_sota_geometric_means.csv',
    'auc': 'aggregated_sota_roc_auc.csv',
}


def _parse_result_txt(path):
    with open(path, 'r') as f:
        content = f.read()

    # str(result) renders numpy arrays as e.g. array([0, 1, ...], dtype=int64)
    # or dtype=float64 for GOIO (its raw labels come from a float64 npz) —
    # strip that down to plain Python literals eval() can read back.
    content = content.replace(" ", "").replace("\n", "")
    content = (content.replace("array(", "")
               .replace(")", "")
               .replace(",dtype=int64", "").replace("dtype=int64,", "")
               .replace(",dtype=float64", "").replace("dtype=float64,", "")
               .replace(",dtype=object", "").replace("dtype=object,", ""))
    return eval(content)


def _geometric_means_per_fold(data):
    if data.get('true_values') is None:
        return None
    return [
        geometric_mean_score(true_values, predicted)
        for true_values, predicted in zip(data['true_values'], data['predicted_values'])
    ]


def _roc_aucs_per_fold(data):
    if data.get('true_values') is None:
        return None
    aucs = []
    for true_values, predicted in zip(data['true_values'], data['predicted_values']):
        # A fold whose test part holds one class only has no defined AUC.
        if len(np.unique(true_values)) == 2:
            aucs.append(roc_auc_score(true_values, predicted))
    return aucs


def _format_cell(values):
    if values is None:
        return ''
    values = [v for v in values if not np.isnan(v)]
    if not values:
        return ''
    mean_pct = np.mean(values) * 100
    std_pct = np.std(values) * 100
    return f'${mean_pct:.1f}_{{{std_pct:.0f}}}$'


def _find_top_level_results():
    found = []
    for method in _TOP_LEVEL_METHODS:
        method_dir = os.path.join(_PROJECT_ROOT, 'results', method)
        if not os.path.isdir(method_dir):
            continue
        prefix = f'{method}_'
        for file_name in os.listdir(method_dir):
            if file_name.startswith(prefix) and file_name.endswith('.txt'):
                dataset = file_name[len(prefix):-len('.txt')]
                found.append((method, dataset, os.path.join(method_dir, file_name)))
    return found


def _find_predictions_folder_results():
    found = []
    predictions_dir = os.path.join(_PROJECT_ROOT, 'predictions')
    if not os.path.isdir(predictions_dir):
        return found
    for ds_folder in os.listdir(predictions_dir):
        for method in _PREDICTIONS_FOLDER_METHODS:
            method_dir = os.path.join(predictions_dir, ds_folder, 'results', method)
            if not os.path.isdir(method_dir):
                continue
            prefix = f'{method}_'
            for fname in os.listdir(method_dir):
                if (fname.startswith(prefix) and fname.endswith('.txt')
                        and not fname.endswith(_AUXILIARY_TXT_SUFFIXES)):
                    dataset = fname[len(prefix):-len('.txt')]
                    found.append((method, dataset, os.path.join(method_dir, fname)))
    return found


def _metric_from_folds_csv(txt_path, metric):
    folds_csv = txt_path[:-len('.txt')] + '_folds.csv'
    return pd.read_csv(folds_csv, index_col=0)[metric].tolist()


def _metric_per_fold_from_txt(path, metric):
    data = _parse_result_txt(path)
    if metric == 'gmean':
        return _geometric_means_per_fold(data)
    return _roc_aucs_per_fold(data)


def _format_summary_cell(summary_csv, metric):
    row = pd.read_csv(summary_csv, index_col=0).loc[metric]
    mean_pct = row['mean_across_folds'] * 100
    # pandas .std() is ddof=1; convert to np.std's ddof=0 like _format_cell.
    std_pct = row['std_across_folds'] * np.sqrt((_N_FOLDS - 1) / _N_FOLDS) * 100
    return f'${mean_pct:.1f}_{{{std_pct:.0f}}}$'


def build_summary(metric='gmean'):
    """metric: 'gmean' or 'auc'."""
    if metric not in _OUTPUT_FILES:
        raise ValueError(f"metric must be one of {sorted(_OUTPUT_FILES)}")
    all_files = _find_top_level_results() + _find_predictions_folder_results()
    if not all_files:
        print("No result .txt files found under results/{cco,dgot,sos}/ or "
              "predictions/*/results/{goio,dgot_baseline,cco_original,sos_original,goio_original}/ yet.")
        return pd.DataFrame()

    rows = {}
    methods_seen = set()
    for method, dataset, path in all_files:
        methods_seen.add(method)
        try:
            summary_csv = path[:-len('.txt')] + '_summary.csv'
            if method in _FOLD_CSV_METHODS:
                cell = _format_cell(_metric_from_folds_csv(path, metric))
            elif method in _SUMMARY_CSV_METHODS and os.path.exists(summary_csv):
                cell = _format_summary_cell(summary_csv, metric)
            else:
                cell = _format_cell(_metric_per_fold_from_txt(path, metric))
        except Exception as exc:
            print(f"Failed to parse {path}: {exc}")
            cell = ''
        rows.setdefault(dataset, {})[method] = cell

    method_order = [m for m in _METHOD_COLUMN_ORDER if m in methods_seen]
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
