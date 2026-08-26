"""
Parses the .txt result files written by the CCO/DGOT/SOS/GOIO baseline
tuners/runners and aggregates them into one gmean summary table, matching
the $mean%_{std%}$ cell format used in results_aggregation/aggregated_real_data_gmeans.csv.

Each .txt is a str()-dumped {'fitness', 'true_values', 'predicted_values'}
dict (see ga_cco_tuner.py / ga_dgot_tuner.py / ga_sos_tuner.py / no_ga_goio_runner.py
on_stop/run_goio_baseline), with true_values/predicted_values holding one
array per CV fold. Two directory conventions exist in this repo:
  - results/{cco,dgot,sos}/{method}_{dataset}.txt
  - predictions/{dataset_folder}/results/goio/goio_{dataset}.txt
"""

import os

import numpy as np
import pandas as pd
from imbalanced_ensemble.metrics import geometric_mean_score

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TOP_LEVEL_METHODS = ['cco', 'dgot', 'sos']
_METHOD_COLUMN_ORDER = ['cco', 'dgot', 'sos', 'goio']


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


def _format_cell(geometric_means):
    if not geometric_means:
        return ''
    mean_pct = np.mean(geometric_means) * 100
    std_pct = np.std(geometric_means) * 100
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


def _find_goio_results():
    found = []
    predictions_dir = os.path.join(_PROJECT_ROOT, 'predictions')
    if not os.path.isdir(predictions_dir):
        return found
    for ds_folder in os.listdir(predictions_dir):
        goio_dir = os.path.join(predictions_dir, ds_folder, 'results', 'goio')
        if not os.path.isdir(goio_dir):
            continue
        for fname in os.listdir(goio_dir):
            if fname.startswith('goio_') and fname.endswith('.txt'):
                dataset = fname[len('goio_'):-len('.txt')]
                found.append(('goio', dataset, os.path.join(goio_dir, fname)))
    return found


def build_summary():
    all_files = _find_top_level_results() + _find_goio_results()
    if not all_files:
        print("No result .txt files found under results/{cco,dgot,sos}/ or "
              "predictions/*/results/goio/ yet.")
        return pd.DataFrame()

    rows = {}
    methods_seen = set()
    for method, dataset, path in all_files:
        methods_seen.add(method)
        try:
            data = _parse_result_txt(path)
            cell = _format_cell(_geometric_means_per_fold(data))
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
    summary = build_summary()
    if not summary.empty:
        out_dir = os.path.join(_PROJECT_ROOT, 'results_aggregation')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'aggregated_sota_geometric_means.csv')
        summary.to_csv(out_path)
        print(summary)
        print(f"\nSaved summary to {out_path}")
