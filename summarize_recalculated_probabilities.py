"""Summarise the <name>_proba.csv files written by recalculate_ga_heso_probabilities.py.

One row per checkpoint:
    dataset | method
    original_gmean        G-mean of the stored 0/1 predictions (<name>.txt)
    original_auc_hard     AUC of the stored 0/1 predictions - no probabilities available
    recalculated_gmean    G-mean of the retrained best chromosome (majority vote)
    recalculated_auc_proba  AUC of its probabilities (mean P(class 1) of the members)
All metrics are computed per fold and averaged over the 5 folds, like the GA fitness.

Usage (repo root):
    python summarize_recalculated_probabilities.py predictions/ecoli/results
"""
import glob
import os
import sys

import pandas as pd
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import roc_auc_score


def _fold_mean(df, metric, score_col):
    return df.groupby('fold').apply(lambda d: metric(d.y_true, d[score_col])).mean()


def summarize(results_dir):
    family = os.path.basename(os.path.dirname(os.path.abspath(results_dir)))  # e.g. "ecoli"
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, '*_proba.csv'))):
        name = os.path.basename(path)[:-len('_proba.csv')]
        method, sep, rest = name.partition('_' + family)
        df = pd.read_csv(path)
        has_stored = 'stored_pred' in df
        rows.append({
            'dataset': family + rest if sep else name,
            'method': method if sep else name,
            'original_gmean': _fold_mean(df, geometric_mean_score, 'stored_pred') if has_stored else None,
            'original_auc_hard': _fold_mean(df, roc_auc_score, 'stored_pred') if has_stored else None,
            'recalculated_gmean': _fold_mean(df, geometric_mean_score, 'y_pred'),
            'recalculated_auc_proba': _fold_mean(df, roc_auc_score, 'proba_mean'),
        })
    return pd.DataFrame(rows).sort_values(['dataset', 'method']).reset_index(drop=True)


if __name__ == '__main__':
    results_dir = sys.argv[1] if len(sys.argv) > 1 else 'predictions/ecoli/results'
    summary = summarize(results_dir)
    out = os.path.join(results_dir, 'ga_heso_recalculated_summary.csv')
    summary.round(4).to_csv(out, index=False)
    print(summary.round(4).to_string(index=False))
    print('\n-> ' + out)
