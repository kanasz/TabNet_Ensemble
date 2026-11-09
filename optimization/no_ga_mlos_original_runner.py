"""
Fixed-baseline (no GA) runner for MLOS, run the way the authors published it.

MLOS ("Synthetic oversampling with Mahalanobis distance and local information
for highly imbalanced class-overlapped data", Yan et al., Expert Systems with
Applications 260, 2025, https://github.com/ytyancp/MLOS) is an oversampler
only: the repository is a single MLOS.py (plus the paper's datasets) with no
training / evaluation pipeline. It is imported from an untouched vendored copy
in ga_heso_sota_methods/MLOS_ORIGINAL (upstream commit UPSTREAM_COMMIT) and
called exactly as upstream's own __main__ does:

    sw = MLOS(len=1)
    X_res, y_res = sw.fit_Sample(x_train, y_train, len(maj) - len(min))

with upstream's module defaults (k=5 nearest neighbours, aer=-0.5), i.e. the
minority class is oversampled up to the majority count. Nothing in MLOS.py is
edited.

Protocol around it (the repository defines none, so this mirrors the GOIO
baseline's evaluation and is the same for every dataset):
  - folds: StratifiedKFold(5, shuffle=True, random_state=42), the GA-HESO split
  - MLOS is fitted on the training fold only; the test fold is never resampled
  - labels: upstream requires majority = 1 and minority = 0 (README); the
    project's labels are mapped to that and predictions mapped back
  - features are scaled to [-1, 1] with the min / max of the original training
    fold (after oversampling), as GOIO's evaluation does
  - classifiers (scikit-learn defaults): GaussianNB, DecisionTree, SVC,
    RandomForest, KNN(5); N_REPEATS repeats per fold, each with a fresh MLOS
    draw (random / np.random seeded with the repeat) and classifier seed
  - reported per fold: mean over 5 classifiers x N_REPEATS repeats; AUC from
    predict_proba, or decision_function for SVC

Upstream's fallback branches are kept as designed and recorded per fit in
_runs.csv / _run_info.json ('mlos_branch'):
  - 'mlos'                 the method proper
  - 'rank_deficient'       the majority training data has linearly dependent
                           columns (e.g. more features than majority rows):
                           upstream returns the data unchanged, no oversampling
  - 'singular_covariance'  np.linalg.inv raised on the majority covariance:
                           upstream adds minority copies with Gaussian noise
Upstream's _msqrt uses np.linalg.eig, which can return a complex array with
zero imaginary parts; such output is cast to real (recorded). Any real
imaginary part or NaN in MLOS's output stops the run.

Outputs (results_file is an absolute or caller-relative path prefix):
  {results_file}.txt                {'fitness', 'true_values', 'predicted_values'}:
                                    fitness = mean G-mean; predictions =
                                    RandomForest, last repeat
  {results_file}_folds.csv          per fold: mean over classifiers x repeats
  {results_file}_summary.csv        per metric: mean/std across folds
  {results_file}_per_classifier.csv per classifier: mean/std across folds
  {results_file}_runs.csv           every (fold, repeat, classifier) evaluation
  {results_file}_run_info.json      upstream commit, settings, branches, timings
"""

import contextlib
import importlib.util
import io
import json
import os
import random
import sys
import time

os.environ.setdefault('MPLBACKEND', 'Agg')   # MLOS.py imports pyplot at module level

import numpy as np
import pandas as pd
from imblearn.metrics import geometric_mean_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MLOS_PATH = os.path.join(_PROJECT_ROOT, 'ga_heso_sota_methods', 'MLOS_ORIGINAL')
UPSTREAM_COMMIT = '2f0a4b0ccbcbbd308faf6a20e28d67b2d79b6cb4'

N_FOLDS = 5
SPLIT_SEED = 42
N_REPEATS = 10
CLASSIFIERS = [
    ('GaussianNB', lambda seed: GaussianNB()),
    ('DecisionTreeClassifier', lambda seed: DecisionTreeClassifier(random_state=seed)),
    ('SVC', lambda seed: SVC()),
    ('RandomForestClassifier', lambda seed: RandomForestClassifier(random_state=seed)),
    ('KNeighborsClassifier', lambda seed: KNeighborsClassifier(n_neighbors=5)),
]
METRICS = ['accuracy', 'balanced_accuracy', 'macro_f1', 'mcc', 'gmean', 'auc']

_mlos_module = None


def load_mlos():
    """The upstream MLOS.py module, loaded from MLOS_ORIGINAL by file path."""
    global _mlos_module
    if _mlos_module is None:
        path = os.path.join(MLOS_PATH, 'MLOS.py')
        spec = importlib.util.spec_from_file_location('MLOS', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _mlos_module = module
    return _mlos_module


def _encode(data, categorical_cols):
    x_df, y_series = data
    x_df = x_df.copy()
    cats = list(categorical_cols or [])
    for col in x_df.columns:
        if col in cats:
            x_df[col] = pd.factorize(x_df[col].astype(str).str.strip())[0].astype(float)
        else:
            x_df[col] = x_df[col].astype(float)
    y = np.asarray(y_series).astype(int)
    if set(np.unique(y).tolist()) != {0, 1}:
        raise ValueError(f"expected binary 0/1 labels, got {np.unique(y).tolist()}")
    return x_df, y


def _oversample(x_train, y_mlos, seed):
    """One call of upstream MLOS as in its __main__. y_mlos: majority 1, minority 0."""
    mlos = load_mlos()
    random.seed(seed)
    np.random.seed(seed)
    n_maj, n_min = int((y_mlos == 1).sum()), int((y_mlos == 0).sum())
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        sw = mlos.MLOS(len=1)
        x_res, y_res = sw.fit_Sample(x_train.copy(), y_mlos.astype(float), n_maj - n_min)
    printed = out.getvalue()
    if 'linearly dependent' in printed:
        branch = 'rank_deficient'
    elif 'singular' in printed:
        branch = 'singular_covariance'
    else:
        branch = 'mlos'

    x_res, y_res = np.asarray(x_res), np.asarray(y_res)
    cast_complex = False
    if np.iscomplexobj(x_res):
        scale = max(float(np.abs(x_res.real).max()), 1.0)
        if float(np.abs(x_res.imag).max()) > 1e-6 * scale:
            raise RuntimeError(f"MLOS returned complex samples (max |imag| "
                               f"{np.abs(x_res.imag).max():.3g})")
        x_res = x_res.real
        cast_complex = True
    if not np.isfinite(x_res).all():
        raise RuntimeError(f"MLOS returned {int((~np.isfinite(x_res)).sum())} non-finite values "
                           f"(branch {branch})")
    return x_res.astype(float), y_res.astype(int), branch, cast_complex


def _score(model, x):
    classes = np.asarray(model.classes_)
    if hasattr(model, 'predict_proba'):
        return np.asarray(model.predict_proba(x))[:, int(np.flatnonzero(classes == 1)[0])]
    score = np.asarray(model.decision_function(x))   # score of classes_[1]
    return score if classes[1] == 1 else -score


def run_mlos_original(data, dataset_name, results_file, categorical_cols=None,
                      n_repeats=N_REPEATS, folds=None):
    """Runs upstream MLOS on the training part of 5 folds, evaluates the five
    classifiers on the test part, and writes the results.

    data              (X_df, y_series) from a base_functions get_*_data().
    dataset_name      used in messages and _run_info.json.
    results_file      output path prefix, without extension.
    categorical_cols  categorical columns (abalone: ['Sex']), integer-coded;
                      upstream MLOS treats every column as numeric.
    n_repeats         MLOS draws per fold (each evaluated with all 5 classifiers).
    folds             subset of fold indices to run (verification only).
    """
    results_file = os.path.abspath(results_file)
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    start = time.time()
    load_mlos()

    x_df, y = _encode(data, categorical_cols)
    x = x_df.to_numpy(dtype=float)
    minority = int(np.argmin(np.bincount(y)))
    # Upstream: majority = 1, minority = 0.
    to_mlos = (lambda labels: (labels != minority).astype(int))
    from_mlos = (lambda labels: np.where(labels == 0, minority, 1 - minority))
    print(f"=== MLOS (original): {dataset_name} (upstream {UPSTREAM_COMMIT[:7]}) ===")
    print(f"  {len(y)} rows, {x.shape[1]} features, classes {np.bincount(y).tolist()} "
          f"(minority label {minority})", flush=True)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SPLIT_SEED)
    rows, true_values, predicted_values, fold_info = [], [], [], []
    for k, (train_idx, test_idx) in enumerate(skf.split(x, y)):
        if folds is not None and k not in folds:
            continue
        t0 = time.time()
        x_train, x_test = x[train_idx], x[test_idx]
        y_train_m, y_test_m = to_mlos(y[train_idx]), to_mlos(y[test_idx])
        vmin, vmax = x_train.min(0), x_train.max(0)

        def scale(a):
            return 2 * (a - vmin) / (vmax - vmin + 1e-8) - 1

        branches, n_added, last_rf = [], [], None
        for rep in range(n_repeats):
            seed = 1000 * k + rep
            x_res, y_res, branch, cast_complex = _oversample(x_train, y_train_m, seed)
            branches.append(branch + (' (complex cast to real)' if cast_complex else ''))
            n_added.append(int(len(y_res) - len(y_train_m)))
            for name, make in CLASSIFIERS:
                model = make(seed).fit(scale(x_res), y_res)
                xs = scale(x_test)
                # Metrics in the project's labels (minority = positive class 1
                # unless the data say otherwise); G-mean / AUC are symmetric.
                pred = from_mlos(model.predict(xs).astype(int))
                y_true = from_mlos(y_test_m)
                score_mlos_1 = _score(model, xs)             # score of MLOS label 1 (majority)
                score = score_mlos_1 if minority == 0 else -score_mlos_1
                rows.append({
                    'fold': k, 'repeat': rep, 'classifier': name, 'mlos_branch': branch,
                    'n_synthetic': n_added[-1],
                    'accuracy': accuracy_score(y_true, pred),
                    'balanced_accuracy': balanced_accuracy_score(y_true, pred),
                    'macro_f1': f1_score(y_true, pred, average='macro'),
                    'mcc': matthews_corrcoef(y_true, pred),
                    'gmean': geometric_mean_score(y_true, pred),
                    'auc': roc_auc_score(y_true, score) if len(np.unique(y_true)) == 2 else np.nan,
                })
                if name == 'RandomForestClassifier':
                    last_rf = (y_true, pred)
        true_values.append(last_rf[0])
        predicted_values.append(last_rf[1])

        fold_df = pd.DataFrame([r for r in rows if r['fold'] == k])
        branch_counts = pd.Series(branches).value_counts().to_dict()
        print(f"  fold {k + 1}/{N_FOLDS}  train={len(train_idx)} (minority "
              f"{int((y_train_m == 0).sum())}, +{n_added[-1]} synthetic)  gmean="
              f"{fold_df['gmean'].mean():.4f}  auc={np.nanmean(fold_df['auc']):.4f}  "
              f"branch={branch_counts}  ({time.time() - t0:.0f}s)", flush=True)
        fold_info.append({'exp': k, 'n_train': int(len(train_idx)), 'n_test': int(len(test_idx)),
                          'mlos_branches': branch_counts, 'n_synthetic': n_added,
                          'seconds': round(time.time() - t0, 1)})

    runs = pd.DataFrame(rows)
    fold_means = runs.groupby('fold')[METRICS].mean()
    summary = pd.DataFrame({'mean_across_folds': fold_means.mean(axis=0),
                            'std_across_folds': fold_means.std(axis=0)})
    per_classifier = (runs.groupby(['classifier', 'fold'])[METRICS].mean()
                      .groupby(level='classifier').agg(['mean', 'std']))
    per_classifier.columns = [f'{m}_{s}' for m, s in per_classifier.columns]

    runs.to_csv(results_file + '_runs.csv', index=False)
    fold_means.to_csv(results_file + '_folds.csv')
    summary.to_csv(results_file + '_summary.csv')
    per_classifier.to_csv(results_file + '_per_classifier.csv')

    fitness = float(fold_means['gmean'].mean())
    with open(results_file + '.txt', 'w') as f:
        f.write(str({'fitness': fitness, 'true_values': true_values,
                     'predicted_values': predicted_values}))

    all_branches = runs.drop_duplicates(['fold', 'repeat'])['mlos_branch'].value_counts().to_dict()
    info = {
        'method': 'MLOS (upstream MLOS.fit_Sample, len=1, numSamples = n_maj - n_min; '
                  'no hyperparameter search)',
        'upstream': 'https://github.com/ytyancp/MLOS', 'upstream_commit': UPSTREAM_COMMIT,
        'upstream_parameters': {'k': load_mlos().k, 'aer': load_mlos().aer, 'len': 1},
        'dataset': dataset_name, 'n_rows': int(len(y)), 'n_features': int(x.shape[1]),
        'categorical_cols': list(categorical_cols or []), 'minority_label': minority,
        'mlos_branch_counts': all_branches,
        'evaluation': {'classifiers': [n for n, _ in CLASSIFIERS], 'repeats_per_fold': n_repeats,
                       'scaling': '[-1, 1] min/max of the original training fold',
                       'split': f'StratifiedKFold({N_FOLDS}, shuffle=True, '
                                f'random_state={SPLIT_SEED})',
                       'note': 'upstream defines no evaluation protocol; same for all datasets'},
        'folds_run': sorted(folds) if folds is not None else list(range(N_FOLDS)),
        'reported': 'mean over GaussianNB/DecisionTree/SVC/RandomForest/KNN x repeats per fold; '
                    'AUC from predict_proba, SVC decision_function',
        'folds': fold_info, 'total_seconds': round(time.time() - start, 1),
    }
    with open(results_file + '_run_info.json', 'w') as f:
        json.dump(info, f, indent=2)

    print(f"  MLOS branches over all fits: {all_branches}")
    print(f"=== {dataset_name}: across folds (mean over 5 classifiers x {n_repeats} repeats) ===")
    print(summary.round(4))
    print(f"evaluated fitness: {fitness:.6f}")
    print(f"--- {dataset_name}: {time.time() - start:.0f} seconds ---", flush=True)
    return {'fitness': fitness, 'true_values': true_values, 'predicted_values': predicted_values}


if __name__ == '__main__':
    sys.exit("import run_mlos_original from this module; see predictions/*/prediction_*_mlos_original.py")
