"""Recalculate per-fold probabilities for finished GA-HESO runs WITHOUT re-running the GA.

A GA-HESO checkpoint (<name>.pkl, written by pygad's ga_instance.save) does not hold
trained models - it holds the GA state: the best chromosome of every generation and,
through the pickled fitness function, the tuner that produced it (data, fold indices,
config-file order, resampler, clustering settings, epochs).

So the probabilities are recovered by training the best chromosome ONCE (5 folds) and
calling predict_proba on every ensemble member - seconds/minutes instead of the
population x generations trainings of the full GA.

Everything that defines the experiment is taken from the checkpoint, not rebuilt:
in particular the order of config_files, which decides what the classifier-mask genes
mean and differs between machines (os.listdir order).

Usage (from the repo root, in the env that produced the checkpoints):
    python recalculate_ga_heso_probabilities.py "predictions/ecoli/results/*OC_TABNET*ecoli_0_vs_1.pkl"
    python recalculate_ga_heso_probabilities.py <a.pkl> <b.pkl>

A CUDA device is required (the loss functions are CUDA-only).

Output, next to each checkpoint:
    <name>_proba.csv   one row per test sample:
        fold, row            fold number and row position in the dataset
        y_true
        y_pred               ensemble majority vote (what the GA scored / the .txt stores)
        vote_fraction        share of members voting 1; y_pred == round(vote_fraction)
        proba_mean           mean of the members' P(class 1) (soft vote)
        stored_pred          prediction from <name>.txt, if that file is readable
"""
import argparse
import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch
from imblearn.metrics import geometric_mean_score
from pygad import pygad
from sklearn.cluster import MeanShift
from sklearn.metrics import roc_auc_score

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from base_functions import get_preprocessor  # noqa: E402
from models.oc_bagging_tabnet_ensemble import OCBaggingTabnetEnsemble  # noqa: E402
from models.oc_bagging_tabnet_ensemble_parallel import _best_overall_solution  # noqa: E402


def _resolve_config(path):
    """Checkpoints store config paths relative to wherever the run started
    ('../../models/configurations\\X.json', 'models/configurations/X.json')."""
    name = path.replace('\\', '/').split('/')[-1]
    return os.path.join(REPO_ROOT, 'models', 'configurations', name)


def _prepare_tuner(tuner, device):
    """Fill in what older checkpoints lack so the current parallel_fit can run them."""
    if device is None:
        device = tuner.device
    if device != 'cuda' or not torch.cuda.is_available():
        # Every class in loss_functions/ builds torch.cuda.FloatTensor, and get_loss swallows
        # the resulting error and returns None - TabNet then trains with its default
        # cross-entropy. A CPU run therefore finishes without complaint but with the WRONG
        # losses, so refuse it instead of writing misleading probabilities.
        raise RuntimeError('a CUDA device is required: the loss functions are CUDA-only and '
                           'silently fall back to plain cross-entropy on CPU')
    tuner.device = device
    tuner.config_files = [_resolve_config(p) for p in tuner.config_files]

    cp = getattr(tuner, 'clustering_params', None)
    if cp is None:
        raise NotImplementedError('checkpoint has no clustering_params (old KMeans layout) - '
                                  'the current parallel_fit cannot evaluate it')
    # Before 2025-01-13 there was no use_cluster_centers switch: centers were always used.
    if not hasattr(tuner, 'use_cluster_centers'):
        tuner.use_cluster_centers = True
    if 'algs' not in cp:
        if cp.get('type') == 'MS' and 'bandwidths' in cp:
            # what the tuner did before the fitted estimators were passed in
            cp['algs'] = [MeanShift(bandwidth=b) for b in cp['bandwidths']]
        else:
            raise NotImplementedError("checkpoint has no clustering_params['algs'] for type {}"
                                      .format(cp.get('type')))
    return tuner


def _read_stored_predictions(txt_path):
    if not os.path.exists(txt_path):
        return None
    try:
        with open(txt_path) as f:
            result = eval(f.read(), {'__builtins__': {}}, {'array': np.array, 'int64': np.int64,
                                                           'int32': np.int32, 'nan': np.nan})
        return [np.asarray(p) for p in result['predicted_values']], float(result['fitness'])
    except Exception as e:
        print('  ! could not parse {}: {}'.format(txt_path, e))
        return None


def recalculate(pkl_path, device=None):
    stem = pkl_path[:-4] if pkl_path.endswith('.pkl') else pkl_path
    print('=' * 100)
    print(stem)
    ga = pygad.load(stem)
    tuner = _prepare_tuner(ga.fitness_func.__self__, device)

    fitness_history = np.asarray(ga.best_solutions_fitness, dtype=float)
    solution = np.asarray(_best_overall_solution(ga))
    n_cfg = len(tuner.config_files)
    members = [c for c, on in zip(tuner.config_files, solution[:n_cfg]) if on]
    print('  generations: {} | GA best fitness: {:.10f} (generation {}) | ensemble members: {}'.format(
        ga.generations_completed, fitness_history.max(), int(fitness_history.argmax()), len(members)))

    X, y = tuner.X_orig, tuner.y_orig
    rows = []
    for fold, (train_index, test_index) in enumerate(zip(tuner.train_indices, tuner.test_indices)):
        ensemble = OCBaggingTabnetEnsemble(members, solution[n_cfg:], tuner.device)
        # Same code path the GA used; called in-process so the fitted members stay in `ensemble`.
        y_valid, y_pred, _ = tuner.parallel_fit(fold, train_index, test_index, X, y, solution,
                                                ensemble, tuner.tabnet_max_epochs)

        preprocessor = get_preprocessor(tuner.numerical_cols, tuner.categorical_cols)
        preprocessor.fit(X.iloc[train_index])
        X_valid_std = preprocessor.transform(X.iloc[test_index])

        member_proba = np.column_stack([m.predict_proba(X_valid_std)[:, 1] for m in ensemble.models])
        member_votes = np.column_stack([m.predict(X_valid_std) for m in ensemble.models])
        vote_fraction = member_votes.mean(axis=1)
        assert np.array_equal(np.round(vote_fraction).astype(int), y_pred)

        rows.append(pd.DataFrame({
            'fold': fold,
            'row': test_index,
            'y_true': np.asarray(y_valid).astype(int),
            'y_pred': y_pred,
            'vote_fraction': vote_fraction,
            'proba_mean': member_proba.mean(axis=1),
        }))
    out = pd.concat(rows, ignore_index=True)

    stored = _read_stored_predictions(stem + '.txt')
    if stored is not None:
        stored_pred, stored_fitness = stored
        if [len(p) for p in stored_pred] == [len(r) for r in rows]:
            out['stored_pred'] = np.concatenate(stored_pred).astype(int)
        else:
            print('  ! stored predictions have different fold sizes - not compared')

    per_fold = out.groupby('fold')
    gmeans = per_fold.apply(lambda d: geometric_mean_score(d.y_true, d.y_pred))
    print('  recalculated G-mean: {:.10f}   per fold: {}'.format(gmeans.mean(), np.round(gmeans.values, 4)))
    for col in ('vote_fraction', 'proba_mean'):
        aucs = per_fold.apply(lambda d: roc_auc_score(d.y_true, d[col]))
        print('  AUC from {:14s}: {:.4f}   per fold: {}'.format(col, aucs.mean(), np.round(aucs.values, 4)))
    if 'stored_pred' in out:
        differ = int((out.stored_pred != out.y_pred).sum())
        print('  stored .txt fitness: {:.10f} | predictions differing from stored .txt: {} of {}'.format(
            stored_fitness, differ, len(out)))

    out.to_csv(stem + '_proba.csv', index=False)
    print('  -> {}_proba.csv'.format(stem))
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('checkpoints', nargs='+', help='GA-HESO .pkl files (globs allowed)')
    parser.add_argument('--device', default=None, help="override the checkpoint's device (cuda/cpu)")
    args = parser.parse_args()
    warnings.filterwarnings('ignore')

    paths = []
    for pattern in args.checkpoints:
        paths.extend(sorted(glob.glob(pattern)) or [pattern])
    failed = []
    for path in paths:
        try:
            recalculate(path, args.device)
        except BaseException as e:  # pygad.load raises BaseException
            if isinstance(e, KeyboardInterrupt):
                raise
            print('  FAILED: {}: {}'.format(type(e).__name__, e))
            failed.append(path)
    if failed:
        print('\nFailed checkpoints:\n  ' + '\n  '.join(failed))
        sys.exit(1)
