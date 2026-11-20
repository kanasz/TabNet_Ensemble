"""Why does the retrained G-mean differ from the one the GA reported?

For SMOTE+XGBoost and SMOTE+TabNet this calls the tuner's OWN eval_func (the GA
fitness code, unpickled from the checkpoint) on best_solutions[-1] in today's
environment, plus variants that undo known library changes:
  * XGBoost >= 2.0 changed the default tree_method to 'hist' (was 'exact' on CPU)
  * TabNet on CPU vs GPU

    python results_analysis/time_complexity/diagnose_gmean.py [--datasets yeast_4 abalone19]

Interpretation:
  eval_func == retrained (JSON 'gmean')  -> measure_one_run.py reproduces the GA code;
                                           the gap comes from the environment
  some variant == GA value               -> that is the library/device that changed
"""
import argparse
import functools
import importlib
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import timing_config as cfg  # noqa: E402
sys.path.insert(0, cfg.REPO_ROOT)
os.chdir(cfg.REPO_ROOT)
warnings.filterwarnings('ignore')
from measure_one_run import _load_checkpoint  # noqa: E402


def _txt_fitness(stem):
    try:
        with open(stem + '.txt') as f:
            head = f.read(200)
        return float(head.split("'fitness':")[1].split(',')[0])
    except Exception:
        return None


def _gmean_of(tuner, ga, sol):
    return float(tuner.eval_func(ga, sol, None)[0])


def diagnose(method, dataset):
    stem, ga, tuner, sol = _load_checkpoint(method, dataset)
    print('\n=== {} / {}'.format(method, dataset))
    print('  best_solutions[-1] raw : {}'.format(list(ga.best_solutions[-1])))
    print('  typed solution         : {}'.format(sol))
    print('  GA best_solutions_fitness[-1] {:.6f}   max {:.6f}   on_stop .txt {}'.format(
        ga.best_solutions_fitness[-1], np.max(ga.best_solutions_fitness), _txt_fitness(stem)))
    print('  eval_func now (default)       {:.6f}'.format(_gmean_of(tuner, ga, sol)))

    if method == 'smote_xgboost':
        mod = importlib.import_module(type(tuner).__module__)
        orig = mod.XGBClassifier
        for tm in ('exact', 'approx', 'hist'):
            mod.XGBClassifier = functools.partial(orig, tree_method=tm)
            try:
                print('  eval_func tree_method={:7s} {:.6f}'.format(tm, _gmean_of(tuner, ga, sol)))
            except Exception as e:
                print('  eval_func tree_method={:7s} failed: {}'.format(tm, e))
            finally:
                mod.XGBClassifier = orig
    elif method == 'smote_tabnet':
        import torch
        mod = importlib.import_module(type(tuner).__module__)
        orig = mod.get_classifier

        def on_cpu(*a, **k):
            clf = orig(*a, **k)
            clf.device_name = 'cpu'
            clf.device = torch.device('cpu')
            return clf
        mod.get_classifier = on_cpu
        try:
            print('  eval_func TabNet on CPU       {:.6f}'.format(_gmean_of(tuner, ga, sol)))
        finally:
            mod.get_classifier = orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--methods', nargs='+', default=['smote_xgboost', 'smote_tabnet'])
    ap.add_argument('--datasets', nargs='+', default=cfg.DEFAULT_DATASETS)
    args = ap.parse_args()
    for mod in ('xgboost', 'torch', 'pytorch_tabnet', 'sklearn', 'imblearn', 'pygad', 'numpy'):
        try:
            print('{:15s} {}'.format(mod, importlib.import_module(mod).__version__))
        except Exception as e:
            print('{:15s} n/a ({})'.format(mod, e))
    for m in args.methods:
        for d in args.datasets:
            try:
                diagnose(m, d)
            except Exception as e:
                print('\n=== {} / {} failed: {}: {}'.format(m, d, type(e).__name__, e))


if __name__ == '__main__':
    main()
