"""
Fixed-baseline (no GA) runner for CCO, run the way the authors designed it.

CCO ("A Cluster Core-based Oversampling Technique for Improved
Class-Imbalanced Learning", https://github.com/priyomondal/CCO) is driven
through its own main.py::main(), from an untouched vendored copy in
ga_heso_sota_methods/CCO_ORIGINAL (upstream commit UPSTREAM_COMMIT). Nothing
in that directory is edited. Unlike optimization/no_ga_cco_runner.py, which
re-implements CCO's training loop, this runner executes upstream's whole
pipeline:

  load_data()   torch.load(PATH/X, PATH/Y), MinMax over the full dataset,
                StratifiedKFold(5, shuffle=True, random_state=state)
  per split     scaling() (MinMax fit on the train split), Cluster(),
                synthetic_generation(), Net + FocalLoss(gamma=1) + Adam(1e-3),
                model_train(): `epochs` epochs, keeping the metrics of the
                epoch with the best TEST balanced accuracy
  output        PATH/TEST.txt: bacc, mcc, weighted F1, gmean, best epoch

This is the README invocation
    python3 main.py --k K --D D --beta BETA --t T --batch_size B
                    --num_workers W --epochs E --state S --split_no 5 --PATH P
reproduced in a worker subprocess (`set_seeds(state, True)` then `main(...)`,
exactly what main.py's __main__ block does), with two additions:

  1. Network width. Upstream's Net is Linear(34, 20) -> (20, 12) -> (12, 6),
     hardwired to the 34-feature, 6-class dermatology set it ships with, so
     main.py cannot run on any other dataset as-is. _SizedNet keeps the hidden
     layers and activations and only sets the input width to the number of
     features and the output width to the number of classes. It replaces the
     `Net` name main.py uses; no file is changed.
  2. Recording. Upstream keeps only the four metrics of the selected epoch and
     discards the predictions. models.test() is wrapped so that, whenever an
     epoch beats the best test balanced accuracy so far (the same strict `>`
     rule model_train uses), the wrapper also stores that epoch's predictions
     and class-1 softmax probabilities. They come from a no_grad forward pass
     over the test split in its original order, so no random state that
     training depends on is consumed. The recorded epoch and metrics are
     checked against upstream's TEST.txt for every split; a mismatch fails
     loudly.

Properties of the published protocol that are kept as designed and worth
knowing when using CCO as a baseline:
  - the reported epoch is the one with the best balanced accuracy on the TEST
    split, so the scores are optimistic;
  - load_data() MinMax-scales the whole dataset before splitting (then each
    split is rescaled on its training part).
optimization/no_ga_cco_runner.py offers leakage-free switches for both; this
runner deliberately does not.

Outputs (results_file is an absolute or caller-relative path prefix):
  {results_file}.txt                {'fitness', 'true_values', 'predicted_values'}:
                                    fitness = mean of upstream's per-split
                                    G-mean; predictions = the selected epoch's
                                    test predictions per split
  {results_file}_folds.csv          per split: bacc, mcc, f1 (weighted), gmean
                                    (all upstream's own values), auc (from the
                                    recorded probabilities), epoch_best
  {results_file}_summary.csv        per metric: mean/std across splits
  {results_file}_upstream_TEST.txt  upstream's own TEST.txt, unchanged
  {results_file}_run_info.json      upstream commit, settings, timings
"""

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CCO_PATH = os.path.join(_PROJECT_ROOT, 'ga_heso_sota_methods', 'CCO_ORIGINAL')
UPSTREAM_COMMIT = '8064bca3b20a18d971be84ce44973d8546595d42'

# The README gives no default values. These are the ones the project's existing
# CCO baseline (predictions/*/prediction_*_cco_baseline.py) uses, kept the
# same so the two CCO runs are comparable.
DEFAULT_K = 0.3
DEFAULT_BETA = 0.5
DEFAULT_T = 0.5
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 32
# --state seeds everything and is StratifiedKFold's random_state; 42 gives the
# same 5 folds GA-HESO uses (same data order, same splitter).
DEFAULT_STATE = 42
SPLIT_NO = 5
NUM_WORKERS = 0  # parsed by main.py but unused (commented out upstream)

_UPSTREAM_EXAMPLE_DATASETS = {'dermatology'}
_RECORDS_FILE = '_runner_records.pkl'


# --------------------------------------------------------------------------
# worker side: runs inside the subprocess, cwd = CCO_PATH
# --------------------------------------------------------------------------

class _SizedNet(nn.Module):
    """Upstream utils.Net with the two outer widths following the dataset.
    Hidden sizes 20 and 12 and the ReLU placement are upstream's. main.py
    calls Net() without arguments, so the widths are class attributes set by
    the worker before main() runs."""
    n_in = 34
    n_out = 6

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(self.n_in, 20)
        self.fc2 = nn.Linear(20, 12)
        self.fc3 = nn.Linear(12, self.n_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def _assert_from_cco(module):
    path = os.path.abspath(getattr(module, '__file__', '') or '')
    if os.path.dirname(path) != os.path.abspath(CCO_PATH):
        raise ImportError(f"'{module.__name__}' was imported from {path or '<unknown>'}, "
                          f"not from {CCO_PATH}")


def _worker(cfg):
    # Upstream imports its modules as top-level `utils`, `models` and `main`.
    # The project root also has a main.py and a models/ folder, so CCO's
    # directory must come first.
    sys.path.insert(0, CCO_PATH)
    import utils as cco_utils
    import models as cco_models
    import main as cco_main
    for module in (cco_utils, cco_models, cco_main):
        _assert_from_cco(module)

    _SizedNet.n_in = cfg['n_features']
    _SizedNet.n_out = cfg['n_classes']
    cco_main.Net = _SizedNet
    cco_utils.Net = _SizedNet

    records = []
    split_state = {}
    upstream_test = cco_models.test
    upstream_model_train = cco_main.model_train

    def recording_test(ep, model, test_loader, device=None):
        result = upstream_test(ep, model, test_loader, device)
        balanced_test = result[0]
        if balanced_test > split_state['best_bacc']:  # model_train's own rule
            rows = test_loader.dataset.data           # original split order
            with torch.no_grad():
                x = torch.stack([r[0] for r in rows]).float()
                out = model(x)
            split_state['best_bacc'] = balanced_test
            split_state['best'] = {
                'epoch': ep,
                'bacc': float(result[0]), 'mcc': float(result[1]),
                'f1': float(result[2]), 'gmean': float(result[3]),
                'y_true': torch.stack([r[1] for r in rows]).cpu().numpy().astype(int),
                'y_pred': out.argmax(dim=1).cpu().numpy().astype(int),
                'y_score': torch.softmax(out, dim=1)[:, 1].cpu().numpy(),
            }
        return result

    def recording_model_train(*args, **kwargs):
        split_state.clear()
        split_state.update(best_bacc=0, best=None)
        output = upstream_model_train(*args, **kwargs)
        records.append(split_state['best'])
        return output

    cco_models.test = recording_test
    cco_main.model_train = recording_model_train

    # Exactly main.py's __main__ block.
    cco_utils.set_seeds(cfg['state'], True)
    cco_main.main(cfg['PATH'], cfg['state'], cfg['k'], cfg['D'], cfg['t'], cfg['beta'],
                  cfg['split_no'], cfg['batch_size'], cfg['num_workers'], cfg['epochs'])

    with open(os.path.join(cfg['PATH'], _RECORDS_FILE), 'wb') as f:
        pickle.dump(records, f)


# --------------------------------------------------------------------------
# caller side
# --------------------------------------------------------------------------

def _encode(data, categorical_cols):
    """Numeric float features (categoricals label-encoded, as for the other
    baselines) and int labels 0..C-1."""
    x_df, y_series = data
    x_df = x_df.copy()
    for col in x_df.columns:
        if col in (categorical_cols or []):
            x_df[col] = pd.factorize(x_df[col].astype(str).str.strip())[0].astype(float)
        else:
            x_df[col] = x_df[col].astype(float)
    y = np.asarray(y_series).astype(int)
    classes = np.unique(y)
    if not np.array_equal(classes, np.arange(len(classes))):
        raise ValueError(f"labels must be 0..C-1, got {classes.tolist()}")
    return x_df.to_numpy(dtype=np.float32), y


def _read_upstream_test_txt(path):
    """The 'ours,bacc,mcc,f1,gmean,epoch' lines main.py appends per split."""
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split(',')
            if parts[0] == 'ours':
                bacc, mcc, f1, gmean, epoch = parts[1:6]
                rows.append({'bacc': float(bacc), 'mcc': float(mcc), 'f1': float(f1),
                             'gmean': float(gmean), 'epoch_best': int(epoch)})
    return rows


def run_cco_original(data, dataset_name, results_file, categorical_cols=None,
                     k=DEFAULT_K, beta=DEFAULT_BETA, t=DEFAULT_T, epochs=DEFAULT_EPOCHS,
                     batch_size=DEFAULT_BATCH_SIZE, state=DEFAULT_STATE, D=None):
    """Runs upstream CCO's main.py pipeline on one dataset and writes the results.

    data              (X_df, y_series) from a base_functions get_*_data().
    dataset_name      folder name used inside CCO_ORIGINAL/data.
    results_file      output path prefix, without extension.
    categorical_cols  names of true categorical columns (abalone: ['Sex']).
    k, beta, t, epochs, batch_size, state
                      main.py's CLI arguments (README).
    D                 main.py's --D, the feature dimensionality in CCO's
                      density term; defaults to the number of features.
    """
    if dataset_name in _UPSTREAM_EXAMPLE_DATASETS:
        raise ValueError(f"'{dataset_name}' is upstream's bundled example; pick another name")

    results_file = os.path.abspath(results_file)
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    start = time.time()

    x, y = _encode(data, categorical_cols)
    n_features, n_classes = x.shape[1], int(y.max()) + 1
    D = float(n_features if D is None else D)

    # Fresh folder every run: main.py APPENDS to TEST.txt, so a leftover file
    # would mix in an earlier run's splits.
    data_dir = os.path.join(CCO_PATH, 'data', dataset_name)
    if os.path.isdir(data_dir):
        shutil.rmtree(data_dir)
    os.makedirs(data_dir)
    # Same on-disk format as upstream's data/dermatology/{X,Y}.
    torch.save(torch.tensor(x), os.path.join(data_dir, 'X'))
    torch.save(torch.tensor(y, dtype=torch.int64), os.path.join(data_dir, 'Y'))

    cfg = {
        'PATH': f'data/{dataset_name}/',  # main.py does PATH + "X", PATH + "TEST.txt"
        'state': state, 'k': k, 'D': D, 't': t, 'beta': beta, 'split_no': SPLIT_NO,
        'batch_size': batch_size, 'num_workers': NUM_WORKERS, 'epochs': epochs,
        'n_features': n_features, 'n_classes': n_classes,
    }
    print(f"=== CCO (original pipeline): {dataset_name} (upstream {UPSTREAM_COMMIT[:7]}) ===")
    print(f"  {len(y)} rows, {n_features} features, classes {np.bincount(y).tolist()}")
    print(f"  main.py --k {k} --D {D} --beta {beta} --t {t} --batch_size {batch_size} "
          f"--num_workers {NUM_WORKERS} --epochs {epochs} --state {state} "
          f"--split_no {SPLIT_NO} --PATH {cfg['PATH']}", flush=True)

    env = dict(os.environ, PYTHONUNBUFFERED='1')
    subprocess.run([sys.executable, '-u', os.path.abspath(__file__), '--worker', json.dumps(cfg)],
                   cwd=CCO_PATH, env=env, check=True)

    upstream_rows = _read_upstream_test_txt(os.path.join(data_dir, 'TEST.txt'))
    with open(os.path.join(data_dir, _RECORDS_FILE), 'rb') as f:
        records = pickle.load(f)
    if len(upstream_rows) != SPLIT_NO or len(records) != SPLIT_NO:
        raise RuntimeError(f"expected {SPLIT_NO} splits, got {len(upstream_rows)} in TEST.txt "
                           f"and {len(records)} recorded")

    folds, true_values, predicted_values = [], [], []
    for i, (up, rec) in enumerate(zip(upstream_rows, records)):
        if rec is None:
            raise RuntimeError(f"split {i}: no epoch beat balanced accuracy 0, nothing recorded")
        # The recorded epoch must be the one upstream reported, and its
        # predictions must reproduce upstream's numbers.
        ours_bacc = balanced_accuracy_score(rec['y_true'], rec['y_pred'])
        ours_gmean = geometric_mean_score(rec['y_true'].tolist(), rec['y_pred'].tolist())
        if (rec['epoch'] != up['epoch_best'] or not np.isclose(ours_bacc, up['bacc'])
                or not np.isclose(ours_gmean, up['gmean'])):
            raise RuntimeError(
                f"split {i}: recorded epoch {rec['epoch']} (bacc {ours_bacc:.6f}, gmean "
                f"{ours_gmean:.6f}) does not match TEST.txt epoch {up['epoch_best']} "
                f"(bacc {up['bacc']:.6f}, gmean {up['gmean']:.6f})")
        auc = np.nan
        if len(np.unique(rec['y_true'])) == 2 and n_classes == 2:
            auc = roc_auc_score(rec['y_true'], rec['y_score'])
        folds.append({**up, 'auc': auc})
        true_values.append(rec['y_true'])
        predicted_values.append(rec['y_pred'])
        print(f"  split {i + 1}/{SPLIT_NO}  gmean={up['gmean']:.4f}  auc={auc:.4f}  "
              f"bacc={up['bacc']:.4f}  (epoch {up['epoch_best']})")

    fold_df = pd.DataFrame(folds, index=pd.Index(range(SPLIT_NO), name='fold'))
    metrics = ['bacc', 'mcc', 'f1', 'gmean', 'auc']
    summary = pd.DataFrame({
        'mean_across_folds': fold_df[metrics].mean(axis=0),
        'std_across_folds': fold_df[metrics].std(axis=0),
    })
    fold_df.to_csv(results_file + '_folds.csv')
    summary.to_csv(results_file + '_summary.csv')
    shutil.copyfile(os.path.join(data_dir, 'TEST.txt'), results_file + '_upstream_TEST.txt')

    fitness = float(fold_df['gmean'].mean())
    with open(results_file + '.txt', 'w') as f:
        f.write(str({'fitness': fitness, 'true_values': true_values,
                     'predicted_values': predicted_values}))

    info = {
        'method': "CCO (upstream main.py pipeline, no hyperparameter search)",
        'upstream': 'https://github.com/priyomondal/CCO', 'upstream_commit': UPSTREAM_COMMIT,
        'dataset': dataset_name, 'n_rows': int(len(y)), 'n_features': int(n_features),
        'n_classes': int(n_classes), 'categorical_cols': categorical_cols or [],
        'main_py_args': {key: cfg[key] for key in ('k', 'D', 'beta', 't', 'batch_size',
                                                   'num_workers', 'epochs', 'state', 'split_no')},
        'adaptations': [f'Net input width {n_features} (upstream 34), output width '
                        f'{n_classes} (upstream 6); hidden layers unchanged'],
        'protocol_notes': ['epoch selected by best TEST balanced accuracy (upstream)',
                           'MinMax over the full dataset before splitting (upstream)'],
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'total_seconds': round(time.time() - start, 1),
    }
    with open(results_file + '_run_info.json', 'w') as f:
        json.dump(info, f, indent=2)

    print(summary.round(4))
    print(f"evaluated fitness: {fitness:.6f}")
    print(f"--- {dataset_name}: {time.time() - start:.0f} seconds ---", flush=True)
    return {'fitness': fitness, 'true_values': true_values, 'predicted_values': predicted_values}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', required=True, help='JSON config; internal use')
    _worker(json.loads(parser.parse_args().worker))
