"""
Fixed-baseline (no GA) runner for GOIO, run the way the authors designed it.

GOIO ("GOIO: Generative Oversampling Approach to Class Imbalance and Overlap
of Tabular Data", https://github.com/MLneuers/GOIO) is driven through its own
main.py from an untouched vendored copy in ga_heso_sota_methods/GOIO_ORIGINAL
(upstream commit UPSTREAM_COMMIT). Nothing in that directory is edited.

Per dataset this runs exactly the README's commands, with all of upstream's
default arguments:

    python main.py --dataname <name> --method data  --mode split
  and per fold k = 0..4:
    python main.py --dataname <name> --method MLVAE --mode train  --exp k
    python main.py --dataname <name> --method CLDM  --mode train  --exp k
    python main.py --dataname <name> --method CLDM  --mode sample --exp k
    python main.py --dataname <name> --method CLDM  --mode eval   --exp k

  split   data/datasets/<name>/<name>.csv -> StratifiedKFold(5, shuffle=True,
          random_state=42) folds (the same split GA-HESO uses; the runner
          checks every fold's test rows against the project's split) plus
          column-type detection (data_description) and the GOIO arrays
  MLVAE   5000 epochs, checkpoint with the lowest reconstruction loss on the
          TEST fold kept
  CLDM    latent diffusion, up to 10001 epochs, early stopping on train loss
  sample  conditional sampling of every class up to 1.1 x the majority count
  eval    the minority (and majority) training data are topped up with
          synthetic rows until both classes have the original majority count;
          GaussianNB, DecisionTree, SVC, RandomForest and KNN are trained 10
          times each and scored on the TEST fold (upstream writes accuracy /
          macro-F1 / MCC to synthetic/<name>/exp<k>/CLDM/<name>.csv)

Inputs. The csv is written in upstream's format (all numeric, label last,
header row). Categorical columns (abalone's 'Sex') are integer-coded 0, 1, 2
like upstream's bundled abalone_15, so upstream's own data_description()
detects them as categorical. Upstream assumes the majority class has label 0
(eval() draws `majority count - minority count` synthetic rows per class,
which is negative otherwise); where label 1 is the majority (ecoli_0_vs_1)
the labels are swapped for GOIO and back for reporting. StratifiedKFold
encodes classes by first appearance, so the folds do not change; every metric
reported is invariant to the swap.

Metrics. Upstream logs accuracy / macro-F1 / MCC only. The five classifiers
of CLDM/evaluation.py are replaced, in the worker, by subclasses that record
what predict() returns plus a continuous score (predict_proba, or
decision_function for SVC, which upstream builds without probability=True)
for AUC; fitting and predicting are untouched and the extra scoring call uses
no random numbers. The recorded accuracy / macro-F1 / MCC of all 50 runs are
checked against upstream's own csv. Reported per fold: mean over 5
classifiers x 10 repeats, as upstream's eval reports it.

Compatibility shims (each only when needed, listed in _run_info.json):
  - libzero missing: src/ imports `zero` (which pins torch < 2) only for
    helpers GOIO never calls; an empty stub module stands in.
  - no CUDA: MLVAE/main.py creates tensors on 'cuda:0' unconditionally and
    CLDM/diffusion_utils.sample() defaults to device='cuda:0'; without a GPU
    both are sent to the CPU (local testing only - the HPC has GPUs).
  - pandas >= 3: CLDM/evaluation.py uses DataFrame._append (removed).

Protocol properties kept as designed (worth stating when using GOIO as a
baseline): MLVAE keeps the checkpoint with the lowest reconstruction loss on
the TEST fold, and upstream's split normalises its TEST/ copy with the
min/max of the whole dataset (that copy is not used by eval, which scales by
the training fold).

Outputs (results_file is an absolute or caller-relative path prefix):
  {results_file}.txt                {'fitness', 'true_values', 'predicted_values'}:
                                    fitness = mean G-mean; predictions =
                                    RandomForest, last repeat
  {results_file}_folds.csv          per fold: mean over classifiers x repeats
  {results_file}_summary.csv        per metric: mean/std across folds
  {results_file}_per_classifier.csv per classifier: mean/std across folds
  {results_file}_runs.csv           every (fold, classifier, repeat) evaluation
  {results_file}_run_info.json      upstream commit, settings, shims, timings
Upstream's stdout of every step goes to synthetic/<name>/exp<k>/_upstream_*.log
(MLVAE prints one line per epoch, CLDM one progress bar per epoch). Finished
folds are cached there and skipped on a rerun with the same data and settings
(resume=False to recompute).
"""

import argparse
import builtins
import hashlib
import json
import os
import pickle
import runpy
import shutil
import subprocess
import sys
import time
import types

import numpy as np
import pandas as pd
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, roc_auc_score)
from sklearn.model_selection import StratifiedKFold

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOIO_PATH = os.path.join(_PROJECT_ROOT, 'ga_heso_sota_methods', 'GOIO_ORIGINAL')
UPSTREAM_COMMIT = '4551ca6164f86c3f7c8964fe935713c5ea6862cf'

N_FOLDS = 5
SPLIT_SEED = 42                     # upstream's and GA-HESO's StratifiedKFold seed
CKPT_DIR = 'ckpt_1_1_1'             # ckpt_{kld}_{proto}_{dist} with upstream defaults
UPSTREAM_EPOCHS = (5000, 10001)     # MLVAE, CLDM num_epochs
N_CLASSIFIERS, N_REPEATS = 5, 10    # CLDM/evaluation.py
FOLD_STEPS = [('MLVAE', 'train'), ('CLDM', 'train'), ('CLDM', 'sample'), ('CLDM', 'eval')]
_UPSTREAM_DATASETS = {'abalone_15'}
_RECORDS_FILE = '_runner_records.pkl'
_DONE_FILE = '_runner_done.json'
METRICS = ['accuracy', 'balanced_accuracy', 'macro_f1', 'mcc', 'gmean', 'auc']


# --------------------------------------------------------------------------
# worker side: runs in a subprocess with cwd = GOIO_PATH
# --------------------------------------------------------------------------

def _version_tuple(version):
    parts = []
    for piece in version.split('.')[:2]:
        digits = ''.join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def _assert_from_goio(module):
    path = os.path.abspath(getattr(module, '__file__', '') or '')
    if not path.startswith(os.path.abspath(GOIO_PATH) + os.sep):
        raise ImportError(f"'{module.__name__}' was imported from {path or '<unknown>'}, "
                          f"not from {GOIO_PATH}")


def _cpu_torch_proxy(torch):
    """torch with device='cuda:0' arguments sent to the CPU (no-GPU testing)."""
    proxy = types.ModuleType('torch_cpu_compat')
    proxy.__dict__.update({k: getattr(torch, k) for k in dir(torch) if not k.startswith('__')})

    def on_cpu(fn):
        def wrapped(*args, **kwargs):
            if str(kwargs.get('device', '')).startswith('cuda'):
                kwargs['device'] = 'cpu'
            return fn(*args, **kwargs)
        return wrapped
    for name in ('tensor', 'zeros'):
        setattr(proxy, name, on_cpu(getattr(torch, name)))
    return proxy


def _epoch_capped_range(cap):
    """range() that shortens only upstream's two training loops (smoke tests)."""
    def capped(*args):
        if len(args) == 1 and args[0] in UPSTREAM_EPOCHS:
            return builtins.range(min(cap, args[0]))
        return builtins.range(*args)
    return capped


def _recording_class(cls, records):
    """Subclass that keeps what upstream's eval gets from predict(), plus a
    continuous score for AUC. Constructor, fit and predict are unchanged."""
    name = cls.__name__

    class Recording(cls):
        def predict(self, X):
            out = super().predict(X)
            classes = np.asarray(self.classes_)
            if hasattr(self, 'predict_proba'):
                score = np.asarray(super().predict_proba(X))[:, int(np.flatnonzero(classes == 1)[0])]
            else:  # SVC(probability=False): decision_function is the score of classes_[1]
                score = np.asarray(super().decision_function(X))
                if classes[1] != 1:
                    score = -score
            records.append({'classifier': name, 'pred': np.asarray(out), 'score': score})
            return out

    Recording.__name__ = name
    Recording.__qualname__ = cls.__qualname__
    return Recording


def _worker(cfg):
    # Upstream imports utils, utils_train, src, data, MLVAE, CLDM as top-level
    # modules; the project root (on PYTHONPATH on the HPC) has a data/ folder
    # too, so it is taken off the path and GOIO's directory put first.
    blocked = {os.path.abspath(_PROJECT_ROOT), os.path.dirname(os.path.abspath(__file__))}
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.curdir) not in blocked]
    sys.path.insert(0, GOIO_PATH)

    import torch
    applied = []
    try:
        import zero  # noqa: F401
    except ImportError:
        # libzero pins torch<2 (installing it downgrades the env's torch);
        # src/ imports it only for helpers GOIO never calls.
        sys.modules['zero'] = types.ModuleType('zero')
        applied.append('libzero not installed: stub module (imported by src/, never used)')
    import utils as goio_utils
    import utils_train as goio_utils_train
    import src as goio_src
    import data.dataprocessing as goio_data
    import MLVAE.main as goio_mlvae
    import CLDM.main as goio_cldm
    import CLDM.sample as goio_sample
    import CLDM.evaluation as goio_eval
    for module in (goio_utils, goio_utils_train, goio_src, goio_data, goio_mlvae, goio_cldm,
                   goio_sample, goio_eval):
        _assert_from_goio(module)

    if not torch.cuda.is_available():
        goio_mlvae.torch = _cpu_torch_proxy(torch)
        upstream_sample = goio_sample.sample
        goio_sample.sample = lambda *a, **kw: upstream_sample(*a, **dict(kw, device='cpu'))
        applied.append("no CUDA: MLVAE 'cuda:0' tensors and the CLDM sampler run on the CPU")
    if not hasattr(pd.DataFrame, '_append'):
        pd.DataFrame._append = lambda self, other, ignore_index=False: pd.concat(
            [self, pd.DataFrame([other])], ignore_index=ignore_index)
        applied.append(f'pandas {pd.__version__}: DataFrame._append restored via concat')
    if cfg.get('epoch_cap'):
        capped = _epoch_capped_range(cfg['epoch_cap'])
        goio_mlvae.range = capped
        goio_cldm.range = capped
        applied.append(f"SMOKE TEST: MLVAE/CLDM epochs capped at {cfg['epoch_cap']}")
    for line in applied:
        print(f'[runner] {line}', flush=True)

    records = []
    if cfg['mode'] == 'eval':
        for spec in goio_eval._MODELS['binary_classification']:
            spec['class'] = _recording_class(spec['class'], records)

    # Exactly `python main.py --dataname ... --method ... --mode ... [--exp k]`.
    sys.argv = ['main.py', '--dataname', cfg['dataname'], '--method', cfg['method'],
                '--mode', cfg['mode']]
    if cfg.get('exp') is not None:
        sys.argv += ['--exp', str(cfg['exp'])]
    runpy.run_path(os.path.join(GOIO_PATH, 'main.py'), run_name='__main__')

    if cfg.get('records_path'):
        with open(cfg['records_path'], 'wb') as f:
            pickle.dump({'records': records, 'shims': applied}, f)


# --------------------------------------------------------------------------
# caller side
# --------------------------------------------------------------------------

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
    swapped = bool(y.sum() * 2 > len(y))
    if swapped:
        y = 1 - y
    return x_df, y, swapped


def _run_step(cfg, log_path):
    """One `python main.py ...` in a subprocess; its output goes to log_path."""
    env = dict(os.environ, PYTHONUNBUFFERED='1')
    t0 = time.time()
    with open(log_path, 'w') as log:
        proc = subprocess.run(
            [sys.executable, '-u', os.path.abspath(__file__), '--worker', json.dumps(cfg)],
            cwd=GOIO_PATH, env=env, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        with open(log_path, errors='replace') as log:
            tail = log.read().splitlines()[-40:]
        print('\n'.join(['  | ' + line for line in tail]), flush=True)
        raise RuntimeError(f"GOIO {cfg['method']} {cfg['mode']} failed (exit {proc.returncode}); "
                           f"full log: {log_path}")
    return time.time() - t0


def _check_split(dataset_dir, dataset_name, full_csv, y, folds_expected):
    """Upstream's split must give the project's folds: same test rows, same order."""
    for k, (train_idx, test_idx) in enumerate(folds_expected):
        npz = np.load(os.path.join(dataset_dir, 'SOS', f'exp{k}', f'{dataset_name}.npz'))
        if not (np.array_equal(npz['test'], full_csv[test_idx])
                and np.array_equal(npz['train'], full_csv[train_idx])):
            raise RuntimeError(f"upstream's fold exp{k} differs from the project's "
                               f"StratifiedKFold split")


def _column_types(dataset_dir, dataset_name):
    with open(os.path.join(dataset_dir, f'{dataset_name}.json')) as f:
        columns = json.load(f)['columns']
    return {c['name']: c['type'] for c in columns}


def _fold_rows(fold, records, upstream_csv, y_test):
    if len(records) != N_CLASSIFIERS * N_REPEATS:
        raise RuntimeError(f"fold {fold}: eval ran {len(records)} classifiers, "
                           f"expected {N_CLASSIFIERS * N_REPEATS}")
    upstream = pd.read_csv(upstream_csv, index_col=0).iloc[:N_CLASSIFIERS * N_REPEATS]
    y = y_test.astype(int)
    rows, last_rf = [], None
    for idx, record in enumerate(records):
        pred = record['pred'].astype(int)
        row = {
            'fold': fold, 'classifier': record['classifier'], 'repeat': idx % N_REPEATS,
            'accuracy': accuracy_score(y, pred),
            'balanced_accuracy': balanced_accuracy_score(y, pred),
            'macro_f1': f1_score(y, pred, average='macro'),
            'mcc': matthews_corrcoef(y, pred),
            'gmean': geometric_mean_score(y, pred),
            'auc': roc_auc_score(y, record['score']) if len(np.unique(y)) == 2 else np.nan,
        }
        for metric in ('accuracy', 'macro_f1', 'mcc'):
            if not np.isclose(row[metric], upstream.iloc[idx][metric]):
                raise RuntimeError(f"fold {fold} run {idx} ({row['classifier']}): recorded "
                                   f"{metric} {row[metric]:.6f} != upstream's "
                                   f"{upstream.iloc[idx][metric]:.6f}")
        rows.append(row)
        if record['classifier'] == 'RandomForestClassifier':
            last_rf = (y, pred)
    return rows, last_rf


def run_goio_original(data, dataset_name, results_file, categorical_cols=None, resume=True,
                      epoch_cap=None, folds=None):
    """Runs upstream GOIO (split, then MLVAE / CLDM train, sample, eval per
    fold) on 5 folds and writes the results.

    data              (X_df, y_series) from a base_functions get_*_data().
    dataset_name      name used inside GOIO_ORIGINAL/data/datasets.
    results_file      output path prefix, without extension.
    categorical_cols  true categorical columns (abalone: ['Sex']), integer-coded.
    epoch_cap         caps MLVAE/CLDM epochs - for smoke tests only; recorded
                      as a deviation.
    folds             subset of fold indices to run (verification only);
                      None runs all 5.
    """
    if dataset_name in _UPSTREAM_DATASETS:
        raise ValueError(f"'{dataset_name}' is upstream's bundled example dataset")
    results_file = os.path.abspath(results_file)
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    start = time.time()

    x_df, y, swapped = _encode(data, categorical_cols)
    frame = x_df.copy()
    frame['label'] = y
    dataset_dir = os.path.join(GOIO_PATH, 'data', 'datasets', dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    csv_path = os.path.join(dataset_dir, f'{dataset_name}.csv')
    frame.to_csv(csv_path, index=False)
    # What upstream reads back (pd.read_csv(...).values.astype(float)).
    full_csv = pd.read_csv(csv_path).values.astype(float)
    data_sha1 = hashlib.sha1(np.ascontiguousarray(full_csv).tobytes()).hexdigest()

    print(f"=== GOIO (original pipeline): {dataset_name} (upstream {UPSTREAM_COMMIT[:7]}) ===")
    print(f"  {len(y)} rows, {x_df.shape[1]} features, classes {np.bincount(y).tolist()}")
    if epoch_cap:
        print(f"  DEVIATION FROM UPSTREAM: MLVAE/CLDM epochs capped at {epoch_cap} (smoke test)")
    if swapped:
        print("  label 1 is the majority class here: labels swapped for GOIO (upstream expects "
              "major=0), predictions swapped back for reporting")

    split_log = os.path.join(dataset_dir, '_upstream_data_split.log')
    print(f"  $ python main.py --dataname {dataset_name} --method data --mode split", flush=True)
    _run_step({'dataname': dataset_name, 'method': 'data', 'mode': 'split'}, split_log)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SPLIT_SEED)
    folds_expected = list(skf.split(full_csv[:, :-1], y))
    _check_split(dataset_dir, dataset_name, full_csv, y, folds_expected)
    column_types = _column_types(dataset_dir, dataset_name)
    detected = {t: [c for c, ct in column_types.items() if ct == t and c != 'label']
                for t in ('continuous', 'ordinal', 'categorical')}
    print("  upstream column types: " + ', '.join(f'{t} {len(c)}' for t, c in detected.items())
          + (f"  (categorical: {detected['categorical']})" if detected['categorical'] else ''))

    cache_key = {'upstream_commit': UPSTREAM_COMMIT, 'data_sha1': data_sha1,
                 'categorical_cols': list(categorical_cols or []), 'epoch_cap': epoch_cap}
    all_rows, true_values, predicted_values, fold_info, shims = [], [], [], [], set()
    for k, (train_idx, test_idx) in enumerate(folds_expected):
        if folds is not None and k not in folds:
            continue
        fold_dir = os.path.join(GOIO_PATH, 'synthetic', dataset_name, f'exp{k}')
        ckpt_dir = os.path.join(GOIO_PATH, CKPT_DIR, dataset_name, f'exp{k}')
        done_path = os.path.join(fold_dir, _DONE_FILE)
        records_path = os.path.join(fold_dir, _RECORDS_FILE)
        print(f"\n--- {dataset_name} exp{k}: train={len(train_idx)} (minority "
              f"{int(y[train_idx].sum())}), test={len(test_idx)} (minority "
              f"{int(y[test_idx].sum())}) ---", flush=True)

        cached = None
        if resume and os.path.exists(done_path):
            with open(done_path) as f:
                done = json.load(f)
            if done.get('key') == cache_key:
                cached = done
                print(f"  reusing finished fold in {fold_dir} (resume=False to recompute)")

        if cached is None:
            # Fresh dirs: MLVAE / CLDM only overwrite a checkpoint when the loss
            # improves, so a stale one could otherwise be sampled from.
            for path in (fold_dir, ckpt_dir):
                if os.path.isdir(path):
                    shutil.rmtree(path)
            os.makedirs(fold_dir)
            seconds = {}
            for method, mode in FOLD_STEPS:
                cfg = {'dataname': dataset_name, 'method': method, 'mode': mode, 'exp': k,
                       'epoch_cap': epoch_cap}
                if mode == 'eval':
                    cfg['records_path'] = records_path
                log_path = os.path.join(fold_dir, f'_upstream_{method}_{mode}.log')
                print(f"  $ python main.py --dataname {dataset_name} --method {method} "
                      f"--mode {mode} --exp {k}", end='', flush=True)
                seconds[f'{method}_{mode}'] = round(_run_step(cfg, log_path), 1)
                print(f"   [{seconds[f'{method}_{mode}']:.0f}s]", flush=True)
            cached = {'key': cache_key, 'seconds': seconds}
            with open(done_path, 'w') as f:
                json.dump(cached, f)

        with open(records_path, 'rb') as f:
            payload = pickle.load(f)
        shims.update(payload['shims'])
        upstream_csv = os.path.join(fold_dir, 'CLDM', f'{dataset_name}.csv')
        rows, last_rf = _fold_rows(k, payload['records'], upstream_csv, y[test_idx])
        all_rows.extend(rows)
        # Back to the dataset's own labels (metrics are swap-invariant).
        true_values.append(1 - last_rf[0] if swapped else last_rf[0])
        predicted_values.append(1 - last_rf[1] if swapped else last_rf[1])

        fold_df = pd.DataFrame(rows)
        print(f"  fold {k + 1}/{N_FOLDS}  gmean={fold_df['gmean'].mean():.4f}  "
              f"auc={np.nanmean(fold_df['auc']):.4f}  "
              f"({sum(cached['seconds'].values()):.0f}s)", flush=True)
        fold_info.append({'exp': k, 'n_train': int(len(train_idx)), 'n_test': int(len(test_idx)),
                          'seconds': cached['seconds']})

    runs = pd.DataFrame(all_rows)
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

    info = {
        'method': 'GOIO (upstream main.py split, MLVAE/CLDM train, sample, eval; '
                  'no hyperparameter search)',
        'upstream': 'https://github.com/MLneuers/GOIO', 'upstream_commit': UPSTREAM_COMMIT,
        'dataset': dataset_name, 'n_rows': int(len(y)), 'n_features': int(x_df.shape[1]),
        'categorical_cols': list(categorical_cols or []),
        'upstream_column_types': detected,
        'arguments': 'upstream defaults (utils.get_args)',
        'deviations_from_upstream': {'epoch_cap': epoch_cap} if epoch_cap else {},
        'compat_shims': sorted(shims),
        'labels_swapped_for_goio': swapped,
        'folds_run': sorted(folds) if folds is not None else list(range(N_FOLDS)),
        'reported': 'mean over GaussianNB/DecisionTree/SVC/RandomForest/KNN x 10 repeats '
                    'per fold (upstream eval); AUC from predict_proba, SVC decision_function',
        'protocol_notes': ['MLVAE checkpoint selected by reconstruction loss on the TEST fold '
                           '(upstream)'],
        'folds': fold_info, 'total_seconds': round(time.time() - start, 1),
    }
    with open(results_file + '_run_info.json', 'w') as f:
        json.dump(info, f, indent=2)

    print(f"\n=== {dataset_name}: across folds (mean over 5 classifiers x 10 repeats) ===")
    print(summary.round(4))
    print(f"evaluated fitness: {fitness:.6f}")
    print(f"--- {dataset_name}: {time.time() - start:.0f} seconds ---", flush=True)
    return {'fitness': fitness, 'true_values': true_values, 'predicted_values': predicted_values}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', required=True, help='JSON config; internal use')
    _worker(json.loads(parser.parse_args().worker))
