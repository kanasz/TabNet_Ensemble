"""
Fixed-baseline (no GA) runner for SOS, run the way the authors designed it.

SOS ("SOS: Score-based Oversampling for Tabular Data", KDD 2022,
https://github.com/JayoungKim408/SOS) is driven through its own main.py from
an untouched vendored copy in ga_heso_sota_methods/SOS_ORIGINAL (upstream
commit UPSTREAM_COMMIT). Nothing in that directory is edited. The project's
other copy, ga_heso_sota_methods/SOS, is the GA-adapted variant and is not used.

Per fold this runs exactly upstream's run.sh:

    python main.py --config <cfg> --mode train     --workdir <wd>
    python main.py --config <cfg> --mode fine_tune --workdir <wd> --workfile <name>

  train      one score network per class (n_iters=10000 each); every
             eval_freq steps the network samples, the 4 classifiers of
             evaluation.py are trained on train+samples and scored on the TEST
             split, and the checkpoint with the best test macro-F1 is kept
  fine_tune  evaluates the minor-class model (5 repeats), fine-tunes it
             against the major-class score field (num_fine_tuning_epochs=5,
             lambda_=0.99, angle=70), evaluates again (5 repeats). Each
             repeat oversamples the minority class up to the majority count
             twice: from perturbed non-minority samples ("z_major", SOS's
             method) and from pure noise ("N").

Inputs. Upstream reads tabular_datasets/{name}.npz (train / test arrays, label
last) + {name}.json (column metadata) with one fixed train/test split. Each of
this project's 5 stratified folds (StratifiedKFold(5, shuffle=True,
random_state=42), the GA-HESO split) is written as its own dataset
{name}_exp{k}. The config is upstream's configs/{sde}/default.py with only
data.dataset and data.image_size set for the fold - all of upstream's configs
are identical apart from those two fields (and nf=128 for ve/weatherAUS).
sde defaults to 've', the one upstream's run.sh uses. Upstream also assumes
the majority class has label 0; where label 1 is the majority (ecoli_0_vs_1)
the labels are swapped for SOS and back for reporting (metrics are invariant).
Constant feature columns are dropped (ecoli_0_vs_1's 'Chg'): the transformer
divides by max - min, so they would become NaN.

Metrics. Upstream logs only weighted-F1. evaluation.py's classifiers
(DecisionTree, AdaBoost, LogisticRegression, MLP) are wrapped in the worker so
their predictions and probabilities are recorded while upstream runs them; the
recorded accuracy is checked against upstream's own score table for every
evaluation. Reported: phase 'after' fine-tuning, variant 'z_major', averaged
over 4 classifiers x 5 repeats per fold (the other phase/variant are kept in
_runs.csv).

Compatibility shims. Upstream pins numpy 1.23, pandas 1.5 and pomegranate
0.14 and breaks on the project env's newer versions. The worker restores the
pinned behaviour at runtime, each only when the installed version needs it,
without touching upstream files or changing results:
  - numpy >= 1.24: datasets.get_dataset() builds np.array() from the
    per-class arrays, which have different lengths; old numpy silently made
    an object array, new numpy raises. Ragged inputs get dtype=object again.
  - pandas >= 2: evaluation.compute_scores() averages a frame with a text
    'name' column; pandas 1.5 dropped such columns, pandas 2 raises.
    groupby().mean() defaults to numeric_only=True again.
  - pomegranate: evaluation.py imports BayesianNetwork but never uses it.
  - scikit-learn >= 1.4: OneHotEncoder(sparse=...) became sparse_output=.
  - torch >= 2.6: torch.load() defaults to weights_only=True, which rejects
    the numpy scalars in upstream's own checkpoints; the old default is
    restored (only checkpoints this same run wrote are ever loaded).
  - no GPU: datasets.get_dataset() does `batch_size % torch.cuda.device_count()`,
    a division by zero without CUDA; the count is reported as at least 1.
The shims applied are listed in each _run_info.json.

Protocol properties kept as designed (worth stating when using SOS as a
baseline): the checkpoint is selected by macro-F1 on the TEST split, and the
GeneralTransformer is fitted on train+test. Both make the scores optimistic.

Outputs (results_file is an absolute or caller-relative path prefix):
  {results_file}.txt                {'fitness', 'true_values', 'predicted_values'}:
                                    fitness = mean G-mean (after / z_major);
                                    predictions = DecisionTree, last repeat
  {results_file}_folds.csv          per fold, after / z_major: mean over
                                    classifiers x repeats
  {results_file}_summary.csv        per metric: mean/std across folds
  {results_file}_per_classifier.csv per classifier: mean/std across folds
  {results_file}_runs.csv           every (fold, phase, variant, repeat,
                                    classifier) evaluation
  {results_file}_run_info.json      upstream commit, settings, shims, timings
Finished folds are cached in SOS_ORIGINAL/runs/{name}/exp{k}/ and skipped on
a rerun with the same data and settings (resume=False to recompute).
"""

import argparse
import hashlib
import json
import os
import pickle
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
SOS_PATH = os.path.join(_PROJECT_ROOT, 'ga_heso_sota_methods', 'SOS_ORIGINAL')
UPSTREAM_COMMIT = 'f1b450cf199d5e0013b401e380008446ef8757f7'

N_FOLDS = 5
SPLIT_SEED = 42                  # GA-HESO's StratifiedKFold seed
DEFAULT_SDE = 've'               # upstream run.sh
N_FINE_TUNE_EVALS = 20           # fine_tune: (before + after) x 5 repeats x (z_major, N)
_UPSTREAM_DATASETS = {'default', 'satimage', 'shoppers', 'weatherAUS'}
_RECORDS_FILE = '_runner_records.pkl'
_DONE_FILE = '_runner_done.json'
METRICS = ['accuracy', 'balanced_accuracy', 'macro_f1', 'mcc', 'gmean', 'auc']


# --------------------------------------------------------------------------
# worker side: runs in a subprocess with cwd = SOS_PATH
# --------------------------------------------------------------------------

def _version_tuple(version):
    parts = []
    for piece in version.split('.')[:2]:
        digits = ''.join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def _apply_compat_shims():
    """Restores the behaviour of upstream's pinned library versions. Each shim
    is applied only when the installed version needs it; see the module
    docstring. Returns the names of the shims applied."""
    applied = []

    try:
        from pomegranate import BayesianNetwork  # noqa: F401
    except Exception:
        stub = types.ModuleType('pomegranate')
        stub.BayesianNetwork = None
        sys.modules['pomegranate'] = stub
        applied.append('pomegranate: stub BayesianNetwork (imported, never used)')

    if _version_tuple(pd.__version__) >= (2, 0):
        from pandas.core.groupby.generic import DataFrameGroupBy
        original_mean = DataFrameGroupBy.mean

        def mean(self, *args, **kwargs):
            if not args:
                kwargs.setdefault('numeric_only', True)
            return original_mean(self, *args, **kwargs)
        DataFrameGroupBy.mean = mean
        applied.append(f'pandas {pd.__version__}: groupby().mean(numeric_only=True)')

    import torch
    if _version_tuple(torch.__version__) >= (2, 6):
        upstream_load = torch.load

        def load(*args, **kwargs):
            # Only checkpoints written by upstream earlier in this same run
            # are loaded, so the pre-2.6 default is safe here.
            kwargs.setdefault('weights_only', False)
            return upstream_load(*args, **kwargs)
        torch.load = load
        applied.append(f'torch {torch.__version__}: torch.load(weights_only=False), the pre-2.6 '
                       f'default (upstream checkpoints hold numpy scalars)')
    if torch.cuda.device_count() == 0:
        torch.cuda.device_count = lambda: 1
        applied.append('no CUDA: torch.cuda.device_count() reported as 1')
    return applied


def _patch_upstream_modules(datasets_mod, evaluation_mod, applied):
    if _version_tuple(np.__version__) >= (1, 24):
        def array(obj, *args, **kwargs):
            if (not args and 'dtype' not in kwargs and isinstance(obj, list) and obj
                    and all(isinstance(a, np.ndarray) for a in obj)
                    and len({a.shape for a in obj}) > 1):
                out = np.empty(len(obj), dtype=object)
                out[:] = obj
                return out
            return np.array(obj, *args, **kwargs)
        numpy_compat = types.ModuleType('numpy_compat')
        numpy_compat.__dict__.update({k: getattr(np, k) for k in dir(np) if not k.startswith('__')})
        numpy_compat.array = array
        datasets_mod.np = numpy_compat
        applied.append(f'numpy {np.__version__}: ragged np.array -> object array in datasets.py')

    import sklearn
    if _version_tuple(sklearn.__version__) >= (1, 4):
        upstream_encoder = evaluation_mod.OneHotEncoder

        def OneHotEncoder(*args, sparse=None, **kwargs):
            if sparse is not None:
                kwargs['sparse_output'] = sparse
            return upstream_encoder(*args, **kwargs)
        evaluation_mod.OneHotEncoder = OneHotEncoder
        applied.append(f'scikit-learn {sklearn.__version__}: OneHotEncoder(sparse=) -> sparse_output=')


def _recording_class(cls, state):
    """Subclass that keeps what upstream's _evaluate_binary_classification gets
    from predict() / predict_proba(). Constructor and fitting are unchanged."""
    name = cls.__name__

    class Recording(cls):
        def predict(self, X):
            out = super().predict(X)
            state['iters'][-1]['preds'].setdefault(name, {})['pred'] = np.asarray(out)
            return out

        def predict_proba(self, X):
            out = super().predict_proba(X)
            entry = state['iters'][-1]['preds'].setdefault(name, {})
            entry['proba'] = np.asarray(out)
            entry['classes'] = np.asarray(self.classes_)
            return out

    Recording.__name__ = name
    Recording.__qualname__ = cls.__qualname__
    return Recording


def _install_recording(evaluation_mod, records):
    state = {'iters': []}
    upstream_prepare = evaluation_mod._prepare_ml_problem
    upstream_compute = evaluation_mod.compute_scores

    def prepare(train, test, metadata, clustering=False):
        x_train, y_train, x_test, y_test, model = upstream_prepare(train, test, metadata, clustering)
        if clustering:
            return x_train, y_train, x_test, y_test, model
        # make_features() shuffles, so y_test is kept in the order predicted.
        state['iters'].append({'y_test': np.asarray(y_test), 'preds': {}})
        wrapped = [dict(spec, **{'class': _recording_class(spec['class'], state)}) for spec in model]
        return x_train, y_train, x_test, y_test, wrapped

    def compute_scores(test, synthesized_data, metadata, test_iter=5):
        state['iters'] = []
        result = upstream_compute(test, synthesized_data, metadata, test_iter)
        records.append({
            'iters': state['iters'],
            'upstream_accuracy': [float(v) for v in result['accuracy'].values],
            'upstream_weighted_f1': [float(v) for v in result['weighted_f1'].values],
        })
        return result

    evaluation_mod._prepare_ml_problem = prepare
    evaluation_mod.compute_scores = compute_scores


def _assert_from_sos(module):
    path = os.path.abspath(getattr(module, '__file__', '') or '')
    if not path.startswith(os.path.abspath(SOS_PATH) + os.sep):
        raise ImportError(f"'{module.__name__}' was imported from {path or '<unknown>'}, "
                          f"not from {SOS_PATH}")


def _worker(cfg):
    # Upstream imports `main`, `run_lib`, `datasets`, `evaluation`, `models`,
    # `utils` as top-level modules; the project root has a main.py and a
    # models/ folder too, so SOS's directory must come first.
    sys.path.insert(0, SOS_PATH)
    applied = _apply_compat_shims()
    import datasets as sos_datasets
    import evaluation as sos_evaluation
    import run_lib as sos_run_lib
    import main as sos_main
    for module in (sos_datasets, sos_evaluation, sos_run_lib, sos_main):
        _assert_from_sos(module)
    _patch_upstream_modules(sos_datasets, sos_evaluation, applied)
    for line in applied:
        print(f'[compat] {line}', flush=True)

    records = []
    if cfg['mode'] == 'fine_tune':
        _install_recording(sos_evaluation, records)

    # Exactly `python main.py --config ... --mode ... --workdir ... [--workfile ...]`.
    argv = ['main.py', '--config', cfg['config_path'], '--mode', cfg['mode'],
            '--workdir', cfg['workdir']]
    if cfg['mode'] == 'fine_tune':
        argv += ['--workfile', cfg['workfile']]
    from absl import app
    try:
        app.run(sos_main.main, argv=argv)
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise

    if cfg['mode'] == 'fine_tune':
        with open(os.path.join(cfg['workdir'], _RECORDS_FILE), 'wb') as f:
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
    # GeneralTransformer scales continuous columns by (x - min) / (max - min),
    # so a constant column becomes NaN and corrupts training. It carries no
    # information, so it is dropped (ecoli_0_vs_1's 'Chg').
    constant = [c for c in x_df.columns if c not in cats and x_df[c].min() == x_df[c].max()]
    x_df = x_df.drop(columns=constant)
    # Upstream assumes the majority class has the smaller label: train() indexes
    # the per-class arrays [major, minor] by label value and fine_tune() decides
    # roles with `label < len(major_label)`. Where label 1 is the majority
    # (ecoli_0_vs_1), the labels are swapped for SOS and swapped back in the
    # stored predictions. Every reported metric is invariant to the swap.
    swapped = bool(y.sum() * 2 > len(y))
    if swapped:
        y = 1 - y
    return x_df, y, swapped, constant


def _metadata(x_df, categorical_cols):
    """Column metadata in upstream's tabular_datasets/*.json format."""
    columns = []
    for col in x_df.columns:
        values = x_df[col].to_numpy()
        if col in (categorical_cols or []):
            codes = sorted(np.unique(values).tolist())
            columns.append({'name': str(col), 'type': 'categorical',
                            'size': len(codes), 'i2s': codes})
        else:
            columns.append({'name': str(col), 'type': 'continuous',
                            'min': float(values.min()), 'max': float(values.max())})
    columns.append({'name': 'label', 'type': 'categorical', 'size': 2, 'i2s': [0, 1]})
    return {'columns': columns, 'problem_type': 'binary_classification'}


def _image_size(meta):
    """GeneralTransformer's output width: 1 per continuous column, one-hot
    width per categorical column (the label included)."""
    return sum(1 if c['type'] == 'continuous' else c['size'] for c in meta['columns'])


def _write_config(path, fold_name, image_size, sde, overrides):
    lines = [
        f'"""Generated by optimization/no_ga_sos_original_runner.py for {fold_name}.',
        f'Upstream configs/{sde}/default.py with only the dataset fields set."""',
        f'from configs.{sde}.default import get_config as _upstream_get_config',
        '',
        '',
        'def get_config():',
        '  config = _upstream_get_config()',
        f'  config.data.dataset = {fold_name!r}',
        f'  config.data.image_size = {image_size}',
    ]
    for key, value in overrides.items():
        lines.append(f'  config.{key} = {value!r}  # deviation from upstream')
    lines += ['  return config', '']
    with open(path, 'w') as f:
        f.write('\n'.join(lines))


def _run_worker(cfg):
    env = dict(os.environ, PYTHONUNBUFFERED='1')
    subprocess.run([sys.executable, '-u', os.path.abspath(__file__), '--worker', json.dumps(cfg)],
                   cwd=SOS_PATH, env=env, check=True)


def _fold_rows(fold, records):
    if len(records) != N_FINE_TUNE_EVALS:
        raise RuntimeError(f"fold {fold}: fine_tune evaluated {len(records)} times, "
                           f"expected {N_FINE_TUNE_EVALS}")
    rows, last_dt = [], None
    for idx, record in enumerate(records):
        phase = 'before' if idx < N_FINE_TUNE_EVALS // 2 else 'after'
        variant = 'z_major' if idx % 2 == 0 else 'N'
        repeat = (idx % (N_FINE_TUNE_EVALS // 2)) // 2
        for it, iteration in enumerate(record['iters']):
            y = iteration['y_test'].astype(int)
            accuracies = []
            for clf, entry in iteration['preds'].items():
                pred = entry['pred'].astype(int)
                auc = np.nan
                if 'proba' in entry and len(np.unique(y)) == 2:
                    col = int(np.flatnonzero(entry['classes'].astype(int) == 1)[0])
                    auc = roc_auc_score(y, entry['proba'][:, col])
                row = {
                    'fold': fold, 'phase': phase, 'variant': variant, 'repeat': repeat,
                    'test_iter': it, 'classifier': clf,
                    'accuracy': accuracy_score(y, pred),
                    'balanced_accuracy': balanced_accuracy_score(y, pred),
                    'macro_f1': f1_score(y, pred, average='macro'),
                    'mcc': matthews_corrcoef(y, pred),
                    'gmean': geometric_mean_score(y, pred),
                    'auc': auc,
                }
                rows.append(row)
                accuracies.append(row['accuracy'])
                if phase == 'after' and variant == 'z_major' and clf == 'DecisionTreeClassifier':
                    last_dt = (y, pred)
            if len(accuracies) != 4:
                raise RuntimeError(f"fold {fold} eval {idx}: recorded {len(accuracies)} "
                                   f"classifiers, expected 4")
            if not np.isclose(np.mean(accuracies), record['upstream_accuracy'][it]):
                raise RuntimeError(
                    f"fold {fold} eval {idx}: recorded mean accuracy {np.mean(accuracies):.6f} "
                    f"!= upstream's {record['upstream_accuracy'][it]:.6f}")
    return rows, last_dt


def run_sos_original(data, dataset_name, results_file, categorical_cols=None,
                     sde=DEFAULT_SDE, resume=True, config_overrides=None, folds=None):
    """Runs upstream SOS (train + fine_tune) on 5 folds and writes the results.

    data              (X_df, y_series) from a base_functions get_*_data().
    dataset_name      base name used inside SOS_ORIGINAL/tabular_datasets.
    results_file      output path prefix, without extension.
    categorical_cols  true categorical columns (abalone: ['Sex']); declared
                      categorical in the metadata, as upstream's datasets do.
    sde               've' (upstream run.sh), 'vp' or 'subvp'.
    config_overrides  {'training.n_iters': 1, ...} - for smoke tests only;
                      anything here is recorded as a deviation.
    folds             subset of fold indices to run (verification only);
                      None runs all 5.
    """
    if dataset_name in _UPSTREAM_DATASETS:
        raise ValueError(f"'{dataset_name}' is one of upstream's bundled datasets")
    if sde not in ('ve', 'vp', 'subvp'):
        raise ValueError("sde must be 've', 'vp' or 'subvp'")
    overrides = dict(config_overrides or {})

    results_file = os.path.abspath(results_file)
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    start = time.time()

    x_df, y, swapped, dropped_constant = _encode(data, categorical_cols)
    meta = _metadata(x_df, categorical_cols)
    image_size = _image_size(meta)
    full = np.column_stack([x_df.to_numpy(dtype=float), y.astype(float)])
    data_sha1 = hashlib.sha1(np.ascontiguousarray(full).tobytes()).hexdigest()

    print(f"=== SOS (original pipeline): {dataset_name} (upstream {UPSTREAM_COMMIT[:7]}, "
          f"sde={sde}) ===")
    print(f"  {len(y)} rows, {x_df.shape[1]} features, classes {np.bincount(y).tolist()}, "
          f"image_size={image_size}")
    if overrides:
        print(f"  DEVIATIONS FROM UPSTREAM CONFIG: {overrides}")
    if swapped:
        print("  label 1 is the majority class here: labels swapped for SOS (upstream expects "
              "major=0), predictions swapped back for reporting")
    if dropped_constant:
        print(f"  dropped constant feature column(s) {dropped_constant} (SOS's transformer "
              f"divides by max - min)")

    cache_key = {'upstream_commit': UPSTREAM_COMMIT, 'data_sha1': data_sha1, 'sde': sde,
                 'categorical_cols': list(categorical_cols or []), 'overrides': overrides}
    tabular_dir = os.path.join(SOS_PATH, 'tabular_datasets')
    run_root = os.path.join(SOS_PATH, 'runs', dataset_name)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SPLIT_SEED)

    all_rows, true_values, predicted_values, fold_info, shims = [], [], [], [], set()
    for k, (train_idx, test_idx) in enumerate(skf.split(full[:, :-1], y)):
        if folds is not None and k not in folds:
            continue
        fold_name = f'{dataset_name}_exp{k}'
        workdir = os.path.join(run_root, f'exp{k}')
        done_path = os.path.join(workdir, _DONE_FILE)
        y_train, y_test = y[train_idx], y[test_idx]
        print(f"\n--- {fold_name}: train={len(train_idx)} (minority {int(y_train.sum())}), "
              f"test={len(test_idx)} (minority {int(y_test.sum())}) ---", flush=True)

        cached = None
        if resume and os.path.exists(done_path):
            with open(done_path) as f:
                done = json.load(f)
            if done.get('key') == cache_key:
                cached = done
                print(f"  reusing finished fold in {workdir} (resume=False to recompute)")

        if cached is None:
            # Fresh workdir: upstream silently resumes from checkpoints-meta,
            # and a stale best checkpoint would be evaluated in place of this run's.
            if os.path.isdir(workdir):
                shutil.rmtree(workdir)
            os.makedirs(workdir)
            np.savez(os.path.join(tabular_dir, f'{fold_name}.npz'),
                     train=full[train_idx], test=full[test_idx])
            with open(os.path.join(tabular_dir, f'{fold_name}.json'), 'w') as f:
                json.dump(meta, f, indent=4)
            config_path = os.path.join(workdir, 'config.py')
            _write_config(config_path, fold_name, image_size, sde, overrides)

            t0 = time.time()
            # Relative to SOS_PATH (the worker's cwd), like upstream's run.sh:
            # ml_collections splits --config on ':' (its override syntax), so
            # an absolute Windows path 'D:\...' would be cut at the drive letter.
            rel_config = os.path.relpath(config_path, SOS_PATH).replace(os.sep, '/')
            rel_workdir = os.path.relpath(workdir, SOS_PATH).replace(os.sep, '/')
            base = {'config_path': rel_config, 'workdir': rel_workdir, 'workfile': fold_name}
            print(f"  $ python main.py --config {rel_config} --mode train --workdir {rel_workdir}",
                  flush=True)
            _run_worker(dict(base, mode='train'))
            train_seconds = time.time() - t0
            t0 = time.time()
            print(f"  $ python main.py --config {rel_config} --mode fine_tune --workdir {rel_workdir} "
                  f"--workfile {fold_name}", flush=True)
            _run_worker(dict(base, mode='fine_tune'))
            cached = {'key': cache_key, 'train_seconds': round(train_seconds, 1),
                      'fine_tune_seconds': round(time.time() - t0, 1)}
            with open(done_path, 'w') as f:
                json.dump(cached, f)

        with open(os.path.join(workdir, _RECORDS_FILE), 'rb') as f:
            payload = pickle.load(f)
        shims.update(payload['shims'])
        rows, last_dt = _fold_rows(k, payload['records'])
        all_rows.extend(rows)
        # Back to the dataset's own labels (metrics are swap-invariant).
        true_values.append(1 - last_dt[0] if swapped else last_dt[0])
        predicted_values.append(1 - last_dt[1] if swapped else last_dt[1])

        head = pd.DataFrame([r for r in rows if r['phase'] == 'after' and r['variant'] == 'z_major'])
        print(f"  fold {k + 1}/{N_FOLDS}  gmean={head['gmean'].mean():.4f}  "
              f"auc={np.nanmean(head['auc']):.4f}  (train {cached['train_seconds']:.0f}s, "
              f"fine_tune {cached['fine_tune_seconds']:.0f}s)", flush=True)
        fold_info.append({'exp': k, 'n_train': int(len(train_idx)), 'n_test': int(len(test_idx)),
                          'train_seconds': cached['train_seconds'],
                          'fine_tune_seconds': cached['fine_tune_seconds']})

    runs = pd.DataFrame(all_rows)
    head = runs[(runs['phase'] == 'after') & (runs['variant'] == 'z_major')]
    fold_means = head.groupby('fold')[METRICS].mean()
    summary = pd.DataFrame({'mean_across_folds': fold_means.mean(axis=0),
                            'std_across_folds': fold_means.std(axis=0)})
    per_classifier = (head.groupby(['classifier', 'fold'])[METRICS].mean()
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
        'method': 'SOS (upstream main.py train + fine_tune, no hyperparameter search)',
        'upstream': 'https://github.com/JayoungKim408/SOS', 'upstream_commit': UPSTREAM_COMMIT,
        'dataset': dataset_name, 'n_rows': int(len(y)), 'n_features': int(x_df.shape[1]),
        'categorical_cols': list(categorical_cols or []), 'image_size': image_size,
        'config_template': f'configs/{sde}/default.py', 'deviations_from_upstream': overrides,
        'compat_shims': sorted(shims),
        'labels_swapped_for_sos': swapped,
        'dropped_constant_columns': dropped_constant,
        'folds_run': sorted(folds) if folds is not None else list(range(N_FOLDS)),
        'reported': "phase 'after' fine-tuning, variant 'z_major', mean over "
                    "DecisionTree/AdaBoost/LogisticRegression/MLP x 5 repeats per fold",
        'protocol_notes': ['checkpoint selected by macro-F1 on the TEST split (upstream)',
                           'GeneralTransformer fitted on train+test (upstream)'],
        'folds': fold_info, 'total_seconds': round(time.time() - start, 1),
    }
    with open(results_file + '_run_info.json', 'w') as f:
        json.dump(info, f, indent=2)

    print(f"\n=== {dataset_name}: across folds (after fine-tuning, z_major) ===")
    print(summary.round(4))
    print(f"evaluated fitness: {fitness:.6f}")
    print(f"--- {dataset_name}: {time.time() - start:.0f} seconds ---", flush=True)
    return {'fitness': fitness, 'true_values': true_values, 'predicted_values': predicted_values}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', required=True, help='JSON config; internal use')
    _worker(json.loads(parser.parse_args().worker))
