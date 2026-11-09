"""
Fixed-baseline (no GA) runner for DGOT, reusable across datasets.

DGOT ("Diffusion GAN-based Oversampling for Imbalanced Tabular Data",
https://github.com/MLneuers/DGOT) is run exactly as its README describes,
from an untouched vendored copy in ga_heso_sota_methods/DGOT_BASELINE
(upstream commit UPSTREAM_COMMIT). Nothing in that directory is edited.

The published pipeline, step by step:

  1. data     scripts/dataprocessing.py::Default_processing(name) reads
              datasets/{name}/{name}.csv and writes the 5-fold layout
              (StratifiedKFold(5, shuffle=True, random_state=42), the same
              split GA-HESO uses). This runner only writes that .csv.
  2. train    `python train.py --dataset {name} --feature_len={F}
              --class_num 2 --exp exp{k}` (README "Example"), one subprocess
              per fold, with every other flag left at its paper default.
  3. evaluate evaluation.py::validation(): the 5 classifiers in its _MODELS
              list x --repeats 10, each with scripts/evaluate_binary.DGOT()
              oversampling every class to 1.2 x the majority count.

evaluation.py only writes accuracy / macro_f1 / mcc and throws the
predictions away. To report this project's usual metrics (G-mean, ROC AUC,
balanced accuracy) without editing upstream, each classifier handed to
evaluate_binary.DGOT() is wrapped in _RecordingClassifier, which passes
fit/predict straight through and keeps a copy of what predict() returned
(plus predict_proba scores for AUC). Upstream's own accuracy is checked
against the recorded predictions for every run, so a misalignment would
fail loudly. evaluation.py's loop is reproduced here, not called, because it
builds its classifiers internally and so cannot be given the wrappers. Its
classifier list (_MODELS), oversample rate (1.2), repeat count (10) and seed
(666) are imported or copied from it unchanged.

Properties of the published protocol that are kept as designed and worth
knowing when using DGOT as a baseline:

  - train.py picks the saved generator by macro-F1 measured on the TEST fold
    (val_data_process reads datasets/{name}/TEST/{exp}), so the scores are
    optimistic. The GA-adapted copy removes this; this runner keeps it.
  - Default_processing min-max scales the whole dataset to [-1, 1] before
    splitting (the comment there says "by the training set", the code uses
    all rows).
  - train.py's DataLoader uses batch_size=512 with drop_last=True. A training
    fold with fewer than 512 rows therefore gets no generator or discriminator
    updates at all, and the "trained" generator is its random
    initialisation. With the paper's defaults that applies to every ecoli and
    glass set, wine_white_9_vs_4, abalone_3_vs_11 and the 200-sample
    synthetic sets. The runner prints a warning and records steps_per_epoch
    in the run info. Pass batch_size= to override it (a deviation from the
    paper, recorded as such).

Two harmless deviations: the device is passed explicitly (upstream hardcodes
'cuda', which crashes on a CPU-only machine), and evaluation is reseeded with
evaluation.py's seed 666 at the start of every fold. That is the same as
running `python evaluation.py --exps k` once per fold, as the README does,
and it lets a fold be resumed on its own.

Outputs (results_file is an absolute or caller-relative path prefix):
  {results_file}.txt               {'fitness', 'true_values',
                                   'predicted_values'} in the same convention
                                   as the GOIO baseline: fitness = mean G-mean
                                   over 5 classifiers x 10 repeats x 5 folds;
                                   predictions = RandomForest's last repeat
                                   per fold
  {results_file}_summary.csv       per metric: mean/std across folds
  {results_file}_folds.csv         per fold: mean over classifiers x repeats
  {results_file}_per_classifier.csv per classifier: mean/std across folds
  {results_file}_runs.csv          every single (fold, classifier, repeat) run
  {results_file}_run_info.json     upstream commit, settings, steps/epoch,
                                   timings, warnings
Upstream's own per-fold evaluation CSV is also written where evaluation.py
would put it: DGOT_BASELINE/saved_log/evaluation/DGOT/exp{k}/{name}.csv.

NOTE: one fold takes a long time with the paper budget. From epoch 101 on,
train.py scores the generator every 5 epochs (5 classifiers x 3 sampling
rounds). Finished folds are therefore cached under
DGOT_BASELINE/saved_log/runner/{name}/exp{k}.pkl and reused when the
dataset, the settings and the upstream commit are all unchanged. Pass
resume=False, or delete that folder, to recompute.
"""

import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, roc_auc_score)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DGOT_PATH = os.path.join(_PROJECT_ROOT, 'ga_heso_sota_methods', 'DGOT_BASELINE')
UPSTREAM_COMMIT = 'bb5ea8b54d8c005ef4651cc17110e034d6bfab77'

# Default_processing hardcodes StratifiedKFold(n_splits=5, random_state=42).
N_FOLDS = 5
# train.py argparse defaults ("the same set of hyper-parameters across all
# experiments", README).
PAPER_NUM_EPOCH = 800
PAPER_BATCH_SIZE = 512
# The generator (models/Generator.py::Unet) halves the feature axis once per
# ch_mult entry with Conv1d(kernel 4, stride 2, padding 1), which needs an
# input of length >= 2 at every step, so feature_len >= 2 ** len(ch_mult).
# The default [1, 2, 2] therefore needs >= 8 features; ecoli (7) needs [1, 2].
PAPER_CH_MULT = [1, 2, 2]
# evaluation.py: --repeats default, the rate hardcoded in validation(), and
# the module-level seed.
PAPER_EVAL_REPEATS = 10
PAPER_OVERSAMPLE_RATE = 1.2
PAPER_EVAL_SEED = 666
# train.py loops `for epoch in range(0, num_epoch + 1)` and only saves netG.pth
# when `epoch % 5 == 0 and epoch > 100`, so the first possible save is epoch 105.
_MIN_EPOCHS_FOR_CHECKPOINT = 105

# Upstream ships this example dataset; never delete or overwrite it.
_UPSTREAM_EXAMPLE_DATASETS = {'abalone_15'}

METRICS = ['accuracy', 'macro_f1', 'mcc', 'gmean', 'auc', 'balanced_accuracy']
_UPSTREAM_METRICS = ['accuracy', 'macro_f1', 'mcc']


# --------------------------------------------------------------------------
# upstream access
# --------------------------------------------------------------------------

@contextmanager
def _inside_dgot():
    """Upstream uses './datasets/...' and './saved_log/...' paths and
    top-level imports (`scripts.*`, `models.*`), so it only works from its
    own directory. Unlike the GA tuners, this changes directory only for the
    duration of the call and restores the caller's working directory."""
    previous = os.getcwd()
    if DGOT_PATH not in sys.path:
        sys.path.insert(0, DGOT_PATH)
    os.chdir(DGOT_PATH)
    try:
        yield
    finally:
        os.chdir(previous)


def _assert_from_dgot(module):
    path = os.path.abspath(getattr(module, '__file__', '') or '')
    if not path.startswith(os.path.abspath(DGOT_PATH) + os.sep):
        raise ImportError(
            f"'{module.__name__}' was imported from {path or '<unknown>'}, not from "
            f"{DGOT_PATH}. Another module of that name was loaded first; DGOT's "
            f"top-level 'scripts', 'models' and 'evaluation' modules must come "
            f"from the vendored copy.")


_UPSTREAM = {}


def _upstream():
    """Imports the unmodified upstream modules once.

    The project root has a `models/` folder too. Both it and DGOT's `models/`
    are namespace packages (no __init__.py), so Python merges them and
    `models.Generator` still resolves into DGOT; the checks below make sure.
    """
    if not _UPSTREAM:
        if not hasattr(pd.DataFrame, '_append'):
            raise ImportError(
                f"pandas {pd.__version__} has no DataFrame._append, which DGOT's "
                f"scripts/evaluate_binary.py uses; upstream pins pandas==2.2.2, "
                f"so install pandas<3")
        with _inside_dgot():
            import scripts.dataprocessing as dataprocessing
            import scripts.evaluate_binary as evaluate_binary
            import evaluation  # module-level: seeds 666, defines _MODELS
            import models.Generator as generator
            for module in (dataprocessing, evaluate_binary, evaluation, generator):
                _assert_from_dgot(module)
        _UPSTREAM.update(dataprocessing=dataprocessing,
                         evaluate_binary=evaluate_binary,
                         evaluation=evaluation)
    return _UPSTREAM


def _reseed(seed):
    """Exactly the seeding evaluation.py does at import."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


class _RecordingClassifier:
    """Pass-through wrapper: upstream's sample_evaluate() calls fit() then
    predict(xtest) and keeps only accuracy/macro-F1/MCC. This keeps the
    predictions too, plus the positive-class probability for ROC AUC.
    predict_proba is deterministic for all five _MODELS classifiers, so the
    extra call does not change any random state upstream depends on."""

    def __init__(self, estimator):
        self.estimator = estimator
        self.records = []

    def fit(self, X, y):
        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        predicted = self.estimator.predict(X)
        score = None
        if hasattr(self.estimator, 'predict_proba'):
            classes = np.asarray(self.estimator.classes_).astype(float)
            positive = np.flatnonzero(classes == 1.0)
            if positive.size:
                score = self.estimator.predict_proba(X)[:, positive[0]]
        self.records.append((np.asarray(predicted).astype(int), score))
        return predicted

    def __getattr__(self, name):
        if name == 'estimator':  # not set yet (e.g. during copy/unpickle)
            raise AttributeError(name)
        return getattr(self.estimator, name)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _metrics(y_true, y_pred, score):
    auc = np.nan
    if score is not None and len(np.unique(y_true)) == 2:
        auc = roc_auc_score(y_true, score)
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro'),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'gmean': geometric_mean_score(y_true, y_pred),
        'auc': auc,
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
    }


def _write_upstream_csv(data, dataset_name, categorical_cols):
    """Writes (X_df, y_series) as datasets/{name}/{name}.csv, the input
    Default_processing() reads: numeric columns, label last, header row.

    Default_processing() does `.values.astype(float)`, so categorical columns
    (abalone's 'Sex') are label-encoded to 0, 1, 2, ... in one column. Upstream's
    own abalone_15 example stores its categorical column the same way, and
    DGOT then treats it like any other scaled feature.
    """
    x_df, y_series = data
    x_df = x_df.copy()
    categorical_cols = categorical_cols or []
    for col in x_df.columns:
        if col in categorical_cols:
            x_df[col] = pd.factorize(x_df[col].astype(str).str.strip())[0].astype(float)
        else:
            x_df[col] = x_df[col].astype(float)

    y = np.asarray(y_series).astype(int)
    labels = set(np.unique(y).tolist())
    if labels != {0, 1}:
        raise ValueError(f"{dataset_name}: expected binary 0/1 labels, got {sorted(labels)}")

    dataset_dir = os.path.join(DGOT_PATH, 'datasets', dataset_name)
    # Remove what a previous run left, including any {name}.xlsx, which
    # Default_processing would read in preference to the .csv.
    if os.path.isdir(dataset_dir):
        shutil.rmtree(dataset_dir)
    os.makedirs(dataset_dir)

    out = x_df
    out['label'] = y
    csv_path = os.path.join(dataset_dir, f'{dataset_name}.csv')
    out.to_csv(csv_path, index=False)
    with open(csv_path, 'rb') as f:
        digest = hashlib.sha1(f.read()).hexdigest()
    return x_df.shape[1] - 1, len(out), digest  # minus the label column


def _fold_shapes(dataset_name, k):
    base = os.path.join(DGOT_PATH, 'datasets', dataset_name)
    y_train = np.load(os.path.join(base, 'DGOT', f'exp{k}', 'ytrain.npy'))
    y_test = np.load(os.path.join(base, 'TEST', f'exp{k}', 'ytest.npy'))
    return y_train, y_test.astype(int)


def _train_fold(dataset_name, k, feature_len, num_epoch, batch_size, ch_mult, device):
    exp_dir = os.path.join(DGOT_PATH, 'saved_log', 'DGOT', dataset_name, f'exp{k}')
    # train.py only writes netG.pth when the test macro-F1 improves, so a
    # model left over from an earlier run could otherwise be evaluated in
    # place of this one.
    if os.path.isdir(exp_dir):
        shutil.rmtree(exp_dir)

    # The README invocation; non-default flags are appended only if overridden.
    cmd = [sys.executable, '-u', 'train.py',
           '--dataset', dataset_name, f'--feature_len={feature_len}',
           '--class_num', '2', '--exp', f'exp{k}', '--device', device]
    if num_epoch != PAPER_NUM_EPOCH:
        cmd += ['--num_epoch', str(num_epoch)]
    if batch_size != PAPER_BATCH_SIZE:
        cmd += ['--batch_size', str(batch_size)]
    if list(ch_mult) != PAPER_CH_MULT:
        cmd += ['--ch_mult', *map(str, ch_mult)]

    env = dict(os.environ, PYTHONUNBUFFERED='1')
    # train.py redraws its tqdm bar every iteration; in a SLURM log that is
    # one ever-growing line. tqdm >= 4.66 reads this to refresh less often.
    env.setdefault('TQDM_MININTERVAL', '30')

    print('    $ ' + ' '.join(cmd[1:]), flush=True)
    subprocess.run(cmd, cwd=DGOT_PATH, env=env, check=True)

    checkpoint = os.path.join(exp_dir, 'netG.pth')
    if not os.path.exists(checkpoint):
        raise RuntimeError(
            f"DGOT wrote no netG.pth for {dataset_name} exp{k}. train.py saves "
            f"one only when epoch > 100, epoch % 5 == 0 and test macro-F1 beats "
            f"0, so this means num_epoch <= 100 or macro-F1 stayed at 0.")


def _evaluate_fold(dataset_name, k, repeats, device):
    up = _upstream()
    _, y_test = _fold_shapes(dataset_name, k)
    rows, upstream_frames, rf_pred = [], [], None

    with _inside_dgot():
        _reseed(PAPER_EVAL_SEED)
        model_dir = f'./saved_log/DGOT/{dataset_name}/exp{k}'
        test_dir = f'./datasets/{dataset_name}/TEST/exp{k}'

        for spec in up['evaluation']._MODELS['binary_classification']:
            name = spec['class'].__name__
            recorder = _RecordingClassifier(spec['class'](**spec['kwargs']))
            result = up['evaluate_binary'].DGOT(
                model_dir, test_dir, recorder, PAPER_OVERSAMPLE_RATE,
                repetitions=repeats, devices=device)
            per_repeat = result.head(repeats)  # drop the appended mean/std rows
            upstream_frames.append(per_repeat)

            if len(recorder.records) != repeats:
                raise RuntimeError(f"{name}: recorded {len(recorder.records)} "
                                   f"predictions, expected {repeats}")
            for rep, (pred, score) in enumerate(recorder.records):
                ours = _metrics(y_test, pred, score)
                upstream_acc = float(per_repeat.iloc[rep]['accuracy'])
                if not np.isclose(upstream_acc, ours['accuracy']):
                    raise RuntimeError(
                        f"{name} repeat {rep}: upstream accuracy {upstream_acc:.6f} "
                        f"!= recorded {ours['accuracy']:.6f}; the recorded "
                        f"predictions do not line up with ytest")
                # Upstream's own numbers for its three metrics.
                for metric in _UPSTREAM_METRICS:
                    ours[metric] = float(per_repeat.iloc[rep][metric])
                rows.append({'fold': k, 'classifier': name, 'repetition': rep, **ours})
            if name == 'RandomForestClassifier':
                rf_pred = recorder.records[-1][0]

            clf_rows = pd.DataFrame([r for r in rows if r['classifier'] == name])
            print(f"    {name:<24} gmean={clf_rows['gmean'].mean():.4f}  "
                  f"auc={np.nanmean(clf_rows['auc']):.4f}  "
                  f"macro_f1={clf_rows['macro_f1'].mean():.4f}", flush=True)

        # What evaluation.py::validation() itself writes.
        upstream_out = os.path.join('.', 'saved_log', 'evaluation', 'DGOT', f'exp{k}')
        os.makedirs(upstream_out, exist_ok=True)
        pd.concat(upstream_frames, ignore_index=True).to_csv(
            os.path.join(upstream_out, f'{dataset_name}.csv'))

    return rows, y_test, rf_pred


def _aggregate(runs):
    fold_means = runs.groupby('fold')[METRICS].mean()
    summary = pd.DataFrame({
        'mean_across_folds': fold_means.mean(axis=0),
        'std_across_folds': fold_means.std(axis=0),
    })
    per_classifier_fold = runs.groupby(['classifier', 'fold'])[METRICS].mean()
    per_classifier = per_classifier_fold.groupby(level='classifier').agg(['mean', 'std'])
    per_classifier.columns = [f'{metric}_{stat}' for metric, stat in per_classifier.columns]
    return fold_means, summary, per_classifier


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def run_dgot_baseline(data, dataset_name, results_file, categorical_cols=None,
                      num_epoch=PAPER_NUM_EPOCH, batch_size=PAPER_BATCH_SIZE,
                      repeats=PAPER_EVAL_REPEATS, ch_mult=None, resume=True):
    """Runs the published DGOT pipeline over 5 folds and writes the results.

    data              (X_df, y_series) from a base_functions get_*_data().
    dataset_name      folder name used inside DGOT_BASELINE/datasets.
    results_file      output path prefix, without extension.
    categorical_cols  names of true categorical columns in data[0] (abalone:
                      ['Sex']); None for purely numeric datasets.
    num_epoch, batch_size, repeats, ch_mult
                      paper defaults (ch_mult None = [1, 2, 2]); anything else
                      is logged as a deviation.
    resume            reuse cached finished folds (see module docstring).
    """
    if dataset_name in _UPSTREAM_EXAMPLE_DATASETS:
        raise ValueError(f"'{dataset_name}' is upstream's bundled example; pick another name")
    if num_epoch < _MIN_EPOCHS_FOR_CHECKPOINT:
        raise ValueError(f"num_epoch={num_epoch}: train.py saves no model below "
                         f"{_MIN_EPOCHS_FOR_CHECKPOINT} epochs")

    results_file = os.path.abspath(results_file)
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    start = time.time()

    deviations = []
    if num_epoch != PAPER_NUM_EPOCH:
        deviations.append(f'num_epoch={num_epoch} (paper {PAPER_NUM_EPOCH})')
    if batch_size != PAPER_BATCH_SIZE:
        deviations.append(f'batch_size={batch_size} (paper {PAPER_BATCH_SIZE})')
    if repeats != PAPER_EVAL_REPEATS:
        deviations.append(f'repeats={repeats} (paper {PAPER_EVAL_REPEATS})')
    ch_mult = list(PAPER_CH_MULT if ch_mult is None else ch_mult)
    if ch_mult != PAPER_CH_MULT:
        deviations.append(f'ch_mult={ch_mult} (paper {PAPER_CH_MULT})')

    print(f"=== DGOT baseline: {dataset_name} (upstream {UPSTREAM_COMMIT[:7]}, device={device}) ===")
    if deviations:
        print("  DEVIATIONS FROM THE PAPER DEFAULTS: " + '; '.join(deviations))

    feature_len, n_rows, data_sha1 = _write_upstream_csv(data, dataset_name, categorical_cols)
    with _inside_dgot():
        _upstream()['dataprocessing'].Default_processing(dataset_name)
    print(f"  {n_rows} rows, feature_len={feature_len}, data sha1={data_sha1[:12]}")
    min_features = 2 ** len(ch_mult)
    if feature_len < min_features:
        raise ValueError(
            f"{dataset_name}: {feature_len} features, but DGOT's generator with "
            f"ch_mult={ch_mult} downsamples the feature axis {len(ch_mult)} times and "
            f"needs at least {min_features}. Pass a shorter ch_mult, e.g. "
            f"ch_mult={ch_mult[:-1]}.")

    cache_key = {
        'upstream_commit': UPSTREAM_COMMIT, 'data_sha1': data_sha1,
        'num_epoch': num_epoch, 'batch_size': batch_size, 'repeats': repeats,
    }
    if ch_mult != PAPER_CH_MULT:  # keeps existing caches valid for default runs
        cache_key['ch_mult'] = ch_mult
    cache_dir = os.path.join(DGOT_PATH, 'saved_log', 'runner', dataset_name)
    os.makedirs(cache_dir, exist_ok=True)

    all_rows, true_values, predicted_values, fold_info, warnings = [], [], [], [], []

    for k in range(N_FOLDS):
        y_train, y_test = _fold_shapes(dataset_name, k)
        steps_per_epoch = len(y_train) // batch_size  # DataLoader(drop_last=True)
        print(f"\n--- {dataset_name} exp{k}: train={len(y_train)} "
              f"(minority {int((y_train == 1).sum())}), test={len(y_test)} "
              f"(minority {int((y_test == 1).sum())}), steps/epoch={steps_per_epoch} ---", flush=True)
        if steps_per_epoch == 0:
            message = (f"exp{k}: {len(y_train)} training rows < batch_size {batch_size} with "
                       f"drop_last=True, so DGOT makes NO training updates; the generator "
                       f"stays at its random initialisation")
            print("  WARNING: " + message, flush=True)
            warnings.append(message)
        if len(np.unique(y_test)) < 2:
            message = f"exp{k}: test fold has a single class; AUC is undefined (NaN)"
            print("  WARNING: " + message, flush=True)
            warnings.append(message)

        cache_file = os.path.join(cache_dir, f'exp{k}.pkl')
        cached = None
        if resume and os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                cached = pickle.load(f)
            if cached.get('key') != cache_key:
                print(f"  cached result in {cache_file} was made with other settings or "
                      f"data; recomputing", flush=True)
                cached = None

        if cached is not None:
            print(f"  reusing cached fold result {cache_file} "
                  f"(delete it or pass resume=False to recompute)", flush=True)
            fold = cached
        else:
            t0 = time.time()
            _train_fold(dataset_name, k, feature_len, num_epoch, batch_size, ch_mult, device)
            train_seconds = time.time() - t0
            t0 = time.time()
            rows, y_true, rf_pred = _evaluate_fold(dataset_name, k, repeats, device)
            fold = {'key': cache_key, 'rows': rows, 'y_test': y_true, 'rf_pred': rf_pred,
                    'train_seconds': train_seconds, 'eval_seconds': time.time() - t0}
            with open(cache_file, 'wb') as f:
                pickle.dump(fold, f)

        fold_df = pd.DataFrame(fold['rows'])
        print(f"  fold {k + 1}/{N_FOLDS}  gmean={fold_df['gmean'].mean():.4f}  "
              f"auc={np.nanmean(fold_df['auc']):.4f}  "
              f"(train {fold['train_seconds']:.0f}s, eval {fold['eval_seconds']:.0f}s)", flush=True)

        all_rows.extend(fold['rows'])
        true_values.append(np.asarray(fold['y_test']).astype(int))
        predicted_values.append(np.asarray(fold['rf_pred']).astype(int))
        fold_info.append({'exp': k, 'n_train': int(len(y_train)), 'n_test': int(len(y_test)),
                          'steps_per_epoch': int(steps_per_epoch),
                          'train_seconds': round(fold['train_seconds'], 1),
                          'eval_seconds': round(fold['eval_seconds'], 1)})

    runs = pd.DataFrame(all_rows)
    fold_means, summary, per_classifier = _aggregate(runs)

    runs.to_csv(results_file + '_runs.csv', index=False)
    fold_means.to_csv(results_file + '_folds.csv')
    summary.to_csv(results_file + '_summary.csv')
    per_classifier.to_csv(results_file + '_per_classifier.csv')

    fitness = float(fold_means['gmean'].mean())
    result = {'fitness': fitness, 'true_values': true_values, 'predicted_values': predicted_values}
    with open(results_file + '.txt', 'w') as f:
        f.write(str(result))

    info = {
        'method': 'DGOT (published pipeline, no hyperparameter search)',
        'upstream': 'https://github.com/MLneuers/DGOT', 'upstream_commit': UPSTREAM_COMMIT,
        'dataset': dataset_name, 'n_rows': n_rows, 'feature_len': feature_len,
        'categorical_cols': categorical_cols or [], 'data_sha1': data_sha1,
        'num_epoch': num_epoch, 'batch_size': batch_size, 'repeats': repeats,
        'ch_mult': ch_mult, 'oversample_rate': PAPER_OVERSAMPLE_RATE, 'eval_seed': PAPER_EVAL_SEED,
        'device': device, 'deviations_from_paper': deviations, 'warnings': warnings,
        'folds': fold_info, 'total_seconds': round(time.time() - start, 1),
        'fitness_definition': 'mean G-mean over 5 classifiers x repeats, averaged over 5 folds',
    }
    with open(results_file + '_run_info.json', 'w') as f:
        json.dump(info, f, indent=2)

    print(f"\n=== {dataset_name}: across folds ===")
    print(summary.round(4))
    print(per_classifier[['gmean_mean', 'gmean_std', 'auc_mean', 'auc_std']].round(4))
    print(f"Saved {results_file}.txt and _summary/_folds/_per_classifier/_runs.csv, _run_info.json")
    print(f"evaluated fitness: {fitness:.6f}")
    print(f"--- {dataset_name}: {time.time() - start:.0f} seconds ---", flush=True)
    return result
