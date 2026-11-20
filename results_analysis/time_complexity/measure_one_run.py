"""ONE timed run of ONE method on ONE dataset, in a fresh process.

    python measure_one_run.py --method ga_heso --dataset abalone19 --run 3 [--warmup]

Normally started by run_timing_experiment.py, which calls this script once per run,
so every measurement starts from a cold interpreter: no warm caches, no leftover
GPU memory, no state shared between runs.

Writes raw/<method>/<dataset>/run_<NN>.json with

  wall_seconds         whole pipeline, 5 folds, end to end (data already loaded)
  phases               seconds per phase, summed over folds:
                         GA methods : setup (GA-HESO MeanShift bandwidths, once),
                                      preprocess, resample, fit, predict
                         SOTA       : what the runner reports (train / eval / ...)
  train_seconds        everything needed to obtain the trained model(s)
  inference_seconds    predicting the test folds (GA methods)
  folds                the same per fold (+ sizes after resampling, G-mean)
  resources            peak RAM, CPU seconds, peak GPU memory, GPU utilisation
  model                ensemble size, #parameters, epochs actually trained
  ga_search_estimate   GA methods: projected cost of the paper's GA search
  gmean                sanity check that the measured pipeline is the real one
  environment          host, CPU, GPU, library versions, thread settings

GA methods retrain the best chromosome from the paper's GA checkpoint with the same
seeds, folds and code as the GA fitness function, but with the folds run one after
another (GA-HESO's GA ran its 5 folds with joblib n_jobs=5) so each fold has its own
time. The fold-parallel time is reported as well.
"""
import argparse
import datetime
import importlib
import json
import os
import platform
import socket
import sys
import time
import traceback
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import timing_config as cfg  # noqa: E402

if cfg.REPO_ROOT not in sys.path:
    sys.path.insert(0, cfg.REPO_ROOT)

from resource_monitor import ResourceMonitor  # noqa: E402


# --------------------------------------------------------------------------- helpers
def _sync():
    """CUDA kernels run asynchronously; wait for them before reading the clock."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


class _Phases:
    """Accumulates wall-clock seconds per named phase."""

    def __init__(self):
        self.seconds = {}

    def time(self, name):
        phases = self

        class _Ctx:
            def __enter__(self):
                _sync()
                self.t0 = time.perf_counter()

            def __exit__(self, *exc):
                _sync()
                phases.seconds[name] = phases.seconds.get(name, 0.0) + time.perf_counter() - self.t0
                return False
        return _Ctx()


def _environment():
    env = {
        'host': socket.gethostname(),
        'platform': platform.platform(),
        'python': platform.python_version(),
        'cpu_count_logical': os.cpu_count(),
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        'slurm_cpus_per_task': os.environ.get('SLURM_CPUS_PER_TASK'),
        'omp_num_threads': os.environ.get('OMP_NUM_THREADS'),
    }
    cpu = platform.processor()
    try:
        with open('/proc/cpuinfo') as f:
            cpu = next(line.split(':', 1)[1].strip() for line in f if line.startswith('model name'))
    except (OSError, StopIteration):
        pass
    env['cpu_model'] = cpu
    try:
        import psutil
        env['ram_total_gb'] = round(psutil.virtual_memory().total / 2 ** 30, 1)
        env['cpu_count_physical'] = psutil.cpu_count(logical=False)
        env['cpu_affinity'] = len(psutil.Process().cpu_affinity()) if hasattr(psutil.Process(), 'cpu_affinity') else None
    except Exception:
        pass
    for mod in ('torch', 'sklearn', 'imblearn', 'xgboost', 'pytorch_tabnet', 'imbalanced_ensemble', 'pygad', 'numpy'):
        try:
            env[mod] = importlib.import_module(mod).__version__
        except Exception:
            env[mod] = None
    try:
        import torch
        env['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env['gpu_name'] = torch.cuda.get_device_name(0)
            env['cuda_version'] = torch.version.cuda
            env['torch_threads'] = torch.get_num_threads()
    except ImportError:
        pass
    return env


def _load_data(dataset):
    import base_functions
    X, y = getattr(base_functions, cfg.DATASETS[dataset]['getter'])()
    y = y.astype(int)
    return X, y


def _gmean(y_true, y_pred):
    from imblearn.metrics import geometric_mean_score
    return float(geometric_mean_score(np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)))


def _tabnet_info(model):
    info = {}
    try:
        info['parameters'] = int(sum(p.numel() for p in model.network.parameters()))
    except Exception:
        pass
    try:
        info['epochs_trained'] = len(model.history['loss'])
    except Exception:
        pass
    return info


# ----------------------------------------------------------------- GA checkpoints
def _load_checkpoint(method, dataset):
    """(ga_instance, tuner, solution the paper reported) from the GA checkpoint."""
    from pygad import pygad
    stem = os.path.join(cfg.REPO_ROOT, cfg.DATASETS[dataset]['checkpoints'][method])
    if not os.path.exists(stem + '.pkl'):
        raise FileNotFoundError(stem + '.pkl')
    ga = pygad.load(stem)
    tuner = ga.fitness_func.__self__
    if method == 'ga_heso':
        # GA-HESO's on_stop reports the best chromosome over the whole run
        from models.oc_bagging_tabnet_ensemble_parallel import _best_overall_solution
        solution = np.asarray(_best_overall_solution(ga))
    else:
        # GaTuner / GaXGBoostTuner on_stop report best_solutions[-1]
        solution = _typed_solution(ga, ga.best_solutions[-1])
    return stem, ga, tuner, solution


def _typed_solution(ga, solution):
    """Applies each gene's declared type, as pygad does before calling the fitness
    function. Some checkpoints store best_solutions as a float array, so int genes
    such as n_estimators come back as 159.0, which the classifiers reject."""
    n = len(solution)
    if getattr(ga, 'gene_type_single', True):
        types = [ga.gene_type] * n
    else:
        types = list(ga.gene_type)
    out = []
    for value, gt in zip(solution, types):
        t = gt[0] if isinstance(gt, (list, tuple)) else gt
        if isinstance(t, type) and issubclass(t, (int, np.integer)):
            out.append(int(float(value)))  # truncation, like pygad's gene_type cast and int() in eval_func
        else:
            out.append(float(value))
    return out


def _ga_search_estimate(ga, fitness_eval_seconds, fitness_eval_seconds_parallel=None):
    """Projected wall time of the GA search the paper ran.

    nominal_evaluations : sol_per_pop for the initial population plus, per generation,
                          every individual that is not carried over with a known fitness
                          (keep_elitism elites / keep_parents parents), plus on_stop's
                          final re-evaluation.
    distinct_evaluations: distinct chromosomes in ga.solutions (save_solutions=True);
                          pygad reuses the fitness of a chromosome it has already seen,
                          so this is closer to what was actually computed.
    """
    pop = int(ga.sol_per_pop)
    gens = int(ga.generations_completed)
    keep_elitism = int(getattr(ga, 'keep_elitism', 1) or 0)
    if keep_elitism > 0:
        carried = keep_elitism
    else:
        keep_parents = int(getattr(ga, 'keep_parents', -1))
        carried = int(ga.num_parents_mating) if keep_parents == -1 else keep_parents
    nominal = pop + gens * (pop - carried) + 1

    distinct = None
    sols = getattr(ga, 'solutions', None)
    if sols is not None and len(sols) > 0:
        distinct = int(len(np.unique(np.asarray(sols, dtype=float), axis=0))) + 1

    out = {'population': pop, 'generations': gens, 'carried_over_per_generation': carried,
           'nominal_evaluations': nominal, 'distinct_evaluations': distinct,
           'fitness_eval_seconds_sequential_folds': fitness_eval_seconds,
           'estimated_seconds_nominal': fitness_eval_seconds * nominal,
           'estimated_seconds_distinct': None if distinct is None else fitness_eval_seconds * distinct}
    if fitness_eval_seconds_parallel is not None:
        out['fitness_eval_seconds_parallel_folds'] = fitness_eval_seconds_parallel
        out['estimated_seconds_nominal_parallel_folds'] = fitness_eval_seconds_parallel * nominal
        if distinct is not None:
            out['estimated_seconds_distinct_parallel_folds'] = fitness_eval_seconds_parallel * distinct
    return out


# ------------------------------------------------------------------------ GA-HESO
def _ga_heso_selected(tuner, solution, index):
    """Cluster-selection genes of fold `index`; same slicing as parallel_fit."""
    cp = tuner.clustering_params
    if cp is None:
        return solution[index * tuner.cluster_count:(index + 1) * tuner.cluster_count]
    n_cfg = len(tuner.config_files)
    starts = np.cumsum([0] + list(cp['clusters'][:-1])) + n_cfg
    ends = np.cumsum(cp['clusters']) + n_cfg
    return solution[starts[index]:ends[index]]


def run_ga_heso(dataset, phases):
    from recalculate_ga_heso_probabilities import _prepare_tuner
    from base_functions import custom_resample_minority_clusters, get_preprocessor, get_meanshift_cluster_counts
    from models.oc_bagging_tabnet_ensemble import OCBaggingTabnetEnsemble
    from optimization.ga_tabnet_functions import GMean

    stem, ga, tuner, solution = _load_checkpoint('ga_heso', dataset)
    tuner = _prepare_tuner(tuner, 'cuda')  # resolves config paths; refuses CPU (losses are CUDA-only)
    n_cfg = len(tuner.config_files)
    members = [c for c, on in zip(tuner.config_files, solution[:n_cfg]) if on]
    X, y = tuner.X_orig, tuner.y_orig

    # One-off before the GA: MeanShift bandwidth / cluster-count estimation per fold.
    # Its result is already in the checkpoint; it is recomputed only to time it.
    if tuner.clustering_params is not None and tuner.clustering_params.get('type') == 'MS':
        with phases.time('setup'):
            get_meanshift_cluster_counts(X, y, tuner.numerical_cols, tuner.categorical_cols,
                                         tuner.resampling_algorithm)

    folds, models_info = [], []
    for index, (train_index, test_index) in enumerate(zip(tuner.train_indices, tuner.test_indices)):
        fp = _Phases()
        X_train, X_valid = X.iloc[train_index], X.iloc[test_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[test_index]
        ensemble = OCBaggingTabnetEnsemble(members, solution[n_cfg:], tuner.device)

        with fp.time('preprocess'):
            preprocessor = get_preprocessor(tuner.numerical_cols, tuner.categorical_cols)
            X_train_std = preprocessor.fit_transform(X_train)
        cls_sum = np.sum(y_train)
        cls_num_list = [len(y_train) - cls_sum, cls_sum]
        with fp.time('resample'):
            X_res, y_res = custom_resample_minority_clusters(
                X_train_std, y_train, _ga_heso_selected(tuner, solution, index),
                cluster_count=tuner.cluster_count,
                syntetic_minority_count=tuner.synthetic_minority_count,
                resampling_algorithm=tuner.resampling_algorithm,
                clustering_algorithm=tuner.clustering_params['algs'][index],
                use_cluster_centers=tuner.use_cluster_centers)
        with fp.time('fit'):
            ensemble.fit(X_res, y_res, solution=solution, cls_num_list=cls_num_list, eval_metric=[GMean],
                         max_epochs=tuner.tabnet_max_epochs, patience=100, batch_size=5000, drop_last=False)
        with fp.time('predict'):
            y_pred = np.array(ensemble.predict(preprocessor.transform(X_valid)))

        member_info = [_tabnet_info(m) for m in ensemble.models]
        models_info.append(member_info)
        folds.append({'fold': index, 'n_train': len(train_index), 'n_train_resampled': int(len(y_res)),
                      'n_test': len(test_index), 'gmean': _gmean(y_valid, y_pred),
                      'seconds': fp.seconds,
                      'epochs_trained': [m.get('epochs_trained') for m in member_info]})
        for k, v in fp.seconds.items():
            phases.seconds[k] = phases.seconds.get(k, 0.0) + v

    params = [sum(m.get('parameters', 0) for m in fold_models) for fold_models in models_info]
    model = {'ensemble_members': len(members), 'config_genes': n_cfg,
             'parameters_per_fold_ensemble': int(np.mean(params)) if params else None,
             'tabnet_max_epochs': tuner.tabnet_max_epochs,
             'synthetic_minority_count': tuner.synthetic_minority_count}
    return stem, ga, folds, model


# ------------------------------------------------------------- GA-tuned baselines
def _baseline_resampler(method, tuner):
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.combine import SMOTEENN
    if method == 'smote_xgboost':  # GaXGBoostTuner: fixed k_neighbors=2
        if getattr(tuner, 'use_smote', False):
            return SMOTE(random_state=42, k_neighbors=2)
        if getattr(tuner, 'use_adasyn', False):
            return ADASYN(random_state=42, n_neighbors=2, sampling_strategy='all')
        if getattr(tuner, 'use_smoteenn', False):
            return SMOTEENN(random_state=42, sampling_strategy='auto')
        return None
    k = getattr(tuner, 'k_neighbors', 2)
    if getattr(tuner, 'use_smote', False):
        return SMOTE(random_state=42, k_neighbors=k)
    if getattr(tuner, 'use_adasyn', False):
        return ADASYN(random_state=42, n_neighbors=k, sampling_strategy='all')
    if getattr(tuner, 'use_smote_enn', False):
        return SMOTEENN(random_state=42, sampling_strategy='auto')
    return None


def _baseline_classifier(method, tuner, solution):
    if method == 'smote_xgboost':  # same construction as GaXGBoostTuner.eval_func
        from xgboost import XGBClassifier
        clf = XGBClassifier(random_state=42, learning_rate=float(solution[0]), n_estimators=int(solution[1]),
                            max_depth=int(solution[2]), min_child_weight=int(solution[3]),
                            gamma=float(solution[4]), subsample=float(solution[5]),
                            colsample_bytree=float(solution[6]))
        if getattr(tuner, 'use_weighting', False):
            clf.scale_pos_weight = int(solution[7])
        return clf
    from base_functions import get_classifier
    return get_classifier(tuner.clf_type, solution)


def _baseline_model_info(method, clf):
    if method == 'smote_tabnet':
        return _tabnet_info(clf)
    if method == 'smote_xgboost':
        try:
            return {'boosting_rounds': int(clf.get_booster().num_boosted_rounds())}
        except Exception:
            return {}
    try:
        return {'estimators': len(clf.estimators_)}
    except Exception:
        return {}


def run_ga_baseline(method, dataset, phases):
    """Same steps as the imblearn Pipeline in GaTuner / GaXGBoostTuner.eval_func
    (preprocess -> resample -> classifier), called one by one so each is timed."""
    from base_functions import get_preprocessor
    stem, ga, tuner, solution = _load_checkpoint(method, dataset)
    X, y = tuner.X_orig, np.asarray(tuner.y_orig).astype(int)

    folds, infos = [], []
    for index, (train_index, test_index) in enumerate(zip(tuner.train_indices, tuner.test_indices)):
        fp = _Phases()
        X_train, X_valid = X.iloc[train_index], X.iloc[test_index]
        y_train, y_valid = y[train_index], y[test_index]
        clf = _baseline_classifier(method, tuner, solution)
        resampler = _baseline_resampler(method, tuner)

        with fp.time('preprocess'):
            preprocessor = get_preprocessor(tuner.numerical_cols, tuner.categorical_cols)
            Xt = preprocessor.fit_transform(X_train)
        yt = y_train
        if resampler is not None:
            with fp.time('resample'):
                Xt, yt = resampler.fit_resample(Xt, yt)
        with fp.time('fit'):
            clf.fit(Xt, yt)
        with fp.time('predict'):
            y_pred = np.asarray(clf.predict(preprocessor.transform(X_valid))).ravel()

        info = _baseline_model_info(method, clf)
        infos.append(info)
        folds.append({'fold': index, 'n_train': len(train_index), 'n_train_resampled': int(len(yt)),
                      'n_test': len(test_index), 'gmean': _gmean(y_valid, y_pred),
                      'seconds': fp.seconds, **{k: v for k, v in info.items() if k == 'epochs_trained'}})
        for k, v in fp.seconds.items():
            phases.seconds[k] = phases.seconds.get(k, 0.0) + v

    model = {'solution': [float(s) for s in solution]}
    for key in ('parameters', 'boosting_rounds', 'estimators'):
        vals = [i[key] for i in infos if key in i]
        if vals:
            model[key] = int(np.mean(vals))
    return stem, ga, folds, model


# ------------------------------------------------------------------------- SOTA
def _sota_phases(run_info):
    """Sums every per-fold '*_seconds' value (and GOIO's 'seconds' dict) by name."""
    sums, per_fold = {}, []
    for fold in run_info.get('folds', run_info.get('folds_run', [])) or []:
        if not isinstance(fold, dict):
            continue
        fs = {}
        for key, val in fold.items():
            if key.endswith('_seconds') and isinstance(val, (int, float)):
                fs[key[:-len('_seconds')]] = float(val)
            elif key == 'seconds' and isinstance(val, dict):
                fs.update({k: float(v) for k, v in val.items()})
            elif key == 'seconds' and isinstance(val, (int, float)):
                fs['fold_total'] = float(val)
        per_fold.append(fs)
        for k, v in fs.items():
            sums[k] = sums.get(k, 0.0) + v
    return sums, per_fold


def run_sota(method, dataset, phases):
    import pandas as pd
    m = cfg.METHODS[method]
    mod_name, fn_name = m['runner'].split(':')
    module = importlib.import_module(mod_name)
    runner = getattr(module, fn_name)

    data = _load_data(dataset)
    name = cfg.SOTA_DATASET_PREFIX + dataset
    results_file = os.path.join(cfg.WORK_DIR, method, dataset, '{}_{}'.format(method, dataset))
    kwargs = dict(categorical_cols=cfg.DATASETS[dataset]['categorical_cols'])
    kwargs.update(cfg.DATASETS[dataset].get('sota_kwargs', {}).get(method, {}))
    if m.get('resume_arg'):
        kwargs['resume'] = False  # never reuse a cached fold

    # MLOS: time the upstream oversampling call itself. The wrapper only reads the
    # clock around the original function, it does not change what it does.
    oversample = {'seconds': 0.0, 'calls': 0}
    if method == 'mlos_original':
        original = module._oversample

        def timed(*a, **kw):
            t0 = time.perf_counter()
            try:
                return original(*a, **kw)
            finally:
                oversample['seconds'] += time.perf_counter() - t0
                oversample['calls'] += 1
        module._oversample = timed

    with phases.time('pipeline'):
        runner(data, name, results_file=results_file, **kwargs)

    with open(results_file + '_run_info.json') as f:
        run_info = json.load(f)
    sums, per_fold = _sota_phases(run_info)
    if oversample['calls']:
        sums['oversample'] = oversample['seconds']
    gmean = None
    try:
        gmean = float(pd.read_csv(results_file + '_folds.csv')['gmean'].mean())
    except Exception:
        pass
    model = {k: run_info[k] for k in ('num_epoch', 'batch_size', 'repeats', 'main_py_args',
                                      'upstream_parameters', 'image_size', 'evaluation') if k in run_info}
    if oversample['calls']:
        model['oversample_calls'] = oversample['calls']
    return sums, per_fold, gmean, model


# ------------------------------------------------------------------------- main
def measure(method, dataset, run_id, warmup):
    warnings.filterwarnings('ignore')
    kind = cfg.METHODS[method]['kind']
    record = {'method': method, 'method_label': cfg.METHODS[method]['label'],
              'dataset': dataset, 'dataset_label': cfg.DATASETS[dataset]['label'],
              'run_id': run_id, 'is_warmup': warmup,
              'started_at': datetime.datetime.now().isoformat(timespec='seconds'),
              'environment': _environment()}
    X, y = _load_data(dataset)
    record.update({'n_rows': int(len(y)), 'n_features': int(X.shape[1]),
                   'n_minority': int(min(np.bincount(y))), 'imbalance_ratio': float(max(np.bincount(y)) / min(np.bincount(y)))})

    phases = _Phases()
    t0 = time.perf_counter()
    with ResourceMonitor() as mon:
        if kind == 'ga':
            if method == 'ga_heso':
                stem, ga, folds, model = run_ga_heso(dataset, phases)
            else:
                stem, ga, folds, model = run_ga_baseline(method, dataset, phases)
        else:
            sota_phases, sota_folds, gmean, model = run_sota(method, dataset, phases)
    _sync()
    wall = time.perf_counter() - t0

    record['wall_seconds'] = wall
    record['resources'] = mon.result()
    try:
        import torch
        if torch.cuda.is_available():
            record['resources']['torch_peak_allocated_mb_this_process'] = round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
    except ImportError:
        pass
    record['model'] = model

    if kind == 'ga':
        p = phases.seconds
        train = sum(p.get(k, 0.0) for k in ('setup', 'preprocess', 'resample', 'fit'))
        record.update({
            'checkpoint': os.path.relpath(stem, cfg.REPO_ROOT).replace('\\', '/'),
            'phases': p,
            'train_seconds': train,
            'inference_seconds': p.get('predict', 0.0),
            'folds': folds,
            'gmean': float(np.mean([f['gmean'] for f in folds])),
            'gmean_reported_by_ga': float(np.max(ga.best_solutions_fitness)) if method == 'ga_heso'
            else float(ga.best_solutions_fitness[-1]),
        })
        fold_totals = [sum(f['seconds'].values()) for f in folds]
        record['fold_seconds_mean'] = float(np.mean(fold_totals))
        record['fold_seconds_max'] = float(np.max(fold_totals))
        # one GA fitness evaluation = the 5 folds (setup is done once, before the GA)
        eval_seq = float(sum(fold_totals))
        eval_par = float(np.max(fold_totals)) if method == 'ga_heso' else None
        record['ga_search_estimate'] = _ga_search_estimate(ga, eval_seq, eval_par)
        if method == 'ga_heso':
            record['ga_search_estimate']['note'] = ('GA-HESO evaluated the 5 folds with joblib n_jobs=5; '
                                                    '*_parallel_folds assumes perfect fold parallelism')
            record['ga_search_estimate']['setup_seconds_once'] = p.get('setup', 0.0)
    else:
        train_keys = [k for k in sota_phases if 'train' in k or k == 'oversample' or 'fine_tune' in k]
        record.update({
            'phases': {'pipeline': phases.seconds['pipeline'], **sota_phases},
            'train_seconds': sum(sota_phases[k] for k in train_keys) if train_keys else None,
            'train_phase_keys': train_keys,
            'inference_seconds': None,
            'folds': sota_folds,
            'gmean': gmean,
            'note': 'SOTA times come from the upstream pipeline incl. its own evaluation protocol; '
                    'phases are as reported per fold by the runner (_run_info.json)',
        })
    record['status'] = 'ok'
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--method', required=True, choices=sorted(cfg.METHODS))
    parser.add_argument('--dataset', required=True, choices=sorted(cfg.DATASETS))
    parser.add_argument('--run', type=int, required=True, help='run number (0 = warm-up by convention)')
    parser.add_argument('--warmup', action='store_true', help='mark this run as warm-up (excluded from statistics)')
    args = parser.parse_args()

    out = cfg.raw_path(args.method, args.dataset, args.run)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    # the upstream SOTA code chdir()s; keep relative paths of the repo working
    os.chdir(cfg.REPO_ROOT)
    try:
        record = measure(args.method, args.dataset, args.run, args.warmup)
    except BaseException as e:  # pygad.load raises BaseException
        if isinstance(e, KeyboardInterrupt):
            raise
        record = {'method': args.method, 'dataset': args.dataset, 'run_id': args.run,
                  'is_warmup': args.warmup, 'status': 'error',
                  'error': '{}: {}'.format(type(e).__name__, e), 'traceback': traceback.format_exc()}
        print(record['traceback'], file=sys.stderr)
        with open(out + '.error', 'w') as f:
            json.dump(record, f, indent=2, default=str)
        sys.exit(1)

    tmp = out + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(record, f, indent=2, default=str)
    os.replace(tmp, out)  # a killed job never leaves a half-written result
    if os.path.exists(out + '.error'):  # stale failure from an earlier attempt
        os.remove(out + '.error')
    print('{} {} run {}: wall {:.1f}s  train {}  G-mean {}  -> {}'.format(
        args.method, args.dataset, args.run, record['wall_seconds'],
        None if record['train_seconds'] is None else round(record['train_seconds'], 1),
        None if record['gmean'] is None else round(record['gmean'], 4), out))


if __name__ == '__main__':
    main()
