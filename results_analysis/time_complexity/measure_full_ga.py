"""MEASURED cost of the whole GA search: re-runs the paper's GA from scratch, timed.
Survives SLURM time limits: a killed run continues from its last generation.

    python measure_full_ga.py --method ga_heso --dataset glass_4 [--run 1] [--seed run]
    python measure_full_ga.py --method selfpe --dataset abalone19 --dry-run    # settings + projection only
    python measure_full_ga.py ... --fresh                                      # discard a started run

The tuner is taken from the paper's GA checkpoint, so population, generations, parents,
gene space, clustering and ensemble configs are exactly the paper's; then the tuner's own
run_experiment() is called - the real GA, with all generations, every fitness evaluation
(5 folds of training each) and on_stop's final re-evaluation. The paper's checkpoints are
only read; the GA writes into work/full_ga/<method>/<dataset>/run_NN/.

Interrupted runs (time limit, node failure) - started again with the same command:
  * the GA checkpoint (saved after every generation, as in the paper runs) is loaded
    and the GA continues with the REMAINING generations only (GaTuner/GaXGBoostTuner
    on their own would run the full budget again on a loaded GA - corrected here
    without touching their code)
  * numpy/random RNG state is saved with every checkpoint and restored, so the search
    continues the way it would have (not guaranteed bit-identical: GPU training and
    any re-evaluation after loading can consume random numbers)
  * timing: each attempt ("segment") counts only up to its last checkpoint; the work
    after it was lost at the kill and is done again, so it is reported separately
    (lost_seconds), not added. Re-evaluations pygad does right after loading are a cost
    of the interruption -> resume_overhead_seconds; search_seconds_uninterrupted
    leaves them out and is the number to report.

Output raw_full_ga/<method>/<dataset>/run_NN.json:
  search_seconds_uninterrupted  the GA search as if never interrupted (report this)
  search_seconds                sum of the counted parts of all segments
  setup_seconds                 GA-HESO: MeanShift cluster estimation done before the GA
  total_seconds / total_hours   setup + search_seconds_uninterrupted
  evaluations, eval_seconds     fitness evaluations kept (pygad skips repeats) and their times
  per_generation                evaluations and seconds per generation
  curve                         best-so-far fitness and cumulative seconds per generation
                                (time-to-quality comparison with SOTA in summarize_timing.py)
  segments                      one entry per attempt (start, generations, counted, lost)
  resources                     peak RAM / GPU over all segments, CPU seconds summed
While it runs, run_NN.progress.json holds the state (it is also what a resume reads).
"""
import argparse
import datetime
import json
import os
import pickle
import random
import shutil
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

from measure_one_run import _load_checkpoint, _environment, _sync  # noqa: E402
from resource_monitor import ResourceMonitor  # noqa: E402

RAW_FULL_GA_DIR = os.path.join(HERE, 'raw_full_ga')


def out_path(method, dataset, run_id):
    return os.path.join(RAW_FULL_GA_DIR, method, dataset, 'run_{:02d}.json'.format(run_id))


def _settings(tuner, ga):
    return {'population': int(tuner.population), 'num_parents': int(tuner.num_parents),
            'num_generations_configured': int(tuner.num_generations),
            'generations_completed_in_paper_run': int(ga.generations_completed),
            'keep_elitism': int(getattr(ga, 'keep_elitism', 1) or 0),
            'tabnet_max_epochs': getattr(tuner, 'tabnet_max_epochs', None)}


def _max_classifier_count(tuner, ga):
    """GA-HESO: the paper run used the custom mutation / initial population (with a cap
    on ensemble members) iff its mutation_type is a function; recover the cap."""
    if not callable(getattr(ga, 'mutation_type', None)):
        return None
    pop = getattr(ga, 'initial_population', None)
    if pop is None:
        return None
    return int(np.sum(np.asarray(pop)[0][:len(tuner.config_files)]))


# ------------------------------------------------------------------ run state
class _State:
    """Everything a resume needs, mirrored to run_NN.progress.json after every
    evaluation and every checkpoint."""

    def __init__(self, path, data):
        self.path = path
        self.d = data
        self.seg = None
        self.t0 = None

    @classmethod
    def load_or_new(cls, path, record):
        if os.path.exists(path):
            with open(path) as f:
                return cls(path, json.load(f))
        return cls(path, dict(record, status='running', segments=[], evals=[], setup_seconds=0.0))

    def start_segment(self, generation_start):
        self.d['segments'].append({'started_at': datetime.datetime.now().isoformat(timespec='seconds'),
                                   'host': _environment().get('host'),
                                   'generation_start': generation_start,
                                   'generation_checkpoint': None, 'elapsed_at_checkpoint': None,
                                   'elapsed_last_seen': 0.0, 'end': 'running'})
        self.seg = len(self.d['segments']) - 1
        self.t0 = time.perf_counter()
        self.write()

    def elapsed(self):
        return time.perf_counter() - self.t0

    def add_eval(self, gen, seconds, fitness):
        self.d['evals'].append([self.seg, gen, seconds, fitness])
        self.d['segments'][self.seg]['elapsed_last_seen'] = self.elapsed()
        self.write()

    def checkpoint(self, gen):
        s = self.d['segments'][self.seg]
        s['generation_checkpoint'] = gen
        s['elapsed_at_checkpoint'] = s['elapsed_last_seen'] = self.elapsed()
        self.write()

    def close_killed_segments(self):
        """Segments still 'running' were killed: keep what their last checkpoint holds."""
        keep = []
        for k, s in enumerate(self.d['segments']):
            if s['end'] != 'running':
                continue
            s['end'] = 'killed'
            ck = s['generation_checkpoint']
            s['counted_seconds'] = s['elapsed_at_checkpoint'] or 0.0
            s['lost_seconds'] = s['elapsed_last_seen'] - s['counted_seconds']
            for e in self.d['evals']:
                if e[0] == k and (ck is None or e[1] > ck):
                    e.append('lost')
        keep = [e for e in self.d['evals'] if len(e) == 4]
        self.d['lost_evals'] = self.d.get('lost_evals', 0) + len(self.d['evals']) - len(keep)
        self.d['evals'] = keep
        self.write()

    def write(self):
        self.d['updated_at'] = datetime.datetime.now().isoformat(timespec='seconds')
        tmp = self.path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(self.d, f, default=str)
        os.replace(tmp, self.path)


_STATE = None
_FITNESS = None
_FNAME = None
_GENERATIONS = None
_SEED = None
_ORIGINAL = {}


def _timed_fitness(ga_instance, solution, solution_idx):
    """A plain 3-argument module-level function: pygad checks the argument count of
    fitness_func and pickles it with the GA every generation (by reference, because
    this module is importable)."""
    gen = int(getattr(ga_instance, 'generations_completed', 0))
    _sync()
    t0 = time.perf_counter()
    fit = _FITNESS(ga_instance, solution, solution_idx)
    _sync()
    _STATE.add_eval(gen, time.perf_counter() - t0, float(np.ravel(fit)[0]))
    return fit


def _rng_path():
    return _FNAME + '.rng.pkl'


def _save_with_state(self, filename):
    """pygad.GA.save + RNG state + checkpoint time (called after every generation)."""
    _ORIGINAL['save'](self, filename)
    if _STATE is not None and os.path.abspath(filename) == os.path.abspath(_FNAME):
        tmp = _rng_path() + '.tmp'
        with open(tmp, 'wb') as f:
            pickle.dump({'np': np.random.get_state(), 'py': random.getstate(),
                         'generation': int(self.generations_completed)}, f)
        os.replace(tmp, _rng_path())
        _STATE.checkpoint(int(self.generations_completed))


def _load_for_resume(filename):
    """pygad.load as the tuners call it on resume, plus: our timing fitness function,
    only the REMAINING generations, and the RNG state of that checkpoint."""
    ga = _ORIGINAL['load'](filename)
    if _STATE is not None and os.path.abspath(filename) == os.path.abspath(_FNAME):
        ga.fitness_func = _timed_fitness
        ga.num_generations = max(_GENERATIONS - int(ga.generations_completed), 0)
        if os.path.exists(_rng_path()):
            with open(_rng_path(), 'rb') as f:
                rng = pickle.load(f)
            if rng.get('generation') == int(ga.generations_completed):
                np.random.set_state(rng['np'])
                random.setstate(rng['py'])
        print('resuming at generation {}/{} ({} to go)'.format(ga.generations_completed, _GENERATIONS,
                                                              ga.num_generations), flush=True)
    return ga


class SeededGA:  # replaced by _install_hooks; module-level name so pickles resolve
    pass


def _install_hooks(seed):
    """Patches pygad.pygad (which all three tuners use as `pygad`) - the tuner code is
    untouched: GA.save also stores RNG state/time, load resumes correctly, and with a
    seed GA(...) gets random_seed=seed instead of the paper's 42."""
    global SeededGA
    from pygad import pygad as pg
    if 'GA' not in _ORIGINAL:
        _ORIGINAL.update(GA=pg.GA, save=pg.GA.save, load=pg.load)
    pg.GA.save = _ORIGINAL['GA'].save = _save_with_state
    pg.load = _load_for_resume
    if seed is not None:
        base = _ORIGINAL['GA']

        class _Seeded(base):
            def __init__(self, *a, **kw):
                kw['random_seed'] = _SEED
                super().__init__(*a, **kw)
        _Seeded.__name__ = _Seeded.__qualname__ = 'SeededGA'
        _Seeded.__module__ = __name__
        SeededGA = _Seeded
        pg.GA = _Seeded


def _per_generation(evals):
    out = {}
    for _, gen, sec, _ in evals:
        g = out.setdefault(str(gen), {'evaluations': 0, 'seconds': 0.0})
        g['evaluations'] += 1
        g['seconds'] += sec
    return out


def _curve(kept, setup, search_uninterrupted):
    """Best-so-far fitness and cumulative time at the end of every generation.
    Time = setup + evaluation seconds up to that generation, scaled so the last point
    equals the whole search (spreads pygad's own work and checkpoint saving evenly)."""
    by_gen = {}
    for _, gen, sec, fit in kept:
        g = by_gen.setdefault(gen, [0.0, -np.inf])
        g[0] += sec
        g[1] = max(g[1], fit)
    total_eval = sum(v[0] for v in by_gen.values())
    scale = search_uninterrupted / total_eval if total_eval > 0 else 1.0
    out, cum, best = [], setup, -np.inf
    for gen in sorted(by_gen):
        cum += by_gen[gen][0] * scale
        best = max(best, by_gen[gen][1])
        out.append({'generation': int(gen), 'best_so_far': float(best), 'cumulative_seconds': float(cum)})
    return out


def _merge_resources(segments):
    res = [s['resources'] for s in segments if s.get('resources')]
    if not res:
        return {}
    out = {'peak_rss_mb': max(r.get('peak_rss_mb') or 0 for r in res),
           'cpu_seconds': sum(r.get('cpu_seconds') or 0 for r in res)}
    gpu = [r['peak_gpu_mb'] for r in res if r.get('peak_gpu_mb') is not None]
    out['peak_gpu_mb'] = max(gpu) if gpu else None
    util = [r['gpu_util_mean'] for r in res if r.get('gpu_util_mean') is not None]
    out['gpu_util_mean'] = float(np.mean(util)) if util else None
    out['note'] = 'cpu_seconds include killed segments' if len(res) > 1 else None
    return out


# ------------------------------------------------------------------------ main
def measure(method, dataset, run_id, workdir, seed=None):
    import pandas as pd
    global _STATE, _FITNESS, _FNAME, _GENERATIONS, _SEED

    stem, ga, tuner, _ = _load_checkpoint(method, dataset)
    record = {'method': method, 'method_label': cfg.METHODS[method]['label'], 'dataset': dataset,
              'dataset_label': cfg.DATASETS[dataset]['label'], 'run_id': run_id,
              'paper_checkpoint': os.path.relpath(stem, cfg.REPO_ROOT).replace('\\', '/'),
              'started_at': datetime.datetime.now().isoformat(timespec='seconds'),
              'environment': _environment(), 'ga_settings': _settings(tuner, ga)}
    data = (tuner.X_orig.copy(), tuner.y_orig.copy() if isinstance(tuner.y_orig, pd.Series) else tuner.y_orig)
    record.update({'n_rows': int(len(data[1])), 'n_features': int(data[0].shape[1])})

    _FNAME = os.path.join(workdir, os.path.basename(stem)).replace('\\', '/')
    resuming = os.path.exists(_FNAME + '.pkl')
    _STATE = _State.load_or_new(out_path(method, dataset, run_id).replace('.json', '.progress.json'), record)
    if resuming:
        _STATE.close_killed_segments()

    kwargs = {}
    if method == 'ga_heso':
        from recalculate_ga_heso_probabilities import _prepare_tuner
        from base_functions import get_meanshift_cluster_counts
        tuner = _prepare_tuner(tuner, 'cuda')
        # The paper's launch script estimated MeanShift clusters before the GA; time it
        # once (first segment), but keep the checkpoint's clustering so the gene layout
        # is identical.
        if (not resuming and tuner.clustering_params is not None
                and tuner.clustering_params.get('type') == 'MS'):
            _sync()
            t0 = time.perf_counter()
            get_meanshift_cluster_counts(data[0], data[1], tuner.numerical_cols, tuner.categorical_cols,
                                         tuner.resampling_algorithm)
            _STATE.d['setup_seconds'] = time.perf_counter() - t0
        mcc = _max_classifier_count(tuner, ga)
        if mcc is not None:
            kwargs['max_classifier_count'] = mcc
        _STATE.d['ga_settings']['max_classifier_count'] = mcc
        tuner.best_fitness_so_far = -np.inf
    if hasattr(tuner, 'save_partial_output'):  # never write the paper run's result files
        tuner.save_partial_output = False
    del ga

    # Seed: None = the paper's random_seed=42 (all runs repeat the same search); an int
    # replaces it. GA-HESO's custom initial population draws from numpy's global RNG
    # before pygad seeds it, like the paper's launch script, which seeded with 42 first.
    effective = 42 if seed is None else int(seed)
    _SEED = effective
    _STATE.d['seed'] = effective
    _STATE.d['seed_source'] = 'paper (42)' if seed is None else '--seed'
    if not resuming:
        np.random.seed(effective)
        random.seed(effective)
    _GENERATIONS = int(tuner.num_generations)
    _install_hooks(seed)
    _FITNESS = tuner.fitness_func
    tuner.fitness_func = _timed_fitness  # run_experiment hands self.fitness_func to pygad

    gen_start = 0
    if resuming:
        with open(_rng_path(), 'rb') if os.path.exists(_rng_path()) else open(os.devnull, 'rb') as f:
            try:
                gen_start = int(pickle.load(f)['generation'])
            except Exception:
                gen_start = None
    _STATE.start_segment(gen_start)
    _sync()
    with ResourceMonitor(interval=2.0) as mon:
        tuner.run_experiment(data, _FNAME, **kwargs)
    _sync()

    seg = _STATE.d['segments'][_STATE.seg]
    seg.update(end='finished', counted_seconds=_STATE.elapsed(), lost_seconds=0.0, resources=mon.result())
    for k, s in enumerate(_STATE.d['segments']):
        s.setdefault('resources', None)
    segments = _STATE.d['segments']
    evals = _STATE.d['evals']

    # re-evaluations pygad does right after loading a checkpoint (labelled with the
    # generation the segment started at) exist only because the run was interrupted
    resume_overhead = sum(e[2] for e in evals if e[0] > 0 and e[1] == segments[e[0]]['generation_start'])
    search = sum(s['counted_seconds'] for s in segments)
    uninterrupted = search - resume_overhead
    kept = [e for e in evals if not (e[0] > 0 and e[1] == segments[e[0]]['generation_start'])]
    secs = np.array([e[2] for e in kept]) if kept else np.array([0.0])
    setup = float(_STATE.d.get('setup_seconds') or 0.0)

    best = None
    try:
        best = float(np.max(_ORIGINAL['load'](_FNAME).best_solutions_fitness))
    except BaseException:
        pass
    rec = {k: v for k, v in _STATE.d.items() if k not in ('evals', 'status')}
    rec.update({
        'status': 'ok', 'finished_at': datetime.datetime.now().isoformat(timespec='seconds'),
        'search_seconds_uninterrupted': uninterrupted, 'search_seconds': search,
        'resume_overhead_seconds': resume_overhead,
        'lost_seconds': sum(s.get('lost_seconds') or 0.0 for s in segments),
        'n_segments': len(segments), 'interrupted': len(segments) > 1,
        'setup_seconds': setup,
        'search_hours': uninterrupted / 3600,
        'total_seconds': uninterrupted + setup, 'total_hours': (uninterrupted + setup) / 3600,
        'evaluations': len(kept),
        'eval_seconds': {'mean': float(secs.mean()), 'median': float(np.median(secs)), 'min': float(secs.min()),
                         'max': float(secs.max()), 'sum': float(secs.sum()),
                         'std': float(secs.std(ddof=1)) if len(secs) > 1 else 0.0},
        'overhead_seconds': uninterrupted - float(secs.sum()),
        'per_generation': _per_generation(kept),
        'curve': _curve(kept, setup, uninterrupted),
        'resources': _merge_resources(segments),
        'best_fitness': best,
        'note': 'evaluations = calls of fitness_func (pygad skips chromosomes it has seen). on_stop final '
                're-evaluation calls eval_func directly, so its time is in overhead_seconds, as is pygad '
                'own work and saving the checkpoint every generation',
    })
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--method', required=True, choices=[m for m, v in cfg.METHODS.items() if v['kind'] == 'ga'])
    ap.add_argument('--dataset', required=True, choices=sorted(cfg.DATASETS))
    ap.add_argument('--run', type=int, default=1)
    ap.add_argument('--seed', default=None,
                    help="GA random seed: an int, 'run' (= 1000 + run number, a different search per run), "
                         "or omitted = the paper's 42 (every run repeats the same search)")
    ap.add_argument('--fresh', action='store_true', help='discard a started (interrupted) run and start over')
    ap.add_argument('--dry-run', action='store_true', help='print settings and the projected time, do not run')
    args = ap.parse_args()
    warnings.filterwarnings('ignore')
    os.chdir(cfg.REPO_ROOT)

    out = out_path(args.method, args.dataset, args.run)
    if os.path.exists(out):
        print('already measured: ' + out)
        return 0
    if args.dry_run:
        stem, ga, tuner, _ = _load_checkpoint(args.method, args.dataset)
        print(json.dumps(_settings(tuner, ga), indent=2))
        timed = os.path.join(cfg.RAW_DIR, args.method, args.dataset, 'run_01.json')
        if os.path.exists(timed):
            with open(timed) as f:
                est = json.load(f).get('ga_search_estimate') or {}
            for k in ('nominal_evaluations', 'distinct_evaluations', 'estimated_seconds_distinct',
                      'estimated_seconds_distinct_parallel_folds', 'estimated_seconds_nominal'):
                if est.get(k) is not None:
                    v = est[k]
                    print('{:45s} {}'.format(k, '{:.1f} h'.format(v / 3600) if 'seconds' in k else v))
        else:
            print('(no timing run yet - run measure_one_run.py first for a projected time)')
        return 0

    workdir = os.path.join(cfg.WORK_DIR, 'full_ga', args.method, args.dataset, 'run_{:02d}'.format(args.run))
    progress = out.replace('.json', '.progress.json')
    has_pkl = os.path.isdir(workdir) and any(f.endswith('.pkl') and not f.endswith('.rng.pkl')
                                             for f in os.listdir(workdir))
    if args.fresh or (has_pkl and not os.path.exists(progress)) or (not has_pkl and os.path.exists(progress)):
        # --fresh, or state that does not belong together (checkpoint without timing
        # record, or a run killed before its first checkpoint): start over cleanly
        if os.path.isdir(workdir) or os.path.exists(progress):
            print('starting run {} from scratch'.format(args.run))
        shutil.rmtree(workdir, ignore_errors=True)
        if os.path.exists(progress):
            os.remove(progress)
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    try:
        seed = None if args.seed is None else (1000 + args.run if args.seed == 'run' else int(args.seed))
        record = measure(args.method, args.dataset, args.run, workdir, seed)
    except BaseException as e:  # pygad.load raises BaseException
        if isinstance(e, KeyboardInterrupt):
            raise
        print(traceback.format_exc(), file=sys.stderr)
        with open(out + '.error', 'w') as f:
            json.dump({'method': args.method, 'dataset': args.dataset, 'run_id': args.run, 'status': 'error',
                       'error': '{}: {}'.format(type(e).__name__, e), 'traceback': traceback.format_exc()},
                      f, indent=2)
        return 1

    tmp = out + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(record, f, indent=2, default=str)
    os.replace(tmp, out)
    for stale in (out + '.error', progress):
        if os.path.exists(stale):
            os.remove(stale)
    print('{} {} run {}: search {:.2f} h in {} segment(s), {} evaluations, mean {:.1f} s/eval -> {}'.format(
        args.method, args.dataset, args.run, record['search_hours'], record['n_segments'],
        record['evaluations'], record['eval_seconds']['mean'], out))
    return 0


if __name__ == '__main__':
    # Run as the importable module `measure_full_ga`, not as __main__: the GA pickles
    # our fitness function / GA class every generation, and a resumed process must
    # find them again under the same module name.
    import measure_full_ga
    sys.exit(measure_full_ga.main())
