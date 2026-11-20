"""Pre-check of the time-complexity runs before summarize_timing.py.

    python results_analysis/time_complexity/check_timing_runs.py
    python results_analysis/time_complexity/check_timing_runs.py --runs 2 --warmup 0      # quick trial
    python results_analysis/time_complexity/check_timing_runs.py --methods ga_heso --logs "logs/slurm_time_*.log"

Only the standard library is needed, so it also runs on a login node.

Checks
  completeness   timed runs and warm-ups per (method, dataset) vs. the plan
  failures       run_NN.json.error (with the error message), leftover .tmp files
  per run        status ok, wall time > 0, 5 folds, G-mean present and > 0
  G-mean         retrained vs. value reported by the GA (GA methods), spread over runs
  timing         coefficient of variation over the runs, runs > 3x the median
  environment    one host type / CPU / GPU per comparison, CUDA for GA-HESO,
                 GPU memory recorded (pynvml), thread settings
  slurm logs     Traceback / FAILED / Killed / CANCELLED / out-of-memory lines (--logs)

Exit code 1 if any FAIL, so it can be used in scripts.
"""
import argparse
import glob
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import timing_config as cfg  # noqa: E402

FAIL, WARN = [], []


def fail(msg):
    FAIL.append(msg)


def warn(msg):
    WARN.append(msg)


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        fail('{}: unreadable JSON ({})'.format(path, e))
        return None


def check_pair(method, dataset, args):
    d = os.path.join(args.raw, method, dataset)
    kind = cfg.METHODS[method]['kind']
    tag = '{:14s} {:10s}'.format(method, dataset)
    expected = set(range(1, args.runs + 1))
    expected_warm = {-w for w in range(args.warmup)}

    for tmp in glob.glob(os.path.join(d, '*.tmp')):
        warn('{} leftover {} (job killed while writing)'.format(tag, os.path.basename(tmp)))
    errors = {}
    for e in glob.glob(os.path.join(d, 'run_*.json.error')):
        r = load(e) or {}
        errors[r.get('run_id')] = r.get('error', '?')

    runs = {}
    for p in glob.glob(os.path.join(d, 'run_*.json')):
        r = load(p)
        if r is not None:
            runs[r.get('run_id')] = r

    timed = {k: r for k, r in runs.items() if not r.get('is_warmup')}
    missing = sorted(expected - set(timed))
    missing_warm = sorted(expected_warm - set(runs))
    for rid, msg in sorted(errors.items(), key=lambda x: str(x[0])):
        if rid not in runs:
            fail('{} run {} FAILED: {}'.format(tag, rid, msg[:200]))
        else:
            warn('{} run {}: stale .error file from an earlier failed attempt (run since succeeded) - '
                 'safe to delete'.format(tag, rid))
    if missing:
        fail('{} missing timed runs: {}'.format(tag, missing if len(missing) < 10 else
                                                 '{} of {}'.format(len(missing), args.runs)))
    if missing_warm:
        warn('{} missing warm-up run(s): {}'.format(tag, missing_warm))

    walls, gmeans, hosts = [], [], set()
    for rid, r in sorted(timed.items()):
        rt = '{} run {}'.format(tag, rid)
        if r.get('status') != 'ok':
            fail('{} status {}'.format(rt, r.get('status')))
            continue
        w = r.get('wall_seconds')
        if not w or w <= 0:
            fail('{} wall_seconds = {}'.format(rt, w))
        else:
            walls.append(w)
        nf = len(r.get('folds') or [])
        if nf != 5:
            fail('{} has {} folds (expected 5)'.format(rt, nf))
        g = r.get('gmean')
        if g is None:
            fail('{} no G-mean'.format(rt))
        else:
            gmeans.append(g)
            if g <= 0:
                fail('{} G-mean = {} (model predicts one class only / did not train)'.format(rt, g))
        if kind == 'ga':
            gr = r.get('gmean_reported_by_ga')
            if g is not None and gr is not None and abs(g - gr) > args.gmean_tol:
                warn('{} G-mean {:.4f} vs GA-reported {:.4f} (diff {:+.4f})'.format(rt, g, gr, g - gr))
            for f in r.get('folds') or []:
                ep = f.get('epochs_trained')
                if isinstance(ep, list) and ep and all(e in (0, 1, None) for e in ep):
                    fail('{} fold {}: TabNet trained {} epochs'.format(rt, f.get('fold'), ep))
        else:
            if r.get('train_seconds') is None:
                warn('{} SOTA runner reported no train phase'.format(rt))
        env = r.get('environment') or {}
        hosts.add((env.get('cpu_model'), env.get('gpu_name')))
        if method == 'ga_heso' and not env.get('cuda_available'):
            fail('{} CUDA not available'.format(rt))
        if env.get('cuda_available') and (r.get('resources') or {}).get('peak_gpu_mb') is None:
            warn('{} peak GPU memory not recorded (pynvml missing?)'.format(rt))
        if env.get('omp_num_threads') is None:
            warn('{} OMP_NUM_THREADS not set'.format(rt))

    # duplicate GPU-memory / thread warnings once per pair instead of once per run
    if len(walls) >= 2:
        med = statistics.median(walls)
        cv = 100 * statistics.stdev(walls) / statistics.mean(walls)
        # CV of a handful of sub-5 s runs is dominated by interpreter/import noise
        if cv > args.cv_tol and len(walls) >= 5 and med >= 5:
            warn('{} wall time CV {:.0f} % (> {} %) - noisy node?'.format(tag, cv, args.cv_tol))
        slow = [round(w) for w in walls if w > 3 * med]
        if slow:
            warn('{} {} run(s) > 3x median ({:.0f}s): {}'.format(tag, len(slow), med, slow))
    if len(gmeans) >= 2 and max(gmeans) - min(gmeans) > args.gmean_tol:
        warn('{} G-mean varies over runs: {:.4f} - {:.4f} (pipeline not deterministic)'.format(
            tag, min(gmeans), max(gmeans)))
    if len(hosts) > 1:
        fail('{} runs come from different hardware: {}'.format(tag, sorted(hosts, key=str)))

    status = 'OK ' if not missing and len(walls) == args.runs else '.. '
    print('{} {}  timed {:2d}/{}  warm-up {}/{}  errors {}  wall median {}'.format(
        status, tag, len(walls), args.runs, len(set(runs) & expected_warm), args.warmup, len(errors),
        '{:.1f}s'.format(statistics.median(walls)) if walls else '-'))
    return hosts


def check_logs(pattern):
    keys = ('Traceback', 'FAILED', 'Killed', 'CANCELLED', 'out of memory', 'OutOfMemory',
            'oom-kill', 'DUE TO TIME LIMIT', 'Error:')
    files = sorted(glob.glob(pattern))
    if not files:
        warn('no slurm logs match {}'.format(pattern))
        return
    for path in files:
        with open(path, errors='replace') as f:
            lines = f.readlines()
        hits = [(i + 1, l.strip()) for i, l in enumerate(lines) if any(k in l for k in keys)]
        done = any(l.startswith('done in') for l in lines)
        print('  {:50s} {:4d} problem line(s){}'.format(os.path.basename(path), len(hits),
                                                         '' if done else '  (no "done in" - still running or killed)'))
        for n, l in hits[:5]:
            print('      {}: {}'.format(n, l[:160]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--methods', nargs='+', default=cfg.DEFAULT_METHODS, choices=sorted(cfg.METHODS))
    ap.add_argument('--datasets', nargs='+', default=cfg.DEFAULT_DATASETS, choices=sorted(cfg.DATASETS))
    ap.add_argument('--runs', type=int, default=cfg.N_RUNS)
    ap.add_argument('--warmup', type=int, default=cfg.N_WARMUP)
    ap.add_argument('--raw', default=cfg.RAW_DIR)
    ap.add_argument('--gmean-tol', type=float, default=0.02)
    ap.add_argument('--cv-tol', type=float, default=20.0, help='warn above this wall-time CV in %%')
    ap.add_argument('--logs', default=None, help='glob of slurm logs, e.g. "logs/slurm_time_*.log"')
    args = ap.parse_args()

    if not os.path.isdir(args.raw):
        sys.exit('no results folder {} - nothing has finished yet (or wrong machine)'.format(args.raw))

    print('Runs in {}\n'.format(args.raw))
    all_hw = set()
    for m in args.methods:
        for d in args.datasets:
            all_hw |= check_pair(m, d, args)
    if len(all_hw) > 1:
        warn('methods ran on different hardware {} - times are not directly comparable'.format(
            sorted(all_hw, key=str)))
    if args.logs:
        print('\nSlurm logs ({})'.format(args.logs))
        check_logs(args.logs)

    # collapse repeated per-run warnings of the same kind
    print('\n{} FAIL, {} WARN'.format(len(FAIL), len(WARN)))
    for msg in FAIL:
        print('FAIL  ' + msg)
    for msg in WARN[:60]:
        print('WARN  ' + msg)
    if len(WARN) > 60:
        print('WARN  ... {} more'.format(len(WARN) - 60))
    if not FAIL:
        print('\nAll planned runs are present and sane -> python results_analysis/time_complexity/summarize_timing.py')
    sys.exit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
