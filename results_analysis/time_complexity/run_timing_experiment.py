"""Runs the time-complexity experiment: N independent timed runs per (method, dataset).

    python run_timing_experiment.py                                   # defaults from timing_config
    python run_timing_experiment.py --methods ga_heso selfpe --datasets abalone19 --runs 20
    python run_timing_experiment.py --methods dgot_baseline --order blocked
    python run_timing_experiment.py --dry-run                         # print the plan only

Independence: every run is a separate `python measure_one_run.py ...` process, so no
run benefits from another's warm caches, compiled kernels or GPU memory.

Order (--order):
  interleaved (default)  run 1 of every (method, dataset) in shuffled order, then run 2 ...
                         Slow drifts of the machine (thermal throttling, other users on
                         the node, file-system load) then spread over all methods instead
                         of biasing whichever method happened to run last.
  blocked                all runs of one (method, dataset) back to back.

Warm-up: --warmup W extra runs (run ids 0, -1, ...) per (method, dataset) are done first
and stored with is_warmup=true; summarize_timing.py leaves them out of the statistics.

Resumable: a run whose JSON already exists is skipped, so a job that hit its time
limit can simply be resubmitted. Failed runs leave run_NN.json.error and are retried.
"""
import argparse
import datetime
import os
import random
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import timing_config as cfg  # noqa: E402


def plan(methods, datasets, runs, warmup, order, seed):
    pairs = [(m, d) for m in methods for d in datasets]
    jobs = []
    # warm-ups first, ids 0, -1, -2 ...
    for w in range(warmup):
        for m, d in pairs:
            jobs.append((m, d, -w, True))
    rng = random.Random(seed)
    if order == 'interleaved':
        for r in range(1, runs + 1):
            block = list(pairs)
            rng.shuffle(block)
            jobs.extend((m, d, r, False) for m, d in block)
    else:
        for m, d in pairs:
            jobs.extend((m, d, r, False) for r in range(1, runs + 1))
    return jobs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--methods', nargs='+', default=cfg.DEFAULT_METHODS, choices=sorted(cfg.METHODS))
    p.add_argument('--datasets', nargs='+', default=cfg.DEFAULT_DATASETS, choices=sorted(cfg.DATASETS))
    p.add_argument('--runs', type=int, default=cfg.N_RUNS)
    p.add_argument('--warmup', type=int, default=cfg.N_WARMUP)
    p.add_argument('--order', choices=['interleaved', 'blocked'], default='interleaved')
    p.add_argument('--seed', type=int, default=2026, help='seed of the run-order shuffle')
    p.add_argument('--python', default=sys.executable, help='interpreter for the measured runs')
    p.add_argument('--timeout', type=float, default=None, help='seconds per run (default: none)')
    p.add_argument('--pause', type=float, default=5.0, help='seconds to idle between runs (GPU cool-down)')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()

    jobs = plan(args.methods, args.datasets, args.runs, args.warmup, args.order, args.seed)
    todo = [j for j in jobs if not os.path.exists(cfg.raw_path(j[0], j[1], j[2]))]
    print('{} runs planned, {} already done, {} to do'.format(len(jobs), len(jobs) - len(todo), len(todo)))
    if args.dry_run:
        for m, d, r, w in todo:
            print('  {:15s} {:10s} run {:3d}{}'.format(m, d, r, '  (warm-up)' if w else ''))
        return

    script = os.path.join(HERE, 'measure_one_run.py')
    failed = []
    t_start = time.time()
    for i, (m, d, r, w) in enumerate(todo, 1):
        out = cfg.raw_path(m, d, r)
        if os.path.exists(out):
            continue
        cmd = [args.python, '-u', script, '--method', m, '--dataset', d, '--run', str(r)]
        if w:
            cmd.append('--warmup')
        print('[{}] ({}/{}) {} {} run {}{}'.format(datetime.datetime.now().strftime('%H:%M:%S'), i, len(todo),
                                                  m, d, r, ' (warm-up)' if w else ''), flush=True)
        try:
            status = subprocess.run(cmd, cwd=cfg.REPO_ROOT, timeout=args.timeout).returncode
        except subprocess.TimeoutExpired:
            status = 'timeout'
        if status != 0:
            print('  FAILED ({}): {} {} run {}'.format(status, m, d, r), flush=True)
            failed.append((m, d, r, status))
        if args.pause and i < len(todo):
            time.sleep(args.pause)

    print('done in {:.1f} h'.format((time.time() - t_start) / 3600))
    if failed:
        print('{} failed run(s):'.format(len(failed)))
        for f in failed:
            print('  ', *f)
        sys.exit(1)


if __name__ == '__main__':
    main()
