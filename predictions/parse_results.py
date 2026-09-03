import argparse
import glob
import os
import numpy as np

from imblearn.metrics import geometric_mean_score
from sklearn.metrics import roc_auc_score

try:
    import parse_results_config as config
except ModuleNotFoundError:
    from predictions import parse_results_config as config

_PREDICTIONS_DIR = os.path.dirname(os.path.abspath(__file__))


def _parse_result_txt(path):
    with open(path, 'r') as f:
        content = f.read()

    # str(result) renders numpy arrays as e.g. array([0, 1, ...], dtype=int64)
    # (dtype=float64 for GOIO) - strip that down to literals eval() can read.
    content = content.replace(" ", "").replace("\n", "")
    content = (content.replace("array(", "")
               .replace(")", "")
               .replace(",dtype=int64", "").replace("dtype=int64,", "")
               .replace(",dtype=float64", "").replace("dtype=float64,", "")
               .replace(",dtype=object", "").replace("dtype=object,", ""))
    return eval(content)


def _per_fold_scores(data, metric_func):
    """Per-fold metric values; np.nan for folds the metric cannot score."""
    true_values = data.get('true_values')
    predicted_values = data.get('predicted_values')
    if not true_values or not predicted_values:
        return []

    scores = []
    for true_fold, predicted_fold in zip(true_values, predicted_values):
        try:
            scores.append(float(metric_func(true_fold, predicted_fold)))
        except ValueError:
            # e.g. a fold that ended up with a single class present
            scores.append(np.nan)
    return scores


def _mean_std(scores):
    if not scores or np.all(np.isnan(scores)):
        return np.nan, np.nan
    return float(np.nanmean(scores)), float(np.nanstd(scores))


def _resolve_targets(domain, targets):
    results_dir = os.path.join(_PREDICTIONS_DIR, domain, 'results')
    if not os.path.isdir(results_dir):
        return None
    if targets == ['*']:
        return sorted(name for name in os.listdir(results_dir)
                      if os.path.isdir(os.path.join(results_dir, name)))
    return targets


def _score_file(path):
    data = _parse_result_txt(path)
    gmean_scores = _per_fold_scores(data, geometric_mean_score)
    auc_scores = _per_fold_scores(data, roc_auc_score)
    gmean_mean, gmean_std = _mean_std(gmean_scores)
    auc_mean, auc_std = _mean_std(auc_scores)
    return {
        'name': os.path.basename(path),
        'folds': len(gmean_scores),
        'fitness': data.get('fitness'),
        'gmean': gmean_mean,
        'gmean_std': gmean_std,
        'roc_auc': auc_mean,
        'roc_auc_std': auc_std,
    }


def collect_target(domain, target):
    """Parses every .txt in one target folder, in on-disk order.

    Returns (rows, failures).
    """
    target_dir = os.path.join(_PREDICTIONS_DIR, domain, 'results', target)
    paths = glob.glob(os.path.join(target_dir, '*.txt'))

    rows, failures = [], []
    for path in paths:
        try:
            rows.append(_score_file(path))
        except Exception as exc:
            failures.append((os.path.basename(path), f'{type(exc).__name__}: {exc}'))
    return rows, failures


def _fmt(value, decimals):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return '-'
    return f'{value:.{decimals}f}'


def _print_target(domain, target, rows, failures, settings):
    decimals = settings['decimals']
    header = f'{domain} / {target}'
    print()
    print(header)
    print('-' * len(header))

    if not rows and not failures:
        print('  no .txt files found')
        return

    if target == 'goio':
        print('  note: goio .txt holds only one RF repetition; see '
              'goio_*_summary.csv for its aggregated scores')

    shown = rows
    total = len(shown)
    if settings['top_n'] is not None:
        shown = shown[:settings['top_n']]

    name_width = max([len(row['name']) for row in shown] + [len('file')])
    value_width = decimals + 3
    print(f'  {"file":<{name_width}}  {"folds":>5}  '
          f'{"gmean":>{value_width}} {"+/-":>{value_width}}  '
          f'{"roc_auc":>{value_width}} {"+/-":>{value_width}}  '
          f'{"fitness":>{value_width}}')
    for row in shown:
        print(f'  {row["name"]:<{name_width}}  {row["folds"]:>5}  '
              f'{_fmt(row["gmean"], decimals):>{value_width}} '
              f'{_fmt(row["gmean_std"], decimals):>{value_width}}  '
              f'{_fmt(row["roc_auc"], decimals):>{value_width}} '
              f'{_fmt(row["roc_auc_std"], decimals):>{value_width}}  '
              f'{_fmt(row["fitness"], decimals):>{value_width}}')

    if len(shown) < total:
        print(f'  ... {total - len(shown)} more file(s) not shown '
              f'(--top-n {settings["top_n"]})')

    for name, error in failures:
        print(f'  FAILED {name}: {error}')


def run(domains, targets, targets_per_domain, settings):
    for domain in domains:
        domain_targets = _resolve_targets(
            domain, targets_per_domain.get(domain, targets))
        if domain_targets is None:
            print(f'\n{domain}: no results/ folder under '
                  f'{os.path.join(_PREDICTIONS_DIR, domain)}')
            continue
        for target in domain_targets:
            rows, failures = collect_target(domain, target)
            _print_target(domain, target, rows, failures, settings)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--domains', nargs='+', default=config.DOMAINS,
                        help='dataset folders under predictions/')
    parser.add_argument('--targets', nargs='+', default=config.TARGETS,
                        help='method folders under predictions/{domain}/results/, '
                             'or "*" for all of them')
    parser.add_argument('--top-n', type=int, default=config.TOP_N,
                        help='show only the first N files per target')
    parser.add_argument('--decimals', type=int, default=config.DECIMALS)
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    run(
        domains=args.domains,
        targets=args.targets,
        targets_per_domain=config.TARGETS_PER_DOMAIN,
        settings={
            'top_n': args.top_n,
            'decimals': args.decimals,
        },
    )
