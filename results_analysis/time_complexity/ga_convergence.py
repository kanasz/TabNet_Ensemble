"""How fast does the GA converge? Reads the paper's GA checkpoints (.pkl), no training.

    python results_analysis/time_complexity/ga_convergence.py      # -> summary/

pygad stores the best fitness of every generation (ga.best_solutions_fitness). From the
best-so-far curve this reports, per GA method x dataset, the generation at which the GA
reaches a share of its final result, in two ways:

  value        best-so-far >= q x final G-mean
               (easy to state, but G-mean starts high, so this is reached early)
  improvement  best-so-far - start >= q x (final - start)
               (share of what the search actually ADDED over the initial population;
               the honest number for "the budget can be cut")

for q = 0.90, 0.95, 0.99, as generation, % of generations, and - when
summary/cost_split.csv from summarize_timing.py exists - projected hours and the share
of the search time that could be saved.

Outputs (summary/):
  convergence.csv          one row per method x dataset x q
  convergence_curves.csv   best-so-far per generation (for your own plots)
  fig_convergence.png      share of the final improvement vs % of generations
  untuned_initial_population.csv / untuned_values.csv
                           fitness of the random first generation = what the method gives
                           WITHOUT tuning (used by summarize_timing.py for the GA-vs-SOTA
                           comparison, so run this script before summarize_timing.py)
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import timing_config as cfg  # noqa: E402
sys.path.insert(0, cfg.REPO_ROOT)
os.chdir(cfg.REPO_ROOT)
warnings.filterwarnings('ignore')

from summarize_timing import INK, INK_MUTED, GRID, _style, _colors  # noqa: E402

QUANTILES = (0.90, 0.95, 0.99)


def _curve(method, dataset):
    from pygad import pygad
    stem = os.path.join(cfg.REPO_ROOT, cfg.DATASETS[dataset]['checkpoints'][method])
    ga = pygad.load(stem)
    f = np.asarray(ga.best_solutions_fitness, dtype=float)
    # best-so-far: GA-HESO reports the best over the whole run; for the others
    # (best_solutions[-1]) the running max is the same thing when elitism keeps the best
    return np.maximum.accumulate(f), int(ga.generations_completed), _initial_population_fitness(ga)


def _initial_population_fitness(ga):
    """Fitness of the first generation = random, UNTUNED configurations. All three
    tuners ran with save_solutions=True, so pygad kept every evaluated chromosome and
    its fitness in order; the first sol_per_pop entries are the initial population."""
    sf = getattr(ga, 'solutions_fitness', None)
    if sf is None or len(sf) < int(ga.sol_per_pop):
        return None
    return np.asarray(sf[:int(ga.sol_per_pop)], dtype=float)


def _first(mask):
    idx = np.flatnonzero(mask)
    return int(idx[0]) if len(idx) else None


def main():
    os.makedirs(cfg.SUMMARY_DIR, exist_ok=True)
    cost_path = os.path.join(cfg.SUMMARY_DIR, 'cost_split.csv')
    cost = pd.read_csv(cost_path) if os.path.exists(cost_path) else None

    rows, curves, untuned, untuned_values = [], [], [], []
    ga_methods = [m for m in cfg.DEFAULT_METHODS if cfg.METHODS[m]['kind'] == 'ga']
    for m in ga_methods:
        for d in cfg.DEFAULT_DATASETS:
            try:
                best, gens, gen0 = _curve(m, d)
            except BaseException as e:  # pygad.load raises BaseException
                if isinstance(e, KeyboardInterrupt):
                    raise
                print('{} {}: cannot read checkpoint ({}: {})'.format(m, d, type(e).__name__, e))
                continue
            if gen0 is not None:
                q1, med, q3 = np.percentile(gen0, [25, 50, 75])
                untuned.append({'method': m, 'method_label': cfg.METHODS[m]['label'], 'dataset': d,
                                'n_configs': len(gen0), 'mean': gen0.mean(), 'std': gen0.std(ddof=1),
                                'median': med, 'q1': q1, 'q3': q3, 'min': gen0.min(), 'max': gen0.max(),
                                'n_zero': int((gen0 <= 0).sum()),
                                'note': 'fitness of the random initial population = untuned configurations '
                                        '(CV G-mean on the GA folds)'})
                untuned_values += [{'method': m, 'dataset': d, 'config': i, 'gmean': float(v)}
                                   for i, v in enumerate(gen0)]
            n = len(best) - 1  # index 0 = first recorded generation
            start, final = best[0], best[-1]
            for g, v in enumerate(best):
                curves.append({'method': m, 'dataset': d, 'generation': g, 'pct_generations': 100 * g / max(n, 1),
                               'best_so_far': v,
                               'share_of_improvement': (v - start) / (final - start) if final > start else 1.0})
            search_h = None
            if cost is not None:
                hit = cost[(cost.method == m) & (cost.dataset == d)]
                if not hit.empty:
                    search_h = float(hit.search_hours.iloc[0])
            for q in QUANTILES:
                for kind, mask in (('value', best >= q * final),
                                   ('improvement', (best - start) >= q * (final - start) - 1e-12)):
                    g = _first(mask)
                    pct = None if g is None else 100 * g / max(n, 1)
                    rows.append({'method': m, 'method_label': cfg.METHODS[m]['label'], 'dataset': d,
                                 'criterion': kind, 'q': q, 'generation': g, 'generations_total': n,
                                 'generations_completed': gens, 'pct_generations': pct,
                                 'gmean_initial_best': start, 'gmean_final': final,
                                 'gmean_at_generation': None if g is None else best[g],
                                 'hours_to_reach': None if (search_h is None or pct is None) else search_h * pct / 100,
                                 'search_hours_total': search_h,
                                 'pct_search_time_saved': None if pct is None else 100 - pct})
            print('{:14s} {:10s} start {:.4f} final {:.4f}  95 % of improvement at gen {} / {}'.format(
                m, d, start, final, _first((best - start) >= 0.95 * (final - start) - 1e-12), n))

    if not rows:
        sys.exit('no checkpoint could be read')
    pd.DataFrame(rows).to_csv(os.path.join(cfg.SUMMARY_DIR, 'convergence.csv'), index=False)
    cv = pd.DataFrame(curves)
    cv.to_csv(os.path.join(cfg.SUMMARY_DIR, 'convergence_curves.csv'), index=False)
    if untuned:
        pd.DataFrame(untuned).to_csv(os.path.join(cfg.SUMMARY_DIR, 'untuned_initial_population.csv'), index=False)
        pd.DataFrame(untuned_values).to_csv(os.path.join(cfg.SUMMARY_DIR, 'untuned_values.csv'), index=False)
    plot(cv)
    print('-> {}'.format(cfg.SUMMARY_DIR))


def plot(cv):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    datasets = [d for d in cfg.DATASETS if d in set(cv.dataset)]
    methods = [m for m in cfg.METHODS if m in set(cv.method)]
    col = _colors(methods)
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.6 * len(datasets), 3.2), squeeze=False, sharey=True)
    for ax, d in zip(axes[0], datasets):
        _style(ax)
        ax.axhline(95, color=INK_MUTED, linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
        for m in methods:
            g = cv[(cv.dataset == d) & (cv.method == m)]
            if not g.empty:
                ax.step(g.pct_generations, 100 * g.share_of_improvement, where='post', color=col[m],
                        linewidth=2, label=cfg.METHODS[m]['label'])
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 102)
        ax.set_title(cfg.DATASETS[d]['label'], fontsize=9, color=INK)
        ax.set_xlabel('Generations [%]', fontsize=8, color=INK)
    axes[0][0].set_ylabel('Share of final G-mean improvement [%]', fontsize=8, color=INK)
    axes[0][0].text(1, 96, '95 %', fontsize=7, color=INK_MUTED, va='bottom')
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, frameon=False, loc='center left', bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.SUMMARY_DIR, 'fig_convergence.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
