"""Aggregates raw/<method>/<dataset>/run_NN.json into statistics, tables and figures.

    python summarize_timing.py            # -> summary/

Outputs (summary/):
  timing_runs.csv          one row per run, every measured quantity as a column
  timing_summary.csv       per (method, dataset, metric): n, mean, std, median, min, max,
                           Q1, Q3, IQR, CV %, 95 % CI of the mean (t), Tukey outliers,
                           and the ratio of the mean to GA-HESO's mean on that dataset
  table_<metric>.csv/.tex  methods x datasets, "mean ± std" with [min - max] below
  scaling.csv              empirical exponent b of time ~ n_rows^b per method
                           (log-log least squares over the datasets; with 3 datasets
                           this is indicative only)
  environment_check.csv    hosts / CPUs / GPUs the runs came from - all runs of a
                           comparison must come from the same hardware
  fig_boxplot_<metric>.png distribution of the runs, one panel per dataset, log scale
  fig_scaling.png          mean wall time vs dataset size, log-log
  fig_phases.png           where the time goes (GA methods): setup/preprocess/resample/fit/predict
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import timing_config as cfg  # noqa: E402

# Categorical palette (validated: adjacent-pair CVD separation passes, light surface).
# Colour follows the method, in the fixed order of timing_config.METHODS.
PALETTE = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
INK, INK_MUTED, GRID = '#1f1f1e', '#6b6a64', '#e4e3de'

# metric column -> (label, unit) for tables and figures
METRICS = {
    'wall_seconds': ('Wall-clock time, 5 folds', 's'),
    'train_seconds': ('Training time, 5 folds', 's'),
    'inference_seconds': ('Prediction time, 5 test folds', 's'),
    'inference_ms_per_1000': ('Prediction time per 1000 samples', 'ms'),
    'fold_seconds_mean': ('Mean time per fold', 's'),
    'cpu_seconds': ('CPU time (process tree)', 's'),
    'cpu_utilisation': ('CPU time / wall time', 'cores'),
    'peak_rss_mb': ('Peak RAM (process tree)', 'MB'),
    'peak_gpu_mb': ('Peak GPU memory', 'MB'),
    'gpu_util_mean': ('Mean GPU utilisation', '%'),
    'ga_estimate_hours': ('Estimated GA search time', 'h'),
    'ga_estimate_hours_parallel': ('Estimated GA search time, folds in parallel', 'h'),
    'gmean': ('G-mean (sanity check)', ''),
}
TABLE_METRICS = ['wall_seconds', 'train_seconds', 'inference_seconds', 'cpu_seconds',
                 'peak_rss_mb', 'peak_gpu_mb', 'ga_estimate_hours']


# ---------------------------------------------------------------------- loading
def load_runs():
    rows = []
    for path in glob.glob(os.path.join(cfg.RAW_DIR, '*', '*', 'run_*.json')):
        with open(path) as f:
            r = json.load(f)
        if r.get('status') != 'ok':
            continue
        res, env, ga = r.get('resources', {}), r.get('environment', {}), r.get('ga_search_estimate') or {}
        row = {
            'method': r['method'], 'dataset': r['dataset'], 'run_id': r['run_id'], 'is_warmup': r['is_warmup'],
            'n_rows': r['n_rows'], 'n_features': r['n_features'], 'imbalance_ratio': r.get('imbalance_ratio'),
            'wall_seconds': r['wall_seconds'], 'train_seconds': r.get('train_seconds'),
            'inference_seconds': r.get('inference_seconds'), 'fold_seconds_mean': r.get('fold_seconds_mean'),
            'gmean': r.get('gmean'), 'gmean_reported_by_ga': r.get('gmean_reported_by_ga'),
            'host': env.get('host'), 'cpu_model': env.get('cpu_model'), 'gpu_name': env.get('gpu_name'),
            'slurm_cpus_per_task': env.get('slurm_cpus_per_task'), 'started_at': r.get('started_at'),
        }
        row.update({k: res.get(k) for k in ('cpu_seconds', 'peak_rss_mb', 'peak_gpu_mb', 'gpu_util_mean')})
        row['cpu_utilisation'] = (row['cpu_seconds'] / row['wall_seconds']) if row['cpu_seconds'] else None
        if row['inference_seconds']:
            row['inference_ms_per_1000'] = 1e6 * row['inference_seconds'] / r['n_rows']  # all rows are tested once
        est = ga.get('estimated_seconds_distinct') or ga.get('estimated_seconds_nominal')
        row['ga_estimate_hours'] = est / 3600 if est else None
        est_p = ga.get('estimated_seconds_distinct_parallel_folds') or ga.get('estimated_seconds_nominal_parallel_folds')
        row['ga_estimate_hours_parallel'] = est_p / 3600 if est_p else None
        row['ga_evaluations'] = ga.get('distinct_evaluations') or ga.get('nominal_evaluations')
        for k, v in (r.get('phases') or {}).items():
            row['phase_' + k] = v
        for k, v in (r.get('model') or {}).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                row['model_' + k] = v
        rows.append(row)
    if not rows:
        sys.exit('no finished runs under ' + cfg.RAW_DIR)
    df = pd.DataFrame(rows)
    order_m = [m for m in cfg.METHODS if m in set(df.method)]
    order_d = [d for d in cfg.DATASETS if d in set(df.dataset)]
    df['method'] = pd.Categorical(df.method, order_m, ordered=True)
    df['dataset'] = pd.Categorical(df.dataset, order_d, ordered=True)
    return df.sort_values(['dataset', 'method', 'run_id']).reset_index(drop=True)


# ------------------------------------------------------------------- statistics
def describe(values):
    x = np.asarray(pd.to_numeric(values, errors='coerce'), dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return None
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    mean, std = x.mean(), (x.std(ddof=1) if n > 1 else 0.0)
    half = stats.t.ppf(0.975, n - 1) * std / np.sqrt(n) if n > 1 else np.nan
    iqr = q3 - q1
    return {'n': n, 'mean': mean, 'std': std, 'median': med, 'min': x.min(), 'max': x.max(),
            'q1': q1, 'q3': q3, 'iqr': iqr, 'cv_pct': 100 * std / mean if mean else np.nan,
            'ci95_low': mean - half, 'ci95_high': mean + half,
            'n_outliers': int(((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).sum())}


def summary_table(df):
    metric_cols = [c for c in list(METRICS) + sorted(c for c in df.columns if c.startswith('phase_'))
                   if c in df.columns]
    out = []
    for (m, d), g in df.groupby(['method', 'dataset'], observed=True):
        for col in metric_cols:
            s = describe(g[col])
            if s:
                out.append({'method': m, 'method_label': cfg.METHODS[m]['label'], 'dataset': d,
                            'n_rows': int(g.n_rows.iloc[0]), 'metric': col, **s})
    s = pd.DataFrame(out)
    ref = s[s.method == 'ga_heso'].set_index(['dataset', 'metric'])['mean']
    s['ratio_to_ga_heso'] = [row['mean'] / ref.get((row['dataset'], row['metric']), np.nan)
                             if ref.get((row['dataset'], row['metric']), 0) else np.nan
                             for _, row in s.iterrows()]
    return s


def _fmt(v):
    if v is None or np.isnan(v):
        return '-'
    if abs(v) >= 100:
        return '{:.0f}'.format(v)
    if abs(v) >= 10:
        return '{:.1f}'.format(v)
    return '{:.2f}'.format(v)


def wide_tables(s, outdir):
    for metric in TABLE_METRICS:
        t = s[s.metric == metric]
        if t.empty:
            continue
        datasets = [d for d in cfg.DATASETS if d in set(t.dataset)]
        methods = [m for m in cfg.METHODS if m in set(t.method)]
        csv_rows, tex_rows = [], []
        for m in methods:
            csv_row, cells = {'method': cfg.METHODS[m]['label']}, []
            for d in datasets:
                r = t[(t.method == m) & (t.dataset == d)]
                if r.empty:
                    csv_row[d] = '-'
                    cells.append('--')
                    continue
                r = r.iloc[0]
                csv_row[d] = '{} ± {} [{} – {}]'.format(_fmt(r['mean']), _fmt(r['std']), _fmt(r['min']), _fmt(r['max']))
                cells.append(r'$\begin{{array}}{{c}}{} \pm {}\\ \scriptstyle [{},\,{}]\end{{array}}$'.format(
                    _fmt(r['mean']), _fmt(r['std']), _fmt(r['min']), _fmt(r['max'])))
            csv_rows.append(csv_row)
            tex_rows.append('{} & {} \\\\'.format(cfg.METHODS[m]['label'], ' & '.join(cells)))
        pd.DataFrame(csv_rows).to_csv(os.path.join(outdir, 'table_{}.csv'.format(metric)), index=False)
        label, unit = METRICS[metric]
        header = ' & '.join('{} ($n$={})'.format(cfg.DATASETS[d]['label'], int(t[t.dataset == d].n_rows.iloc[0]))
                            for d in datasets)
        n_runs = int(t.n.max())
        tex = '\n'.join([
            r'\begin{table}[ht]', r'\centering',
            r'\caption{{{} [{}]: mean $\pm$ std over {} independent runs, [min, max] below.}}'.format(label, unit, n_runs),
            r'\begin{{tabular}}{{l{}}}'.format('c' * len(datasets)), r'\hline',
            'Method & {} \\\\'.format(header), r'\hline', *tex_rows, r'\hline', r'\end{tabular}',
            r'\label{{tab:time_{}}}'.format(metric), r'\end{table}', ''])
        with open(os.path.join(outdir, 'table_{}.tex'.format(metric)), 'w', encoding='utf-8') as f:
            f.write(tex)


def scaling_table(s):
    rows = []
    for metric in ('wall_seconds', 'train_seconds'):
        t = s[s.metric == metric]
        for m, g in t.groupby('method', observed=True):
            g = g[g['mean'] > 0]
            if len(g) < 2:
                continue
            b, a = np.polyfit(np.log(g.n_rows), np.log(g['mean']), 1)
            rows.append({'method': m, 'method_label': cfg.METHODS[m]['label'], 'metric': metric,
                         'exponent_b': b, 'n_datasets': len(g),
                         'note': 'time ~ n_rows^b fitted on dataset means; datasets also differ in features/imbalance'})
    return pd.DataFrame(rows)


def environment_check(df):
    return (df.groupby(['method', 'host', 'cpu_model', 'gpu_name'], observed=True, dropna=False)
              .size().rename('runs').reset_index())


# ------------------------------------------------------------------------ plots
def _style(ax):
    ax.set_facecolor('white')
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(INK_MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK, labelsize=8)
    ax.grid(axis='y', which='major', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _colors(methods):
    return {m: PALETTE[list(cfg.METHODS).index(m) % len(PALETTE)] for m in methods}


def plot_boxplots(df, metric, outdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data = df.dropna(subset=[metric])
    if data.empty:
        return
    datasets = [d for d in cfg.DATASETS if d in set(data.dataset)]
    methods = [m for m in cfg.METHODS if m in set(data.method)]
    col = _colors(methods)
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.6 * len(datasets), 3.6), squeeze=False)
    rng = np.random.default_rng(0)
    for ax, d in zip(axes[0], datasets):
        _style(ax)
        present = [m for m in methods if not data[(data.dataset == d) & (data.method == m)].empty]
        vals = [data[(data.dataset == d) & (data.method == m)][metric].to_numpy(float) for m in present]
        bp = ax.boxplot(vals, widths=0.55, patch_artist=True, showfliers=False,
                        medianprops={'color': INK, 'linewidth': 1.5},
                        whiskerprops={'color': INK_MUTED, 'linewidth': 1},
                        capprops={'color': INK_MUTED, 'linewidth': 1},
                        boxprops={'linewidth': 1})
        for patch, m in zip(bp['boxes'], present):
            patch.set_facecolor(col[m] + '40')
            patch.set_edgecolor(col[m])
        for i, (m, v) in enumerate(zip(present, vals), 1):
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=10, color=col[m],
                       edgecolor='white', linewidth=0.5, zorder=3)
        ax.set_yscale('log')
        ax.set_xticks(range(1, len(present) + 1))
        ax.set_xticklabels([cfg.METHODS[m]['label'] for m in present], rotation=40, ha='right')
        n = int(data[data.dataset == d].n_rows.iloc[0])
        ax.set_title('{}  (n = {})'.format(cfg.DATASETS[d]['label'], n), fontsize=9, color=INK)
    label, unit = METRICS[metric]
    axes[0][0].set_ylabel('{} [{}], log scale'.format(label, unit), fontsize=8, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'fig_boxplot_{}.png'.format(metric)), dpi=200)
    plt.close(fig)


def plot_scaling(s, outdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t = s[s.metric == 'wall_seconds']
    if t.dataset.nunique() < 2:
        return
    methods = [m for m in cfg.METHODS if m in set(t.method)]
    col = _colors(methods)
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    _style(ax)
    for m in methods:
        g = t[t.method == m].sort_values('n_rows')
        ax.errorbar(g.n_rows, g['mean'], yerr=g['std'], color=col[m], linewidth=2, marker='o', markersize=5,
                    markeredgecolor='white', capsize=2, label=cfg.METHODS[m]['label'])
    ax.set_xscale('log')
    ax.set_yscale('log')
    sizes = sorted(t.drop_duplicates('dataset')[['n_rows', 'dataset']].itertuples(index=False))
    ax.set_xticks([n for n, _ in sizes])
    ax.set_xticklabels(['{}\n{}'.format(n, cfg.DATASETS[d]['label']) for n, d in sizes], fontsize=7)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel('Dataset size (rows), log scale', fontsize=8, color=INK)
    ax.set_ylabel('Wall-clock time, 5 folds [s], log scale', fontsize=8, color=INK)
    ax.legend(fontsize=7, frameon=False, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'fig_scaling.png'), dpi=200)
    plt.close(fig)


def plot_phases(s, outdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    phases = ['setup', 'preprocess', 'resample', 'fit', 'predict']
    t = s[s.metric.isin(['phase_' + p for p in phases])]
    t = t[t.method.isin([m for m, v in cfg.METHODS.items() if v['kind'] == 'ga'])]
    if t.empty:
        return
    datasets = [d for d in cfg.DATASETS if d in set(t.dataset)]
    methods = [m for m in cfg.METHODS if m in set(t.method)]
    phase_col = dict(zip(phases, PALETTE))
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.6 * len(datasets), 2.8), squeeze=False, sharey=True)
    for ax, d in zip(axes[0], datasets):
        _style(ax)
        ax.grid(axis='y', visible=False)
        ax.grid(axis='x', color=GRID, linewidth=0.8)
        for i, m in enumerate(methods):
            g = t[(t.dataset == d) & (t.method == m)].set_index('metric')['mean']
            total = sum(g.get('phase_' + p, 0.0) for p in phases)
            left = 0.0
            for p in phases:
                v = g.get('phase_' + p, 0.0)
                if v > 0 and total > 0:
                    ax.barh(i, 100 * v / total, left=left, color=phase_col[p], edgecolor='white',
                            linewidth=1, height=0.6)
                    left += 100 * v / total
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels([cfg.METHODS[m]['label'] for m in methods])
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel('Share of time [%]', fontsize=8, color=INK)
        ax.set_title(cfg.DATASETS[d]['label'], fontsize=9, color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=phase_col[p]) for p in phases]
    fig.legend(handles, phases, ncol=len(phases), fontsize=7, frameon=False, loc='lower center')
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(outdir, 'fig_phases.png'), dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------------- main
def main():
    os.makedirs(cfg.SUMMARY_DIR, exist_ok=True)
    df_all = load_runs()
    df_all.to_csv(os.path.join(cfg.SUMMARY_DIR, 'timing_runs.csv'), index=False)
    df = df_all[~df_all.is_warmup]

    counts = df.groupby(['method', 'dataset'], observed=True).size().unstack(fill_value=0)
    print('Timed runs (warm-ups excluded):\n{}\n'.format(counts))

    env = environment_check(df_all)
    env.to_csv(os.path.join(cfg.SUMMARY_DIR, 'environment_check.csv'), index=False)
    if env[['host', 'cpu_model', 'gpu_name']].drop_duplicates().shape[0] > 1:
        print('WARNING: runs come from more than one machine/CPU/GPU - see environment_check.csv; '
              'times are only comparable on identical hardware.\n')

    s = summary_table(df)
    s.to_csv(os.path.join(cfg.SUMMARY_DIR, 'timing_summary.csv'), index=False)
    wide_tables(s, cfg.SUMMARY_DIR)
    sc = scaling_table(s)
    sc.to_csv(os.path.join(cfg.SUMMARY_DIR, 'scaling.csv'), index=False)

    for metric in ('wall_seconds', 'train_seconds', 'peak_rss_mb'):
        plot_boxplots(df, metric, cfg.SUMMARY_DIR)
    plot_scaling(s, cfg.SUMMARY_DIR)
    plot_phases(s, cfg.SUMMARY_DIR)

    view = s[s.metric == 'wall_seconds'][['method_label', 'dataset', 'n', 'mean', 'std', 'median', 'min',
                                          'max', 'cv_pct', 'ratio_to_ga_heso']]
    with pd.option_context('display.width', 160, 'display.float_format', '{:.2f}'.format):
        print(view.to_string(index=False))
        if not sc.empty:
            print('\nScaling exponents (time ~ n^b):\n' + sc[['method_label', 'metric', 'exponent_b']].to_string(index=False))
    print('\n-> {}'.format(cfg.SUMMARY_DIR))


if __name__ == '__main__':
    main()
