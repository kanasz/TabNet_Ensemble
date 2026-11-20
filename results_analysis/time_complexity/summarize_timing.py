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
  cost_split.csv/table_cost_split.tex
                           (a) one-off search cost (h, CPU-core-h, GPU-h) vs (b) final model
                           (training s, prediction ms/1000, RAM)
  fig_accuracy_vs_time.png G-mean vs time per dataset; filled = final training, hollow =
                           GA search + final training. G-mean from paper_gmean.csv if present
                           (accuracy_used_in_figure.csv says which source was used)
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
        # one fitness evaluation as the GA does it (GA-HESO: folds in parallel)
        row['fitness_eval_seconds'] = (ga.get('fitness_eval_seconds_parallel_folds')
                                       or ga.get('fitness_eval_seconds_sequential_folds'))
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


def _uses_gpu(g):
    return any(c in g and (pd.to_numeric(g[c], errors='coerce').fillna(0) > 0).any()
               for c in ('gpu_util_mean', 'peak_gpu_mb'))


def load_full_ga():
    """Measured full-GA searches (measure_full_ga.py): (method, dataset) -> list of records."""
    out = {}
    for path in glob.glob(os.path.join(HERE, 'raw_full_ga', '*', '*', 'run_*.json')):
        if path.endswith('.progress.json'):
            continue
        with open(path) as f:
            r = json.load(f)
        if r.get('status') == 'ok':
            out.setdefault((r['method'], r['dataset']), []).append(r)
    return out


def cost_split(df, outdir):
    """(a) one-off search cost vs (b) cost of the final model, per method x dataset.

    search_hours           GA methods: one fitness evaluation x number of evaluations
                           (distinct chromosomes when known); SOTA: 0, they are not tuned
    search_cpu_core_hours  search_hours x measured CPU/wall ratio
    search_gpu_hours       search_hours if the method trains on the GPU, else 0
    final_train_s          training the final model, 5 folds (median over runs)
    predict_ms_per_1000    prediction time per 1000 samples (GA methods)
    total_hours            search + final training
    """
    rows = []
    full = load_full_ga()
    for (m, d), g in df.groupby(['method', 'dataset'], observed=True):
        kind = cfg.METHODS[m]['kind']
        est_h = float(g.ga_estimate_hours.median()) if kind == 'ga' and g.ga_estimate_hours.notna().any() else np.nan
        measured = full.get((m, d), [])
        meas_h = float(np.mean([r['total_hours'] for r in measured])) if measured else np.nan
        if measured:
            search_h, source = meas_h, 'measured ({} full GA run{})'.format(len(measured), 's' if len(measured) > 1 else '')
        elif kind == 'ga' and not np.isnan(est_h):
            search_h, source = est_h, 'estimated (1 evaluation x number of evaluations)'
        else:
            search_h, source = 0.0, 'not tuned'
        if m == 'ga_heso' and 'ga_estimate_hours_parallel' in g and g.ga_estimate_hours_parallel.notna().any():
            search_h_par = float(g.ga_estimate_hours_parallel.median())
        else:
            search_h_par = np.nan
        train_s = float(g.train_seconds.median()) if g.train_seconds.notna().any() else float(g.wall_seconds.median())
        cpu_ratio = float(g.cpu_utilisation.median()) if g.cpu_utilisation.notna().any() else np.nan
        rows.append({
            'method': m, 'method_label': cfg.METHODS[m]['label'], 'kind': kind, 'dataset': d,
            'n_rows': int(g.n_rows.iloc[0]),
            'ga_evaluations': g.ga_evaluations.median() if 'ga_evaluations' in g else np.nan,
            'search_hours': search_h,
            'search_source': source,
            'search_hours_measured': meas_h,
            'search_runs_measured': len(measured),
            'search_hours_measured_std': float(np.std([r['total_hours'] for r in measured], ddof=1))
            if len(measured) > 1 else np.nan,
            'search_hours_measured_min': float(np.min([r['total_hours'] for r in measured])) if measured else np.nan,
            'search_hours_measured_max': float(np.max([r['total_hours'] for r in measured])) if measured else np.nan,
            'search_hours_estimated': est_h,
            'estimate_error_pct': 100 * (est_h - meas_h) / meas_h if measured and not np.isnan(est_h) else np.nan,
            'search_evaluations_measured': float(np.mean([r['evaluations'] for r in measured])) if measured else np.nan,
            'search_hours_folds_parallel': search_h_par,
            'search_cpu_core_hours': search_h * cpu_ratio if search_h else 0.0,
            'search_gpu_hours': search_h if (search_h and _uses_gpu(g)) else 0.0,
            'final_train_s': train_s,
            'final_wall_s': float(g.wall_seconds.median()),
            'predict_ms_per_1000': float(g.inference_ms_per_1000.median())
            if 'inference_ms_per_1000' in g and g.inference_ms_per_1000.notna().any() else np.nan,
            'peak_rss_mb': float(g.peak_rss_mb.median()) if g.peak_rss_mb.notna().any() else np.nan,
            'peak_gpu_mb': float(g.peak_gpu_mb.median()) if g.peak_gpu_mb.notna().any() else np.nan,
            'total_hours': search_h + train_s / 3600,
        })
    c = pd.DataFrame(rows)
    c.to_csv(os.path.join(outdir, 'cost_split.csv'), index=False)

    # LaTeX: one block per dataset, (a) search | (b) final model
    lines = [r'\begin{table}[ht]', r'\centering',
             r'\caption{One-off search cost (a) vs.\ cost of the final model (b). Final model: medians over '
             r'the timed runs. Search: measured full GA run; $^\dagger$ projected from one fitness evaluation '
             r'$\times$ number of evaluations. SOTA methods are not tuned.}',
             r'\begin{tabular}{llrrrrr}', r'\hline',
             r' & & \multicolumn{2}{c}{(a) search} & \multicolumn{3}{c}{(b) final model} \\',
             r'Dataset & Method & time [h] & GPU [h] & train [s] & predict [ms/1000] & RAM [MB] \\', r'\hline']
    for d in [x for x in cfg.DATASETS if x in set(c.dataset)]:
        for i, (_, r) in enumerate(c[c.dataset == d].iterrows()):
            lines.append('{} & {} & {} & {} & {} & {} & {} \\\\'.format(
                cfg.DATASETS[d]['label'] if i == 0 else '', r.method_label,
                (_fmt(r.search_hours) + ('' if r.search_source.startswith('measured') else r'$^\dagger$'))
                if r.kind == 'ga' else '--',
                _fmt(r.search_gpu_hours) if r.kind == 'ga' else '--',
                _fmt(r.final_train_s), _fmt(r.predict_ms_per_1000), _fmt(r.peak_rss_mb)))
        lines.append(r'\hline')
    lines += [r'\end{tabular}', r'\label{tab:cost_split}', r'\end{table}', '']
    with open(os.path.join(outdir, 'table_cost_split.tex'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    return c


# G-mean for the accuracy-vs-time figure. This experiment does not measure accuracy;
# the figure should show the G-mean the PAPER reports. Put those values into
# PAPER_GMEAN_CSV (columns method,dataset,gmean); without it, GA methods use the
# fitness the GA reported and SOTA methods the G-mean of the timed upstream run.
PAPER_GMEAN_CSV = os.path.join(HERE, 'paper_gmean.csv')


def accuracy_values(df, outdir):
    rows = []
    for (m, d), g in df.groupby(['method', 'dataset'], observed=True):
        if cfg.METHODS[m]['kind'] == 'ga' and g.gmean_reported_by_ga.notna().any():
            rows.append({'method': m, 'dataset': d, 'gmean': float(g.gmean_reported_by_ga.median()),
                         'source': 'GA-reported fitness (checkpoint)'})
        elif g.gmean.notna().any():
            rows.append({'method': m, 'dataset': d, 'gmean': float(g.gmean.median()),
                         'source': 'timed upstream run'})
    acc = pd.DataFrame(rows, columns=['method', 'dataset', 'gmean', 'source'])
    if os.path.exists(PAPER_GMEAN_CSV):
        paper = pd.read_csv(PAPER_GMEAN_CSV)
        for _, p in paper.iterrows():
            hit = (acc.method == p.method) & (acc.dataset == p.dataset)
            if hit.any():
                acc.loc[hit, ['gmean', 'source']] = [float(p.gmean), 'paper_gmean.csv']
            else:
                acc.loc[len(acc)] = [p.method, p.dataset, float(p.gmean), 'paper_gmean.csv']
    acc.to_csv(os.path.join(outdir, 'accuracy_used_in_figure.csv'), index=False)
    return acc


def plot_accuracy_vs_time(cost, acc, outdir):
    """G-mean vs time, one panel per dataset. Filled marker: final training only.
    GA-tuned methods also get a hollow marker at search + final training, joined by
    a thin line - the horizontal distance is what the search costs."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t = cost.merge(acc[['method', 'dataset', 'gmean']], on=['method', 'dataset'], how='inner')
    if t.empty:
        return
    datasets = [d for d in cfg.DATASETS if d in set(t.dataset)]
    methods = [m for m in cfg.METHODS if m in set(t.method)]
    col = _colors(methods)
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.8 * len(datasets), 3.6), squeeze=False)
    for ax, d in zip(axes[0], datasets):
        _style(ax)
        ax.grid(axis='x', which='major', color=GRID, linewidth=0.8)
        for _, r in t[t.dataset == d].iterrows():
            c = col[r.method]
            ax.scatter(r.final_train_s, r.gmean, s=64, color=c, edgecolor='white', linewidth=2, zorder=4)
            if r.kind == 'ga' and r.search_hours > 0:
                total_s = r.total_hours * 3600
                ax.plot([r.final_train_s, total_s], [r.gmean, r.gmean], color=c, linewidth=1, alpha=0.6, zorder=2)
                ax.scatter(total_s, r.gmean, s=64, facecolor='white', edgecolor=c, linewidth=2, zorder=4)
        ax.set_xscale('log')
        ax.set_title('{}  (n = {})'.format(cfg.DATASETS[d]['label'], int(t[t.dataset == d].n_rows.iloc[0])),
                     fontsize=9, color=INK)
        ax.set_xlabel('Time [s], log scale', fontsize=8, color=INK)
    axes[0][0].set_ylabel('G-mean', fontsize=8, color=INK)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker='o', linestyle='', markersize=7, color=col[m], markeredgecolor='white',
                      label=cfg.METHODS[m]['label']) for m in methods]
    handles += [Line2D([], [], marker='o', linestyle='', markersize=7, color=INK_MUTED, label='final model training'),
                Line2D([], [], marker='o', linestyle='', markersize=7, markerfacecolor='white',
                       markeredgecolor=INK_MUTED, label='GA search + final training')]
    fig.legend(handles=handles, fontsize=7, frameon=False, loc='center left', bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'fig_accuracy_vs_time.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)


def _ga_curves(cost):
    """(method, dataset) -> list of curves [(seconds array, best-so-far array), ...] and source.
    Measured full GA runs when present; otherwise the paper's checkpoint curve
    (convergence_curves.csv from ga_convergence.py) with the ESTIMATED search time spread
    evenly over the generations."""
    out = {}
    for (m, d), recs in load_full_ga().items():
        cs = [(np.array([p['cumulative_seconds'] for p in r['curve']]), np.array([p['best_so_far'] for p in r['curve']]))
              for r in recs if r.get('curve')]
        if cs:
            out[(m, d)] = (cs, 'measured ({} run{})'.format(len(cs), 's' if len(cs) > 1 else ''))
    path = os.path.join(cfg.SUMMARY_DIR, 'convergence_curves.csv')
    if os.path.exists(path):
        cv = pd.read_csv(path)
        for (m, d), g in cv.groupby(['method', 'dataset']):
            if (m, d) in out:
                continue
            hit = cost[(cost.method == m) & (cost.dataset == d)]
            if hit.empty or not hit.search_hours_estimated.notna().any():
                continue
            g = g.sort_values('generation')
            total = float(hit.search_hours_estimated.iloc[0]) * 3600
            secs = total * (g.generation.to_numpy() + 1) / (g.generation.max() + 1)
            out[(m, d)] = ([(secs, g.best_so_far.to_numpy(float))], 'paper checkpoint, estimated time')
    return out


def ga_vs_sota(df, cost, acc, outdir):
    """Compares every GA-tuned method with every SOTA method on each dataset, two ways:

    A  untuned   the random first generation of the paper's GA = the method WITHOUT
                 tuning; cost = one fitness evaluation. Share of untuned configurations
                 that already reach the SOTA G-mean.
    C  time-to-quality  along the GA's best-so-far curve: search time until it reaches
                 the SOTA G-mean (and as a multiple of the SOTA training time), and the
                 best G-mean reached within the SOTA training time.
    """
    gm = acc.set_index(['method', 'dataset'])['gmean'].to_dict()
    path = os.path.join(cfg.SUMMARY_DIR, 'untuned_values.csv')
    untuned = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=['method', 'dataset', 'gmean'])
    curves = _ga_curves(cost)
    eval_s = df.groupby(['method', 'dataset'], observed=True)['fitness_eval_seconds'].median().to_dict() \
        if 'fitness_eval_seconds' in df else {}
    # a measured first generation gives the real time of an untuned evaluation
    for (m, d), recs in load_full_ga().items():
        g0 = [r['per_generation']['0'] for r in recs if '0' in (r.get('per_generation') or {})]
        if g0:
            eval_s[(m, d)] = float(np.median([g['seconds'] / g['evaluations'] for g in g0]))

    rows = []
    sota = cost[cost.kind == 'sota']
    for (m, d), g in cost[cost.kind == 'ga'].groupby(['method', 'dataset'], observed=True):
        u = untuned[(untuned.method == m) & (untuned.dataset == d)].gmean.to_numpy(float)
        cs, src = curves.get((m, d), ([], None))
        for _, s in sota[sota.dataset == d].iterrows():
            target, s_time = gm.get((s.method, d)), float(s.final_train_s)
            row = {'ga_method': m, 'ga_label': cfg.METHODS[m]['label'], 'dataset': d,
                   'sota_method': s.method, 'sota_label': s.method_label,
                   'sota_gmean': target, 'sota_train_s': s_time,
                   # A
                   'untuned_n': len(u), 'untuned_gmean_median': float(np.median(u)) if len(u) else np.nan,
                   'untuned_gmean_q1': float(np.percentile(u, 25)) if len(u) else np.nan,
                   'untuned_gmean_q3': float(np.percentile(u, 75)) if len(u) else np.nan,
                   'untuned_share_beating_sota_pct': 100 * float(np.mean(u >= target))
                   if len(u) and target is not None else np.nan,
                   'untuned_eval_s': eval_s.get((m, d), np.nan),
                   'untuned_eval_vs_sota_train': eval_s.get((m, d), np.nan) / s_time if s_time else np.nan,
                   # C
                   'curve_source': src, 'curve_runs': len(cs)}
            if cs and target is not None:
                t_beat = []
                within = []
                for secs, best in cs:
                    hit = np.flatnonzero(best >= target)
                    t_beat.append(float(secs[hit[0]]) if len(hit) else np.nan)
                    ok = secs <= s_time
                    within.append(float(best[ok].max()) if ok.any() else np.nan)
                t_beat = np.array(t_beat)
                reached = ~np.isnan(t_beat)
                row.update({
                    'runs_reaching_sota': int(reached.sum()),
                    'time_to_sota_s_median': float(np.median(t_beat[reached])) if reached.any() else np.nan,
                    'time_to_sota_s_min': float(np.min(t_beat[reached])) if reached.any() else np.nan,
                    'time_to_sota_s_max': float(np.max(t_beat[reached])) if reached.any() else np.nan,
                    'time_to_sota_vs_sota_train': float(np.median(t_beat[reached])) / s_time
                    if reached.any() and s_time else np.nan,
                    'gmean_within_sota_train_time': float(np.nanmedian(within)) if not np.all(np.isnan(within))
                    else np.nan,
                })
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(outdir, 'ga_vs_sota.csv'), index=False)
    if not out.empty:
        lines = [r'\begin{table}[ht]', r'\centering',
                 r'\caption{GA-tuned methods vs.\ SOTA. Untuned: share of random (initial-population) '
                 r'configurations reaching the SOTA G-mean, cost of one evaluation. Search: GA time until '
                 r'the best-so-far G-mean reaches the SOTA G-mean, in hours and as a multiple of the SOTA '
                 r'training time (--: not reached).}',
                 r'\begin{tabular}{lllrrrr}', r'\hline',
                 r' & & & \multicolumn{2}{c}{untuned} & \multicolumn{2}{c}{GA search to SOTA level} \\',
                 r'Dataset & Method & vs. & reach [\%] & eval [s] & time [h] & $\times$ SOTA train \\', r'\hline']
        for d in [x for x in cfg.DATASETS if x in set(out.dataset)]:
            first = True
            for _, r in out[out.dataset == d].iterrows():
                lines.append('{} & {} & {} & {} & {} & {} & {} \\\\'.format(
                    cfg.DATASETS[d]['label'] if first else '', r.ga_label, r.sota_label,
                    _fmt(r.untuned_share_beating_sota_pct), _fmt(r.untuned_eval_s),
                    _fmt(r.get('time_to_sota_s_median', np.nan) / 3600), _fmt(r.get('time_to_sota_vs_sota_train', np.nan))))
                first = False
            lines.append(r'\hline')
        lines += [r'\end{tabular}', r'\label{tab:ga_vs_sota}', r'\end{table}', '']
        with open(os.path.join(outdir, 'table_ga_vs_sota.tex'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
    return out, curves, untuned


def plot_time_to_quality(gvs, curves, untuned, cost, acc, outdir):
    """Per dataset: GA best-so-far G-mean vs cumulative search time (lines, one per run),
    the untuned method (hollow diamond: median, bar: IQR of the random first generation,
    at the time of one evaluation) and the SOTA methods (dots at their training time)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    if not curves and untuned.empty:
        return
    gm = acc.set_index(['method', 'dataset'])['gmean'].to_dict()
    datasets = [d for d in cfg.DATASETS if d in set(cost.dataset)]
    methods = [m for m in cfg.METHODS if m in set(cost.method)]
    col = _colors(methods)
    eval_s = gvs.drop_duplicates(['ga_method', 'dataset']).set_index(['ga_method', 'dataset'])['untuned_eval_s'] \
        .to_dict() if not gvs.empty else {}
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.9 * len(datasets), 3.8), squeeze=False)
    for ax, d in zip(axes[0], datasets):
        _style(ax)
        ax.grid(axis='x', which='major', color=GRID, linewidth=0.8)
        for m in [x for x in methods if cfg.METHODS[x]['kind'] == 'ga']:
            cs, _ = curves.get((m, d), ([], None))
            for secs, best in cs:
                ax.step(secs, best, where='post', color=col[m], linewidth=2 if len(cs) == 1 else 1.2,
                        alpha=1 if len(cs) == 1 else 0.55, zorder=3)
            u = untuned[(untuned.method == m) & (untuned.dataset == d)].gmean.to_numpy(float)
            x = eval_s.get((m, d))
            if len(u) and x and not np.isnan(x):
                q1, med, q3 = np.percentile(u, [25, 50, 75])
                ax.plot([x, x], [q1, q3], color=col[m], linewidth=2, zorder=4)
                ax.scatter(x, med, marker='D', s=48, facecolor='white', edgecolor=col[m], linewidth=2, zorder=5)
        for _, s in cost[(cost.dataset == d) & (cost.kind == 'sota')].iterrows():
            y = gm.get((s.method, d))
            if y is not None:
                ax.scatter(s.final_train_s, y, s=64, color=col[s.method], edgecolor='white', linewidth=2, zorder=6)
        ax.set_xscale('log')
        ax.set_title(cfg.DATASETS[d]['label'], fontsize=9, color=INK)
        ax.set_xlabel('Time [s], log scale', fontsize=8, color=INK)
    axes[0][0].set_ylabel('G-mean', fontsize=8, color=INK)
    handles = [Line2D([], [], color=col[m], linewidth=2, label=cfg.METHODS[m]['label'] + ' (GA search)')
               for m in methods if cfg.METHODS[m]['kind'] == 'ga']
    handles += [Line2D([], [], marker='o', linestyle='', markersize=7, color=col[m], markeredgecolor='white',
                       label=cfg.METHODS[m]['label']) for m in methods if cfg.METHODS[m]['kind'] == 'sota']
    handles += [Line2D([], [], marker='D', linestyle='', markersize=6, markerfacecolor='white',
                       markeredgecolor=INK_MUTED, label='untuned: median, IQR of random configs')]
    fig.legend(handles=handles, fontsize=7, frameon=False, loc='center left', bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'fig_time_to_quality.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)


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
    cost = cost_split(df, cfg.SUMMARY_DIR)
    acc = accuracy_values(df, cfg.SUMMARY_DIR)
    plot_accuracy_vs_time(cost, acc, cfg.SUMMARY_DIR)
    gvs, curves, untuned = ga_vs_sota(df, cost, acc, cfg.SUMMARY_DIR)
    plot_time_to_quality(gvs, curves, untuned, cost, acc, cfg.SUMMARY_DIR)
    with pd.option_context('display.width', 160, 'display.float_format', '{:.3g}'.format):
        print('\nCost split (a) search vs (b) final model:\n' + cost[
            ['method_label', 'dataset', 'search_hours', 'search_gpu_hours', 'final_train_s',
             'predict_ms_per_1000', 'total_hours']].to_string(index=False))

    view = s[s.metric == 'wall_seconds'][['method_label', 'dataset', 'n', 'mean', 'std', 'median', 'min',
                                          'max', 'cv_pct', 'ratio_to_ga_heso']]
    with pd.option_context('display.width', 160, 'display.float_format', '{:.2f}'.format):
        print(view.to_string(index=False))
        if not sc.empty:
            print('\nScaling exponents (time ~ n^b):\n' + sc[['method_label', 'metric', 'exponent_b']].to_string(index=False))
    print('\n-> {}'.format(cfg.SUMMARY_DIR))


if __name__ == '__main__':
    main()
