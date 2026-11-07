"""
Verifies that the SOS "original way" baseline runs on every GA-HESO dataset.

Runs optimization/no_ga_sos_original_runner.py::run_sos_original() on all 24
datasets with training.n_iters=1 (default) on fold exp0 only. That exercises
every data-dependent step of upstream SOS (metadata, GeneralTransformer,
both score networks, checkpoint save/restore, sampling from z_major and N,
fine-tuning, evaluation.py's classifiers, the recording cross-check, AUC,
output files) in minutes instead of hours. Only the number of training steps
and folds differ from the real run, so a PASS here means the real run will
not fail on the data. The metrics printed are meaningless at this budget.

    python test_sos_original_all_datasets.py                  # all 24 datasets, fold 0
    python test_sos_original_all_datasets.py --only glass_2 ecoli_0_vs_1
    python test_sos_original_all_datasets.py --n-iters 50 --folds 0 1 2 3 4

Outputs go to a temporary directory and the per-dataset files it creates in
ga_heso_sota_methods/SOS_ORIGINAL/{runs,tabular_datasets}/ are removed
afterwards. Exit code is 1 if any dataset fails.
"""

import argparse
import os
import shutil
import sys
import tempfile
import time
import traceback

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT_ROOT)

import base_functions as bf  # noqa: E402
from optimization.no_ga_sos_original_runner import SOS_PATH, run_sos_original  # noqa: E402

# (family, dataset_name, loader, categorical_cols) - the same 24 datasets as
# predictions/*/prediction_*_sos_original.py.
DATASETS = [
    ('wine', 'wine_red_3_vs_5', bf.get_wine_quality_red_3_vs_5_data, None),
    ('wine', 'wine_red_8_vs_6', bf.get_wine_quality_red_8_vs_6_data, None),
    ('wine', 'wine_white_3_vs_7', bf.get_wine_quality_white_3_vs_7_data, None),
    ('wine', 'wine_white_9_vs_4', bf.get_wine_quality_white_9_vs_4_data, None),
    ('yeast', 'yeast_3', bf.get_yeast_3_data, None),
    ('yeast', 'yeast_4', bf.get_yeast_4_data, None),
    ('yeast', 'yeast_5', bf.get_yeast_5_data, None),
    ('yeast', 'yeast_6', bf.get_yeast_6_data, None),
    ('glass', 'glass_0_1_6_vs_5', bf.get_glass_0_1_6_vs_5_data, None),
    ('glass', 'glass_2', bf.get_glass_2_data, None),
    ('glass', 'glass_4', bf.get_glass_4_data, None),
    ('glass', 'glass_5', bf.get_glass_5_data, None),
    ('ecoli', 'ecoli_0_vs_1', bf.get_ecoli_0_vs_1_data, None),
    ('ecoli', 'ecoli_0_2_3_4_vs_5', bf.get_ecoli_0_2_3_4_vs_5_data, None),
    ('ecoli', 'ecoli_0_3_4_vs_5', bf.get_ecoli_0_3_4_vs_5_data, None),
    ('ecoli', 'ecoli_0_4_6_vs_5', bf.get_ecoli_0_4_6_vs_5_data, None),
    ('abalone', 'abalone19', bf.get_abalone19_data, ['Sex']),
    ('abalone', 'abalone_9_vs_18', bf.get_abalone_9_vs_18_data, ['Sex']),
    ('abalone', 'abalone_3_vs_11', bf.get_abalone_3_vs_11_data, ['Sex']),
    ('abalone', 'abalone_19_vs_10_11_12_13', bf.get_abalone_19_vs_10_11_12_13_data, ['Sex']),
    ('abalone', 'abalone_20_vs_8_9_10', bf.get_abalone_20_vs_8_9_10_data, ['Sex']),
    ('synthetic', '01_synthetic_0.3_contamination_50_features',
     lambda: bf.get_synthetic_data(prefix='01', contamination='0.3', features=50), None),
    ('synthetic', '02_synthetic_0.1_contamination_100_features_200_samples',
     lambda: bf.get_synthetic_data(prefix='02', contamination='0.1', features=100, samples=200), None),
    ('synthetic', '03_synthetic_0.02_contamination_200_features_200_samples',
     lambda: bf.get_synthetic_data(prefix='03', contamination='0.02', features=200, samples=200), None),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--n-iters', type=int, default=1, help='training.n_iters per score network')
    parser.add_argument('--folds', type=int, nargs='+', default=[0])
    parser.add_argument('--only', nargs='+', help='dataset names to run (default: all)')
    args = parser.parse_args()

    selected = [d for d in DATASETS if not args.only or d[1] in args.only]
    if not selected:
        raise SystemExit(f"no dataset matches {args.only}")

    out_dir = tempfile.mkdtemp(prefix='sos_original_verify_')
    results = []
    try:
        for family, name, loader, cats in selected:
            run_name = f'verify_{name}'
            t0 = time.time()
            print(f"\n######## {family}/{name} ########", flush=True)
            try:
                x_df, y = loader()
                y_int = np.asarray(y).astype(int)
                result = run_sos_original(
                    (x_df, y), run_name, os.path.join(out_dir, f'sos_original_{run_name}'),
                    categorical_cols=cats, resume=False, folds=args.folds,
                    config_overrides={'training.n_iters': args.n_iters})
                folds = os.path.join(out_dir, f'sos_original_{run_name}_folds.csv')
                n_nan_auc = int(np.isnan(np.genfromtxt(folds, delimiter=',', names=True)['auc']).sum())
                note = f"{n_nan_auc} split(s) with single-class test fold (AUC NaN)" if n_nan_auc else ''
                results.append((name, 'PASS', x_df.shape[0], x_df.shape[1],
                                f"{int((y_int == 1).sum())}/{len(y_int)}",
                                f"{result['fitness']:.3f}", time.time() - t0, note))
            except Exception as exc:  # report and continue with the next dataset
                traceback.print_exc()
                results.append((name, 'FAIL', '', '', '', '', time.time() - t0,
                                f"{type(exc).__name__}: {str(exc).splitlines()[0][:120]}"))
            finally:
                shutil.rmtree(os.path.join(SOS_PATH, 'runs', run_name), ignore_errors=True)
                for k in range(5):
                    for ext in ('npz', 'json'):
                        path = os.path.join(SOS_PATH, 'tabular_datasets', f'{run_name}_exp{k}.{ext}')
                        if os.path.exists(path):
                            os.remove(path)
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)

    print(f"\n==== SOS original: verification with --n-iters {args.n_iters} --folds {args.folds} ====")
    print(f"{'dataset':<58}{'status':<7}{'rows':>6}{'feat':>6}{'label1':>10}{'gmean':>8}{'sec':>7}  note")
    for name, status, rows, feats, label1, gm, sec, note in results:
        print(f"{name:<58}{status:<7}{rows:>6}{feats:>6}{label1:>10}{gm:>8}{sec:>7.0f}  {note}")
    failed = [r[0] for r in results if r[1] == 'FAIL']
    print(f"\n{len(results) - len(failed)}/{len(results)} passed"
          + (f"; FAILED: {', '.join(failed)}" if failed else ''))
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
