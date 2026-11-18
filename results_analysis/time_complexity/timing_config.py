"""What the time-complexity experiment measures: methods, datasets, number of runs.

Edit this file (or pass --methods / --datasets / --runs to run_timing_experiment.py)
to change the selection. Everything else reads from here.

Method kinds
------------
ga        GA-tuned in the paper. A timed run retrains the BEST chromosome stored in the
          GA checkpoint (.pkl) on the same 5 stratified folds, with resampling, fit and
          predict timed separately. The cost of the whole GA search is estimated from
          it (time of one fitness evaluation x number of fitness evaluations).
sota      Untuned upstream pipeline, run exactly as for the paper results through
          optimization/no_ga_*_runner.py (resume=False, so nothing is reused).
"""
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
HERE = os.path.dirname(os.path.abspath(__file__))

RAW_DIR = os.path.join(HERE, 'raw')          # one JSON per (method, dataset, run)
WORK_DIR = os.path.join(HERE, 'work')        # scratch output of SOTA runners (overwritten)
SUMMARY_DIR = os.path.join(HERE, 'summary')  # tables and figures

N_RUNS = 20      # timed, independent runs per (method, dataset)
N_WARMUP = 1     # extra untimed-for-statistics run(s) first: CUDA init, disk cache, JIT

# ---------------------------------------------------------------------------
# Datasets. 'getter' is a base_functions function name; 'checkpoints' are GA
# checkpoints (without .pkl) relative to the repo root, per GA method.
# ---------------------------------------------------------------------------
DATASETS = {
    'glass_4': {
        'getter': 'get_glass_4_data', 'label': 'glass4', 'size': 'small',
        'categorical_cols': None,
        'checkpoints': {
            'ga_heso': 'predictions/glass/results/UNCLUSTERED_OC_TABNET_ENSEMBLE_SMOTE_MEANSHIFT_glass_4',
            'smote_tabnet': 'predictions/glass/results/tabnet_smote/tabnet_smote_glass_4',
            'selfpe': 'predictions/glass/results/self_paced_ensemble/spe_glass_4',
            'smote_xgboost': 'predictions/glass/results/xgboost_smote/xgboost_smote_glass_4',
        },
        # upstream DGOT never trains on folds < 512 rows (batch 512 + drop_last);
        # the paper runs used 64 for the small datasets, so the timing does too
        'sota_kwargs': {'dgot_baseline': {'batch_size': 64}},
    },
    'yeast_4': {
        'getter': 'get_yeast_4_data', 'label': 'yeast4', 'size': 'medium',
        'categorical_cols': None,
        'checkpoints': {
            'ga_heso': 'predictions/yeast/results/UNCLUSTERED_OC_TABNET_ENSEMBLE_SMOTE_MEANSHIFT_yeast_4',
            'smote_tabnet': 'predictions/yeast/results/tabnet_smote/tabnet_smote_yeast_4',
            'selfpe': 'predictions/yeast/results/self_paced_ensemble/spe_yeast_4',
            'smote_xgboost': 'predictions/yeast/results/xgboost_smote/xgboost_smote_yeast_4',
        },
        'sota_kwargs': {},
    },
    'abalone19': {
        'getter': 'get_abalone19_data', 'label': 'A4 (abalone19)', 'size': 'large',
        'categorical_cols': ['Sex'],
        'checkpoints': {
            'ga_heso': 'predictions/abalone/results/unclustered_smote_meanshift_abalone19/'
                       'UNCLUSTERED_OC_TABNET_ENSEMBLE_SMOTE_MEANSHIFT_abalone19',
            'smote_tabnet': 'predictions/abalone/results/tabnet_smote/tabnet_smote_abalone19',
            'selfpe': 'predictions/abalone/results/self_paced_ensemble/spe_abalone19',
            'smote_xgboost': 'predictions/abalone/results/xgboost_smote/xgboost_smote_abalone19',
        },
        'sota_kwargs': {},
    },
}

# ---------------------------------------------------------------------------
# Methods. Order here = order in tables and figures.
# ---------------------------------------------------------------------------
METHODS = {
    'ga_heso':       {'kind': 'ga', 'label': 'GA-HESO', 'group': 'proposed'},
    'smote_tabnet':  {'kind': 'ga', 'label': 'SMOTE+TabNet', 'group': 'baseline'},
    'selfpe':        {'kind': 'ga', 'label': 'SelfPE', 'group': 'baseline'},
    'smote_xgboost': {'kind': 'ga', 'label': 'SMOTE+XGBoost', 'group': 'baseline'},
    'dgot_baseline': {'kind': 'sota', 'label': 'DGOT', 'group': 'sota',
                      'runner': 'optimization.no_ga_dgot_runner:run_dgot_baseline', 'resume_arg': True},
    'sos_original':  {'kind': 'sota', 'label': 'SOS', 'group': 'sota',
                      'runner': 'optimization.no_ga_sos_original_runner:run_sos_original', 'resume_arg': True},
    'mlos_original': {'kind': 'sota', 'label': 'MLOS', 'group': 'sota',
                      'runner': 'optimization.no_ga_mlos_original_runner:run_mlos_original', 'resume_arg': False},
    # Available but off by default (CCO takes ~4 h per run on abalone19):
    'cco_original':  {'kind': 'sota', 'label': 'CCO', 'group': 'sota',
                      'runner': 'optimization.no_ga_cco_original_runner:run_cco_original', 'resume_arg': False},
    'goio_original': {'kind': 'sota', 'label': 'GOIO', 'group': 'sota',
                      'runner': 'optimization.no_ga_goio_original_runner:run_goio_original', 'resume_arg': True},
}

DEFAULT_METHODS = ['ga_heso', 'smote_tabnet', 'selfpe', 'smote_xgboost',
                   'dgot_baseline', 'sos_original', 'mlos_original']
DEFAULT_DATASETS = ['glass_4', 'yeast_4', 'abalone19']

# Upstream SOTA code writes into its own folders under this name, so timing runs
# never touch the folders/caches behind the paper results.
SOTA_DATASET_PREFIX = 'timing_'


def raw_path(method, dataset, run_id):
    return os.path.join(RAW_DIR, method, dataset, 'run_{:02d}.json'.format(run_id))
