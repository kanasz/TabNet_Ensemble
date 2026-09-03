"""
Configuration for parse_results.py.

Point DOMAINS at the dataset folders under predictions/ and TARGETS at the
method folders under predictions/{domain}/results/ that should be parsed.
Every setting here can be overridden on the command line, e.g.

    python parse_results.py --domains wine yeast --targets svc xgboost_smote
    python parse_results.py --domains abalone --targets "*" --top-n 3
"""

# Dataset folders under predictions/ (e.g. 'wine', 'yeast', 'abalone', 'ecoli').
DOMAINS = ['wine', 'ecoli']

# Method folders under predictions/{domain}/results/.
# Use ['*'] to parse every method folder found for the domain.
TARGETS = [
    'self_paced_ensemble',
    'balanced_cascade',
    'svc',
    'svc_smote',
    'svc_smoteenn',
    'xgboost',
    'xgboost_smote',
    'xgboost_smoteenn'
]

# Optional per-domain override of TARGETS, e.g. {'yeast': ['cco', 'dgot']}.
TARGETS_PER_DOMAIN = {}

# Keep only the first N files per target. None keeps all of them.
TOP_N = None

# Number of decimals for the printed scores.
DECIMALS = 4
