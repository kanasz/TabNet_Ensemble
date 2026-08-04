"""
Fixed-baseline (no GA) run of GOIO on the abalone datasets — thin wrapper
around optimization/no_ga_goio_runner.py, which holds the reusable
GOIO-driving logic (dataset info / output path are inputs there, not
hardcoded), matching the project's CCO/DGOT/SOS tuner-vs-prediction-script
split.

Abalone has one true categorical feature ('Sex': M/F/I) — passed through as
categorical_cols so it gets label-encoded (0/1/2) rather than written as raw
strings, matching what GOIO's own data_description() auto-detection expects.
"""

import os
from base_functions import get_abalone_9_vs_18_data, get_abalone_3_vs_11_data, get_abalone_19_vs_10_11_12_13_data
from base_functions import get_abalone_20_vs_8_9_10_data, get_abalone19_data
from optimization.no_ga_goio_runner import run_goio_baseline

_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'goio')
_CATEGORICAL_COLS = ['Sex']


def train_and_evaluate_goio_method(data, dataset_name):
    return run_goio_baseline(
        data,
        dataset_name=dataset_name,
        results_file=os.path.join(_RESULTS_DIR, f'goio_{dataset_name}'),
        categorical_cols=_CATEGORICAL_COLS,
    )


if __name__ == '__main__':

    # abalone19
    train_and_evaluate_goio_method(data=get_abalone19_data(), dataset_name='abalone19')

    # abalone_9_vs_18
    train_and_evaluate_goio_method(data=get_abalone_9_vs_18_data(), dataset_name='abalone_9_vs_18')

    # abalone_3_vs_11
    train_and_evaluate_goio_method(data=get_abalone_3_vs_11_data(), dataset_name='abalone_3_vs_11')

    # abalone_19_vs_10_11_12_13
    train_and_evaluate_goio_method(data=get_abalone_19_vs_10_11_12_13_data(), dataset_name='abalone_19_vs_10_11_12_13')

    # abalone_20_vs_8_9_10
    train_and_evaluate_goio_method(data=get_abalone_20_vs_8_9_10_data(), dataset_name='abalone_20_vs_8_9_10')