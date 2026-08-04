"""
Fixed-baseline (no GA) run of GOIO on the glass datasets — thin wrapper
around optimization/no_ga_goio_runner.py, which holds the reusable
GOIO-driving logic (dataset info / output path are inputs there, not
hardcoded), matching the project's CCO/DGOT/SOS tuner-vs-prediction-script
split.

All glass features are continuous (no categorical columns), same as yeast.
"""

import os
from base_functions import (get_glass_0_1_6_vs_5_data, get_glass_2_data,
                             get_glass_4_data, get_glass_5_data)
from optimization.no_ga_goio_runner import run_goio_baseline

_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'goio')


def train_and_evaluate_goio_method(data, dataset_name):
    return run_goio_baseline(
        data,
        dataset_name=dataset_name,
        results_file=os.path.join(_RESULTS_DIR, f'goio_{dataset_name}'),
    )


if __name__ == '__main__':

    # glass_0_1_6_vs_5
    train_and_evaluate_goio_method(data=get_glass_0_1_6_vs_5_data(), dataset_name='glass_0_1_6_vs_5')

    # glass_2
    train_and_evaluate_goio_method(data=get_glass_2_data(), dataset_name='glass_2')

    # glass_4
    train_and_evaluate_goio_method(data=get_glass_4_data(), dataset_name='glass_4')

    # glass_5
    train_and_evaluate_goio_method(data=get_glass_5_data(), dataset_name='glass_5')