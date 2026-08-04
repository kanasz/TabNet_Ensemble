"""
Fixed-baseline (no GA) run of GOIO on the ecoli datasets — thin wrapper
around optimization/no_ga_goio_runner.py, which holds the reusable
GOIO-driving logic (dataset info / output path are inputs there, not
hardcoded), matching the project's CCO/DGOT/SOS tuner-vs-prediction-script
split.

All ecoli features are continuous (no categorical columns), same as yeast.
"""

import os
from base_functions import (get_ecoli_0_vs_1_data, get_ecoli_0_2_3_4_vs_5_data,
                             get_ecoli_0_3_4_vs_5_data, get_ecoli_0_4_6_vs_5_data)
from optimization.no_ga_goio_runner import run_goio_baseline

_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'goio')


def train_and_evaluate_goio_method(data, dataset_name):
    return run_goio_baseline(
        data,
        dataset_name=dataset_name,
        results_file=os.path.join(_RESULTS_DIR, f'goio_{dataset_name}'),
    )


if __name__ == '__main__':

    # ecoli_0_vs_1
    train_and_evaluate_goio_method(data=get_ecoli_0_vs_1_data(), dataset_name='ecoli_0_vs_1')

    # ecoli_0_2_3_4_vs_5
    train_and_evaluate_goio_method(data=get_ecoli_0_2_3_4_vs_5_data(), dataset_name='ecoli_0_2_3_4_vs_5')

    # ecoli_0_3_4_vs_5
    train_and_evaluate_goio_method(data=get_ecoli_0_3_4_vs_5_data(), dataset_name='ecoli_0_3_4_vs_5')

    # ecoli_0_4_6_vs_5
    train_and_evaluate_goio_method(data=get_ecoli_0_4_6_vs_5_data(), dataset_name='ecoli_0_4_6_vs_5')