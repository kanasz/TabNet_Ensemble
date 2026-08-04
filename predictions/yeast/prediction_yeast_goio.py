"""
Fixed-baseline (no GA) run of GOIO on the yeast datasets — thin wrapper
around optimization/no_ga_goio_runner.py, which holds the reusable
GOIO-driving logic (dataset info / output path are inputs there, not
hardcoded), matching the project's CCO/DGOT/SOS tuner-vs-prediction-script
split.
"""

import os
from base_functions import get_yeast_3_data, get_yeast_4_data, get_yeast_5_data, get_yeast_6_data
from optimization.no_ga_goio_runner import run_goio_baseline

_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'goio')

def train_and_evaluate_goio_method(data, dataset_name):
    return run_goio_baseline(
        data,
        dataset_name=dataset_name,
        results_file=os.path.join(_RESULTS_DIR, f'goio_{dataset_name}'),
    )


if __name__ == '__main__':

    # yeast_3
    train_and_evaluate_goio_method(data=get_yeast_3_data(), dataset_name='yeast_3')

    # yeast_4
    train_and_evaluate_goio_method(data=get_yeast_4_data(), dataset_name='yeast_4')

    # yeast_5
    train_and_evaluate_goio_method(data=get_yeast_5_data(), dataset_name='yeast_5')

    # yeast_6
    train_and_evaluate_goio_method(data=get_yeast_6_data(), dataset_name='yeast_6')
