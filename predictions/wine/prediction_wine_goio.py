"""
Fixed-baseline (no GA) run of GOIO on the wine datasets — thin wrapper
around optimization/no_ga_goio_runner.py, which holds the reusable
GOIO-driving logic (dataset info / output path are inputs there, not
hardcoded), matching the project's CCO/DGOT/SOS tuner-vs-prediction-script
split.

All wine features are continuous (no categorical columns), same as yeast.
"""

import os
from base_functions import (get_wine_quality_red_3_vs_5_data, get_wine_quality_red_8_vs_6_data,
                             get_wine_quality_white_3_vs_7_data, get_wine_quality_white_9_vs_4_data)
from optimization.no_ga_goio_runner import run_goio_baseline

_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'goio')


def train_and_evaluate_goio_method(data, dataset_name):
    return run_goio_baseline(
        data,
        dataset_name=dataset_name,
        results_file=os.path.join(_RESULTS_DIR, f'goio_{dataset_name}'),
    )


if __name__ == '__main__':

    # wine_red_3_vs_5
    train_and_evaluate_goio_method(data=get_wine_quality_red_3_vs_5_data(), dataset_name='wine_red_3_vs_5')

    # wine_red_8_vs_6
    train_and_evaluate_goio_method(data=get_wine_quality_red_8_vs_6_data(), dataset_name='wine_red_8_vs_6')

    # wine_white_3_vs_7
    train_and_evaluate_goio_method(data=get_wine_quality_white_3_vs_7_data(), dataset_name='wine_white_3_vs_7')

    # wine_white_9_vs_4
    train_and_evaluate_goio_method(data=get_wine_quality_white_9_vs_4_data(), dataset_name='wine_white_9_vs_4')