"""
Fixed-baseline (no GA) run of GOIO on the yeast datasets — thin wrapper
around optimization/no_ga_goio_runner.py, which holds the reusable
GOIO-driving logic (dataset info / output path are inputs there, not
hardcoded), matching the project's CCO/DGOT/SOS tuner-vs-prediction-script
split.
"""

import os
from optimization.no_ga_goio_runner import run_goio_baseline
from base_functions import get_synthetic_data

_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'goio')


def train_and_evaluate_goio_method(data, dataset_name):
    return run_goio_baseline(
        data,
        dataset_name=dataset_name,
        results_file=os.path.join(_RESULTS_DIR, f'goio_{dataset_name}'),
    )


if __name__ == '__main__':

    # Dataset synthetic1: data/synthetic_data/01_synthetic_0.3_contamination_50_features.csv
    train_and_evaluate_goio_method(data=get_synthetic_data(prefix='01', contamination='0.3', features=50),
                                   dataset_name='01_synthetic_0.3_contamination_50_features')

    # Dataset synthetic2: data/synthetic_data/02_synthetic_0.1_contamination_100_features_200_samples.csv
    train_and_evaluate_goio_method(data=get_synthetic_data(prefix='02', contamination='0.1', features=100),
                                   dataset_name='02_synthetic_0.1_contamination_100_features_200_samples')
