"""
MLOS run the original way (no GA) on the synthetic datasets: upstream MLOS.fit_Sample
from the untouched copy in ga_heso_sota_methods/MLOS_ORIGINAL, called as in
upstream's own __main__, on the training part of each fold. A thin wrapper
around optimization/no_ga_mlos_original_runner.py, which holds the evaluation.

Where the majority training data has more features than rows, upstream MLOS
finds them linearly dependent and returns the data unchanged (no
oversampling); the runner records this per fit ('mlos_branch').

Results go to predictions/synthetic/results/mlos_original/mlos_original_{dataset}.*
"""

import os
import time

from base_functions import get_synthetic_data
from optimization.no_ga_mlos_original_runner import run_mlos_original

METHOD = 'mlos_original'
_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', METHOD)
_CATEGORICAL_COLS = None


def __run(data, dataset_name):
    start_time = time.time()
    run_mlos_original(
        data, dataset_name,
        results_file=os.path.join(_RESULTS_DIR, '{}_{}'.format(METHOD, dataset_name)),
        categorical_cols=_CATEGORICAL_COLS,
    )
    print("--- {}: {:.0f} seconds ---".format(dataset_name, time.time() - start_time))


if __name__ == '__main__':
    __run(get_synthetic_data(prefix='01', contamination='0.3', features=50),
          '01_synthetic_0.3_contamination_50_features')
    __run(get_synthetic_data(prefix='02', contamination='0.1', features=100, samples=200),
          '02_synthetic_0.1_contamination_100_features_200_samples')
    __run(get_synthetic_data(prefix='03', contamination='0.02', features=200, samples=200),
          '03_synthetic_0.02_contamination_200_features_200_samples')
