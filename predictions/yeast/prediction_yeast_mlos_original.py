"""
MLOS run the original way (no GA) on the yeast datasets: upstream MLOS.fit_Sample
from the untouched copy in ga_heso_sota_methods/MLOS_ORIGINAL, called as in
upstream's own __main__, on the training part of each fold. A thin wrapper
around optimization/no_ga_mlos_original_runner.py, which holds the evaluation.

Results go to predictions/yeast/results/mlos_original/mlos_original_{dataset}.*
"""

import os
import time

from base_functions import get_yeast_3_data, get_yeast_4_data, get_yeast_5_data, get_yeast_6_data
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
    __run(get_yeast_3_data(),
          'yeast_3')
    __run(get_yeast_4_data(),
          'yeast_4')
    __run(get_yeast_5_data(),
          'yeast_5')
    __run(get_yeast_6_data(),
          'yeast_6')
