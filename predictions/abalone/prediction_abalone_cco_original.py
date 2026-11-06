"""
CCO run the original way (no GA) on the abalone datasets: upstream CCO's own
main.py pipeline from the untouched copy in ga_heso_sota_methods/CCO_ORIGINAL,
one run per dataset. A thin wrapper around
optimization/no_ga_cco_original_runner.py, which holds the CCO-driving logic.

Abalone has one true categorical feature ('Sex': M/F/I). CCO measures raw
distances, so it is label-encoded to a single 0/1/2 column first.

Results go to predictions/abalone/results/cco_original/cco_original_{dataset}.*
"""

import os
import time

from base_functions import (get_abalone19_data, get_abalone_9_vs_18_data, get_abalone_3_vs_11_data,
                            get_abalone_19_vs_10_11_12_13_data, get_abalone_20_vs_8_9_10_data)
from optimization.no_ga_cco_original_runner import run_cco_original

METHOD = 'cco_original'
_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', METHOD)
_CATEGORICAL_COLS = ['Sex']


def __run(data, dataset_name):
    start_time = time.time()
    run_cco_original(
        data, dataset_name,
        results_file=os.path.join(_RESULTS_DIR, '{}_{}'.format(METHOD, dataset_name)),
        categorical_cols=_CATEGORICAL_COLS,
    )
    print("--- {}: {:.0f} seconds ---".format(dataset_name, time.time() - start_time))


if __name__ == '__main__':
    __run(get_abalone19_data(),
          'abalone19')
    __run(get_abalone_9_vs_18_data(),
          'abalone_9_vs_18')
    __run(get_abalone_3_vs_11_data(),
          'abalone_3_vs_11')
    __run(get_abalone_19_vs_10_11_12_13_data(),
          'abalone_19_vs_10_11_12_13')
    __run(get_abalone_20_vs_8_9_10_data(),
          'abalone_20_vs_8_9_10')
