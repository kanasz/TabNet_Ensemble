"""
SOS run the original way (no GA) on the abalone datasets: upstream SOS's own
main.py (train, then fine_tune) from the untouched copy in
ga_heso_sota_methods/SOS_ORIGINAL, upstream's VE config, one run per dataset.
A thin wrapper around optimization/no_ga_sos_original_runner.py, which holds
the SOS-driving logic.

Abalone has one true categorical feature ('Sex': M/F/I), declared categorical
in SOS's column metadata the way upstream's own datasets declare theirs.

Results go to predictions/abalone/results/sos_original/sos_original_{dataset}.*
"""

import os
import time

from base_functions import (get_abalone19_data, get_abalone_9_vs_18_data, get_abalone_3_vs_11_data,
                            get_abalone_19_vs_10_11_12_13_data, get_abalone_20_vs_8_9_10_data)
from optimization.no_ga_sos_original_runner import run_sos_original

METHOD = 'sos_original'
_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', METHOD)
_CATEGORICAL_COLS = ['Sex']


def __run(data, dataset_name):
    start_time = time.time()
    run_sos_original(
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
