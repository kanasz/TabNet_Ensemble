"""
GOIO run the original way (no GA) on the ecoli datasets: upstream GOIO's own
main.py (data split, MLVAE train, CLDM train / sample / eval) from the
untouched copy in ga_heso_sota_methods/GOIO_ORIGINAL with upstream's default
arguments, one run per dataset. A thin wrapper around
optimization/no_ga_goio_original_runner.py, which holds the GOIO-driving logic.

ecoli_0_vs_1 has label 1 as the majority class; the runner swaps the labels
for GOIO (upstream expects majority = 0) and back for reporting.

Results go to predictions/ecoli/results/goio_original/goio_original_{dataset}.*
"""

import os
import time

from base_functions import (get_ecoli_0_vs_1_data, get_ecoli_0_2_3_4_vs_5_data,
                            get_ecoli_0_3_4_vs_5_data, get_ecoli_0_4_6_vs_5_data)
from optimization.no_ga_goio_original_runner import run_goio_original

METHOD = 'goio_original'
_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', METHOD)
_CATEGORICAL_COLS = None


def __run(data, dataset_name):
    start_time = time.time()
    run_goio_original(
        data, dataset_name,
        results_file=os.path.join(_RESULTS_DIR, '{}_{}'.format(METHOD, dataset_name)),
        categorical_cols=_CATEGORICAL_COLS,
    )
    print("--- {}: {:.0f} seconds ---".format(dataset_name, time.time() - start_time))


if __name__ == '__main__':
    __run(get_ecoli_0_vs_1_data(),
          'ecoli_0_vs_1')
    __run(get_ecoli_0_2_3_4_vs_5_data(),
          'ecoli_0_2_3_4_vs_5')
    __run(get_ecoli_0_3_4_vs_5_data(),
          'ecoli_0_3_4_vs_5')
    __run(get_ecoli_0_4_6_vs_5_data(),
          'ecoli_0_4_6_vs_5')
