"""
MLOS run the original way (no GA) on the ecoli datasets: upstream MLOS.fit_Sample
from the untouched copy in ga_heso_sota_methods/MLOS_ORIGINAL, called as in
upstream's own __main__, on the training part of each fold. A thin wrapper
around optimization/no_ga_mlos_original_runner.py, which holds the evaluation.

In ecoli_0_vs_1 label 1 is the majority; the runner maps labels to upstream's
convention (majority 1, minority 0) per dataset and back for reporting.

Results go to predictions/ecoli/results/mlos_original/mlos_original_{dataset}.*
"""

import os
import time

from base_functions import (get_ecoli_0_vs_1_data, get_ecoli_0_2_3_4_vs_5_data,
                            get_ecoli_0_3_4_vs_5_data, get_ecoli_0_4_6_vs_5_data)
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
    __run(get_ecoli_0_vs_1_data(),
          'ecoli_0_vs_1')
    __run(get_ecoli_0_2_3_4_vs_5_data(),
          'ecoli_0_2_3_4_vs_5')
    __run(get_ecoli_0_3_4_vs_5_data(),
          'ecoli_0_3_4_vs_5')
    __run(get_ecoli_0_4_6_vs_5_data(),
          'ecoli_0_4_6_vs_5')
