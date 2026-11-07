"""
SOS run the original way (no GA) on the ecoli datasets: upstream SOS's own
main.py (train, then fine_tune) from the untouched copy in
ga_heso_sota_methods/SOS_ORIGINAL, upstream's VE config, one run per dataset.
A thin wrapper around optimization/no_ga_sos_original_runner.py, which holds
the SOS-driving logic.

All ecoli features are continuous. In ecoli_0_vs_1 label 1 is the majority
class and 'Chg' is constant; the runner handles both (see its docstring).

Results go to predictions/ecoli/results/sos_original/sos_original_{dataset}.*
"""

import os
import time

from base_functions import (get_ecoli_0_vs_1_data, get_ecoli_0_2_3_4_vs_5_data,
                            get_ecoli_0_3_4_vs_5_data, get_ecoli_0_4_6_vs_5_data)
from optimization.no_ga_sos_original_runner import run_sos_original

METHOD = 'sos_original'
_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', METHOD)
_CATEGORICAL_COLS = None


def __run(data, dataset_name):
    start_time = time.time()
    run_sos_original(
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
