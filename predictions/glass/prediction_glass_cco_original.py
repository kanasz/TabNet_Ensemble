"""
CCO run the original way (no GA) on the glass datasets: upstream CCO's own
main.py pipeline from the untouched copy in ga_heso_sota_methods/CCO_ORIGINAL,
one run per dataset. A thin wrapper around
optimization/no_ga_cco_original_runner.py, which holds the CCO-driving logic.

All glass features are continuous.

Results go to predictions/glass/results/cco_original/cco_original_{dataset}.*
"""

import os
import time

from base_functions import get_glass_0_1_6_vs_5_data, get_glass_2_data, get_glass_4_data, get_glass_5_data
from optimization.no_ga_cco_original_runner import run_cco_original

METHOD = 'cco_original'
_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', METHOD)
_CATEGORICAL_COLS = None


def __run(data, dataset_name):
    start_time = time.time()
    run_cco_original(
        data, dataset_name,
        results_file=os.path.join(_RESULTS_DIR, '{}_{}'.format(METHOD, dataset_name)),
        categorical_cols=_CATEGORICAL_COLS,
    )
    print("--- {}: {:.0f} seconds ---".format(dataset_name, time.time() - start_time))


if __name__ == '__main__':
    __run(get_glass_0_1_6_vs_5_data(),
          'glass_0_1_6_vs_5')
    __run(get_glass_2_data(),
          'glass_2')
    __run(get_glass_4_data(),
          'glass_4')
    __run(get_glass_5_data(),
          'glass_5')
