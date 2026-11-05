"""
DGOT baseline (no GA) on the yeast datasets: the published DGOT pipeline with the
paper's default hyperparameters, one run per dataset. A thin wrapper around
optimization/no_ga_dgot_runner.py, which holds the DGOT-driving logic.

All yeast features are continuous.

Results go to predictions/yeast/results/dgot_baseline/dgot_baseline_{dataset}.*
"""

import os
import time

from base_functions import get_yeast_3_data, get_yeast_4_data, get_yeast_5_data, get_yeast_6_data
from optimization.no_ga_dgot_runner import PAPER_BATCH_SIZE, run_dgot_baseline

METHOD = 'dgot_baseline'
# Absolute: the runner changes into the DGOT directory while it works.
_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', METHOD)
_CATEGORICAL_COLS = None


def __run(data, dataset_name, batch_size=PAPER_BATCH_SIZE):
    start_time = time.time()
    run_dgot_baseline(
        data, dataset_name,
        results_file=os.path.join(_RESULTS_DIR, '{}_{}'.format(METHOD, dataset_name)),
        categorical_cols=_CATEGORICAL_COLS,
        batch_size=batch_size,
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
