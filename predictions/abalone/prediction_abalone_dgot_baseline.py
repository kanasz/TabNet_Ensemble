"""
DGOT baseline (no GA) on the abalone datasets: the published DGOT pipeline with the
paper's default hyperparameters, one run per dataset. A thin wrapper around
optimization/no_ga_dgot_runner.py, which holds the DGOT-driving logic.

Abalone has one true categorical feature ('Sex': M/F/I). It is label-encoded
to a single 0/1/2 column, the way upstream's own abalone_15 example stores it.

Results go to predictions/abalone/results/dgot_baseline/dgot_baseline_{dataset}.*
"""

import os
import time

from base_functions import (get_abalone19_data, get_abalone_9_vs_18_data, get_abalone_3_vs_11_data,
                            get_abalone_19_vs_10_11_12_13_data, get_abalone_20_vs_8_9_10_data)
from optimization.no_ga_dgot_runner import PAPER_BATCH_SIZE, run_dgot_baseline

METHOD = 'dgot_baseline'
# Absolute: the runner changes into the DGOT directory while it works.
_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', METHOD)
_CATEGORICAL_COLS = ['Sex']

# Upstream trains with batch_size=512 and drop_last=True, so a training fold
# with fewer than 512 rows gets NO updates at all and the generator stays at
# its random initialisation. Datasets whose folds are that small run with
# batch_size=SMALL_DATASET_BATCH_SIZE instead: a deliberate deviation from the
# paper, recorded under deviations_from_paper in each _run_info.json.
SMALL_DATASET_BATCH_SIZE = 64


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
    __run(get_abalone19_data(),
          'abalone19')
    __run(get_abalone_9_vs_18_data(),
          'abalone_9_vs_18')
    __run(get_abalone_3_vs_11_data(),
          'abalone_3_vs_11', batch_size=SMALL_DATASET_BATCH_SIZE)
    __run(get_abalone_19_vs_10_11_12_13_data(),
          'abalone_19_vs_10_11_12_13')
    __run(get_abalone_20_vs_8_9_10_data(),
          'abalone_20_vs_8_9_10')
