"""
DGOT baseline (no GA) on the glass datasets: the published DGOT pipeline with the
paper's default hyperparameters, one run per dataset. A thin wrapper around
optimization/no_ga_dgot_runner.py, which holds the DGOT-driving logic.

All glass features are continuous.

Results go to predictions/glass/results/dgot_baseline/dgot_baseline_{dataset}.*
"""

import os
import time

from base_functions import get_glass_0_1_6_vs_5_data, get_glass_2_data, get_glass_4_data, get_glass_5_data
from optimization.no_ga_dgot_runner import PAPER_BATCH_SIZE, run_dgot_baseline

METHOD = 'dgot_baseline'
# Absolute: the runner changes into the DGOT directory while it works.
_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', METHOD)
_CATEGORICAL_COLS = None

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
    __run(get_glass_0_1_6_vs_5_data(),
          'glass_0_1_6_vs_5', batch_size=SMALL_DATASET_BATCH_SIZE)
    __run(get_glass_2_data(),
          'glass_2', batch_size=SMALL_DATASET_BATCH_SIZE)
    __run(get_glass_4_data(),
          'glass_4', batch_size=SMALL_DATASET_BATCH_SIZE)
    __run(get_glass_5_data(),
          'glass_5', batch_size=SMALL_DATASET_BATCH_SIZE)
