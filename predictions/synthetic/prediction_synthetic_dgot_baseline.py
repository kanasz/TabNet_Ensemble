"""
DGOT baseline (no GA) on the synthetic datasets: the published DGOT pipeline with the
paper's default hyperparameters, one run per dataset. A thin wrapper around
optimization/no_ga_dgot_runner.py, which holds the DGOT-driving logic.

All synthetic features are continuous. Same three sets as prediction_synthetic_dgot.py.

Results go to predictions/synthetic/results/dgot_baseline/dgot_baseline_{dataset}.*
"""

import os
import time

from base_functions import get_synthetic_data
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
    __run(get_synthetic_data(prefix='01', contamination='0.3', features=50),
          '01_synthetic_0.3_contamination_50_features')
    __run(get_synthetic_data(prefix='02', contamination='0.1', features=100, samples=200),
          '02_synthetic_0.1_contamination_100_features_200_samples', batch_size=SMALL_DATASET_BATCH_SIZE)
    __run(get_synthetic_data(prefix='03', contamination='0.02', features=200, samples=200),
          '03_synthetic_0.02_contamination_200_features_200_samples', batch_size=SMALL_DATASET_BATCH_SIZE)
