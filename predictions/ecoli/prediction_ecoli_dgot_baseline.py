"""
DGOT baseline (no GA) on the ecoli datasets: the published DGOT pipeline with the
paper's default hyperparameters, one run per dataset. A thin wrapper around
optimization/no_ga_dgot_runner.py, which holds the DGOT-driving logic.

All ecoli features are continuous.

Results go to predictions/ecoli/results/dgot_baseline/dgot_baseline_{dataset}.*
"""

import os
import time

from base_functions import (get_ecoli_0_vs_1_data, get_ecoli_0_2_3_4_vs_5_data,
                            get_ecoli_0_3_4_vs_5_data, get_ecoli_0_4_6_vs_5_data)
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

# ecoli has 7 features. DGOT's generator halves the feature axis once per
# ch_mult entry with a size-4 kernel, so the paper's ch_mult=[1, 2, 2] needs
# >= 8 features (7 -> 3 -> 1 -> crash). Two levels fit: 7 -> 3 -> 1, and the
# upsampling restores 1 -> 3 -> 7. Passed through DGOT's own --ch_mult flag and
# recorded as a deviation in each _run_info.json.
ECOLI_CH_MULT = [1, 2]


def __run(data, dataset_name, batch_size=PAPER_BATCH_SIZE):
    start_time = time.time()
    run_dgot_baseline(
        data, dataset_name,
        results_file=os.path.join(_RESULTS_DIR, '{}_{}'.format(METHOD, dataset_name)),
        categorical_cols=_CATEGORICAL_COLS,
        batch_size=batch_size,
        ch_mult=ECOLI_CH_MULT,
    )
    print("--- {}: {:.0f} seconds ---".format(dataset_name, time.time() - start_time))


if __name__ == '__main__':
    __run(get_ecoli_0_vs_1_data(),
          'ecoli_0_vs_1', batch_size=SMALL_DATASET_BATCH_SIZE)
    __run(get_ecoli_0_2_3_4_vs_5_data(),
          'ecoli_0_2_3_4_vs_5', batch_size=SMALL_DATASET_BATCH_SIZE)
    __run(get_ecoli_0_3_4_vs_5_data(),
          'ecoli_0_3_4_vs_5', batch_size=SMALL_DATASET_BATCH_SIZE)
    __run(get_ecoli_0_4_6_vs_5_data(),
          'ecoli_0_4_6_vs_5', batch_size=SMALL_DATASET_BATCH_SIZE)
