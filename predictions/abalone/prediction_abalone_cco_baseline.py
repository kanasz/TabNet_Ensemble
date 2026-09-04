import os
import time

import numpy as np

from base_functions import get_abalone19_data, get_abalone_9_vs_18_data, get_abalone_3_vs_11_data, \
    get_abalone_19_vs_10_11_12_13_data, get_abalone_20_vs_8_9_10_data, set_seed
from optimization.no_ga_cco_runner import run_cco_baseline

set_seed(seed=42)
np.set_printoptions(threshold=np.inf)

# CCO's published defaults (see its README usage line). No search is done here -
# this is the fixed baseline, so one run per dataset.
K = 0.3
BETA = 0.5
T = 0.5
EPOCHS = 100
BATCH_SIZE = 32

# Abalone's one true categorical feature ('Sex': M/F/I). CCO works on raw
# pairwise distances, so it has to be label encoded rather than left as text.
CATEGORICAL_COLS = ['Sex']

# One .txt (metrics) and one _summary.csv per dataset, named {METHOD}_{dataset}
# so results_parser can read the folder as a method. No GA here, so there is no
# transient output and no ga_logs/ directory.
METHOD = 'cco_baseline'
RESULTS_DIR = 'results/{}'.format(METHOD)


def __run(data, dataset_name):
    start_time = time.time()
    results_file = os.path.join(RESULTS_DIR, '{}_{}'.format(METHOD, dataset_name))
    print(f"Starting CCO baseline run for {dataset_name}...")
    run_cco_baseline(
        data, dataset_name, results_file,
        k=K, beta=BETA, t=T, epochs=EPOCHS, batch_size=BATCH_SIZE,
        categorical_cols=CATEGORICAL_COLS,
    )
    print("--- {}: {} seconds ---".format(dataset_name, time.time() - start_time))


if __name__ == '__main__':
    __run(get_abalone19_data(), 'abalone19')
    __run(get_abalone_9_vs_18_data(), 'abalone_9_vs_18')
    __run(get_abalone_3_vs_11_data(), 'abalone_3_vs_11')
    __run(get_abalone_19_vs_10_11_12_13_data(), 'abalone_19_vs_10_11_12_13')
    __run(get_abalone_20_vs_8_9_10_data(), 'abalone_20_vs_8_9_10')