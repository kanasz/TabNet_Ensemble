import os
import time

import numpy as np

from base_functions import get_ecoli_0_vs_1_data, get_ecoli_0_2_3_4_vs_5_data, get_ecoli_0_3_4_vs_5_data, \
    get_ecoli_0_4_6_vs_5_data, set_seed
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
    )
    print("--- {}: {} seconds ---".format(dataset_name, time.time() - start_time))


if __name__ == '__main__':
    __run(get_ecoli_0_vs_1_data(), 'ecoli_0_vs_1')
    __run(get_ecoli_0_2_3_4_vs_5_data(), 'ecoli_0_2_3_4_vs_5')
    __run(get_ecoli_0_3_4_vs_5_data(), 'ecoli_0_3_4_vs_5')
    __run(get_ecoli_0_4_6_vs_5_data(), 'ecoli_0_4_6_vs_5')