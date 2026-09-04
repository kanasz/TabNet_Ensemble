import os
import time

import numpy as np

from base_functions import get_glass_0_1_6_vs_5_data, get_glass_2_data, get_glass_4_data, get_glass_5_data, set_seed
from constants import GASotaRunConfig
from optimization.ga_cco_tuner import GaCCOTuner

set_seed(seed=42)
np.set_printoptions(threshold=np.inf)

# GA-tuned CCO: searches k, beta, t, focal gamma and batch_size (constants.genes_cco)
# with mean G-mean over the 5 folds as fitness. The counterpart fixed-default run
# is prediction_glass_cco_baseline.py.
METHOD = 'cco'
RESULTS_DIR = 'results/{}'.format(METHOD)
# Transient GA output (the per-generation resume checkpoint) - can be wiped
# without losing a result.
LOG_DIR = 'ga_logs/{}'.format(METHOD)


def __run(data, dataset_name):
    start_time = time.time()
    results_file = os.path.join(RESULTS_DIR, '{}_{}'.format(METHOD, dataset_name))
    log_dir = os.path.join(LOG_DIR, dataset_name)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f"Starting GA-tuned CCO run for {dataset_name}...")
    tuner = GaCCOTuner(
        GASotaRunConfig.NUM_GENERATIONS.value,
        GASotaRunConfig.NUM_PARENTS.value,
        GASotaRunConfig.POPULATION.value,
        input_dim=data[0].shape[1],
    )
    tuner.run_experiment(data, results_file, log_dir=log_dir)
    print("--- {}: {} seconds ---".format(dataset_name, time.time() - start_time))


if __name__ == '__main__':
    __run(get_glass_0_1_6_vs_5_data(), 'glass_0_1_6_vs_5')
    __run(get_glass_2_data(), 'glass_2')
    __run(get_glass_4_data(), 'glass_4')
    __run(get_glass_5_data(), 'glass_5')