import os
import time

import numpy as np

from base_functions import get_wine_quality_red_3_vs_5_data, get_wine_quality_red_8_vs_6_data, \
    get_wine_quality_white_3_vs_7_data, get_wine_quality_white_9_vs_4_data, set_seed
from constants import GASotaRunConfig
from optimization.ga_sos_tuner import GaSOSTuner

set_seed(seed=42)
np.set_printoptions(threshold=np.inf)

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# GA-tuned SOS: searches lr, beta1, beta_min, beta_max, num_scales and ema_rate
# (constants.genes_sos) with mean G-mean over the 5 folds as fitness.
# RESULTS_DIR keeps one .txt (metrics) and one .pkl (final model) per dataset,
# named {METHOD}_{dataset} so results_parser can read the folder as a method.
# LOG_DIR takes the transient GA output - the per-improvement dumps and the
# resume checkpoint - and can be wiped without losing a result.
METHOD = 'sos'
RESULTS_DIR = os.path.join(_PROJECT_ROOT, 'results', METHOD)
LOG_DIR = os.path.join(_PROJECT_ROOT, 'ga_logs', METHOD)


def __run(data, dataset_name):
    start_time = time.time()
    x_df, y_series = data

    # image_size = n_features + n_label_classes (categorical label -> one-hot)
    image_size = x_df.shape[1] + len(y_series.unique())

    results_file = os.path.join(RESULTS_DIR, '{}_{}'.format(METHOD, dataset_name))
    log_dir = os.path.join(LOG_DIR, dataset_name)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f"Starting GA-tuned SOS run for {dataset_name}...")
    tuner = GaSOSTuner(
        num_generations=GASotaRunConfig.NUM_GENERATIONS.value,
        num_parents=GASotaRunConfig.NUM_PARENTS.value,
        population=GASotaRunConfig.POPULATION.value,
        dataset_name=dataset_name,
        image_size=image_size,
    )
    tuner.run_experiment(data, results_file, log_dir=log_dir)
    print("--- {}: {} seconds ---".format(dataset_name, time.time() - start_time))


if __name__ == '__main__':
    __run(get_ecoli_0_vs_1_data(), 'ecoli_0_vs_1')
    __run(get_ecoli_0_2_3_4_vs_5_data(), 'ecoli_0_2_3_4_vs_5')
    __run(get_ecoli_0_3_4_vs_5_data(), 'ecoli_0_3_4_vs_5')
    __run(get_ecoli_0_4_6_vs_5_data(), 'ecoli_0_4_6_vs_5')