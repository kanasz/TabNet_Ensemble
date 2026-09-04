import os
import time

import numpy as np
from imblearn.over_sampling import SMOTE

from base_functions import get_config_files, get_ecoli_0_vs_1_data, get_ecoli_0_2_3_4_vs_5_data, \
    get_ecoli_0_3_4_vs_5_data, get_ecoli_0_4_6_vs_5_data, set_seed
from constants import SYNTHETIC_MINORITY_COUNT
from models.oc_bagging_tabnet_ensemble_parallel_no_clustering import GaOCBaggingTabnetEnsembleTunerParallelNoClustering

set_seed(seed=42)
np.set_printoptions(threshold=np.inf)

tabnet_max_epochs = 50
num_generations = 50
num_parents = 20
population = 50

# Where output goes. RESULTS_DIR keeps one .txt (metrics) and one .pkl (final
# model) per dataset, named {METHOD}_{dataset} so results_parser can read the
# folder as a method. LOG_DIR takes the transient GA output - partial dumps and
# the resume checkpoint - and can be wiped without losing a result.
METHOD = 'ga_heso_no_clustering_smote'
RESULTS_DIR = 'results/{}'.format(METHOD)
LOG_DIR = 'ga_logs/{}'.format(METHOD)


def __run_experiment(data, dataset_name, results_file, log_dir):
    start_time = time.time()
    numerical_cols = list(data[0].columns.values)
    categorical_cols = None
    sampling_algorithm = SMOTE(random_state=42)

    config_files = get_config_files("../../models/configurations")
    tuner = GaOCBaggingTabnetEnsembleTunerParallelNoClustering(
        tabnet_max_epochs, num_generations, num_parents, population,
        config_files=config_files, device='cuda', sampling_algorithm=sampling_algorithm,
        numerical_cols=numerical_cols, categorical_cols=categorical_cols,
        save_partial_output=True, synthetic_minority_count=SYNTHETIC_MINORITY_COUNT,
    )

    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    print(f"Starting GA-HESO with no clustering ablation run for {dataset_name}...")
    tuner.run_experiment(data, results_file, log_dir=log_dir)
    print("--- {}: {} seconds ---".format(dataset_name, time.time() - start_time))


def __run(data, dataset_name):
    __run_experiment(
        data, dataset_name,
        results_file=os.path.join(RESULTS_DIR, '{}_{}'.format(METHOD, dataset_name)),
        log_dir=os.path.join(LOG_DIR, dataset_name),
    )


if __name__ == '__main__':
    __run(get_ecoli_0_vs_1_data(), 'ecoli_0_vs_1')
    __run(get_ecoli_0_2_3_4_vs_5_data(), 'ecoli_0_2_3_4_vs_5')
    __run(get_ecoli_0_3_4_vs_5_data(), 'ecoli_0_3_4_vs_5')
    __run(get_ecoli_0_4_6_vs_5_data(), 'ecoli_0_4_6_vs_5')