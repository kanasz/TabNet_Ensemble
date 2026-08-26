import os
import random
import time

import numpy as np
import torch
from imblearn.over_sampling import SMOTE

from base_functions import get_config_files, get_glass_0_1_6_vs_5_data, get_glass_2_data, \
    get_glass_4_data, get_glass_5_data
from constants import SYNTHETIC_MINORITY_COUNT
from models.oc_bagging_tabnet_ensemble_parallel_no_clustering import GaOCBaggingTabnetEnsembleTunerParallelNoClustering

seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Same GA budget as the clustering-based main experiments (tabnet_max_epochs=50,
# num_generations=50, num_parents=20, population=50) — this ablation only
# removes clustering, everything else must stay comparable.
tabnet_max_epochs = 50
num_generations = 50
num_parents = 20
population = 50


def __run_experiment(data, dataset_name, results_file):
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
    print(f"Starting no-clustering ablation run for {dataset_name}...")
    tuner.run_experiment(data, results_file)
    print("--- {}: {} seconds ---".format(dataset_name, time.time() - start_time))


if __name__ == '__main__':
    __run_experiment(get_glass_0_1_6_vs_5_data(), 'glass_0_1_6_vs_5',
                     'results/no_clustering_smote_glass_0_1_6_vs_5/NO_CLUSTERING_OC_TABNET_ENSEMBLE_SMOTE_glass_0_1_6_vs_5')

    __run_experiment(get_glass_2_data(), 'glass_2',
                     'results/no_clustering_smote_glass_2/NO_CLUSTERING_OC_TABNET_ENSEMBLE_SMOTE_glass_2')

    __run_experiment(get_glass_4_data(), 'glass_4',
                     'results/no_clustering_smote_glass_4/NO_CLUSTERING_OC_TABNET_ENSEMBLE_SMOTE_glass_4')

    __run_experiment(get_glass_5_data(), 'glass_5',
                     'results/no_clustering_smote_glass_5/NO_CLUSTERING_OC_TABNET_ENSEMBLE_SMOTE_glass_5')