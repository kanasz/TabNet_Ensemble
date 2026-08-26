import os
import random
import time

import numpy as np
import torch
from imblearn.over_sampling import SMOTE

from base_functions import get_config_files, get_abalone19_data, get_abalone_9_vs_18_data, \
    get_abalone_3_vs_11_data, get_abalone_19_vs_10_11_12_13_data, get_abalone_20_vs_8_9_10_data
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

# Abalone's one true categorical feature ('Sex': M/F/I) — everything else numeric.
_CATEGORICAL_COLS = ['Sex']


def __run_experiment(data, dataset_name, results_file):
    start_time = time.time()
    numerical_cols = [col for col in data[0].columns.values if col not in _CATEGORICAL_COLS]
    categorical_cols = _CATEGORICAL_COLS
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
    __run_experiment(get_abalone19_data(), 'abalone19',
                     'results/no_clustering_smote_abalone19/NO_CLUSTERING_OC_TABNET_ENSEMBLE_SMOTE_abalone19')

    __run_experiment(get_abalone_9_vs_18_data(), 'abalone_9_vs_18',
                     'results/no_clustering_smote_abalone_9_vs_18/NO_CLUSTERING_OC_TABNET_ENSEMBLE_SMOTE_abalone_9_vs_18')

    __run_experiment(get_abalone_3_vs_11_data(), 'abalone_3_vs_11',
                     'results/no_clustering_smote_abalone_3_vs_11/NO_CLUSTERING_OC_TABNET_ENSEMBLE_SMOTE_abalone_3_vs_11')

    __run_experiment(get_abalone_19_vs_10_11_12_13_data(), 'abalone_19_vs_10_11_12_13',
                     'results/no_clustering_smote_abalone_19_vs_10_11_12_13/NO_CLUSTERING_OC_TABNET_ENSEMBLE_SMOTE_abalone_19_vs_10_11_12_13')

    __run_experiment(get_abalone_20_vs_8_9_10_data(), 'abalone_20_vs_8_9_10',
                     'results/no_clustering_smote_abalone_20_vs_8_9_10/NO_CLUSTERING_OC_TABNET_ENSEMBLE_SMOTE_abalone_20_vs_8_9_10')