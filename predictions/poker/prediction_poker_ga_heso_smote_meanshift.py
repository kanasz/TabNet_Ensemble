import random
import time
import numpy as np
import torch
from imblearn.over_sampling import SMOTE

from base_functions import get_config_files, get_meanshift_cluster_counts
from base_functions import get_poker_9_vs_7, get_poker_8_vs_6, get_poker_8_9_vs_5, get_poker_8_9_vs_6
from models.oc_bagging_tabnet_ensemble_parallel import GaOCBaggingTabnetEnsembleTunerParallel

seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.rand(seed)
random.SystemRandom(seed)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # for multiGPUs.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def run_experiment_poker_9_vs_7():
    tabnet_max_epochs = 50
    num_generations = 50
    num_parents = 20
    population = 50
    start_time = time.time()
    data = get_poker_9_vs_7()
    numerical_cols = list(data[0].columns.values)
    categorical_cols = None
    sampling_algorithm = SMOTE(random_state=42)
    clusters, bandwidths, algs = get_meanshift_cluster_counts(data[0], data[1], numerical_cols, categorical_cols)
    clustering_params = {
        "bandwidths": bandwidths,
        "clusters": clusters,
        "type": "MS",
        "algs": algs
    }
    config_files = get_config_files("../../models/configurations")
    tuner = GaOCBaggingTabnetEnsembleTunerParallel(tabnet_max_epochs=tabnet_max_epochs,
                                                   num_generations=num_generations,
                                                   num_parents=num_parents,
                                                   population=population,
                                                   config_files=config_files,
                                                   device='cuda',
                                                   sampling_algorithm=sampling_algorithm,
                                                   numerical_cols=numerical_cols,
                                                   categorical_cols=categorical_cols,
                                                   save_partial_output=True,
                                                   clustering_params=clustering_params,
                                                   use_cluster_centers=False)
    tuner.run_experiment(data, 'results/ga_heso_smote_meanshift_poker_9_vs_7/GA_HESO_SMOTE_MEANSHIFT_poker_9_vs_7')
    print("--- total: %s seconds ---" % (time.time() - start_time))


def run_experiment_poker_8_vs_6():
    tabnet_max_epochs = 50
    num_generations = 50
    num_parents = 20
    population = 50
    start_time = time.time()
    data = get_poker_8_vs_6()
    numerical_cols = list(data[0].columns.values)
    categorical_cols = None
    sampling_algorithm = SMOTE(random_state=42)
    clusters, bandwidths, algs = get_meanshift_cluster_counts(data[0], data[1], numerical_cols, categorical_cols)
    clustering_params = {
        "bandwidths": bandwidths,
        "clusters": clusters,
        "type": "MS",
        "algs": algs
    }
    config_files = get_config_files("../../models/configurations")
    tuner = GaOCBaggingTabnetEnsembleTunerParallel(tabnet_max_epochs=tabnet_max_epochs,
                                                   num_generations=num_generations,
                                                   num_parents=num_parents,
                                                   population=population,
                                                   config_files=config_files,
                                                   device='cuda',
                                                   sampling_algorithm=sampling_algorithm,
                                                   numerical_cols=numerical_cols,
                                                   categorical_cols=categorical_cols,
                                                   save_partial_output=True,
                                                   clustering_params=clustering_params,
                                                   use_cluster_centers=False)
    tuner.run_experiment(data, 'results/ga_heso_smote_meanshift_poker_8_vs_6/GA_HESO_SMOTE_MEANSHIFT_poker_8_vs_6')
    print("--- total: %s seconds ---" % (time.time() - start_time))


def run_experiment_poker_8_9_vs_5():
    tabnet_max_epochs = 50
    num_generations = 50
    num_parents = 20
    population = 50
    start_time = time.time()
    data = get_poker_8_9_vs_5()
    numerical_cols = list(data[0].columns.values)
    categorical_cols = None
    sampling_algorithm = SMOTE(random_state=42)
    clusters, bandwidths, algs = get_meanshift_cluster_counts(data[0], data[1], numerical_cols, categorical_cols)
    clustering_params = {
        "bandwidths": bandwidths,
        "clusters": clusters,
        "type": "MS",
        "algs": algs
    }
    config_files = get_config_files("../../models/configurations")
    tuner = GaOCBaggingTabnetEnsembleTunerParallel(tabnet_max_epochs=tabnet_max_epochs,
                                                   num_generations=num_generations,
                                                   num_parents=num_parents,
                                                   population=population,
                                                   config_files=config_files,
                                                   device='cuda',
                                                   sampling_algorithm=sampling_algorithm,
                                                   numerical_cols=numerical_cols,
                                                   categorical_cols=categorical_cols,
                                                   save_partial_output=True,
                                                   clustering_params=clustering_params,
                                                   use_cluster_centers=False)
    tuner.run_experiment(data, 'results/ga_heso_smote_meanshift_poker_8_9_vs_5/GA_HESO_SMOTE_MEANSHIFT_poker_8_9_vs_5')
    print("--- total: %s seconds ---" % (time.time() - start_time))


def run_experiment_poker_8_9_vs_6():
    tabnet_max_epochs = 50
    num_generations = 50
    num_parents = 20
    population = 50
    start_time = time.time()
    data = get_poker_8_9_vs_6()
    numerical_cols = list(data[0].columns.values)
    categorical_cols = None
    sampling_algorithm = SMOTE(random_state=42)
    clusters, bandwidths, algs = get_meanshift_cluster_counts(data[0], data[1], numerical_cols, categorical_cols)
    clustering_params = {
        "bandwidths": bandwidths,
        "clusters": clusters,
        "type": "MS",
        "algs": algs
    }
    config_files = get_config_files("../../models/configurations")
    tuner = GaOCBaggingTabnetEnsembleTunerParallel(tabnet_max_epochs=tabnet_max_epochs,
                                                   num_generations=num_generations,
                                                   num_parents=num_parents,
                                                   population=population,
                                                   config_files=config_files,
                                                   device='cuda',
                                                   sampling_algorithm=sampling_algorithm,
                                                   numerical_cols=numerical_cols,
                                                   categorical_cols=categorical_cols,
                                                   save_partial_output=True,
                                                   clustering_params=clustering_params,
                                                   use_cluster_centers=False)
    tuner.run_experiment(data, 'results/ga_heso_smote_meanshift_poker_8_9_vs_6/GA_HESO_SMOTE_MEANSHIFT_poker_8_9_vs_6')
    print("--- total: %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    # data: poker_9_vs_7
    run_experiment_poker_9_vs_7()

    # data: poker_8_vs_6
    run_experiment_poker_8_vs_6()
    
    # data: poker_8_9_vs_5
    run_experiment_poker_8_9_vs_5()

    # data: poker_8_9_vs_6
    run_experiment_poker_8_9_vs_6()
