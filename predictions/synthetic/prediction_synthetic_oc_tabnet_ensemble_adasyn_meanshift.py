import random
import time

import numpy as np
import torch
from sklearn.cluster import KMeans

from base_functions import get_abalone_9_vs_18_data, \
    get_config_files, get_meanshift_cluster_counts, get_wine_quality_white_9_vs_4_data, get_synthetic_data, \
    get_sensitivity_synthetic_data
from constants import LossFunction, SMOTE_K_NEIGHBORS, SYNTHETIC_MINORITY_COUNT
from models.oc_bagging_tabnet_ensemble_parallel import GaOCBaggingTabnetEnsembleTunerParallel
from sklearn import cluster, datasets, mixture
from imblearn.over_sampling import SMOTE
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
from imblearn.over_sampling import ADASYN
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# warnings.filterwarnings("ignore")

if __name__ == '__main__':
    tabnet_max_epochs = 50
    num_generations = 50
    num_parents = 20  # 10
    population = 50  # 20
    start_time = time.time()
    actual_loss_function = LossFunction.CROSSENTROPYLOSS

    contamination = '0.05'
    features = 100
    samples = 300
    id = '07'

    data = get_sensitivity_synthetic_data(id, contamination, features, samples)
    numerical_cols = numerical_cols = list(data[0].columns.values)
    categorical_cols = None
    sampling_strategy = {1: sum(data[1] == 1) + SYNTHETIC_MINORITY_COUNT}
    n_neighbors = SMOTE_K_NEIGHBORS
    sampling_algorithm = ADASYN(sampling_strategy=sampling_strategy,
                                random_state=42, n_neighbors=n_neighbors)

    clusters, bandwidths, algs = get_meanshift_cluster_counts(data[0], data[1], numerical_cols, categorical_cols,
                                                              smote=sampling_algorithm)

    clustering_params = {
        "bandwidths": bandwidths,
        "clusters": clusters,
        "type": "MS",
        "algs": algs
    }
    config_files = get_config_files("../../models/configurations")
    tuner = GaOCBaggingTabnetEnsembleTunerParallel(tabnet_max_epochs, num_generations, num_parents, population,
                                                    config_files=config_files, device='cuda', sampling_algorithm=sampling_algorithm,
                                                    numerical_cols=numerical_cols, categorical_cols=categorical_cols,
                                                    save_partial_output=True,clustering_params = clustering_params)
    tuner.run_experiment(data, 'results/mean_shift/TEST_OC_TABNET_ENSEMBLE_ADASYN_MEANSHIFT_synthetic_{}'.format(contamination))
    print("--- total: %s seconds ---" % (time.time() - start_time))