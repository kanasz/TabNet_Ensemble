import random
import time

import numpy as np
import torch
from sklearn.cluster import KMeans

from base_functions import get_abalone_9_vs_18_data, \
    get_config_files, get_ecoli_0_vs_1_data
from constants import LossFunction, SMOTE_K_NEIGHBORS
from models.oc_bagging_tabnet_ensemble_parallel import GaOCBaggingTabnetEnsembleTunerParallel
from sklearn import cluster, datasets, mixture

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
    data = get_ecoli_0_vs_1_data()
    numerical_cols = list(data[0].columns.values)
    categorical_cols = None
    synthetic_count = int(len((data[1]) - np.sum(data[1])) / 5)
    cluster_count = int(synthetic_count / SMOTE_K_NEIGHBORS)

    sampling_strategy = {1: sum(data[1] == 1) + synthetic_count}
    n_neighbors = SMOTE_K_NEIGHBORS

    sampling_algorithm = None # SMOTE will be used
    clustering_algorithm = mixture.GaussianMixture(n_components=cluster_count, covariance_type="full", random_state=42)

    config_files = get_config_files("../../models/configurations")
    tuner = GaOCBaggingTabnetEnsembleTunerParallel(tabnet_max_epochs, num_generations, num_parents, population,
                                           config_files=config_files, device='cuda', clustering_algorithm=clustering_algorithm,
                                           sampling_algorithm=sampling_algorithm, numerical_cols=numerical_cols, categorical_cols=categorical_cols,save_partial_output=True, cluster_count=cluster_count)
    tuner.run_experiment(data, 'results/smote_gmm/OC_TABNET_ENSEMBLE_SMOTE_GMM_ecoli_0_vs_1')
    print("--- total: %s seconds ---" % (time.time() - start_time))