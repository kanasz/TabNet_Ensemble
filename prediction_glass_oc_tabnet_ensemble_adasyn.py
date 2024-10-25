import random
import time
import numpy as np
import torch
from sklearn.cluster import KMeans
from base_functions import get_config_files, get_glass_2_data
from constants import SMOTE_K_NEIGHBORS
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
from imblearn.over_sampling import ADASYN

if __name__ == '__main__':
    tabnet_max_epochs = 50
    num_generations = 50
    num_parents = 20  # 10
    population = 50  # 20
    start_time = time.time()

    data = get_glass_2_data()

    synthetic_count = int(len((data[1]) - np.sum(data[1])) / 5)
    numerical_cols = list(data[0].columns.values)
    categorical_cols = None
    sampling_strategy = {1: sum(data[1] == 1) + synthetic_count}
    cluster_count = int(synthetic_count / SMOTE_K_NEIGHBORS)
    sampling_algorithm = ADASYN(sampling_strategy=sampling_strategy, random_state=42, n_neighbors=SMOTE_K_NEIGHBORS)
    clustering_algorithm = KMeans(random_state=42, n_clusters=cluster_count)

    config_files = get_config_files("models/configurations")
    tuner = GaOCBaggingTabnetEnsembleTunerParallel(tabnet_max_epochs, num_generations, num_parents, population,
                                                   config_files=config_files, device='cuda',
                                                   clustering_algorithm=clustering_algorithm,
                                                   sampling_algorithm=sampling_algorithm, numerical_cols=numerical_cols,
                                                   categorical_cols=categorical_cols, save_partial_output=True,
                                                   synthetic_minority_count=synthetic_count,
                                                   cluster_count=cluster_count)
    tuner.run_experiment(data, 'results/glass_2/OC_TABNET_ENSEMBLE_ADASYN_KMEANS_glass_2_SYNTHETIC_COUNT_{}_CLUSTER_COUNT_{}'.format(synthetic_count, cluster_count))
    print("--- total: %s seconds ---" % (time.time() - start_time))