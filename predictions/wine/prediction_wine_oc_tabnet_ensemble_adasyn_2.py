import random
import time

import numpy as np
import torch
from sklearn.cluster import KMeans

from base_functions import get_wine_quality_white_3_vs_7_data, get_config_files, get_wine_quality_red_3_vs_5_data
from constants import Classifier, SMOTE_K_NEIGHBORS, SYNTHETIC_MINORITY_COUNT, CLUSTER_COUNT, LossFunction
from models.oc_bagging_tabnet_ensemble_parallel import GaOCBaggingTabnetEnsembleTunerParallel
from optimization.ga_tuner import GaTuner
from imblearn.over_sampling import ADASYN
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

if __name__ == '__main__':
    tabnet_max_epochs = 50
    num_generations = 50
    num_parents = 20  # 10
    population = 50  # 20
    start_time = time.time()
    actual_loss_function = LossFunction.CROSSENTROPYLOSS
    data = get_wine_quality_red_3_vs_5_data()
    numerical_columns = list(data[0].columns.values)

    sampling_strategy = {1: sum(data[1] == 1) + SYNTHETIC_MINORITY_COUNT}
    n_neighbors = SMOTE_K_NEIGHBORS

    sampling_algorithm = ADASYN(sampling_strategy=sampling_strategy,
                                random_state=42, n_neighbors=n_neighbors)

    clustering_algorithm = KMeans(n_clusters=CLUSTER_COUNT, random_state=42)

    config_files = get_config_files("../../models/configurations")
    tuner = GaOCBaggingTabnetEnsembleTunerParallel(tabnet_max_epochs, num_generations, num_parents, population,
                                                   config_files=config_files, device='cuda',
                                                   clustering_algorithm=clustering_algorithm,
                                                   sampling_algorithm=sampling_algorithm, numerical_cols=numerical_columns,
                                                   categorical_cols=None)
    tuner.run_experiment(data, 'results/OC_TABNET_ENSEMBLE_ADASYN_KMEANS_wine_3_vs_7',
                         loss_function=actual_loss_function)
    print("--- total: %s seconds ---" % (time.time() - start_time))



