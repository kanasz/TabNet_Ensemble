import random
import time

import numpy as np
import torch
from sklearn.cluster import KMeans

from base_functions import get_abalone_9_vs_18_data, \
    get_config_files, get_meanshift_cluster_counts, get_abalone_3_vs_11_data, get_abalone_19_vs_10_11_12_13_data, \
    get_abalone_20_vs_8_9_10_data, get_yeast_3_data
from constants import LossFunction, SMOTE_K_NEIGHBORS
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
    data = get_yeast_3_data()
    numerical_cols = numerical_cols = list(data[0].columns.values)
    categorical_cols = None
    cfl_count = 2
    clusters, bandwidths = get_meanshift_cluster_counts(data[0], data[1], numerical_cols, categorical_cols)
    sampling_algorithm = SMOTE(random_state=42)
    clustering_params = {
        "bandwidths":bandwidths,
        "clusters":clusters,
        "type":"MS"
    }
    config_files = get_config_files("../../models/configurations")
    tuner = GaOCBaggingTabnetEnsembleTunerParallel(tabnet_max_epochs, num_generations, num_parents, population,
                                                    config_files=config_files, device='cuda', sampling_algorithm=sampling_algorithm,
                                                    numerical_cols=numerical_cols, categorical_cols=categorical_cols,
                                                    save_partial_output=True,clustering_params = clustering_params)
    tuner.run_experiment(data, 'results_sensitivity/{}/OC_TABNET_ENSEMBLE_SMOTE_MEANSHIFT_{}_yeast_3'.format(cfl_count, cfl_count), max_classifier_count=cfl_count)
    print("--- total: %s seconds ---" % (time.time() - start_time))