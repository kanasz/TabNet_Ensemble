import random
import time

import numpy as np
import torch
from imblearn.over_sampling import ADASYN
from sklearn.cluster import KMeans

from base_functions import get_synthetic_data, get_slovak_data, get_config_files
from constants import LossFunction, CLUSTER_COUNT, WEAK_CLASSIFIERS_COUNT, SYNTHETIC_MINORITY_COUNT, SMOTE_K_NEIGHBORS
from models.oc_bagging_tabnet_ensemble_parallel import GaOCBaggingTabnetEnsembleTunerParallel
from optimization.ga_boosting_tabnet_tuner import GaBoostingTabnetTuner
from optimization.ga_oc_bagging_tabnet_ensemble_tuner import GaOCBaggingTabnetEnsembleTuner

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


# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# warnings.filterwarnings("ignore")

if __name__ == '__main__':
    tabnet_max_epochs = 50
    num_generations = 50
    num_parents = 20  # 10
    population = 50  # 20
    start_time = time.time()
    contamination = '0.02'
    features = 200
    samples = 200
    actual_loss_function = LossFunction.CROSSENTROPYLOSS


    data = get_synthetic_data('03', contamination, features, samples)
    numerical_cols = []
    for i in range(features):
        numerical_cols.append('feature{}'.format(i + 1))

    sampling_algorithm = ADASYN( sampling_strategy={1: sum(data[1] == 1) + SYNTHETIC_MINORITY_COUNT},
                               random_state=42, n_neighbors=SMOTE_K_NEIGHBORS)

    #synthetic_count = int(len((data[1]) - np.sum(data[1])) / 5)
    #sampling_strategy = {1: sum(data[1] == 1) + synthetic_count}
    #cluster_count = int(synthetic_count / SMOTE_K_NEIGHBORS)

    #clustering_algorithm = KMeans(n_clusters=CLUSTER_COUNT, random_state=42)

    config_files = get_config_files("../../models/configurations")
    tuner = GaOCBaggingTabnetEnsembleTunerParallel(tabnet_max_epochs, num_generations, num_parents, population,
                                           config_files=config_files, device='cuda',
                                           sampling_algorithm=sampling_algorithm, numerical_cols=numerical_cols)
    tuner.run_experiment(data,
                         'results_adasyn/OC_TABNET_ENSEMBLE_ADASYN_KMEANS_synthetic_{}_CLUSTER_COUNT_{}_CLASSIFIER_COUNT_{}_SYNTH_COUNT_{}'
                         .format(contamination, CLUSTER_COUNT, WEAK_CLASSIFIERS_COUNT, SYNTHETIC_MINORITY_COUNT))
    print("--- total: %s seconds ---" % (time.time() - start_time))
    print("Experiment info -> data: {}, features: {}, loss function: {}".format(contamination, features,
                                                                                actual_loss_function))






