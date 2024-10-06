import random
import time

import numpy as np
import torch
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans

from base_functions import get_abalone_20_vs_8_9_10_data
from constants import Classifier, SMOTE_K_NEIGHBORS, SYNTHETIC_MINORITY_COUNT, CLUSTER_COUNT
from models.oc_generic_model import GaOCGenericTunerParallel

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

    data = get_abalone_20_vs_8_9_10_data()
    numerical_cols = ['Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight',
                      'Shell_weight']
    categorical_cols = ['Sex']

    sampling_algorithm = SMOTE(sampling_strategy={1: sum(data[1] == 1) + SYNTHETIC_MINORITY_COUNT},
          random_state=42, k_neighbors=SMOTE_K_NEIGHBORS)

    clustering_algorithm = KMeans(n_clusters=CLUSTER_COUNT, random_state=42)

    tuner = GaOCGenericTunerParallel(50, num_generations, num_parents, population,
                    clf_type=Classifier.SVC, numerical_cols=numerical_cols,
                    categorical_cols=categorical_cols,
                                     sampling_algorithm=sampling_algorithm,
                                     clustering_algorithm=clustering_algorithm)

    tuner.run_experiment(data, 'results/OC_SVM_abalone_20_vs_8_9_10',loss_function=None)
    print("--- total: %s seconds ---" % (time.time() - start_time))