import random
import sys
import time

import numpy as np
import torch

from base_functions import get_synthetic_data, get_slovak_data, get_config_files, get_abalone_9_vs_18_data, \
    get_abalone_19_vs_10_11_12_13_data, get_abalone_3_vs_11_data, get_wine_quality_red_3_vs_5_data, \
    get_wine_quality_red_8_vs_6_data, get_ecoli_0_vs_1_data, get_ecoli_0_4_6_vs_5_data, get_glass_2_data, \
    get_glass_0_1_6_vs_5_data, get_yeast_3_data
from constants import LossFunction, CLUSTER_COUNT, WEAK_CLASSIFIERS_COUNT, SYNTHETIC_MINORITY_COUNT, Classifier
from models.oc_bagging_tabnet_ensemble_new import GaOCBaggingTabnetEnsembleTunerNew
from models.oc_bagging_tabnet_ensemble_parallel import GaOCBaggingTabnetEnsembleTunerParallel
from optimization.ga_boosting_tabnet_tuner import GaBoostingTabnetTuner
from optimization.ga_oc_bagging_tabnet_ensemble_tuner import GaOCBaggingTabnetEnsembleTuner
from optimization.ga_tuner import GaTuner

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
np.set_printoptions(threshold=np.inf)

if __name__ == '__main__':
    num_generations = 50
    num_parents = 20  # 10
    population = 50  # 20
    start_time = time.time()
    data = get_yeast_3_data()
    numerical_cols = list(data[0].columns.values)
    tuner = GaTuner(num_generations, num_parents, population,
                    use_smote=False,
                    clf_type=Classifier.BalancedCascade, numerical_cols=numerical_cols)
    tuner.run_experiment(data,'results/BCascade_yeast_3')
    print("--- total: %s seconds ---" % (time.time() - start_time))




