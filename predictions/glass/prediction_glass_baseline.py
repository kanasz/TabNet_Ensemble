import random
import time

import numpy as np
import torch

from base_functions import get_synthetic_data, get_slovak_data, get_abalone_9_vs_18_data, \
    get_abalone_19_vs_10_11_12_13_data, get_abalone_20_vs_8_9_10_data, get_ecoli_0_vs_1_data, get_ecoli_0_4_6_vs_5_data, \
    get_ecoli_0_3_4_vs_5_data, get_ecoli_0_2_3_4_vs_5_data, get_glass_0_1_6_vs_5_data, get_glass_2_data, \
    get_glass_4_data, get_glass_5_data
from constants import LossFunction, Classifier
from optimization.ga_boosting_tabnet_tuner import GaBoostingTabnetTuner
from optimization.ga_tuner import GaTuner
from optimization.ga_xgboost_tuner import GaXGBoostTuner

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

    '''
    data = get_glass_0_1_6_vs_5_data()
    numerical_cols = list(data[0].columns.values)
    categorical_cols = None
    tuner = GaTuner(num_generations, num_parents, population,
                    use_smote=True,
                    clf_type=Classifier.SVC, numerical_cols=numerical_cols, categorical_cols=categorical_cols)
    tuner.run_experiment(data, 'results/SVC_SMOTE_glass_0_1_6_vs_5')
    print("--- total: %s seconds ---" % (time.time() - start_time))

    tuner = GaTuner(num_generations, num_parents, population,
                    use_smote=False,
                    clf_type=Classifier.WeightedSVC, numerical_cols=numerical_cols, categorical_cols=categorical_cols)
    tuner.run_experiment(data, 'results/SVC_WEIGHTED_glass_0_1_6_vs_5')
    print("--- total: %s seconds ---" % (time.time() - start_time))
    '''
    #----------------------------------------------------------------------------------------
    '''
    data = get_glass_2_data()
    numerical_cols = list(data[0].columns.values)
    categorical_cols = None
    tuner = GaTuner(num_generations, num_parents, population,
                    use_smote=True,
                    clf_type=Classifier.SVC, numerical_cols=numerical_cols, categorical_cols=categorical_cols)
    tuner.run_experiment(data, 'results/SVC_SMOTE_glass_2')
    print("--- total: %s seconds ---" % (time.time() - start_time))

    tuner = GaTuner(num_generations, num_parents, population,
                    use_smote=False,
                    clf_type=Classifier.WeightedSVC, numerical_cols=numerical_cols, categorical_cols=categorical_cols)
    tuner.run_experiment(data, 'results/SVC_WEIGHTED_glass_2')
    print("--- total: %s seconds ---" % (time.time() - start_time))
    '''
    #----------------------------------------------------------------------------------------
    '''
    data = get_glass_4_data()
    numerical_cols = list(data[0].columns.values)
    categorical_cols = None
    tuner = GaTuner(num_generations, num_parents, population,
                    use_smote=True,
                    clf_type=Classifier.SVC, numerical_cols=numerical_cols, categorical_cols=categorical_cols)
    tuner.run_experiment(data, 'results/SVC_SMOTE_glass_4')
    print("--- total: %s seconds ---" % (time.time() - start_time))

    tuner = GaTuner(num_generations, num_parents, population,
                    use_smote=False,
                    clf_type=Classifier.WeightedSVC, numerical_cols=numerical_cols, categorical_cols=categorical_cols)
    tuner.run_experiment(data, 'results/SVC_WEIGHTED_glass_4')
    print("--- total: %s seconds ---" % (time.time() - start_time))

    #----------------------------------------------------------------------------------------
    '''
    data = get_glass_5_data()
    numerical_cols = list(data[0].columns.values)
    categorical_cols = None
    tuner = GaTuner(num_generations, num_parents, population,
                    use_smote=True,
                    clf_type=Classifier.SVC, numerical_cols=numerical_cols, categorical_cols=categorical_cols)
    tuner.run_experiment(data, 'results/SVC_SMOTE_glass_5')
    print("--- total: %s seconds ---" % (time.time() - start_time))

    tuner = GaTuner(num_generations, num_parents, population,
                    use_smote=False,
                    clf_type=Classifier.WeightedSVC, numerical_cols=numerical_cols, categorical_cols=categorical_cols)
    tuner.run_experiment(data, 'results/SVC_WEIGHTED_glass_5')
    print("--- total: %s seconds ---" % (time.time() - start_time))