import random
import time

import numpy as np
import torch

from base_functions import get_synthetic_data, get_slovak_data
from constants import LossFunction
from optimization.ga_boosting_tabnet_tuner import GaBoostingTabnetTuner

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
    num_generations = 30
    num_parents = 20  # 10
    population = 50  # 20

    business_area = 'construction'
    year = '16'
    postfix = '13_14_15'
    actual_loss_function = LossFunction.CROSSENTROPYLOSS


    #CONSTRUCTION 2013 - 0.8232794624171836
    '''
    data = get_slovak_data(business_area, '13', '10_11_12')
    tuner = GaBoostingTabnetTuner(tabnet_max_epochs, num_generations, num_parents, population, device='cuda',
                                  use_smote=False,
                                  use_new_model=False)
    tuner.evaluate_experiment_from_pkl(data, actual_loss_function,
                                       "results/BOOSTING_CROSSENTROPYLOSS_2_construction_13_10_11_12_features_50_epochs_50_population")
    '''
    # CONSTRUCTION 2014 - 0.8476422252
    '''
    tabnet_max_epochs = 70
    data = get_slovak_data(business_area, '14', '11_12_13')
    tuner = GaBoostingTabnetTuner(tabnet_max_epochs, num_generations, num_parents, population, device='cuda',
                                  use_smote=False,
                                  use_new_model=False)
    tuner.evaluate_experiment_from_pkl(data, actual_loss_function,
                                       "results/BOOSTING_CROSSENTROPYLOSS_construction_14_11_12_13_features_70_epochs")
    '''
    # CONSTRUCTION 2016 - 0.9616855784

    tabnet_max_epochs = 50
    data = get_slovak_data(business_area, '13', '10_11_12')
    tuner = GaBoostingTabnetTuner(tabnet_max_epochs, num_generations, num_parents, population, device='cuda',
                                  use_smote=False,
                                  use_new_model=False)
    tuner.evaluate_experiment_from_pkl(data, actual_loss_function,
                                       "results/BOOSTING_CROSSENTROPYLOSS_3_construction_13_10_11_12_features_50_epochs_50_population")
