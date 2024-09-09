import random
import time

import numpy as np
import torch

from base_functions import get_synthetic_data, get_slovak_data, get_config_files
from constants import LossFunction
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
    business_area = 'construction'
    year = '13'
    postfix = '10_11_12'

    start_time = time.time()
    actual_loss_function = LossFunction.CROSSENTROPYLOSS
    data = get_slovak_data(business_area, year, postfix)
    config_files = get_config_files("../../models/configurations")
    tuner = GaOCBaggingTabnetEnsembleTuner(tabnet_max_epochs, num_generations, num_parents, population, config_files=config_files, device='cuda')
    tuner.run_experiment(data, 'results/OC_TABNET_ENSEMBLE_CROSSENTROPYLOSS_{}_{}_{}_features_{}_epochs_50_population'
                         .format(business_area, year, postfix, tabnet_max_epochs), actual_loss_function)
    print("--- total: %s seconds ---" % (time.time() - start_time))












