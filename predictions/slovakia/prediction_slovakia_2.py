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
    tabnet_max_epochs = 70
    num_generations = 30
    num_parents = 20  # 10
    population = 40  # 20
    business_area = 'construction'
    year = '16'
    postfix = '13_14_15'
    start_time = time.time()
    actual_loss_function = LossFunction.CROSSENTROPYLOSS
    data = get_slovak_data(business_area, year, postfix)
    tuner = GaBoostingTabnetTuner(tabnet_max_epochs, num_generations, num_parents, population, device='cuda',
                                  use_smote=False,
                                  use_new_model=True)
    #tuner.run_experiment(data, 'results/BOOSTING_CROSSENTROPYLOSS_{}_{}_{}_features_{}_epochs_40_population'
    #                     .format(business_area, year, postfix, tabnet_max_epochs), actual_loss_function)
    print("--- total: %s seconds ---" % (time.time() - start_time))

    solution = [55, 60, 2, 3.880629843723761, 0.009510274927448648, 0.29414369336622354, 2, 4, 0.17570905225866779, 7, 1.8643825460108894, 250.54394013109854]

    tuner.evaluate_experiment(data,actual_loss_function,solution)










