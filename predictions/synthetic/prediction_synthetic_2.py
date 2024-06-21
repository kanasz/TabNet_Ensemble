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
    population = 50  # 20
    start_time = time.time()
    contamination = '0.02'
    features = '200'
    samples = 200
    actual_loss_function = LossFunction.CROSSENTROPYLOSS

    data = get_synthetic_data('03', contamination, features,samples)
    tuner = GaBoostingTabnetTuner(tabnet_max_epochs, num_generations, num_parents, population, device='cuda',
                                  use_smote=False,
                                  use_new_model=False)
    tuner.run_experiment(data, 'results_200_samples/CROSSENTROPYLOSS_2.2_synthetic_{}_contamination_{}_features_{}_epochs_{}_population_{}'
                         .format(contamination, features, samples, tabnet_max_epochs, population), actual_loss_function)
    print("--- total: %s seconds ---" % (time.time() - start_time))
    print("Experiment info -> data: {}, features: {}, loss function: {}".format(contamination, features,
                                                                                actual_loss_function))









