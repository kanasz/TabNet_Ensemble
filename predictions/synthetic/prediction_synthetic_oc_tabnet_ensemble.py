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
    start_time = time.time()
    contamination = '0.3'
    features = '50'
    samples = 200
    actual_loss_function = LossFunction.CROSSENTROPYLOSS

    data = get_synthetic_data('01', contamination, features)
    config_files = get_config_files("../../models/configurations")
    tuner = GaOCBaggingTabnetEnsembleTuner(tabnet_max_epochs, num_generations, num_parents, population, config_files=config_files, device='cuda')
    tuner.run_experiment(data,
                         'results_200_samples/OC_TABNET_ENSEMBLE_CROSSENTROPYLOSS_synthetic_{}_contamination_{}_features_{}_epochs_{}_population_{}'
                         .format(contamination, features, samples, tabnet_max_epochs, population), actual_loss_function)
    print("--- total: %s seconds ---" % (time.time() - start_time))
    print("Experiment info -> data: {}, features: {}, loss function: {}".format(contamination, features,
                                                                                actual_loss_function))
    tuner.evaluate_experiment_from_pkl(data, actual_loss_function,
                                       'results_200_samples/CROSSENTROPYLOSS_01_synthetic_{}_contamination_{}_features_{}_epochs_{}_population_{}'.format(contamination, features, samples, tabnet_max_epochs, population))









