import random
import time

import numpy as np
import torch

from base_functions import get_synthetic_data, get_slovak_data, get_abalone_9_vs_18_data, \
    get_abalone_19_vs_10_11_12_13_data, get_abalone_20_vs_8_9_10_data, get_abalone_3_vs_11_data
from constants import LossFunction, Classifier
from optimization.ga_boosting_tabnet_tuner import GaBoostingTabnetTuner
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

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# warnings.filterwarnings("ignore")

if __name__ == '__main__':
    tabnet_max_epochs = 50
    num_generations = 50
    num_parents = 20  # 10
    population = 50  # 20

    contamination = '0.3'
    features = 50
    samples = 200

    start_time = time.time()

    numerical_cols = []

    data = get_synthetic_data('01', contamination, features)
    for i in range(features):
        numerical_cols.append('feature{}'.format(i + 1))

    tuner = GaTuner(num_generations, num_parents, population,
                    use_smote=False,
                    use_adasyn=True,
                    clf_type=Classifier.SVC, numerical_cols=numerical_cols,
                    categorical_cols=None)
    tuner.run_experiment(data, 'results/AdasynSVC_synthetic_01')
    print("--- total: %s seconds ---" % (time.time() - start_time))

