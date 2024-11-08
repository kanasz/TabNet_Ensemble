import random
import time

import numpy as np
import torch

from base_functions import get_synthetic_data, get_slovak_data, get_abalone_9_vs_18_data, \
    get_abalone_19_vs_10_11_12_13_data, get_abalone_20_vs_8_9_10_data, get_abalone_3_vs_11_data, \
    get_wine_quality_white_9_vs_4_data, get_ecoli_0_vs_1_data, get_ecoli_0_4_6_vs_5_data, get_ecoli_0_3_4_vs_5_data
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
    start_time = time.time()

    numerical_columns = []

    data = get_ecoli_0_3_4_vs_5_data()
    numerical_cols = list(data[0].columns.values)
    categorical_cols = None

    tuner = GaTuner(num_generations, num_parents, population,
                    use_smote=True,
                    clf_type=Classifier.TabNet, numerical_cols=numerical_cols,
                    categorical_cols=categorical_cols, save_partial_output=True)
    tuner.run_experiment(data, 'tabnet_smote_ecoli/04/TabNetSmote_ecoli_0_3_4_vs_5')
    print("--- total: %s seconds ---" % (time.time() - start_time))