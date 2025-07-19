import random
import time
import numpy as np
import torch

from base_functions import get_glass_2_data, get_glass_4_data, get_glass_5_data, get_glass_0_1_6_vs_5_data
from constants import Classifier
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


def run_experiment(glass_data, file_name):
    num_generations = 50
    num_parents = 20
    population = 50
    start_time = time.time()
    data = glass_data
    numerical_cols = list(data[0].columns.values)

    print(f"Starting simulation run...")
    tuner = GaTuner(num_generations=num_generations,
                    num_parents=num_parents,
                    population=population,
                    use_smote=False,
                    clf_type=Classifier.EasyEnsemble,
                    numerical_cols=numerical_cols)
    tuner.run_experiment(data, file_name)
    print("--- total: %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    # glass_2
    run_experiment(glass_data=get_glass_2_data(),
                   file_name="results/easy_ensemble_glass_2")
    
    # glass_4
    run_experiment(glass_data=get_glass_4_data(),
                   file_name="results/easy_ensemble_glass_4")
    
    # glass_5
    run_experiment(glass_data=get_glass_5_data(),
                   file_name="results/easy_ensemble_glass_5")

    # glass_0_1_6_vs_5
    run_experiment(glass_data=get_glass_0_1_6_vs_5_data(),
                   file_name="results/easy_ensemble_glass_0_1_6_vs_5")
