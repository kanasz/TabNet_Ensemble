import random
import time
import numpy as np
import torch

from base_functions import get_wine_quality_white_3_vs_7_data, get_wine_quality_white_9_vs_4_data
from base_functions import get_wine_quality_red_8_vs_6_data, get_wine_quality_red_3_vs_5_data
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


def run_experiment(wine_data, file_name):
    num_generations = 50
    num_parents = 20
    population = 50
    start_time = time.time()
    data = wine_data
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
    # white_3_vs_7
    run_experiment(wine_data=get_wine_quality_white_3_vs_7_data(),
                   file_name="results/easy_ensemble_white_3_vs_7")

    # white_9_vs_4
    run_experiment(wine_data=get_wine_quality_white_9_vs_4_data(),
                   file_name="results/easy_ensemble_white_9_vs_4")
    
    # red_8_vs_6
    run_experiment(wine_data=get_wine_quality_red_8_vs_6_data(),
                   file_name="results/easy_ensemble_red_8_vs_6")

    # red_3_vs_5
    run_experiment(wine_data=get_wine_quality_red_3_vs_5_data(),
                   file_name="results/easy_ensemble_red_3_vs_5")
