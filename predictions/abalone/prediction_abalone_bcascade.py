import random
import time
import numpy as np
import torch

from base_functions import get_abalone_9_vs_18_data, get_abalone_19_vs_10_11_12_13_data
from base_functions import get_abalone_20_vs_8_9_10_data, get_abalone_3_vs_11_data
from base_functions import get_abalone19_data
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


def run_experiment(abalone_data, file_name):
    num_generations = 50
    num_parents = 20
    population = 50
    start_time = time.time()
    data = abalone_data
    numerical_cols = ['Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight',
                      'Shell_weight']
    categorical_cols = ['Sex']

    print(f"Starting simulation run...")
    tuner = GaTuner(num_generations=num_generations,
                    num_parents=num_parents,
                    population=population,
                    use_smote=False, use_adasyn=False, use_smote_enn=False,
                    numerical_cols=numerical_cols,
                    categorical_cols=categorical_cols,
                    clf_type=Classifier.BalancedCascade)
    tuner.run_experiment(data, file_name)
    print("--- total: %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    # abalone_9_vs_18
    run_experiment(abalone_data=get_abalone_9_vs_18_data(),
                   file_name="results/BCascade_abalone_9_vs_18")

    # abalone_19_vs_10_11_12_13
    run_experiment(abalone_data=get_abalone_19_vs_10_11_12_13_data(),
                   file_name="results/BCascade_abalone_19_vs_10_11_12_13")

    # abalone_20_vs_8_9_10
    run_experiment(abalone_data=get_abalone_20_vs_8_9_10_data(),
                   file_name="results/BCascade_abalone_20_vs_8_9_10")

    # abalone_3_vs_11
    run_experiment(abalone_data=get_abalone_3_vs_11_data(),
                   file_name="results/BCascade_abalone_3_vs_11")

    # abalone19
    run_experiment(abalone_data=get_abalone19_data(),
                   file_name="results/BCascade_abalone19")
