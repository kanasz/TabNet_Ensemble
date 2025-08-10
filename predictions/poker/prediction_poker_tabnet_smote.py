import random
import time
import numpy as np
import torch

from base_functions import get_poker_8_vs_6, get_poker_9_vs_7, get_poker_8_9_vs_5, get_poker_8_9_vs_6
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


def run_experiment(poker_data, file_name):
    num_generations = 50
    num_parents = 20
    population = 50
    start_time = time.time()
    data = poker_data
    numerical_cols = list(data[0].columns.values)
    categorical_cols = None

    print(f"Starting simulation run - TabNet + SMOTE...")
    tuner = GaTuner(num_generations=num_generations,
                    num_parents=num_parents,
                    population=population,
                    use_smote=True, use_adasyn=False, use_smote_enn=False,
                    clf_type=Classifier.TabNet,
                    numerical_cols=numerical_cols,
                    categorical_cols=categorical_cols)
    tuner.run_experiment(data, file_name)
    print("--- total: %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    # data: poker_8_vs_6
    run_experiment(poker_data=get_poker_8_vs_6(),
                   file_name="results/tabnet_smote_poker_8_vs_6")
    
    # data: poker_9_vs_7
    run_experiment(poker_data=get_poker_9_vs_7(),
                   file_name="results/tabnet_smote_poker_9_vs_7")
    
    # data: poker_8_9_vs_5
    run_experiment(poker_data=get_poker_8_9_vs_5(),
                   file_name="results/tabnet_smote_poker_8_9_vs_5")

    # data: poker_8_9_vs_6
    run_experiment(poker_data=get_poker_8_9_vs_6(),
                   file_name="results/tabnet_smote_poker_8_9_vs_6")
