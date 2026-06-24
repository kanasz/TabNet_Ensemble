import os
import random
import time

import numpy as np
import torch

from base_functions import (get_wine_quality_red_3_vs_5_data,
                             get_wine_quality_red_8_vs_6_data,
                             get_wine_quality_white_3_vs_7_data,
                             get_wine_quality_white_9_vs_4_data)
from constants import GARunConfig
from optimization.ga_cco_tuner import GaCCOTuner

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_RESULTS_DIR  = os.path.join(_PROJECT_ROOT, 'results')

seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def __run_experiment(wine_data, results_file):
    start_time = time.time()
    input_dim = wine_data[0].shape[1]

    tuner = GaCCOTuner(
        GARunConfig.NUM_GENERATIONS.value,
        GARunConfig.NUM_PARENTS.value,
        GARunConfig.POPULATION.value,
        input_dim=input_dim,
    )
    tuner.run_experiment(wine_data, results_file)
    print("--- total: %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    _CCO_RESULTS_DIR = os.path.join(_RESULTS_DIR, 'cco')
    os.makedirs(_CCO_RESULTS_DIR, exist_ok=True)

    __run_experiment(get_wine_quality_red_3_vs_5_data(),
                     os.path.join(_CCO_RESULTS_DIR, 'cco_wine_red_3_vs_5'))

    __run_experiment(get_wine_quality_red_8_vs_6_data(),
                     os.path.join(_CCO_RESULTS_DIR, 'cco_wine_red_8_vs_6'))

    __run_experiment(get_wine_quality_white_3_vs_7_data(),
                     os.path.join(_CCO_RESULTS_DIR, 'cco_wine_white_3_vs_7'))

    __run_experiment(get_wine_quality_white_9_vs_4_data(),
                     os.path.join(_CCO_RESULTS_DIR, 'cco_wine_white_9_vs_4'))


