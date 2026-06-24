import os
import random
import time

import numpy as np
import torch

from base_functions import get_yeast_3_data
from constants import GARunConfig
from optimization.ga_cco_tuner import GaCCOTuner

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_RESULTS_FILE = os.path.join(_PROJECT_ROOT, 'results', 'CCO_yeast_3')

seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    start_time = time.time()
    data = get_yeast_3_data()
    input_dim = data[0].shape[1]  # D = number of features (8 for yeast3)

    tuner = GaCCOTuner(
        GARunConfig.NUM_GENERATIONS.value,
        GARunConfig.NUM_PARENTS.value,
        GARunConfig.POPULATION.value,
        input_dim=input_dim,
    )
    tuner.run_experiment(data, _RESULTS_FILE)

    print("--- total: %s seconds ---" % (time.time() - start_time))



