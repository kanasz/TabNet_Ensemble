import os
import random
import time

import numpy as np
import torch

# Compute absolute results path BEFORE importing ga_dgot_tuner, because that
# module calls os.chdir(_DGOT_PATH) at import time, which would make any later
# relative path resolve inside the DGOT directory instead of the project root.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RESULTS_FILE = os.path.join(_PROJECT_ROOT, 'results', 'DGOT_yeast_3')

from base_functions import get_yeast_3_data
from optimization.ga_dgot_tuner import GaDGOTTuner

seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    num_generations = 50
    num_parents     = 20
    population      = 50

    start_time = time.time()
    data      = get_yeast_3_data()
    input_dim = data[0].shape[1]  # 8 features for yeast3

    tuner = GaDGOTTuner(
        num_generations=num_generations,
        num_parents=num_parents,
        population=population,
        dataset_name='yeast3',
        feature_len=input_dim,
    )
    tuner.run_experiment(data, _RESULTS_FILE)

    print("--- total: %s seconds ---" % (time.time() - start_time))