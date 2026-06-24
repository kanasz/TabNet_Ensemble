import os
import random
import time

import numpy as np
import torch

from base_functions import get_yeast_3_data
from constants import GARunConfig
from optimization.ga_sos_tuner import GaSOSTuner

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_RESULTS_FILE = os.path.join(_PROJECT_ROOT, 'results', 'SOS_yeast_3')

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
    X_df, y_series = data

    # image_size = n_features (continuous, 1 col each in transformed space)
    #            + n_label_classes (categorical label -> one-hot)
    n_classes = len(y_series.unique())
    image_size = X_df.shape[1] + n_classes   # 8 + 2 = 10 for yeast3

    tuner = GaSOSTuner(
        num_generations=GARunConfig.NUM_GENERATIONS.value,
        num_parents=GARunConfig.NUM_PARENTS.value,
        population=GARunConfig.POPULATION.value,
        dataset_name='yeast3',
        image_size=image_size,
    )
    tuner.run_experiment(data, _RESULTS_FILE)

    print("--- total: %s seconds ---" % (time.time() - start_time))



