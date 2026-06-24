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
from optimization.ga_sos_tuner import GaSOSTuner

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


def __run_experiment(wine_data, dataset_name, results_file):
    start_time = time.time()
    X_df, y_series = wine_data

    # image_size = n_features + n_label_classes (categorical label -> one-hot)
    n_classes = len(y_series.unique())
    image_size = X_df.shape[1] + n_classes

    print(f"Starting simulation run for {dataset_name}...")
    tuner = GaSOSTuner(
        num_generations=GARunConfig.NUM_GENERATIONS.value,
        num_parents=GARunConfig.NUM_PARENTS.value,
        population=GARunConfig.POPULATION.value,
        dataset_name=dataset_name,
        image_size=image_size,
    )
    tuner.run_experiment(wine_data, results_file)
    print("--- total: %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    _SOS_RESULTS_DIR = os.path.join(_RESULTS_DIR, 'sos')
    os.makedirs(_SOS_RESULTS_DIR, exist_ok=True)

    __run_experiment(get_wine_quality_red_3_vs_5_data(),
                     dataset_name='wine_red_3_vs_5',
                     results_file=os.path.join(_SOS_RESULTS_DIR, 'sos_wine_red_3_vs_5'))

    __run_experiment(get_wine_quality_red_8_vs_6_data(),
                     dataset_name='wine_red_8_vs_6',
                     results_file=os.path.join(_SOS_RESULTS_DIR, 'sos_wine_red_8_vs_6'))

    __run_experiment(get_wine_quality_white_3_vs_7_data(),
                     dataset_name='wine_white_3_vs_7',
                     results_file=os.path.join(_SOS_RESULTS_DIR, 'sos_wine_white_3_vs_7'))

    __run_experiment(get_wine_quality_white_9_vs_4_data(),
                     dataset_name='wine_white_9_vs_4',
                     results_file=os.path.join(_SOS_RESULTS_DIR, 'sos_wine_white_9_vs_4'))


