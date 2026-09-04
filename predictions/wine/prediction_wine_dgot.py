import os
import random
import time
import numpy as np
import torch

from base_functions import get_wine_quality_red_3_vs_5_data, get_wine_quality_red_8_vs_6_data
from base_functions import get_wine_quality_white_3_vs_7_data, get_wine_quality_white_9_vs_4_data
from constants import GASotaRunConfig
from optimization.ga_dgot_tuner import GaDGOTTuner


_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_RESULTS_DIR  = os.path.join(_PROJECT_ROOT, 'results')
# Transient GA output (the per-generation resume checkpoint) - absolute,
# because ga_dgot_tuner chdir()s into the DGOT directory on import.
_LOG_DIR = os.path.join(_PROJECT_ROOT, 'ga_logs', 'dgot')

seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def __run_experiment(wine_data, dataset_name, results_file):
    num_cols = list(wine_data[0].columns.values)
    categorical_cols = None
    input_dim = len(num_cols)
    start_time = time.time()

    print(f"Starting simulation run for {dataset_name}...")
    tuner = GaDGOTTuner(
        num_generations=GASotaRunConfig.NUM_GENERATIONS.value,
        num_parents=GASotaRunConfig.NUM_PARENTS.value,
        population=GASotaRunConfig.POPULATION.value,
        feature_len=input_dim,
        dataset_name=dataset_name,
        numerical_cols=num_cols,
        categorical_cols=categorical_cols,
    )
    log_dir = os.path.join(_LOG_DIR, dataset_name)
    os.makedirs(log_dir, exist_ok=True)
    tuner.run_experiment(data=wine_data, fname=results_file, log_dir=log_dir)
    print("--- total: %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    _DGOT_RESULTS_DIR = os.path.join(_RESULTS_DIR, 'dgot')
    os.makedirs(_DGOT_RESULTS_DIR, exist_ok=True)

    __run_experiment(get_wine_quality_red_3_vs_5_data(), dataset_name='wine_red_3_vs_5',
                     results_file=os.path.join(_DGOT_RESULTS_DIR, 'dgot_wine_red_3_vs_5'))

    __run_experiment(get_wine_quality_red_8_vs_6_data(), dataset_name='wine_red_8_vs_6',
                     results_file=os.path.join(_DGOT_RESULTS_DIR, 'dgot_wine_red_8_vs_6'))

    __run_experiment(get_wine_quality_white_3_vs_7_data(), dataset_name='wine_white_3_vs_7',
                     results_file=os.path.join(_DGOT_RESULTS_DIR, 'dgot_wine_white_3_vs_7'))

    __run_experiment(get_wine_quality_white_9_vs_4_data(), dataset_name='wine_white_9_vs_4',
                     results_file=os.path.join(_DGOT_RESULTS_DIR, 'dgot_wine_white_9_vs_4'))



