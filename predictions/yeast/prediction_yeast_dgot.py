import os
import random
import time
import numpy as np
import torch

from base_functions import get_yeast_3_data, get_yeast_4_data, get_yeast_5_data, get_yeast_6_data
from constants import GARunConfig
from optimization.ga_dgot_tuner import GaDGOTTuner

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RESULTS_DIR  = os.path.join(_PROJECT_ROOT, 'results')

seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def __run_experiment(yeast_data, dataset_name, results_file):
    num_cols = list(yeast_data[0].columns.values)
    categorical_cols = None
    # feature_len is a pre-encoding estimate; run_experiment updates it after OHE
    input_dim = len(num_cols) + (len(categorical_cols) if categorical_cols else 0)
    start_time = time.time()

    print(f"Starting simulation run...")
    tuner = GaDGOTTuner(
        num_generations=GARunConfig.NUM_GENERATIONS.value,
        num_parents=GARunConfig.NUM_PARENTS.value,
        population=GARunConfig.POPULATION.value,
        feature_len=input_dim,
        dataset_name=dataset_name,
        numerical_cols=num_cols,
        categorical_cols=categorical_cols,
    )
    tuner.run_experiment(data=yeast_data, fname=results_file)
    print("--- total: %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    _DGOT_RESULTS_DIR = os.path.join(_RESULTS_DIR, 'dgot')
    os.makedirs(_DGOT_RESULTS_DIR, exist_ok=True)

    # yeast 3
    __run_experiment(yeast_data=get_yeast_3_data(), dataset_name='yeast3',
                     results_file=os.path.join(_DGOT_RESULTS_DIR, 'dgot_yeast_3'))

    # yeast 4
    __run_experiment(yeast_data=get_yeast_4_data(), dataset_name='yeast4',
                     results_file=os.path.join(_DGOT_RESULTS_DIR, 'dgot_yeast_4'))

    # yeast 5
    __run_experiment(yeast_data=get_yeast_5_data(), dataset_name='yeast5',
                     results_file=os.path.join(_DGOT_RESULTS_DIR, 'dgot_yeast_5'))

    # yeast 6
    __run_experiment(yeast_data=get_yeast_6_data(), dataset_name='yeast6',
                     results_file=os.path.join(_DGOT_RESULTS_DIR, 'dgot_yeast_6'))
