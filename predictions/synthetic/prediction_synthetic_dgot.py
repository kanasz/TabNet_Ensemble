import os
import random
import time
import numpy as np
import torch

from base_functions import get_synthetic_data
from constants import GARunConfig
from optimization.ga_dgot_tuner import GaDGOTTuner

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_RESULTS_DIR = os.path.join(_PROJECT_ROOT, 'results')
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


def __run_experiment(data_prefix, data_contamination, data_features, dataset_name, results_file, samples=None):

    synthetic_data = get_synthetic_data(prefix=data_prefix, contamination=data_contamination, features=data_features,
                                        samples=samples)
    num_cols = list(synthetic_data[0].columns.values)
    categorical_cols = None
    input_dim = len(num_cols)
    start_time = time.time()

    print(f"Starting simulation run for {dataset_name}...")
    tuner = GaDGOTTuner(
        num_generations=GARunConfig.NUM_GENERATIONS.value,
        num_parents=GARunConfig.NUM_PARENTS.value,
        population=GARunConfig.POPULATION.value,
        feature_len=input_dim,
        dataset_name=dataset_name,
        numerical_cols=num_cols,
        categorical_cols=categorical_cols,
    )
    log_dir = os.path.join(_LOG_DIR, dataset_name)
    os.makedirs(log_dir, exist_ok=True)
    tuner.run_experiment(data=synthetic_data, fname=results_file, log_dir=log_dir)
    print("--- total: %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    _DGOT_RESULTS_DIR = os.path.join(_RESULTS_DIR, 'dgot')
    os.makedirs(_DGOT_RESULTS_DIR, exist_ok=True)

    # Dataset synthetic1: data/synthetic_data/01_synthetic_0.3_contamination_50_features.csv
    __run_experiment(data_prefix='01', data_contamination='0.3', data_features=50,
                     dataset_name="01_synthetic_0.3_contamination_50_features",
                     results_file=os.path.join(_DGOT_RESULTS_DIR, 'dgot_01_synthetic_0.3_contamination_50_features'))

    # Dataset synthetic2: data/synthetic_data/02_synthetic_0.1_contamination_100_features_200_samples.csv
    __run_experiment(data_prefix='02', data_contamination='0.1', data_features=100,
                     dataset_name="02_synthetic_0.1_contamination_100_features_200_samples",
                     results_file=os.path.join(_DGOT_RESULTS_DIR,
                                               'dgot_02_synthetic_0.1_contamination_100_features_200_samples'),
                     samples=200)

    # Dataset synthetic3: data/synthetic_data/03_synthetic_0.02_contamination_200_features_200_samples.csv
    __run_experiment(data_prefix='03', data_contamination='0.02', data_features=200,
                     dataset_name="03_synthetic_0.02_contamination_200_features_200_samples",
                     results_file=os.path.join(_DGOT_RESULTS_DIR,
                                               'dgot_02_synthetic_0.1_contamination_100_features_200_samples'),
                     samples=200)
