import os
import random
import time
import numpy as np
import torch

from base_functions import get_abalone19_data, get_abalone_9_vs_18_data, get_abalone_3_vs_11_data
from base_functions import get_abalone_20_vs_8_9_10_data, get_abalone_19_vs_10_11_12_13_data
from constants import GARunConfig

# Compute absolute paths BEFORE importing ga_dgot_tuner — that module calls
# os.chdir(_DGOT_PATH) at import time, so any abspath() call after it would
# resolve relative __file__ against the DGOT directory instead of project root.
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_RESULTS_DIR = os.path.join(_PROJECT_ROOT, 'results')
# Transient GA output (the per-generation resume checkpoint) - absolute,
# because ga_dgot_tuner chdir()s into the DGOT directory on import.
_LOG_DIR = os.path.join(_PROJECT_ROOT, 'ga_logs', 'dgot')

from optimization.ga_dgot_tuner import GaDGOTTuner  # noqa: E402

seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def __run_experiment(abalone_data, dataset_name, results_file):
    # num_cols = list(abalone_data[0].columns.values)
    # categorical_cols = None
    num_cols = ['Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight']
    categorical_cols = ['Sex']
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
    log_dir = os.path.join(_LOG_DIR, dataset_name)
    os.makedirs(log_dir, exist_ok=True)
    tuner.run_experiment(data=abalone_data, fname=results_file, log_dir=log_dir)
    print("--- total: %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    _DGOT_RESULTS_DIR = os.path.join(_RESULTS_DIR, 'dgot')
    os.makedirs(_DGOT_RESULTS_DIR, exist_ok=True)

    # abalone19
    __run_experiment(abalone_data=get_abalone19_data(), dataset_name='abalone19',
                     results_file=os.path.join(_DGOT_RESULTS_DIR, 'dgot_abalone19'))
    
    # abalone_3_vs_11
    __run_experiment(abalone_data=get_abalone_3_vs_11_data(), dataset_name='abalone_3_vs_11',
                     results_file=os.path.join(_DGOT_RESULTS_DIR, 'dgot_abalone_3_vs_11'))
    
    # abalone_9_vs_18
    __run_experiment(abalone_data=get_abalone_9_vs_18_data(), dataset_name='abalone_9_vs_18',
                     results_file=os.path.join(_DGOT_RESULTS_DIR, 'dgot_abalone_9_vs_18'))
    
    # abalone_20_vs_8_9_10
    __run_experiment(abalone_data=get_abalone_20_vs_8_9_10_data(), dataset_name='abalone_20_vs_8_9_10',
                     results_file=os.path.join(_DGOT_RESULTS_DIR, 'dgot_abalone_20_vs_8_9_10'))

    # abalone_19_vs_10_11_12_13
    __run_experiment(abalone_data=get_abalone_19_vs_10_11_12_13_data(), dataset_name='abalone_19_vs_10_11_12_13',
                     results_file=os.path.join(_DGOT_RESULTS_DIR, 'dgot_abalone_19_vs_10_11_12_13'))
