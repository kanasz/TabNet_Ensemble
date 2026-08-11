import os
import random
import time
import numpy as np
import torch

from base_functions import get_ecoli_0_vs_1_data, get_ecoli_0_4_6_vs_5_data, get_ecoli_0_3_4_vs_5_data
from base_functions import get_ecoli_0_2_3_4_vs_5_data
from constants import GARunConfig

# Compute absolute paths BEFORE importing ga_dgot_tuner — that module calls
# os.chdir(_DGOT_PATH) at import time, so any abspath() call after it would
# resolve relative __file__ against the DGOT directory instead of project root.
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_RESULTS_DIR = os.path.join(_PROJECT_ROOT, 'results')

from optimization.ga_dgot_tuner import GaDGOTTuner  # noqa: E402

seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def __run_experiment(ecoli_data, dataset_name, results_file):
    num_cols = list(ecoli_data[0].columns.values)
    categorical_cols = None
    input_dim = len(num_cols)
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
    tuner.run_experiment(data=ecoli_data, fname=results_file)
    print("--- total: %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    _DGOT_RESULTS_DIR = os.path.join(_RESULTS_DIR, 'dgot')
    os.makedirs(_DGOT_RESULTS_DIR, exist_ok=True)

    # ecoli_0_vs_1
    __run_experiment(ecoli_data=get_ecoli_0_vs_1_data(), dataset_name='ecoli_0_vs_1',
                     results_file=os.path.join(_DGOT_RESULTS_DIR, 'dgot_ecoli_0_vs_1'))
    """
    # ecoli_0_3_4_vs_5
    __run_experiment(ecoli_data=get_ecoli_0_3_4_vs_5_data(), dataset_name='ecoli_0_3_4_vs_5',
                     results_file=os.path.join(_DGOT_RESULTS_DIR, 'dgot_ecoli_0_3_4_vs_5'))

    # ecoli_0_4_6_vs_5
    __run_experiment(ecoli_data=get_ecoli_0_4_6_vs_5_data(), dataset_name='ecoli_0_4_6_vs_5',
                     results_file=os.path.join(_DGOT_RESULTS_DIR, 'dgot_ecoli_0_4_6_vs_5'))
    # ecoli_0_2_3_4_vs_5
    __run_experiment(ecoli_data=get_ecoli_0_2_3_4_vs_5_data(), dataset_name='ecoli_0_2_3_4_vs_5',
                     results_file=os.path.join(_DGOT_RESULTS_DIR, 'dgot_ecoli_0_2_3_4_vs_5'))
    """
