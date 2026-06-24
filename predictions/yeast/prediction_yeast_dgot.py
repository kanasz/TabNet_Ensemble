import os
import random
import time
import numpy as np
import torch

from base_functions import get_yeast_3_data
from optimization.ga_dgot_tuner import GaDGOTTuner

# Compute absolute results path BEFORE importing ga_dgot_tuner, because that
# module calls os.chdir(_DGOT_PATH) at import time, which would make any later
# relative path resolve inside the DGOT directory instead of the project root.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RESULTS_FILE = os.path.join(_PROJECT_ROOT, 'results', 'DGOT_yeast_3')

seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def __run_experiment(yeast_data, file_name):
    num_generations = 50
    num_parents = 20
    population = 50
    start_time = time.time()
    data = yeast_data
    input_dim = data[0].shape[1]
    numerical_cols = list(data[0].columns.values)

    print(f"Starting simulation run...")
    tuner = GaDGOTTuner(num_generations=num_generations, num_parents=num_parents, population=population,
                        feature_len=input_dim, dataset_name=file_name)
    tuner.run_experiment(data=data, file_name=file_name)
    print("--- total: %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':

    __run_experiment(yeast_data=get_yeast_3_data(), file_name="")
