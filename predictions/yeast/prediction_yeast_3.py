#SBATCH --account=p482-24-1
import random
import sys
import time

import numpy as np
import torch

from base_functions import get_config_files, get_yeast_3_data
from constants import LossFunction, CLUSTER_COUNT, WEAK_CLASSIFIERS_COUNT, SYNTHETIC_MINORITY_COUNT
from models.oc_bagging_tabnet_ensemble_parallel import GaOCBaggingTabnetEnsembleTunerParallel

seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.rand(seed)
random.SystemRandom(seed)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # for multiGPUs.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
np.set_printoptions(threshold=np.inf)


if __name__ == '__main__':
    # test pytorch
    # Check if CUDA (NVIDIA GPU support) is available
    if torch.cuda.is_available():
        print("CUDA is available.")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"Memory Allocated: {torch.cuda.memory_allocated(i) / 1024 ** 3:.2f} GB")
            print(f"Memory Cached: {torch.cuda.memory_reserved(i) / 1024 ** 3:.2f} GB")
    else:
        print("CUDA is not available. No GPU found.")
    sys.stdout.flush()

    tabnet_max_epochs = 50
    num_generations = 50
    num_parents = 20  # 10
    population = 50  # 20
    start_time = time.time()
    samples = 200
    actual_loss_function = LossFunction.CROSSENTROPYLOSS

    data = get_yeast_3_data()
    numerical_cols = list(data[0].columns.values)
    categorical_cols = None
    config_files = get_config_files("models/configurations")
    tuner = GaOCBaggingTabnetEnsembleTunerParallel(tabnet_max_epochs, num_generations, num_parents, population, config_files=config_files, device='cuda',
                                              numerical_cols=numerical_cols, categorical_cols=categorical_cols)
    tuner.run_experiment(data,
                         'results/OC_TABNET_ENSEMBLE_CROSSENTROPYLOSS_yeast_3_CLUSTER_COUNT_{}_CLASSIFIER_COUNT_{}_SYNTH_COUNT_{}_2'
                         .format( CLUSTER_COUNT, WEAK_CLASSIFIERS_COUNT, SYNTHETIC_MINORITY_COUNT), actual_loss_function)
    print("--- total: %s seconds ---" % (time.time() - start_time))





