import random
import time
import numpy as np
import torch
from base_functions import get_polish_bankruptcy_data
from constants import LossFunction
from optimization.ga_boosting_tabnet_tuner import GaBoostingTabnetTuner


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


# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# warnings.filterwarnings("ignore")

if __name__ == '__main__':
    tabnet_max_epochs = 50
    num_generations = 50
    num_parents = 10  # 10
    population = 20  # 20

    start_time = time.time()
    actual_loss_function = LossFunction.CROSSENTROPYLOSS
    data = get_polish_bankruptcy_data(index=5)
    tuner = GaBoostingTabnetTuner(tabnet_max_epochs, num_generations, num_parents, population, device='cuda',
                                  use_smote=False, use_new_model=True)
    tuner.run_experiment(data=data, fname="results/BOOSTING_TABNET_CROSS_ENTROPY_LOSS",
                         loss_function=actual_loss_function)
    print("--- total: %s seconds ---" % (time.time() - start_time))
