import random
import time

import numpy as np
import torch

from base_functions import get_wine_quality_white_3_vs_7_data
from constants import Classifier
from optimization.ga_tuner import GaTuner

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

if __name__ == '__main__':
    tabnet_max_epochs = 50
    num_generations = 50
    num_parents = 20  # 10
    population = 50  # 20
    start_time = time.time()

    data = get_wine_quality_white_3_vs_7_data()
    numerical_columns = list(data[0].columns.values)
    tuner = GaTuner(num_generations, num_parents, population,
                    use_smote=False,
                    use_adasyn=True,
                    clf_type=Classifier.SVC, numerical_cols=numerical_columns)
    tuner.run_experiment(data, 'results/AdasynSVC_white_3_vs_7')
    print("--- total: %s seconds ---" % (time.time() - start_time))



