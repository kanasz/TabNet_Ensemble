import random
import time
import numpy as np
import torch

from base_functions import get_yeast_3_data, get_yeast_4_data, get_yeast_5_data, get_yeast_6_data
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
np.set_printoptions(threshold=np.inf)


def run_experiment(yeast_data, file_name):
    num_generations = 50
    num_parents = 20
    population = 50
    start_time = time.time()
    data = yeast_data
    numerical_cols = list(data[0].columns.values)

    print(f"Starting simulation run...")
    tuner = GaTuner(num_generations=num_generations,
                    num_parents=num_parents,
                    population=population,
                    use_smote=False,
                    use_adasyn=False,
                    use_smote_enn=True,
                    clf_type=Classifier.SVC,
                    numerical_cols=numerical_cols)
    tuner.run_experiment(data, file_name)
    print("--- total: %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    # yeast 3
    run_experiment(yeast_data=get_yeast_3_data(), file_name="results/svc_smoteenn_yeast_3")

    # yeast 4
    run_experiment(yeast_data=get_yeast_4_data(), file_name="results/svc_smoteenn_yeast_4")

    # yeast 5
    run_experiment(yeast_data=get_yeast_5_data(), file_name="results/svc_smoteenn_yeast_5")

    # yeast 6
    run_experiment(yeast_data=get_yeast_6_data(), file_name="results/svc_smoteenn_yeast_6")
