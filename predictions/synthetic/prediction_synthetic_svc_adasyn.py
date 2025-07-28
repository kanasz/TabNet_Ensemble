import random
import time
import numpy as np
import torch

from base_functions import get_synthetic_data
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


def run_experiment(data_prefix, data_contamination, data_features, file_name, data_samples=None):
    num_generations = 50
    num_parents = 20
    population = 50
    start_time = time.time()
    data = get_synthetic_data(prefix=data_prefix, contamination=data_contamination, features=data_features,
                              samples=data_samples)
    numerical_cols = list(data[0].columns.values)

    print(f"Starting simulation run...")
    tuner = GaTuner(num_generations=num_generations,
                    num_parents=num_parents,
                    population=population,
                    use_smote=False, use_adasyn=True, use_smote_enn=False,
                    clf_type=Classifier.SVC,
                    numerical_cols=numerical_cols)
    tuner.run_experiment(data, file_name)
    print("--- total: %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    # Dataset synthetic1: data/synthetic_data/01_synthetic_0.3_contamination_50_features.csv
    run_experiment(data_prefix='01', data_contamination='0.3', data_features=50,
                   file_name="results/svc_adasyn_01_synthetic_0.3_contamination_50_features")
    
    # Dataset synthetic2: data/synthetic_data/02_synthetic_0.1_contamination_100_features_200_samples.csv
    run_experiment(data_prefix='02', data_contamination='0.1', data_features=100,
                   file_name="results/svc_adasyn_02_synthetic_0.1_contamination_100_features_200_samples",
                   data_samples=200)

    # Dataset synthetic3: data/synthetic_data/03_synthetic_0.02_contamination_200_features_200_samples.csv
    run_experiment(data_prefix='03', data_contamination='0.02', data_features=200,
                   file_name="results/svc_adasyn_03_synthetic_0.02_contamination_200_features_200_samples")
