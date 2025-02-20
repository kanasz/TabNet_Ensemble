import random
import time

import numpy as np
import torch

from base_functions import get_synthetic_data, get_slovak_data, get_sensitivity_synthetic_data
from constants import LossFunction, Classifier
from optimization.ga_boosting_tabnet_tuner import GaBoostingTabnetTuner
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


# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# warnings.filterwarnings("ignore")

if __name__ == '__main__':
    tabnet_max_epochs = 50
    num_generations = 50
    num_parents = 20  # 10
    population = 50  # 20
    start_time = time.time()
    contamination = '0.05'
    features = 50
    samples = 250
    numerical_columns = []

    contamination = '0.05'
    features = 20
    samples = 250
    id = '01'

    data = get_sensitivity_synthetic_data(id, contamination, features, samples)
    numerical_cols = list(data[0].columns.values)
    '''
    tuner = GaTuner(num_generations, num_parents, population,
                    use_smote=True,
                    clf_type=Classifier.SVC, numerical_cols=numerical_cols)
    tuner.run_experiment(data, 'results/SVC_SMOTE_Synthetic_04'
                         .format(contamination, features, samples))

    
    tuner = GaTuner(num_generations, num_parents, population,
                    use_smote=False,
                    clf_type=Classifier.WeightedSVC, numerical_cols=numerical_cols)
    tuner.run_experiment(data, 'results/SVC_WEIGHTED_Synthetic_04'
                         .format(contamination, features, samples))

    '''
    tuner = GaTuner(num_generations, num_parents, population,
                    use_smote=False,
                    use_adasyn=True,
                    clf_type=Classifier.SVC, numerical_cols=numerical_cols)
    tuner.run_experiment(data, 'results/AdasynSVC_Synthetic_04'
                         .format(contamination, features, samples))

    '''
    for i in range(features):
        numerical_columns.append('feature{}'.format(i+1))
    data = get_synthetic_data('01', contamination, features)

    tuner = GaTuner( num_generations, num_parents, population,
                                  use_smote=False,
                                  clf_type=Classifier.SVC, numerical_cols=numerical_columns)
    tuner.run_experiment(data, 'results_200_samples/SVC_01_synthetic_{}_contamination_{}_features_{}'
                         .format(contamination, features, samples))
    print("--- total: %s seconds ---" % (time.time() - start_time))
    print("Experiment info -> data: {}, features: {}".format(contamination, features))

    tuner = GaTuner(num_generations, num_parents, population,
                    use_smote=True,
                    clf_type=Classifier.SVC, numerical_cols=numerical_columns)
    tuner.run_experiment(data, 'results_200_samples/SVC_SMOTE_01_synthetic_{}_contamination_{}_features_{}'
                         .format(contamination, features, samples))
    print("--- total: %s seconds ---" % (time.time() - start_time))
    print("Experiment info -> data: {}, features: {}".format(contamination, features))

    tuner = GaTuner(num_generations, num_parents, population,
                    use_smote=False,
                    clf_type=Classifier.WeightedSVC, numerical_cols=numerical_columns)
    tuner.run_experiment(data, 'results_200_samples/SVC_WEIGHTED_01_synthetic_{}_contamination_{}_features_{}'
                         .format(contamination, features, samples))
    print("--- total: %s seconds ---" % (time.time() - start_time))
    print("Experiment info -> data: {}, features: {}".format(contamination, features))
   
    contamination = '0.1'
    features = 100
    samples = 200
    numerical_columns = []
    for i in range(features):
        numerical_columns.append('feature{}'.format(i + 1))
    data = get_synthetic_data('02', contamination, features, samples)

    tuner = GaTuner(num_generations, num_parents, population,
                    use_smote=False,
                    clf_type=Classifier.SVC, numerical_cols=numerical_columns)
    tuner.run_experiment(data, 'results_200_samples/SVC_02_synthetic_{}_contamination_{}_features_{}'
                         .format(contamination, features, samples))
    print("--- total: %s seconds ---" % (time.time() - start_time))
    print("Experiment info -> data: {}, features: {}".format(contamination, features))

    tuner = GaTuner(num_generations, num_parents, population,
                    use_smote=True,
                    clf_type=Classifier.SVC, numerical_cols=numerical_columns)
    tuner.run_experiment(data, 'results_200_samples/SVC_SMOTE_02_synthetic_{}_contamination_{}_features_{}'
                         .format(contamination, features, samples))
    print("--- total: %s seconds ---" % (time.time() - start_time))
    print("Experiment info -> data: {}, features: {}".format(contamination, features))

    tuner = GaTuner(num_generations, num_parents, population,
                    use_smote=False,
                    clf_type=Classifier.WeightedSVC, numerical_cols=numerical_columns)
    tuner.run_experiment(data, 'results_200_samples/SVC_WEIGHTED_02_synthetic_{}_contamination_{}_features_{}'
                         .format(contamination, features, samples))
    print("--- total: %s seconds ---" % (time.time() - start_time))
    print("Experiment info -> data: {}, features: {}".format(contamination, features))

    contamination = '0.02'
    features = 200
    samples = 200
    numerical_columns = []
    for i in range(features):
        numerical_columns.append('feature{}'.format(i + 1))
    data = get_synthetic_data('03', contamination, features, samples)

    tuner = GaTuner(num_generations, num_parents, population,
                    use_smote=False,
                    clf_type=Classifier.SVC, numerical_cols=numerical_columns)
    tuner.run_experiment(data, 'results_200_samples/SVC_03_synthetic_{}_contamination_{}_features_{}'
                         .format(contamination, features, samples))
    print("--- total: %s seconds ---" % (time.time() - start_time))
    print("Experiment info -> data: {}, features: {}".format(contamination, features))

    tuner = GaTuner(num_generations, num_parents, population,
                    use_smote=True,
                    clf_type=Classifier.SVC, numerical_cols=numerical_columns)
    tuner.run_experiment(data, 'results_200_samples/SVC_SMOTE_03_synthetic_{}_contamination_{}_features_{}'
                         .format(contamination, features, samples))
    print("--- total: %s seconds ---" % (time.time() - start_time))
    print("Experiment info -> data: {}, features: {}".format(contamination, features))

    tuner = GaTuner(num_generations, num_parents, population,
                    use_smote=False,
                    clf_type=Classifier.WeightedSVC, numerical_cols=numerical_columns)
    tuner.run_experiment(data, 'results_200_samples/SVC_WEIGHTED_03_synthetic_{}_contamination_{}_features_{}'
                         .format(contamination, features, samples))
    print("--- total: %s seconds ---" % (time.time() - start_time))
    print("Experiment info -> data: {}, features: {}".format(contamination, features))
    '''








