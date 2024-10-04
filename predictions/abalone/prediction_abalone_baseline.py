import random
import time

import numpy as np
import torch

from base_functions import get_synthetic_data, get_slovak_data, get_abalone_9_vs_18_data, \
    get_abalone_19_vs_10_11_12_13_data, get_abalone_20_vs_8_9_10_data, get_abalone_3_vs_11_data
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
    contamination = '0.3'
    features = 50
    samples = 200
    numerical_columns = []

    data = get_abalone_3_vs_11_data()
    numerical_cols = ['Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight',
                      'Shell_weight']
    categorical_cols = ['Sex']
    '''
    tuner = GaTuner(num_generations, num_parents, population,
                    use_smote=False,
                    clf_type=Classifier.WeightedSVC, numerical_cols=numerical_columns, categorical_cols=categorical_cols)
    tuner.run_experiment(data, 'results/SVC_WEIGHTED_abalone_3_vs_11')
    print("--- total: %s seconds ---" % (time.time() - start_time))
    '''
    tuner = GaTuner(num_generations, num_parents, population,
                    use_smote=True,
                    clf_type=Classifier.SVC, numerical_cols=numerical_columns,
                    categorical_cols=categorical_cols)
    tuner.run_experiment(data, 'results/SVC_SMOTE_abalone_3_vs_11')
    print("--- total: %s seconds ---" % (time.time() - start_time))


    '''
    data  = get_abalone_9_vs_18_data()
    numerical_cols = ['Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight',
                      'Shell_weight']
    categorical_cols = ['Sex']

    tuner = GaTuner(num_generations, num_parents, population,
                    use_smote=False,
                    clf_type=Classifier.SVC, numerical_cols=numerical_columns, categorical_cols=categorical_cols)
    tuner.run_experiment(data, 'results/SVC_abalone_9_vs_18')
    print("--- total: %s seconds ---" % (time.time() - start_time))
    print("Experiment info -> data: {}, features: {}".format(contamination, features))

    tuner = GaTuner(num_generations, num_parents, population,
                    use_smote=True,
                    clf_type=Classifier.SVC, numerical_cols=numerical_columns, categorical_cols=categorical_cols)
    tuner.run_experiment(data, 'results/SVC_SMOTE_abalone_9_vs_18')
    print("--- total: %s seconds ---" % (time.time() - start_time))
    print("Experiment info -> data: {}, features: {}".format(contamination, features))

    tuner = GaTuner(num_generations, num_parents, population,
                    use_smote=False,
                    clf_type=Classifier.WeightedSVC, numerical_cols=numerical_columns, categorical_cols=categorical_cols)
    tuner.run_experiment(data, 'results/SVC_WEIGHTED_abalone_9_vs_18'
                         .format(contamination, features, samples))
    print("--- total: %s seconds ---" % (time.time() - start_time))
    print("Experiment info -> data: {}, features: {}".format(contamination, features))

    
    data  = get_abalone_19_vs_10_11_12_13_data()
    numerical_cols = ['Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight',
                      'Shell_weight']
    categorical_cols = ['Sex']

    tuner = GaTuner(num_generations, num_parents, population,
                    use_smote=False,
                    clf_type=Classifier.SVC, numerical_cols=numerical_columns, categorical_cols=categorical_cols)
    tuner.run_experiment(data, 'results/SVC_abalone_19_vs_10_11_12_13')
    print("--- total: %s seconds ---" % (time.time() - start_time))
    print("Experiment info -> data: {}, features: {}".format(contamination, features))

    tuner = GaTuner(num_generations, num_parents, population,
                    use_smote=True,
                    clf_type=Classifier.SVC, numerical_cols=numerical_columns, categorical_cols=categorical_cols)
    tuner.run_experiment(data, 'results/SVC_SMOTE_abalone_19_vs_10_11_12_13')
    print("--- total: %s seconds ---" % (time.time() - start_time))
    print("Experiment info -> data: {}, features: {}".format(contamination, features))

    tuner = GaTuner(num_generations, num_parents, population,
                    use_smote=False,
                    clf_type=Classifier.WeightedSVC, numerical_cols=numerical_columns, categorical_cols=categorical_cols)
    tuner.run_experiment(data, 'results/SVC_WEIGHTED_abalone_19_vs_10_11_12_13'
                         .format(contamination, features, samples))
    print("--- total: %s seconds ---" % (time.time() - start_time))
    print("Experiment info -> data: {}, features: {}".format(contamination, features))


    data = get_abalone_20_vs_8_9_10_data()
    numerical_cols = ['Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight',
                      'Shell_weight']
    categorical_cols = ['Sex']

    tuner = GaTuner(num_generations, num_parents, population,
                    use_smote=False,
                    clf_type=Classifier.SVC, numerical_cols=numerical_columns, categorical_cols=categorical_cols)
    tuner.run_experiment(data, 'results/SVC_abalone_20_vs_8_9_10')
    print("--- total: %s seconds ---" % (time.time() - start_time))
    print("Experiment info -> data: {}, features: {}".format(contamination, features))

    tuner = GaTuner(num_generations, num_parents, population,
                    use_smote=True,
                    clf_type=Classifier.SVC, numerical_cols=numerical_columns, categorical_cols=categorical_cols)
    tuner.run_experiment(data, 'results/SVC_SMOTE_abalone_20_vs_8_9_10')
    print("--- total: %s seconds ---" % (time.time() - start_time))
    print("Experiment info -> data: {}, features: {}".format(contamination, features))

    tuner = GaTuner(num_generations, num_parents, population,
                    use_smote=False,
                    clf_type=Classifier.WeightedSVC, numerical_cols=numerical_columns,
                    categorical_cols=categorical_cols)
    tuner.run_experiment(data, 'results/SVC_WEIGHTED_abalone_20_vs_8_9_10'
                         .format(contamination, features, samples))
    print("--- total: %s seconds ---" % (time.time() - start_time))
    print("Experiment info -> data: {}, features: {}".format(contamination, features))
    '''