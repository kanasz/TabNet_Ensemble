import random
import time
import numpy as np
import torch

from base_functions import get_abalone19_data, get_config_files, get_meanshift_cluster_counts
from constants import LossFunction
from models.oc_bagging_tabnet_ensemble_parallel import GaOCBaggingTabnetEnsembleTunerParallel
from imblearn.over_sampling import SMOTE

seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.rand(seed)
random.SystemRandom(seed)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    tabnet_max_epochs = 50
    num_generations = 50
    num_parents = 20
    population = 50
    start_time = time.time()
    actual_loss_function = LossFunction.CROSSENTROPYLOSS
    data = get_abalone19_data()
    numerical_cols = ['Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight',
                      'Shell_weight']
    categorical_cols = ['Sex']
    clusters, bandwidths, algs = get_meanshift_cluster_counts(data[0], data[1], numerical_cols, categorical_cols)
    sampling_algorithm = SMOTE(random_state=42)
    clustering_params = {
        "bandwidths": bandwidths,
        "clusters": clusters,
        "type": "MS",
        "algs": algs
    }
    config_files = get_config_files("../../models/configurations")
    tuner = GaOCBaggingTabnetEnsembleTunerParallel(tabnet_max_epochs=tabnet_max_epochs,
                                                   num_generations=num_generations,
                                                   num_parents=num_parents,
                                                   population=population,
                                                   config_files=config_files,
                                                   device='cuda',
                                                   sampling_algorithm=sampling_algorithm,
                                                   numerical_cols=numerical_cols,
                                                   categorical_cols=categorical_cols,
                                                   save_partial_output=True,
                                                   clustering_params=clustering_params,
                                                   use_cluster_centers=False)
    tuner.run_experiment(data=data, fname='results/unclustered_smote_meanshift_abalone19/UNCLUSTERED_OC_TABNET_ENSEMBLE_SMOTE_MEANSHIFT_abalone19')
    print("--- total: %s seconds ---" % (time.time() - start_time))
