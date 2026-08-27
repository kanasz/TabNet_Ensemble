import time
import numpy as np

from base_functions import get_yeast_3_data, get_yeast_4_data, get_yeast_5_data, get_yeast_6_data
from base_functions import get_config_files, set_seed
from constants import LossFunction, CLUSTER_COUNT, WEAK_CLASSIFIERS_COUNT, SYNTHETIC_MINORITY_COUNT
from models.oc_bagging_tabnet_ensemble_parallel import GaOCBaggingTabnetEnsembleTunerParallel

set_seed(seed=42)
np.set_printoptions(threshold=np.inf)
DIR_NAME = 'results/oc_tabnet_ensemble_cross_entropy'
actual_loss_function = LossFunction.CROSSENTROPYLOSS


def run_experiment(yeast_data, file_name):
    tabnet_max_epochs = 50
    num_generations = 50
    num_parents = 20
    population = 50
    start_time = time.time()
    data = yeast_data
    numerical_cols = list(data[0].columns.values)
    categorical_cols = None

    print(f"Starting simulation run...")
    config_files = get_config_files("../../models/configurations")
    tuner = GaOCBaggingTabnetEnsembleTunerParallel(tabnet_max_epochs, num_generations, num_parents, population,
                                                   config_files=config_files, device='cuda',
                                                   numerical_cols=numerical_cols, categorical_cols=categorical_cols)
    tuner.run_experiment(data, file_name, actual_loss_function)
    print("--- total: %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':

    # Cluster/classifier/synthetic counts are baked into the result file name,
    # matching the naming the per-dataset scripts used before consolidation.
    suffix = "CLUSTER_COUNT_{}_CLASSIFIER_COUNT_{}_SYNTH_COUNT_{}_2".format(
        CLUSTER_COUNT, WEAK_CLASSIFIERS_COUNT, SYNTHETIC_MINORITY_COUNT)

    # yeast_3
    run_experiment(yeast_data=get_yeast_3_data(),
                   file_name=f"{DIR_NAME}/OC_TABNET_ENSEMBLE_CROSSENTROPYLOSS_yeast_3_{suffix}")

    # yeast_4
    run_experiment(yeast_data=get_yeast_4_data(),
                   file_name=f"{DIR_NAME}/OC_TABNET_ENSEMBLE_CROSSENTROPYLOSS_yeast_4_{suffix}")

    # yeast_5
    run_experiment(yeast_data=get_yeast_5_data(),
                   file_name=f"{DIR_NAME}/OC_TABNET_ENSEMBLE_CROSSENTROPYLOSS_yeast_5_{suffix}")

    # yeast_6
    run_experiment(yeast_data=get_yeast_6_data(),
                   file_name=f"{DIR_NAME}/OC_TABNET_ENSEMBLE_CROSSENTROPYLOSS_yeast_6_{suffix}")
