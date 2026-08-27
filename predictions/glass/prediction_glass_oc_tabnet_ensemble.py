import time
import numpy as np

from base_functions import get_glass_2_data, get_glass_4_data, get_glass_5_data, get_glass_0_1_6_vs_5_data
from base_functions import get_config_files, set_seed
from constants import LossFunction, CLUSTER_COUNT, WEAK_CLASSIFIERS_COUNT, SYNTHETIC_MINORITY_COUNT
from models.oc_bagging_tabnet_ensemble_parallel import GaOCBaggingTabnetEnsembleTunerParallel

set_seed(42)
np.set_printoptions(threshold=np.inf)
DIR_NAME = 'results/oc_tabnet_ensemble'

actual_loss_function = LossFunction.CROSSENTROPYLOSS


def run_experiment(glass_data, file_name):
    tabnet_max_epochs = 50
    num_generations = 50
    num_parents = 20
    population = 50
    start_time = time.time()
    data = glass_data
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

    # glass_2
    run_experiment(glass_data=get_glass_2_data(),
                   file_name=f"{DIR_NAME}/OC_TABNET_ENSEMBLE_CROSSENTROPYLOSS_glass_2_{suffix}")

    # glass_4
    run_experiment(glass_data=get_glass_4_data(),
                   file_name=f"{DIR_NAME}/OC_TABNET_ENSEMBLE_CROSSENTROPYLOSS_glass_4_{suffix}")

    # glass_5
    run_experiment(glass_data=get_glass_5_data(),
                   file_name=f"{DIR_NAME}/OC_TABNET_ENSEMBLE_CROSSENTROPYLOSS_glass_5_{suffix}")

    # glass_0_1_6_vs_5
    run_experiment(glass_data=get_glass_0_1_6_vs_5_data(),
                   file_name=f"{DIR_NAME}/OC_TABNET_ENSEMBLE_CROSSENTROPYLOSS_glass_0_1_6_vs_5_{suffix}")