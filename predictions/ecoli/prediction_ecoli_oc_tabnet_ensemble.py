import time
import numpy as np

from base_functions import get_ecoli_0_vs_1_data, get_ecoli_0_4_6_vs_5_data, get_ecoli_0_2_3_4_vs_5_data
from base_functions import get_ecoli_0_3_4_vs_5_data, get_config_files, set_seed
from constants import LossFunction, CLUSTER_COUNT, WEAK_CLASSIFIERS_COUNT, SYNTHETIC_MINORITY_COUNT
from models.oc_bagging_tabnet_ensemble_parallel import GaOCBaggingTabnetEnsembleTunerParallel

set_seed(42)
np.set_printoptions(threshold=np.inf)
DIR_NAME = 'results/oc_tabnet_ensemble'

actual_loss_function = LossFunction.CROSSENTROPYLOSS


def run_experiment(ecoli_data, file_name):
    tabnet_max_epochs = 50
    num_generations = 50
    num_parents = 20
    population = 50
    start_time = time.time()
    data = ecoli_data
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

    # ecoli_0_vs_1
    run_experiment(ecoli_data=get_ecoli_0_vs_1_data(),
                   file_name=f"{DIR_NAME}/OC_TABNET_ENSEMBLE_CROSSENTROPYLOSS_ecoli_0_vs_1_{suffix}")

    # ecoli_0_2_3_4_vs_5
    run_experiment(ecoli_data=get_ecoli_0_2_3_4_vs_5_data(),
                   file_name=f"{DIR_NAME}/OC_TABNET_ENSEMBLE_CROSSENTROPYLOSS_ecoli_0_2_3_4_vs_5_{suffix}")

    # ecoli_0_3_4_vs_5
    run_experiment(ecoli_data=get_ecoli_0_3_4_vs_5_data(),
                   file_name=f"{DIR_NAME}/OC_TABNET_ENSEMBLE_CROSSENTROPYLOSS_ecoli_0_3_4_vs_5_{suffix}")

    # ecoli_0_4_6_vs_5
    run_experiment(ecoli_data=get_ecoli_0_4_6_vs_5_data(),
                   file_name=f"{DIR_NAME}/OC_TABNET_ENSEMBLE_CROSSENTROPYLOSS_ecoli_0_4_6_vs_5_{suffix}")