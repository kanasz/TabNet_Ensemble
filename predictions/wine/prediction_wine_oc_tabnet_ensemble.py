import time
import numpy as np

from base_functions import get_wine_quality_white_3_vs_7_data, get_wine_quality_white_9_vs_4_data
from base_functions import get_wine_quality_red_8_vs_6_data, get_wine_quality_red_3_vs_5_data
from base_functions import get_config_files, set_seed
from constants import LossFunction, CLUSTER_COUNT, WEAK_CLASSIFIERS_COUNT, SYNTHETIC_MINORITY_COUNT
from models.oc_bagging_tabnet_ensemble_parallel import GaOCBaggingTabnetEnsembleTunerParallel

set_seed(42)
np.set_printoptions(threshold=np.inf)
DIR_NAME = 'results/oc_tabnet_ensemble'

actual_loss_function = LossFunction.CROSSENTROPYLOSS


def run_experiment(wine_data, file_name):
    tabnet_max_epochs = 50
    num_generations = 50
    num_parents = 20
    population = 50
    start_time = time.time()
    data = wine_data
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

    # wine_white_3_vs_7
    run_experiment(wine_data=get_wine_quality_white_3_vs_7_data(),
                   file_name=f"{DIR_NAME}/OC_TABNET_ENSEMBLE_CROSSENTROPYLOSS_wine_white_3_vs_7_{suffix}")

    # wine_white_9_vs_4
    run_experiment(wine_data=get_wine_quality_white_9_vs_4_data(),
                   file_name=f"{DIR_NAME}/OC_TABNET_ENSEMBLE_CROSSENTROPYLOSS_wine_white_9_vs_4_{suffix}")

    # wine_red_8_vs_6
    run_experiment(wine_data=get_wine_quality_red_8_vs_6_data(),
                   file_name=f"{DIR_NAME}/OC_TABNET_ENSEMBLE_CROSSENTROPYLOSS_wine_red_8_vs_6_{suffix}")

    # wine_red_3_vs_5
    run_experiment(wine_data=get_wine_quality_red_3_vs_5_data(),
                   file_name=f"{DIR_NAME}/OC_TABNET_ENSEMBLE_CROSSENTROPYLOSS_wine_red_3_vs_5_{suffix}")