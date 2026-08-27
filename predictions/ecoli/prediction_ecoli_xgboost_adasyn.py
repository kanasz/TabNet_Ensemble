import time
import numpy as np

from base_functions import get_ecoli_0_vs_1_data, get_ecoli_0_4_6_vs_5_data, get_ecoli_0_3_4_vs_5_data
from base_functions import get_ecoli_0_2_3_4_vs_5_data, set_seed
from optimization.ga_xgboost_tuner import GaXGBoostTuner

set_seed(42)
np.set_printoptions(threshold=np.inf)
DIR_NAME = 'results/xgboost_adasyn'


def run_experiment(ecoli_data, file_name):
    num_generations = 50
    num_parents = 20
    population = 50
    start_time = time.time()
    data = ecoli_data
    numerical_cols = list(data[0].columns.values)

    print(f"Starting simulation run...")
    tuner = GaXGBoostTuner(num_generations=num_generations,
                           num_parents=num_parents,
                           population=population,
                           use_smote=False,
                           use_adasyn=True,
                           use_smoteenn=False,
                           numerical_cols=numerical_cols)
    tuner.run_experiment(data, file_name)
    print("--- total: %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':

    # ecoli_0_vs_1
    run_experiment(ecoli_data=get_ecoli_0_vs_1_data(),
                   file_name=f"{DIR_NAME}/xgboost_adasyn_ecoli_0_vs_1")

    # ecoli_0_4_6_vs_5
    run_experiment(ecoli_data=get_ecoli_0_4_6_vs_5_data(),
                   file_name=f"{DIR_NAME}/xgboost_adasyn_ecoli_0_4_6_vs_5")

    # ecoli_0_3_4_vs_5
    run_experiment(ecoli_data=get_ecoli_0_3_4_vs_5_data(),
                   file_name=f"{DIR_NAME}/xgboost_adasyn_ecoli_0_3_4_vs_5")

    # ecoli_0_2_3_4_vs_5
    run_experiment(ecoli_data=get_ecoli_0_2_3_4_vs_5_data(),
                   file_name=f"{DIR_NAME}/xgboost_adasyn_ecoli_0_2_3_4_vs_5")
