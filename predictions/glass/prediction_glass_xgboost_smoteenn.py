import time
import numpy as np

from base_functions import get_glass_2_data, get_glass_4_data, get_glass_5_data, get_glass_0_1_6_vs_5_data
from base_functions import set_seed
from optimization.ga_xgboost_tuner import GaXGBoostTuner

set_seed(seed=42)
np.set_printoptions(threshold=np.inf)
DIR_NAME = 'results/xgboost_smoteenn'


def run_experiment(glass_data, file_name):
    num_generations = 50
    num_parents = 20
    population = 50
    start_time = time.time()
    data = glass_data
    numerical_cols = list(data[0].columns.values)

    print(f"Starting simulation run...")
    tuner = GaXGBoostTuner(num_generations=num_generations,
                           num_parents=num_parents,
                           population=population,
                           use_smote=False,
                           use_adasyn=False,
                           use_smoteenn=True,
                           numerical_cols=numerical_cols
                           )
    tuner.run_experiment(data, file_name)
    print("--- total: %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':

    # glass_2
    run_experiment(glass_data=get_glass_2_data(),
                   file_name=f"{DIR_NAME}/xgboost_smoteenn_glass_2")

    # glass_4
    run_experiment(glass_data=get_glass_4_data(),
                   file_name=f"{DIR_NAME}/xgboost_smoteenn_glass_4")

    # glass_5
    run_experiment(glass_data=get_glass_5_data(),
                   file_name=f"{DIR_NAME}/xgboost_smoteenn_glass_5")

    # glass_0_1_6_vs_5
    run_experiment(glass_data=get_glass_0_1_6_vs_5_data(),
                   file_name=f"{DIR_NAME}/xgboost_smoteenn_glass_0_1_6_vs_5")
