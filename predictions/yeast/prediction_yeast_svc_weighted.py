import time
import numpy as np

from base_functions import get_yeast_3_data, get_yeast_4_data, get_yeast_5_data, get_yeast_6_data
from base_functions import set_seed
from constants import Classifier
from optimization.ga_tuner import GaTuner

set_seed(seed=42)
np.set_printoptions(threshold=np.inf)
DIR_NAME = 'results/svc_weighted'


def run_experiment(yeast_data, file_name):
    num_generations = 50
    num_parents = 20
    population = 50
    start_time = time.time()
    data = yeast_data
    numerical_cols = list(data[0].columns.values)

    print(f"Starting simulation run...")
    tuner = GaTuner(num_generations=num_generations,
                    num_parents=num_parents,
                    population=population,
                    use_smote=False,
                    numerical_cols=numerical_cols,
                    clf_type=Classifier.WeightedSVC)
    tuner.run_experiment(data, file_name)
    print("--- total: %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':

    # yeast_3
    run_experiment(yeast_data=get_yeast_3_data(),
                   file_name=f"{DIR_NAME}/svc_weighted_yeast_3")

    # yeast_4
    run_experiment(yeast_data=get_yeast_4_data(),
                   file_name=f"{DIR_NAME}/svc_weighted_yeast_4")

    # yeast_5
    run_experiment(yeast_data=get_yeast_5_data(),
                   file_name=f"{DIR_NAME}/svc_weighted_yeast_5")

    # yeast_6
    run_experiment(yeast_data=get_yeast_6_data(),
                   file_name=f"{DIR_NAME}/svc_weighted_yeast_6")
