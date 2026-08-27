import time
import numpy as np

from base_functions import get_ecoli_0_vs_1_data, get_ecoli_0_4_6_vs_5_data, get_ecoli_0_3_4_vs_5_data
from base_functions import get_ecoli_0_2_3_4_vs_5_data, set_seed
from optimization.ga_tuner import GaTuner
from base_functions import Classifier

set_seed(seed=42)
np.set_printoptions(threshold=np.inf)
DIR_NAME = "results/svc_smoteenn"


def run_experiment(ecoli_data, file_name):
    num_generations = 50
    num_parents = 20
    population = 50
    start_time = time.time()
    data = ecoli_data
    numerical_cols = list(data[0].columns.values)

    print(f"Starting simulation run...")
    tuner = GaTuner(num_generations=num_generations,
                    num_parents=num_parents,
                    population=population,
                    use_smote=False,
                    use_adasyn=False,
                    use_smote_enn=True,
                    numerical_cols=numerical_cols,
                    clf_type=Classifier.SVC)
    tuner.run_experiment(data, file_name)
    print("--- total: %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    # ecoli_0_vs_1
    run_experiment(ecoli_data=get_ecoli_0_vs_1_data(),
                   file_name=f"{DIR_NAME}/svc_smoteenn_ecoli_0_vs_1")
    
    # ecoli_0_4_6_vs_5
    run_experiment(ecoli_data=get_ecoli_0_4_6_vs_5_data(),
                   file_name=f"{DIR_NAME}/svc_smoteenn_ecoli_0_4_6_vs_5")
    
    # ecoli_0_3_4_vs_5
    run_experiment(ecoli_data=get_ecoli_0_3_4_vs_5_data(),
                   file_name=f"{DIR_NAME}/svc_smoteenn_ecoli_0_3_4_vs_5")

    # ecoli_0_2_3_4_vs_5
    run_experiment(ecoli_data=get_ecoli_0_2_3_4_vs_5_data(),
                   file_name=f"{DIR_NAME}/svc_smoteenn_ecoli_0_2_3_4_vs_5")
