from base_functions import get_fraudulent_claim_on_cars_physical_damage_data, get_aids_data
import time

from constants import LossFunction
from optimization.ga_boosting_aids_tabnet_tuner import GaBoostingAidsTabnetTuner
from optimization.ga_oc_bagging_tabnet_ensemble_tuner import GaOCBaggingTabnetEnsembleTuner

if __name__ == '__main__':
    categorical_cols = ['trt', 'hemo', 'homo', 'drugs', 'oprior', 'z30', 'race', 'gender', 'str2', 'strat', 'symptom',
                        'treat', 'offtrt']
    numerical_cols = ['time', 'age', 'wtkg', 'karnof', 'preanti', 'cd40', 'cd420', 'cd80', 'cd820']

    tabnet_max_epochs = 50
    num_generations = 1
    num_parents = 20  # 10
    population = 50  # 20
    start_time = time.time()
    samples = 2139
    data = get_aids_data(samples)
    actual_loss_function = LossFunction.CROSSENTROPYLOSS
    tuner = GaOCBaggingTabnetEnsembleTuner(tabnet_max_epochs, num_generations, num_parents, population, device='cuda')
    tuner.run_experiment(data, 'results/OC_TABNET_ENSEMBLE_CROSSENTROPYLOSS_AIDS_{}_samples_{}_epochs'.format(samples, tabnet_max_epochs),
                         actual_loss_function)
    print("--- total: %s seconds ---" % (time.time() - start_time))
    print("Experiment info -> data: {}, samples: {}, loss function: {}".format('AIDS', samples, actual_loss_function))


