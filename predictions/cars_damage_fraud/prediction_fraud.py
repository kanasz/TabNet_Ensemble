from base_functions import get_fraudulent_claim_on_cars_physical_damage_data
import time

from constants import LossFunction
from optimization.ga_boosting_fraud_tabnet_tuner import GaBoostingFraudTabnetTuner
from optimization.ga_boosting_tabnet_tuner import GaBoostingTabnetTuner

if __name__ == '__main__':
    tabnet_max_epochs = 50
    num_generations = 30
    num_parents = 20  # 10
    population = 40  # 20
    start_time = time.time()
    data = get_fraudulent_claim_on_cars_physical_damage_data()



    #columns_to_scale = ['age_of_driver', 'safty_rating', 'annual_income', 'liab_prct', 'claim_est_payout', 'age_of_vehicle','vehicle_price','vehicle_weight']
    columns_to_scale = [0]
    actual_loss_function = LossFunction.CROSSENTROPYLOSS
    tuner = GaBoostingFraudTabnetTuner(tabnet_max_epochs, num_generations, num_parents, population, device='cuda', use_smote=False)
    tuner.run_experiment(data,
                         'results/BOOSTING_CROSSENTROPYLOSS_2_fraud'
                         , actual_loss_function)
    print("--- total: %s seconds ---" % (time.time() - start_time))
    print("Experiment info -> data: FRAUD, loss function: {}".format(actual_loss_function))
