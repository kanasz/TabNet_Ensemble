import time

from base_functions import get_abalone_9_vs_18_data, get_abalone_3_vs_11_data, get_abalone_19_vs_10_11_12_13_data, \
    get_abalone_20_vs_8_9_10_data, get_wine_quality_red_3_vs_5_data, get_wine_quality_red_8_vs_6_data, \
    get_wine_quality_white_3_vs_7_data, get_wine_quality_white_9_vs_4_data, get_sensitivity_synthetic_data
from optimization.ga_xgboost_tuner import GaXGBoostTuner

features, labels = get_abalone_9_vs_18_data()


if __name__ == '__main__':

    numerical_cols = None#['Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight']
    categorical_cols = None

    num_generations = 50
    num_parents = 20  # 10
    population = 50  # 20
    start_time = time.time()

    contamination = '0.05'
    features = 20
    samples = 250
    id = '01'

    data = get_sensitivity_synthetic_data(id, contamination, features, samples)

    numerical_cols = list(data[0].columns.values)
    tuner = GaXGBoostTuner(num_generations, num_parents, population, True, False, numerical_cols, categorical_cols)
    tuner.run_experiment(data, 'results/XGBOOST_SMOTE_Synthetic_04')

    '''
    start_time = time.time()
    data = get_wine_quality_red_3_vs_5_data()
    numerical_cols = list(data[0].columns.values)
    tuner = GaXGBoostTuner(num_generations, num_parents, population, False, False,numerical_cols, categorical_cols)
    tuner.run_experiment(data, 'results/XGBOOST_Wine_Red_3_vs_5')
    print("--- total: %s seconds ---" % (time.time() - start_time))
    

    start_time = time.time()
    data = get_wine_quality_red_3_vs_5_data()
    numerical_cols = list(data[0].columns.values)
    tuner = GaXGBoostTuner(num_generations, num_parents, population, True, False, numerical_cols, categorical_cols)
    tuner.run_experiment(data, 'results/XGBOOST_SMOTE_Red_3_vs_5')
    print("--- total: %s seconds ---" % (time.time() - start_time))
    
    start_time = time.time()
    data = get_wine_quality_red_3_vs_5_data()
    numerical_cols = list(data[0].columns.values)
    tuner = GaXGBoostTuner(num_generations, num_parents, population, False, True, numerical_cols, categorical_cols)
    tuner.run_experiment(data, 'results/XGBOOST_WEIGHTED_Red_3_vs_5')
    print("--- total: %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    data = get_wine_quality_red_8_vs_6_data()
    numerical_cols = list(data[0].columns.values)
    tuner = GaXGBoostTuner(num_generations, num_parents, population, False, False, numerical_cols, categorical_cols)
    tuner.run_experiment(data, 'results/XGBOOST_Wine_Red_8_vs_6')
    print("--- total: %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    data = get_wine_quality_red_8_vs_6_data()
    numerical_cols = list(data[0].columns.values)
    tuner = GaXGBoostTuner(num_generations, num_parents, population, True, False, numerical_cols, categorical_cols)
    tuner.run_experiment(data, 'results/XGBOOST_SMOTE_Red_8_vs_6')
    print("--- total: %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    data = get_wine_quality_red_8_vs_6_data()
    numerical_cols = list(data[0].columns.values)
    tuner = GaXGBoostTuner(num_generations, num_parents, population, False, True, numerical_cols, categorical_cols)
    tuner.run_experiment(data, 'results/XGBOOST_WEIGHTED_Red_8_vs_6')
    print("--- total: %s seconds ---" % (time.time() - start_time))
    

    start_time = time.time()
    data = get_wine_quality_white_3_vs_7_data()
    numerical_cols = list(data[0].columns.values)
    tuner = GaXGBoostTuner(num_generations, num_parents, population, False, False, numerical_cols, categorical_cols)
    tuner.run_experiment(data, 'results/XGBOOST_Wine_White_3_vs_7')
    print("--- total: %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    data = get_wine_quality_white_3_vs_7_data()
    numerical_cols = list(data[0].columns.values)
    tuner = GaXGBoostTuner(num_generations, num_parents, population, True, False, numerical_cols, categorical_cols)
    tuner.run_experiment(data, 'results/XGBOOST_SMOTE_White_3_vs_7')
    print("--- total: %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    data = get_wine_quality_white_3_vs_7_data()
    numerical_cols = list(data[0].columns.values)
    tuner = GaXGBoostTuner(num_generations, num_parents, population, False, True, numerical_cols, categorical_cols)
    tuner.run_experiment(data, 'results/XGBOOST_WEIGHTED_White_3_vs_7')
    print("--- total: %s seconds ---" % (time.time() - start_time))




    start_time = time.time()
    data = get_wine_quality_white_9_vs_4_data()
    numerical_cols = list(data[0].columns.values)
    tuner = GaXGBoostTuner(num_generations, num_parents, population, False, False, numerical_cols, categorical_cols)
    tuner.run_experiment(data, 'results/XGBOOST_Wine_White_9_vs_4')
    print("--- total: %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    data = get_wine_quality_white_9_vs_4_data()
    numerical_cols = list(data[0].columns.values)
    tuner = GaXGBoostTuner(num_generations, num_parents, population, True, False, numerical_cols, categorical_cols)
    tuner.run_experiment(data, 'results/XGBOOST_SMOTE_White_9_vs_4')
    print("--- total: %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    data = get_wine_quality_white_9_vs_4_data()
    numerical_cols = list(data[0].columns.values)
    tuner = GaXGBoostTuner(num_generations, num_parents, population, False, True, numerical_cols, categorical_cols)
    tuner.run_experiment(data, 'results/XGBOOST_WEIGHTED_White_9_vs_4')
    print("--- total: %s seconds ---" % (time.time() - start_time))
    '''



