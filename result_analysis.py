import numpy as np
from imbalanced_ensemble.metrics import geometric_mean_score
from sklearn.metrics import roc_auc_score


def process_results(path):
    # Open the file in read mode
    with open(path, 'r') as file:
        # Read the content of the file
        file_content = file.read()

    file_content = file_content.replace(" ","").replace("\n","")
    file_content = file_content.replace("array(","").replace(")","").replace(",dtype=int64", "").replace("dtype=int64,","")
    # Evaluate the content to convert it to a dictionary
    data = eval(file_content)

    gmeans = []
    roc_aucs = []
    for idx, true_values in enumerate(data['true_values']):
        predicions = data['predicted_values'][idx]
        gmean = geometric_mean_score(true_values, predicions)
        roc_auc = roc_auc_score(true_values, predicions)
        roc_aucs.append(roc_auc)
        gmeans.append(gmean)

    print("gmean: {:.2f}±{}, auc: {:.2f}±{}".format(np.average(gmeans)*100,int(round(np.std(gmeans)*100)),
                                                    np.average(roc_aucs)*100,int(round(np.std(roc_aucs)*100))))

"""
print("CONSTRUCTION 2013")
process_results("predictions/slovakia/results/BOOSTING_CROSSENTROPYLOSS_3_construction_13_10_11_12_features_50_epochs_50_population.txt")
print("CONSTRUCTION 2014")
process_results("predictions/slovakia/results/BOOSTING_CROSSENTROPYLOSS_2_construction_14_11_12_13.txt")
print("CONSTRUCTION 2016")
process_results("predictions/slovakia/results/BOOSTING_CROSSENTROPYLOSS_2_construction_16_13_14_15.txt")
print("FRAUD")
process_results("predictions/cars_damage_fraud/results/BOOSTING_CROSSENTROPYLOSS_4_fraud.txt")
print("AIDS")
process_results("predictions/aids_classification/results/CROSSENTROPYLOSS_AIDS_2_2139_samples_50_epochs.txt")
print("SYNTHETIC 01")
process_results("predictions/synthetic/results_200_samples/CROSSENTROPYLOSS_01_synthetic_0.3_contamination_50_features_200_epochs_50_population_50.txt")
print("SYNTHETIC 02")
process_results("predictions/synthetic/results_200_samples/CROSSENTROPYLOSS_02_synthetic_0.1_contamination_100_features_200_epochs_50_population_50_samples_200.txt")
print("SYNTHETIC 03")
process_results("predictions/synthetic/results_200_samples/CROSSENTROPYLOSS_synthetic_0.02_contamination_200_features_200_epochs_70_population_50.txt")
"""

print("Taiwan - boosting TabNet:")
process_results("predictions/taiwan/results/BOOSTING_TABNET_CROSS_ENTROPY_LOSS_2.txt")
