import os
import math
import numpy as np
import pandas as pd

from imbalanced_ensemble.metrics import geometric_mean_score
from sklearn.metrics import roc_auc_score


def process_results(path):
    with open(path, 'r') as _file:
        file_content = _file.read()

    file_content = file_content.replace(" ", "").replace("\n", "")
    file_content = (file_content.replace("array(", "")
                    .replace(")", "")
                    .replace(",dtype=int64", "")
                    .replace("dtype=int64,", "")
                    .replace(",dtype=object", "")
                    )
    # Evaluate the content to convert it to a dictionary
    data = eval(file_content)

    _geometric_means_scores = []
    _roc_auc_scores = []
    if data['true_values'] is None:
        return "0"
    for idx, true_values in enumerate(data['true_values']):
        _predicted_values = data['predicted_values'][idx]
        _g_mean_ = geometric_mean_score(true_values, _predicted_values)
        roc_auc = roc_auc_score(true_values, _predicted_values)
        _roc_auc_scores.append(roc_auc)
        _geometric_means_scores.append(_g_mean_)

    # prints for CIFAR 2025 conference
    print("GM score: {}±{}".format(math.ceil((np.average(_geometric_means_scores) * 100) * 10) / 10,
                                   math.ceil(np.std(_geometric_means_scores) * 100)))
    print("AUC score: {}±{}".format(math.ceil((np.average(_roc_auc_scores) * 100) * 10) / 10,
                                    math.ceil(np.std(_roc_auc_scores) * 100)))
    print("#####################")

    return "${:.1f}_{{{}}}$".format(math.ceil((np.average(_geometric_means_scores) * 100) * 10) / 10,
                                    math.ceil(np.std(_geometric_means_scores) * 100))
    # AUC scores (if needed)
    # return "${:.1f}_{{{}}}$".format(math.ceil((np.average(_roc_auc_scores) * 100) * 10) / 10,
    #                                math.ceil(np.std(_roc_auc_scores) * 100))


# This method summarizes all results required for CIFAR 2025 in Trondheim
def print_results_for_cifar_2025_conference():
    print("Taiwan: ")
    process_results("predictions/taiwan/results/BOOSTING_TABNET_CROSS_ENTROPY_LOSS_2.txt")
    print("Poland (5): ")
    process_results("predictions/polish/results/BOOSTING_TABNET_CROSS_ENTROPY_LOSS_2.txt")
    print("Slovakia (2013 + 3 years)")
    process_results("predictions/slovakia/results"
                    "/BOOSTING_CROSSENTROPYLOSS_3_construction_13_10_11_12_features_50_epochs_50_population.txt")
    print("Slovakia (2014 + 3 years)")
    process_results("predictions/slovakia/results/BOOSTING_CROSSENTROPYLOSS_2_construction_14_11_12_13.txt")
    print("Slovakia (2016 + 3 years)")
    process_results("predictions/slovakia/results/BOOSTING_CROSSENTROPYLOSS_2_construction_16_13_14_15.txt")
    print("Synthetic-01")
    process_results("predictions/synthetic/results_200_samples/CROSSENTROPYLOSS_01_synthetic_0"
                    ".3_contamination_50_features_200_epochs_50_population_50.txt")
    print("Synthetic-02")
    process_results("predictions/synthetic/results_200_samples/CROSSENTROPYLOSS_02_synthetic_0"
                    ".1_contamination_100_features_200_epochs_50_population_50_samples_200.txt")


#######################
# new paper from here #
#######################

# oversampling clustering - OC
# OC_TABNET_ENSEMBLE_CROSSENTROPYLOSS -> SMOTE + K-MEAN
# classifiers = ['XGBOOST', 'XGBOOST_SMOTE', 'SVC_SMOTE', 'SVC_WEIGHTED', 'AdasynSVC', 'Adacost', 'SelfPaced',
#               'TabNetSmote', 'OC_TABNET_ENSEMBLE_CROSSENTROPYLOSS', 'OC_TABNET_ENSEMBLE_ADASYN_MEANSHIFT',
#               'OC_TABNET_ENSEMBLE_SMOTE_MEANSHIFT'
#               ]

# updated order for paper
"""
classifiers = ['SVC_WEIGHTED',  'SVC_SMOTE', 'AdasynSVC', 'XGBOOST_SMOTE', 'Adacost', 'SelfPaced',
               'TabNetSmote', 'OC_TABNET_ENSEMBLE_SMOTE_MEANSHIFT']
"""
# ablation study
classifiers = ['OC_TABNET_ENSEMBLE_SMOTE_MEANSHIFT',
               'OC_TABNET_ENSEMBLE_ADASYN_MEANSHIFT',
               'OC_TABNET_ENSEMBLE_CROSSENTROPYLOSS',
               ]

folder_path = "predictions/{}/results"
results_folders = {
    "synthetic": ['synthetic_01', 'synthetic_02', 'synthetic_03'],
    # "abalone": ['abalone_3_vs_11', 'abalone_9_vs_18', 'abalone_19_vs_10_11_12_13', 'abalone_20_vs_8_9_10'],
    # "wine": ['white_9_vs_4', 'white_3_vs_7', 'red_8_vs_6', 'red_3_vs_5'],
    # "ecoli": ['ecoli_0_vs_1', 'ecoli_0_4_6_vs_5', 'ecoli_0_3_4_vs_5', 'ecoli_0_2_3_4_vs_5'],
    # "glass": ['glass_2', 'glass_4', 'glass_5', 'glass_0_1_6_vs_5'],
    # 'yeast': ['yeast_3', 'yeast_4', 'yeast_5', 'yeast_6']
}

df = pd.DataFrame(columns=['dataset'] + classifiers)

for folder in results_folders.keys():
    for dataset in results_folders[folder]:
        row = {
            "dataset": [dataset]
        }
        results_path = folder_path.format(folder)
        for classifier in classifiers:

            file = str.upper("{}_{}".format(classifier, dataset))
            for file_name in os.listdir(results_path):
                if file_name.endswith('.txt'):
                    if file in str.upper(file_name):
                        file_path = os.path.join(results_path, file_name)
                        if os.path.exists(file_path):
                            row[classifier] = [process_results(file_path)]
        new_row = pd.DataFrame(row)
        df = pd.concat([df, new_row], ignore_index=True)

print(df)
df.to_csv("./results_aggregation/aggregated_synthetic_data_gmeans_ablation_study.csv", index=None)
