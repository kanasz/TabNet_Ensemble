import os
import math
import numpy as np
import pandas as pd
from imbalanced_ensemble.metrics import geometric_mean_score
from sklearn.metrics import roc_auc_score


def process_results(path):
    # Open the file in read mode
    with open(path, 'r') as _file:
        # Read the content of the file
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

    _geometric_mean_scores = []
    _auc_scores = []
    if data['true_values'] is None:
        return "0"
    for idx, _true_values in enumerate(data['true_values']):
        _predicted_values = data['predicted_values'][idx]
        _geometric_mean_scores.append(geometric_mean_score(_true_values, _predicted_values))
        _auc_scores.append(roc_auc_score(_true_values, _predicted_values))

    # geometric mean score and standard deviation
    _final_g_mean_score = math.ceil((np.average(_geometric_mean_scores) * 100) * 10) / 10
    _final_g_mean_std = math.ceil((np.std(_geometric_mean_scores) * 100))

    # auc score and standard deviation
    _final_auc_score = math.ceil((np.average(_auc_scores) * 100) * 10) / 10
    _final_auc_std = math.ceil((np.std(_auc_scores) * 100))

    print(f"GM score from .txt file: {_final_g_mean_score}±{_final_g_mean_std}")
    print(f"AUC score from .txt file: {_final_auc_score}±{_final_auc_std}\n")


    #print("\t{}".format(np.average(gmeans)) )
    #print("gmean: {:.2f}±{}, auc: {:.2f}±{}".format(np.average(gmeans)*100,int(round(np.std(gmeans)*100)),
    #                                                np.average(roc_aucs)*100,int(round(np.std(roc_aucs)*100))))
    # return f"{_final_g_mean_score}_{{{_final_g_mean_std}}}"
    # return f"{_final_auc_score}_{{{_final_auc_std}}}"
    return float(_final_auc_score)
    # return float(_final_g_mean_score)

#######################
# new paper from here #
#######################

# oversampling clustering - OC
# OC_TABNET_ENSEMBLE_CROSSENTROPYLOSS -> SMOTE + K-MEAN
# classifiers = ['XGBOOST', 'XGBOOST_SMOTE', 'SVC_SMOTE', 'SVC_WEIGHTED', 'AdasynSVC', 'Adacost', 'SelfPaced',
#               'TabNetSmote', 'OC_TABNET_ENSEMBLE_CROSSENTROPYLOSS', 'OC_TABNET_ENSEMBLE_ADASYN_MEANSHIFT',
#               'OC_TABNET_ENSEMBLE_SMOTE_MEANSHIFT'
#               ]


"""
'UNCLUSTERED_OC_TABNET_ENSEMBLE_SMOTE_MEANSHIFT', # tento ide do main tabulky
'UNCLUSTERED_OC_TABNET_ENSEMBLE_ADASYN_MEANSHIFT',
'UNCLUSTERED_OC_TABNET_ENSEMBLE_SMOTE_DBSCAN'
"""

# classifiers = ['SVC_WEIGHTED', 'SVC_SMOTE', 'AdasynSVC', 'XGBOOST_SMOTE', 'Adacost', 'SelfPaced', 'TabNetSmote',
#                'OC_TABNET_ENSEMBLE_SMOTE_MEANSHIFT']

"""
classifiers = ['SVC_WEIGHTED', 'SVC_SMOTE', 'AdasynSVC', 'XGBOOST_SMOTE', 'Adacost', 'SelfPaced', 'TabNetSmote',
               'UNCLUSTERED_OC_TABNET_ENSEMBLE_SMOTE_MEANSHIFT']
"""

classifiers = ['UNCLUSTERED_OC_TABNET_ENSEMBLE_SMOTE_MEANSHIFT',
               'UNCLUSTERED_OC_TABNET_ENSEMBLE_ADASYN_MEANSHIFT',
               'UNCLUSTERED_OC_TABNET_ENSEMBLE_SMOTE_DBSCAN']

folder_path = "predictions/{}/results"
results_folders = {
    # "synthetic": ['synthetic_01', 'synthetic_02', 'synthetic_04'],
    "abalone": ['abalone_3_vs_11', 'abalone_9_vs_18', 'abalone_19_vs_10_11_12_13', 'abalone_20_vs_8_9_10'],
    "wine": ['white_9_vs_4', 'white_3_vs_7', 'red_8_vs_6', 'red_3_vs_5'],
    "ecoli": ['ecoli_0_vs_1', 'ecoli_0_4_6_vs_5', 'ecoli_0_3_4_vs_5', 'ecoli_0_2_3_4_vs_5'],
    "glass": ['glass_2', 'glass_4', 'glass_5', 'glass_0_1_6_vs_5'],
    'yeast': ['yeast_3', 'yeast_4', 'yeast_5', 'yeast_6']
}


if __name__ == "__main__":
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
    df.to_csv("./visualizations/aggregated_ablation_real_data_auc_scores.csv")
