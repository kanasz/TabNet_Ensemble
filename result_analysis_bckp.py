import numpy as np
import pandas as pd
from imbalanced_ensemble.metrics import geometric_mean_score
from sklearn.metrics import roc_auc_score
import os

def process_results(path):
    # Open the file in read mode
    with open(path, 'r') as file:
        # Read the content of the file
        file_content = file.read()

    file_content = file_content.replace(" ","").replace("\n","")
    file_content = (file_content.replace("array(","")
                    .replace(")","")
                    .replace(",dtype=int64", "")
                    .replace("dtype=int64,","")
                    .replace(",dtype=object","")
                    )
    # Evaluate the content to convert it to a dictionary
    data = eval(file_content)

    gmeans = []
    roc_aucs = []
    if data['true_values']==None:
        return "0"
    for idx, true_values in enumerate(data['true_values']):
        predicions = data['predicted_values'][idx]
        gmean = geometric_mean_score(true_values, predicions)
        roc_auc = roc_auc_score(true_values, predicions)
        roc_aucs.append(roc_auc)
        gmeans.append(gmean)
    #print("\t{}".format(np.average(gmeans)) )
    #print("gmean: {:.2f}±{}, auc: {:.2f}±{}".format(np.average(gmeans)*100,int(round(np.std(gmeans)*100)),
    #                                                np.average(roc_aucs)*100,int(round(np.std(roc_aucs)*100))))

    return "{}".format(np.average(gmeans))

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

#print("Taiwan - boosting TabNet:")
#process_results("predictions/wine/results/OC_TABNET_ENSEMBLE_CROSSENTROPYLOSS_red_3_vs_5_CLUSTER_COUNT_800_CLASSIFIER_COUNT_35_SYNTH_COUNT_1500_2.txt")

classifiers = ['XGBOOST','XGBOOST_SMOTE', 'XGBOOST_WEIGHTED',
                'SVC_SMOTE','SVC_WEIGHTED','BCascade',
               'OC_TABNET_ENSEMBLE_CROSSENTROPYLOSS'
               ]
#classifiers = ['BCascade', 'OC_TABNET_ENSEMBLE_CROSSENTROPYLOSS']

#datasets = ['red_3_vs_5','red_8_vs_6','white_3_vs_7','white_9_vs_4']
#folder_path = "predictions/wine/results"

datasets = ['abalone_3_vs_11','abalone_9_vs_18','abalone_19_vs_10_11_12_13','abalone_20_vs_8_9_10']
folder_path = "predictions/abalone/results"

classifiers = ["AdasynSVC"]
datasets = ['ecoli_0_vs_1','ecoli_0_4_6_vs_5','ecoli_0_3_4_vs_5','ecoli_0_2_3_4_vs_5']
folder_path = "predictions/ecoli/results"


#datasets = ['glass_2','glass_4','glass_5','glass_0_1_6_vs_5']
#folder_path = "predictions/glass/results"

#datasets = ['yeast_3','yeast_4','yeast_5','yeast_6']
#folder_path = "predictions/yeast/results"
'''
for dataset in datasets:
    print(dataset)
    for classifier in classifiers:
        file = str.upper("{}_{}".format(classifier, dataset))
        for file_name in os.listdir(folder_path):
            # Check if the file is a .txt file
            if file_name.endswith('.txt'):
                # Construct the full file path
                #file_path = os.path.join(folder_path, file_name)
                if file in str.upper(file_name):
                    file_path = os.path.join(folder_path, file_name)
                    try:
                        print("\t{}".format( classifier))
                        process_results(file_path)
                    except:
                        print(file_path)
                        process_results(file_path)
'''
classifiers = ['XGBOOST','XGBOOST_SMOTE',
                'SVC_SMOTE','SVC_WEIGHTED','AdasynSVC','Adacost','SelfPaced', 'TabNetSmote',
               'OC_TABNET_ENSEMBLE_CROSSENTROPYLOSS',
               'OC_TABNET_ENSEMBLE_ADASYN_KMEANS',
               'OC_TABNET_ENSEMBLE_SMOTE_GMM',
               'OC_TABNET_ENSEMBLE_SMOTE_MEANSHIFT',
                'OC_TABNET_ENSEMBLE_ADASYN_MEANSHIFT',
                'UNCLUSTERED_OC_TABNET_ENSEMBLE_SMOTE_MEANSHIFT',
                'UNCLUSTERED_OC_TABNET_ENSEMBLE_ADASYN_MEANSHIFT',
                'UNCLUSTERED_OC_TABNET_ENSEMBLE_SMOTE_DBSCAN'
               ]


folder_path = "predictions/{}/results"
results_folders = {
    "synthetic":['synthetic_01', 'synthetic_02', 'synthetic_03'],
    "abalone":['abalone_3_vs_11','abalone_9_vs_18','abalone_19_vs_10_11_12_13','abalone_20_vs_8_9_10'],
    "wine":['white_9_vs_4', 'white_3_vs_7','red_8_vs_6','red_3_vs_5'],
    "ecoli":['ecoli_0_vs_1','ecoli_0_4_6_vs_5','ecoli_0_3_4_vs_5','ecoli_0_2_3_4_vs_5'],
    "glass": ['glass_2', 'glass_4', 'glass_5', 'glass_0_1_6_vs_5'],
    'yeast':['yeast_3','yeast_4','yeast_5','yeast_6']
}

df = pd.DataFrame(columns=['dataset'] + classifiers)

for folder in results_folders.keys():
    for dataset in results_folders[folder]:
        row = {
            "dataset":[dataset]
        }
        results_path  = folder_path.format(folder)
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
df.to_csv("aggregated.csv", index=None)




