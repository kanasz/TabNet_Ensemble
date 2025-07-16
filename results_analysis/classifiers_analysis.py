import numpy as np
import pandas as pd
import pygad
from imbalanced_ensemble.metrics import geometric_mean_score
from sklearn.metrics import roc_auc_score
import os

from constants import WEAK_CLASSIFIERS_COUNT

def split_array(selected_samples, samples_counts):
    subarrays = []
    start = 0
    for count in samples_counts:
        subarrays.append(selected_samples[start:start + count])
        start += count
    return subarrays

base_folder = '../predictions/{}/results'
model_name = 'UNCLUSTERED_OC_TABNET_ENSEMBLE_SMOTE_MEANSHIFT_{}'
datasets = [
            #('abalone','abalone_9_vs_18'),
            #('abalone','abalone_19_vs_10_11_12_13'),
            #('abalone','abalone_20_vs_8_9_10'),
            ('ecoli','ecoli_0_2_3_4_vs_5'),
            ('ecoli','ecoli_0_3_4_vs_5'),
            ('ecoli','ecoli_0_4_6_vs_5'),
            ('ecoli','ecoli_0_vs_1'),
            #('glass','glass_0_1_6_vs_5'),
            #('glass','glass_2'),
            #('glass', 'glass_4'),
            #('glass','glass_5'),
            #('wine','red_3_vs_5'),
            #('wine','red_8_vs_6'),
            #('wine','white_3_vs_7'),
            #('wine','white_9_vs_4'),
            #('yeast','yeast_3'),
            #('yeast','yeast_4'),
            #('yeast','yeast_5'),
            #('yeast','yeast_6')
]
#datasets = [('ecoli','ecoli_0_vs_1')]
#datasets = [('glass','glass_5')]
#datasets = [('wine','red_8_vs_6')]
#datasets = [('yeast','yeast_3')]

def process_ga(path):
    ga_instance = pygad.load(path)
    best_solution = ga_instance.best_solutions[-1]
    best_fitness = ga_instance.best_solutions_fitness[-1]
    return best_solution, best_fitness
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
    return "{}".format(np.average(gmeans))
all_counts = None
all_counts_unaggregated = None
for ds in datasets:
    print(ds)
    ds_base_folder = base_folder.format(ds[0])
    array = None
    selected_samples = None

    file_name = model_name.format(ds[1])
    file_path = "{}/{}".format(ds_base_folder, file_name)
    results = process_results(file_path+".txt")
    best_solution, best_fitness = process_ga(file_path)
    clf_count = np.sum(best_solution[0:WEAK_CLASSIFIERS_COUNT])
    samples_count = np.sum(best_solution[WEAK_CLASSIFIERS_COUNT:])
    if array is None:
        array =best_solution[0:WEAK_CLASSIFIERS_COUNT]
    else:
        array = np.vstack((array, best_solution[0:WEAK_CLASSIFIERS_COUNT]))

    print("{} {} / {}".format(ds[1], results, clf_count))



    clf_sum = (np.sum(array, axis=0))
    if all_counts_unaggregated is None:
        all_counts_unaggregated = array
    else:
        all_counts_unaggregated = np.vstack((all_counts_unaggregated, array))
    if all_counts is None:
        all_counts = clf_sum
    else:
        all_counts = np.vstack((all_counts, clf_sum))


    print("---------------------------------------------")
print(all_counts)

column_names = ['Column1', 'Column2', 'Column3', 'Column4']

# Create a pandas DataFrame
df = pd.DataFrame(all_counts)
df_unaggregated = pd.DataFrame(all_counts_unaggregated)

print(df)
print(df_unaggregated)
df.to_csv("counts_real.csv")
df_unaggregated.to_csv("unaggregated_real.csv")